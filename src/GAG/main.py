# -*- coding: utf-8 -*-
"""
Created on 17/9/2019
@author: RuihongQiu
"""

import argparse
import logging
import time
from tqdm import tqdm
from model import GNNModel
from train import forward
from torch.utils.tensorboard import SummaryWriter
from se_data_process import load_data_valid, load_testdata
from reservoir import Reservoir
from sampling import *

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla', help='dataset name: gowalla/lastfm')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=200, help='hidden state size')  # SARA: embedding size
parser.add_argument('--epoch', type=int, default=4, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=1.0, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')  # SARA: this is fixed. This value is not used anywhere
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--u', type=int, default=1, help='the number of layer with u')
# SARA: res_size determines the storage requirement of the online update.
# In the paper: default (and best) value is |D|/100. Other values: {|D|/5, |D|/20, |D|/400}
# In the paper: when the reservoir size is set to |D|/100, our GAG model achieves the best performance.
parser.add_argument('--res_size', type=int, default=100, help='the denominator of the reservoir size')
# SARA: win_size restricts how many data instances will be sampled for the online training.
# In the paper: default value is |C|/2. Other values: {|C|, |C|/4, |C|/8, |C|/16, |C|/32}. C denote the reservoir, which contains |C| sessions.
# In the paper: when the window size is larger, the model can achieve a better recommendation performance because it can utilize more data to update itself.
parser.add_argument('--win_size', type=int, default=1, help='the denominator of the window size')
opt = parser.parse_args()
logging.warning(opt)


def main():
    assert opt.dataset in ['gowalla', 'lastfm']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()

    train_dataset = MultiSessionsGraph(cur_dir + '/../../datasets/' + opt.dataset, phrase='train')
    # train_dataset = MultiSessionsGraph(cur_dir + '\\..\\..\\datasets\\' + opt.dataset, phrase='train')  # windows
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)  # SARA: data for training
    train_for_res, _ = load_data_valid(
        os.path.expanduser(os.path.normpath(cur_dir + '/../../datasets/' + opt.dataset + '/raw/train.txt.csv')), 0)  # SARA: data for constructing the reservoir
        # os.path.expanduser(os.path.normpath(cur_dir + '\\..\\..\\datasets\\' + opt.dataset + '\\raw\\train.txt.csv')), 0)  # windows
    max_train_item = max(max(max(train_for_res[0])), max(train_for_res[1]))
    max_train_user = max(train_for_res[2])

    test_dataset = MultiSessionsGraph(cur_dir + '/../../datasets/' + opt.dataset, phrase='test1')  # SARA: data for 1st test
    # test_dataset = MultiSessionsGraph(cur_dir + '\\..\\..\\datasets\\' + opt.dataset, phrase='test1')  # windows
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_for_res = load_testdata(
        os.path.expanduser(os.path.normpath(cur_dir + '/../../datasets/' + opt.dataset + '/raw/test1.txt.csv')))
        # os.path.expanduser(os.path.normpath(cur_dir + '\\..\\..\\datasets\\' + opt.dataset + '\\raw\\test1.txt.csv')))  # windows
    max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
    max_user = max(test_for_res[2])
    pre_max_item = max_train_item
    pre_max_user = max_train_user

    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/paper200/' + str(
        opt) + '_fix_new_entropy(rank)_on_union+' + str(opt.u) + 'tanh*u_AGCN***GAG-win' + str(opt.win_size) \
              + '***concat3_linear_tanh_in_e2s_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # log_dir = cur_dir + '\\..\\..\\log\\' + str(opt.dataset) + '\\paper200\\logFile'  # windows
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    if opt.dataset == 'gowalla':
        n_item = 30000
        n_user = 33005
    else:  # lastfm
        n_item = 10000
        n_user = 984

    model = GNNModel(hidden_size=opt.hidden_size, n_item=n_item, n_user=n_user, u=opt.u).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=opt.lr_dc)

    logging.warning(model)

    # offline training on 'train' and test on 'test1'
    logging.warning('*********Begin offline training*********')
    updates_per_epoch = len(train_loader)
    updates_count = 0
    for train_epoch in tqdm(range(opt.epoch)):  # SARA: tqdm to create load bar
        forward(model, train_loader, device, writer, train_epoch, optimizer=optimizer,
                train_flag=True, max_item_id=max_train_item,
                last_update=updates_count)  # SARA: train_flag=True (& by default: last_update=0) -> train
        scheduler.step()
        updates_count += updates_per_epoch
        with torch.no_grad():  # SARA: disables the gradient calculation
            forward(model, test_loader, device, writer, train_epoch, train_flag=False,
                    max_item_id=max_item)  # SARA: train_flag=False -> test

    # reservoir construction with 'train'
    logging.warning('*********Constructing the reservoir with offline training data*********')
    res = Reservoir(train_for_res, opt.res_size)
    res.update(train_for_res)

    # test and online training on 'test2~5'
    logging.warning('*********Begin online training*********')
    now = time.asctime()
    for test_epoch in tqdm(range(1, 6)):  # SARA: tqdm to create load bar
        if test_epoch != 1:  # SARA: for test 2, 3, 4, 5
            test_dataset = MultiSessionsGraph(cur_dir + '/../../datasets/' + opt.dataset, phrase='test' + str(test_epoch))
            # test_dataset = MultiSessionsGraph(cur_dir + '\\..\\..\\datasets\\' + opt.dataset, phrase='test' + str(test_epoch))  # windows
            test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

            test_for_res = load_testdata(
                os.path.expanduser(os.path.normpath(
                    cur_dir + '/../../datasets/' + opt.dataset + '/raw/test' + str(test_epoch) + '.txt.csv')))
                # os.path.expanduser(os.path.normpath(
                #     cur_dir + '\\..\\..\\datasets\\' + opt.dataset + '\\raw\\test' + str(test_epoch) + '.txt.csv')))
            pre_max_item = max_item
            pre_max_user = max_user
            max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
            max_user = max(test_for_res[2])

            # test on the current test set
            # no need to test on test1 because it's done in the online training part
            # epoch + 10 is a number only for the visualization convenience
            with torch.no_grad():
                forward(model, test_loader, device, writer, test_epoch + 10,
                        train_flag=False, max_item_id=max_item)

        # reservoir sampling
        # SARA: win_size = len(test_for_res[0]) // opt.win_size
        print("win_size::" + str(len(test_for_res[0]) // opt.win_size))  # SARA
        sampled_data = fix_new_entropy_on_union(cur_dir, now, opt, model, device, res.data, test_for_res,
                                                len(test_for_res[0]) // opt.win_size, pre_max_item, pre_max_user,
                                                ent='wass')

        # cast the sampled set to dataset
        sampled_dataset = MultiSessionsGraph(cur_dir + '/../../datasets/' + opt.dataset,
                                             phrase='sampled' + now,
                                             sampled_data=sampled_data)
        sampled_dataset = MultiSessionsGraph(cur_dir + '\\..\\..\\datasets\\' + opt.dataset,
                                             phrase='sampled' + now,
                                             sampled_data=sampled_data)  # windows
        sampled_loader = DataLoader(sampled_dataset, batch_size=opt.batch_size, shuffle=True)

        # update with the sampled set
        forward(model, sampled_loader, device, writer, test_epoch + opt.epoch, optimizer=optimizer,
                train_flag=True, max_item_id=max_item,
                last_update=updates_count)  # SARA: train_flag=True & last_update=updates_count -> online learning

        updates_count += len(test_loader)

        scheduler.step()

        res.update(test_for_res)
        os.remove('../../datasets/' + opt.dataset + '/processed/sampled' + now + '.pt')
        # os.remove('..\\..\\datasets\\' + opt.dataset + '\\processed\\sampled' + now + '.pt')  # windows


if __name__ == '__main__':
    main()
