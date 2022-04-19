import time
import os.path
import numpy as np
import pandas as pd
from _datetime import timezone, datetime
# from src.se_data_process import load_data_valid, load_testdata
import sys
sys.path.insert(0, '../../src/GAG/')
from se_data_process import load_data_valid, load_testdata


def load_data(path, file, rows_train=None, rows_test=None, slice_num=None, density=1, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    density_appendix = ''
    if (density < 1):  # create sample

        if not os.path.isfile(path + file + train_appendix + split + '.txt.' + str(density)):
            train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
            test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})

            sessions = train.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            train = train[~train.SessionId.isin(drop_sessions)]
            train.to_csv(path + file + train_appendix + split + '.txt.' + str(density), sep='\t', index=False)

            sessions = test.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            test = test[~test.SessionId.isin(drop_sessions)]
            test = test[np.in1d(test.ItemId, train.ItemId)]
            test.to_csv(path + file + test_appendix + split + '.txt.' + str(density), sep='\t', index=False)

        density_appendix = '.' + str(density)

    if (rows_train == None):
        train = pd.read_csv(path + file + train_appendix + split + '.txt' + density_appendix, sep='\t',
                            dtype={'ItemId': np.int64})
    else:
        train = pd.read_csv(path + file + train_appendix + split + '.txt' + density_appendix, sep='\t',
                            dtype={'ItemId': np.int64}, nrows=rows_train)

    if (rows_test == None):
        test = pd.read_csv(path + file + test_appendix + split + '.txt' + density_appendix, sep='\t',
                           dtype={'ItemId': np.int64})
    else:
        test = pd.read_csv(path + file + test_appendix + split + '.txt' + density_appendix, sep='\t',
                           dtype={'ItemId': np.int64}, nrows=rows_test)

    test = test[np.in1d(test.ItemId, train.ItemId)]

    session_lengths = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, session_lengths[session_lengths > 1].index)]

    test.sort_values(['SessionId', 'ItemId'], inplace=True)
    train.sort_values(['SessionId', 'ItemId'], inplace=True)

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    check_data(train, test)

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def prepare_data_session(train, test, sessions_train=None, sessions_test=None):
    train = rename_cols(train)
    test = rename_cols(test)

    if (sessions_train != None):
        keep = train.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train - 1)]
        train = train[np.in1d(train.SessionId, keep)]
        test = test[np.in1d(test.ItemId, train.ItemId)]

    if (sessions_test != None):
        keep = test.SessionId.unique()[:(sessions_test - 1)]
        test = test[np.in1d(test.SessionId, keep)]

    session_lengths = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, session_lengths[session_lengths > 1].index)]

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    check_data(train, test)

    return (train, test)

def prepare_data_stream(train, test, test_split_num = 1, sessions_train=None, sessions_test=None):

    # prepare test data
    session_items = train[0]
    target_items = train[1]
    users = train[2]
    # session_nums_train = len(train[0])
    list_sizes = len(train[0])

    user_id_list = list()
    session_id_list = list()
    session_items_id_list = list()
    time_list = list()

    session_id = 0  # start of the session ids for training set
    order = 0
    # add data of the first list (of session items)
    for item_id in session_items[session_id]:
        user_id_list.append(users[session_id])
        session_id_list.append(session_id)
        session_items_id_list.append(item_id)
        time_list.append(order)
        order = order + 1
    user_id_list.append(users[session_id])
    session_id_list.append(session_id)
    session_items_id_list.append(target_items[session_id])
    time_list.append(order)
    order = order + 1

    for x in range(1, list_sizes):  # continue from the second list (of session items)
        if users[x] != users[x-1]:  # if the current user is not the same as the previous user
            # new user
            session_id = session_id + 1  # next session_id
            order = 0  # reset orders
            # for item_id in session_items[session_id]:
            for item_id in session_items[x]:
                # user_id_list.append(users[session_id])
                user_id_list.append(users[x])
                session_id_list.append(session_id)
                session_items_id_list.append(item_id)
                time_list.append(order)
                order = order+1
        else:  # if it is the same user, just increase the order
            order = order + 1
        # either it is a new user or not
        # user_id_list.append(users[session_id])
        user_id_list.append(users[x])
        session_id_list.append(session_id)
        # session_items_id_list.append(target_items[session_id])
        session_items_id_list.append(target_items[x])
        time_list.append(order)

    data = {
        "UserId": user_id_list,
        "SessionId": session_id_list,
        "ItemId": session_items_id_list,
        "Time": time_list
    }

    train = pd.DataFrame(data)

    # prepare (first) test data
    session_items = test[0]
    target_items = test[1]
    users = test[2]
    list_sizes = len(test[0])
    user_id_list = list()
    session_id_list = list()
    session_items_id_list = list()
    time_list = list()

    session_id = 0
    start_id = len(train.SessionId.unique())  # start of the session
    order = 0
    # add data of the first list (of session items)
    for item_id in session_items[session_id]:
        user_id_list.append(users[session_id])
        session_id_list.append(session_id + start_id)
        session_items_id_list.append(item_id)
        time_list.append(order)
        order = order + 1
    user_id_list.append(users[session_id])
    session_id_list.append(session_id + start_id)
    session_items_id_list.append(target_items[session_id])
    time_list.append(order)
    order = order + 1

    for x in range(1, list_sizes):  # continue from the second list (of session items)
        if users[x] != users[x - 1]:  # if the current user is not the same as the previous user
            # new user
            session_id = session_id + 1  # next session_id
            order = 0  # reset orders
            # for item_id in session_items[session_id]:
            for item_id in session_items[x]:
                # user_id_list.append(users[session_id])
                user_id_list.append(users[x])
                session_id_list.append(session_id + start_id)
                session_items_id_list.append(item_id)
                time_list.append(order)
                order = order + 1
        else:  # if it is the same user, just increase the order
            order = order + 1
        # either it is a new user or not
        # user_id_list.append(users[session_id])
        user_id_list.append(users[x])
        session_id_list.append(session_id + start_id)
        # session_items_id_list.append(target_items[session_id])
        session_items_id_list.append(target_items[x])
        time_list.append(order)

    data = {
        "UserId": user_id_list,
        "SessionId": session_id_list,
        "ItemId": session_items_id_list,
        "Time": time_list
    }

    test = pd.DataFrame(data)


    train = rename_cols(train)
    test = rename_cols(test)

    if (sessions_train != None):
        keep = train.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train - 1)]
        train = train[np.in1d(train.SessionId, keep)]
        test = test[np.in1d(test.ItemId, train.ItemId)]

    if (sessions_test != None):
        keep = test.SessionId.unique()[:(sessions_test - 1)]
        test = test[np.in1d(test.SessionId, keep)]

    session_lengths = test.groupby('SessionId').size()
    # filter sessions which their length 1 (less than 2)
    test = test[np.in1d(test.SessionId, session_lengths[session_lengths > 1].index)]

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    check_data(train, test)

    return (train, test)


def prepare_data_stream_data(train, candidate_list, sessions_train=None, sessions_test=None):
    '''
       Loads a tuple of training and test sets (list of candidates) with the given parameters.

       Parameters
       --------
       train : tuple of 3 lists: session_items, target_items, and user_ids
           Base path to look in for the prepared data files
       candidate_list : list of tuples (for each candidate) with the same format as "train" (i.e., tuple of 3 lists: session_items, target_items, and user_ids)
       sessions_train : int or None
            Number of sessions to load from the training set file.
            This option will automatically filter the test set to only retain items included in the training set.
        sessions_test : int or None
            Number of sessions to load from the test set file.
       Returns
       --------
       out : tuple of pandas.DataFrame
           (train, candidate_list)

       '''

    # prepare test data
    session_items = train[0]
    target_items = train[1]
    users = train[2]
    # session_nums_train = len(train[0])
    list_sizes = len(train[0])

    user_id_list = list()
    session_id_list = list()
    session_items_id_list = list()
    time_list = list()

    # for x in range(session_nums_train):
    #     order = 0
    #     for item_id in session_items[x]:
    #         user_id_list.append(users[x])
    #         session_id_list.append(x)
    #         session_items_id_list.append(item_id)
    #         time_list.append(order)
    #         order = order+1
    #     user_id_list.append(users[x])
    #     session_id_list.append(x)
    #     session_items_id_list.append(target_items[x])
    #     time_list.append(order)

    session_id = 0  # start of the session ids for training set
    order = 0
    # add data of the first list (of session items)
    for item_id in session_items[session_id]:
        user_id_list.append(users[session_id])
        session_id_list.append(session_id)
        session_items_id_list.append(item_id)
        time_list.append(order)
        order = order + 1
    user_id_list.append(users[session_id])
    session_id_list.append(session_id)
    session_items_id_list.append(target_items[session_id])
    time_list.append(order)
    order = order + 1

    for x in range(1, list_sizes):  # continue from the second list (of session items)
        if users[x] != users[x-1]:  # if the current user is not the same as the previous user
            # new user
            session_id = session_id + 1  # next session_id
            order = 0  # reset orders
            # for item_id in session_items[session_id]:
            for item_id in session_items[x]:
                # user_id_list.append(users[session_id])
                user_id_list.append(users[x])
                session_id_list.append(session_id)
                session_items_id_list.append(item_id)
                time_list.append(order)
                order = order+1
        else:  # if it is the same user, just increase the order
            order = order + 1
        # either it is a new user or not
        # user_id_list.append(users[session_id])
        user_id_list.append(users[x])
        session_id_list.append(session_id)
        # session_items_id_list.append(target_items[session_id])
        session_items_id_list.append(target_items[x])
        time_list.append(order)

    data = {
        "UserId": user_id_list,
        "SessionId": session_id_list,
        "ItemId": session_items_id_list,
        "Time": time_list
    }

    train = pd.DataFrame(data)
    train = rename_cols(train)

    start_id = train.iloc[-1]['SessionId']+1  # sessionId of the last row (last session id in the training set)
    for idx in range(len(candidate_list)):
        test = candidate_list[idx]
        session_items = test[0]
        target_items = test[1]
        users = test[2]
        list_sizes = len(test[0])
        user_id_list = list()
        session_id_list = list()
        session_items_id_list = list()
        time_list = list()

        session_id = 0
        order = 0
        # add data of the first list (of session items)
        for item_id in session_items[session_id]:
            user_id_list.append(users[session_id])
            session_id_list.append(session_id + start_id)
            session_items_id_list.append(item_id)
            time_list.append(order)
            order = order + 1
        user_id_list.append(users[session_id])
        session_id_list.append(session_id + start_id)
        session_items_id_list.append(target_items[session_id])
        time_list.append(order)
        order = order + 1

        for x in range(1, list_sizes):  # continue from the second list (of session items)
            if users[x] != users[x - 1]:  # if the current user is not the same as the previous user
                # new user
                session_id = session_id + 1  # next session_id
                order = 0  # reset orders
                # for item_id in session_items[session_id]:
                for item_id in session_items[x]:
                    # user_id_list.append(users[session_id])
                    user_id_list.append(users[x])
                    session_id_list.append(session_id + start_id)
                    session_items_id_list.append(item_id)
                    time_list.append(order)
                    order = order + 1
            else:  # if it is the same user, just increase the order
                order = order + 1
            # either it is a new user or not
            # user_id_list.append(users[session_id])
            user_id_list.append(users[x])
            session_id_list.append(session_id + start_id)
            # session_items_id_list.append(target_items[session_id])
            session_items_id_list.append(target_items[x])
            time_list.append(order)

        data = {
            "UserId": user_id_list,
            "SessionId": session_id_list,
            "ItemId": session_items_id_list,
            "Time": time_list
        }

        test = pd.DataFrame(data)
        test = rename_cols(test)

        if (sessions_train != None):
            keep = train.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train - 1)]
            train = train[np.in1d(train.SessionId, keep)]
            test = test[np.in1d(test.ItemId, train.ItemId)]

        if (sessions_test != None):
            keep = test.SessionId.unique()[:(sessions_test - 1)]
            test = test[np.in1d(test.SessionId, keep)]

        session_lengths = test.groupby('SessionId').size()
        # filter sessions which their length 1 (less than 2)
        test = test[np.in1d(test.SessionId, session_lengths[session_lengths > 1].index)]

        candidate_list[idx] = test
        start_id = test.iloc[-1]['SessionId']+1  # sessionId of the last row (last session id in this test set)

    # output
    print('Loaded train set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n'.
          format(len(train), train.UserId.nunique(), train.SessionId.nunique(), train.ItemId.nunique()))

    data_total = train.copy()
    for test in candidate_list:

        print('Loaded test set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n'.
              format(len(test), test.UserId.nunique(), test.SessionId.nunique(), test.ItemId.nunique()))

        data_total = pd.concat([data_total, test], ignore_index=True)

        check_data(train, test)

    print('Loaded dataset set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tAvg Session Length: {}\n'.
          format(len(data_total), data_total.UserId.nunique(), data_total.SessionId.nunique(), data_total.ItemId.nunique(), data_total.groupby('SessionId').size().mean()))

    return (train, candidate_list)

def load_data_stream_pickle(path, file='', sessions_train=None, sessions_test=None, candidate_num=None, train_eval=False):
    '''
       [pickle format] Loads a tuple of training and test sets (list of candidates) with the given parameters.

       Parameters
       --------
       path : string
           Base path to look in for the prepared data files
       file : string
           Prefix of the file of the dataset you want to use, e.g., "test" or "train"
       candidate_num :
           load the candidate set with this number, e.g., 1 for test1.txt.csv
       train_eval : boolean
           True:  loads the training and validation sets for optimization
           False: loads the training set for training the model
       Returns
       --------
       out : tuple of pandas.DataFrame
           (train, candidate_list)

       '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    if train_eval:  # hyper-parameter tuning
        valid_portion = 0.1  # training (tr) & validation
        # train, test = load_data_valid(os.path.expanduser(os.path.normpath(path + file + 'train.txt.csv')), valid_portion, shuffle=False)
        train, validation = load_data_valid(os.path.expanduser(os.path.normpath(path + file + 'train.txt.csv')), valid_portion, shuffle=False)
        candidate_1 = load_testdata(os.path.expanduser(os.path.normpath(path + file + 'test1.txt.csv')))
        # concatenate 10% of the training data and first candidate set to create the validation set
        validation[0].extend(candidate_1[0])
        validation[1].extend(candidate_1[1])
        validation[2].extend(candidate_1[2])
        candidate_list = list()
        candidate_list.append(validation)
    else:  # train and test
        valid_portion = 0  # training (full) & test
        train, _ = load_data_valid(os.path.expanduser(os.path.normpath(path + file + 'train.txt.csv')), valid_portion, shuffle=False)

        candidate_list = list()
        for idx in range(1, candidate_num+1):
            candidate_list.append(load_testdata(os.path.expanduser(os.path.normpath(path + file + 'test' + str(idx) + '.txt.csv'))))

    train, candidate_list = prepare_data_stream_data(train, candidate_list)

    print('!!!!!!!!! File: ' + path + file + "including candidates "+ str(candidate_num))
    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, candidate_list)


def load_data_session_pickle(path, file='', sessions_train=None, sessions_test=None, slice_num=None, train_eval=False):
    '''
       [pickle format] Loads a tuple of training and test set with the given parameters.

       Parameters
       --------
       path : string
           Base path to look in for the prepared data files
       file : string
           Prefix of  the dataset you want to use.
           "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
       rows_train : int or None
           Number of rows to load from the training set file.
           This option will automatically filter the test set to only retain items included in the training set.
       rows_test : int or None
           Number of rows to load from the test set file.
       slice_num :
           Adds a slice index to the constructed file_path
           yoochoose-clicks-full_train_full.0.txt
       density : float
           Percentage of the sessions to randomly retain from the original data (0-1).
           The result is cached for the execution of multiple experiments.
       Returns
       --------
       out : tuple of pandas.DataFrame
           (train, test)

       '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = '1'  # first test set
    if (slice_num != None and isinstance(slice_num, int)):  # slice num will be: 1, 2, 3, 4, 5
        split = str(slice_num)

    if train_eval:
        valid_portion = 0.1  # training (tr) & validation
        train, test = load_data_valid(os.path.expanduser(os.path.normpath(path + file + 'train.txt.csv')), valid_portion, shuffle=False)

    else:
        valid_portion = 0  # training (full) & test
        train, _ = load_data_valid(os.path.expanduser(os.path.normpath(path + file + 'train.txt.csv')), valid_portion, shuffle=False)
        test = load_testdata(os.path.expanduser(os.path.normpath(path + file + 'test' + split + '.txt.csv')))

    train, test = prepare_data_stream(train, test, split)
    # train, test = prepare_data_session(train, test)

    print('!!!!!!!!! File: ' + path + file + split)
    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def load_data_session_hdf(path, file, sessions_train=None, sessions_test=None, slice_num=None,
                          train_eval=False):
    '''
       [HDF5 format] Loads a tuple of training and test set with the given parameters.

       Parameters
       --------
       path : string
           Base path to look in for the prepared data files
       file : string
           Prefix of  the dataset you want to use.
           "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
       rows_train : int or None
           Number of rows to load from the training set file.
           This option will automatically filter the test set to only retain items included in the training set.
       rows_test : int or None
           Number of rows to load from the test set file.
       slice_num :
           Adds a slice index to the constructed file_path
           yoochoose-clicks-full_train_full.0.txt
       density : float
           Percentage of the sessions to randomly retain from the original data (0-1).
           The result is cached for the execution of multiple experiments.
       Returns
       --------
       out : tuple of pandas.DataFrame
           (train, test)

       '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    # train_appendix = '_train_full'
    # test_appendix = '_test'
    train_key = 'train'
    test_key = 'test'
    if train_eval:
        # train_appendix = '_train_tr'
        # test_appendix = '_train_valid'
        train_key = 'valid_train'
        test_key = 'valid_test'

    # train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
    # test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})

    sessions_path = os.path.join(path, file + split + '.hdf')
    train = pd.read_hdf(sessions_path, train_key)
    test = pd.read_hdf(sessions_path, test_key)

    train, test = prepare_data_session(train, test, sessions_train,
                                       sessions_test)  # (train, test, sessions_train=None, sessions_test=None)

    print('!!!!!!!!! File: ' + file + split + '.hdf')
    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def load_data_session(path, file, sessions_train=None, sessions_test=None, slice_num=None, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
    test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})

    train, test = prepare_data_session(train, test, sessions_train, sessions_test)

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def load_buys(path, file):
    '''
    Load all buy events from the youchoose file, retains events fitting in the given test set and merges both data sets into one

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt

    Returns
    --------
    out : pandas.DataFrame
        test with buys

    '''

    print('START load buys')
    st = time.time()
    sc = time.clock()

    # load csv
    buys = pd.read_csv(path + file + '.txt', sep='\t', dtype={'ItemId': np.int64})

    print('END load buys ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return buys


def load_data_session_retrain(path, file, trian_set, test_num, sessions_train=None, sessions_test=None, slice_num=None,
                              train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    sessions_train : int or None
        Number of sessions to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    sessions_test : int or None
        Number of sessions to load from the test set file.
    trian_set: int
        The postfix of the train set
        train-item-views_train_full.0.txt
    test_num: int
        Number of days included in test data. Adds another postfix to the postfix of the train set for test data
        for ex test_num: 14 will create train-item-views_test.0_0.txt to train-item-views_test.0_13.txt
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    train_eval : boolean
        shows if it is an experiment or optimization, to return the proper data
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    split = '.' + str(trian_set)
    train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
    test_list = []
    for n in range(0, test_num):
        split = '.' + str(trian_set) + '_' + str(n)
        test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
        test_list.append(test)

    if (sessions_train != None):
        keep = train.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train - 1)]
        train = train[np.in1d(train.SessionId, keep)]
        for i in range(0, len(test_list)):
            test = test_list[i]
            test_list[i] = test[
                np.in1d(test.ItemId, train.ItemId)]

    if (sessions_test != None):
        for i in range(0, len(test_list)):
            test = test_list[i]
            keep = test.SessionId.unique()[:(sessions_test - 1)]
            test_list[i] = test[
                np.in1d(test.SessionId, keep)]

    for i in range(0, len(test_list)):
        test = test_list[i]
        session_lengths = test.groupby('SessionId').size()
        test_list[i] = test[np.in1d(test.SessionId, session_lengths[
            session_lengths > 1].index)]

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    for test in test_list:
        data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
        data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

        print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
              format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                     data_end.date().isoformat()))

        check_data(train, test)

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test_list)


def load_data_userbased(path, file, rows_train=None, rows_test=None, slice_num=None, density=1, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    train_appendix = '_train'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_valid'
        test_appendix = '_test_valid'

    density_appendix = ''
    if (density < 1):  # create sample

        if not os.path.isfile(path + file + train_appendix + split + '.csv.' + str(density)):
            train = pd.read_csv(path + file + train_appendix + split + '.csv', sep='\t', dtype={'item_id': np.int64})
            test = pd.read_csv(path + file + test_appendix + split + '.csv', sep='\t', dtype={'item_id': np.int64})

            sessions = train.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            train = train[~train.SessionId.isin(drop_sessions)]
            train.to_csv(path + file + train_appendix + split + '.csv.' + str(density), sep='\t', index=False)

            sessions = test.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            test = test[~test.SessionId.isin(drop_sessions)]
            test = test[np.in1d(test.ItemId, train.ItemId)]
            test.to_csv(path + file + test_appendix + split + '.csv.' + str(density), sep='\t', index=False)

        density_appendix = '.' + str(density)

    if (rows_train == None):
        train = pd.read_csv(path + file + train_appendix + split + '.csv' + density_appendix, sep='\t',
                            dtype={'item_id': np.int64})
    else:
        train = pd.read_csv(path + file + train_appendix + split + '.csv' + density_appendix, sep='\t',
                            dtype={'item_id': np.int64}, nrows=rows_train)

    if (rows_test == None):
        test = pd.read_csv(path + file + test_appendix + split + '.csv' + density_appendix, sep='\t',
                           dtype={'item_id': np.int64})
    else:
        test = pd.read_csv(path + file + test_appendix + split + '.csv' + density_appendix, sep='\t',
                           dtype={'item_id': np.int64}, nrows=rows_test)

    train = rename_cols(train)
    test = rename_cols(test)

    if (rows_train != None):
        test = test[np.in1d(test.ItemId, train.ItemId)]

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)


def check_data(train, test):
    if 'ItemId' in train.columns and 'SessionId' in train.columns:

        new_in_test = set(test.ItemId.unique()) - set(train.ItemId.unique())
        if len(new_in_test) > 0:
            print('WAAAAAARRRNIIIIING: new items in test set')

        session_min_train = train.groupby('SessionId').size().min()
        if session_min_train == 0:
            print('WAAAAAARRRNIIIIING: session length 1 in train set')

        session_min_test = test.groupby('SessionId').size().min()
        if session_min_test == 0:
            print('WAAAAAARRRNIIIIING: session length 1 in train set')

        sess_train = train.SessionId.unique()
        sess_test = test.SessionId.unique()

        if not all(sess_train[i] <= sess_train[i + 1] for i in range(len(sess_train) - 1)):
            print('WAAAAAARRRNIIIIING: train sessions not sorted by id')
            train.sort_values(['SessionId', 'Time'], inplace=True)
            print(' -- corrected the order')

        if not all(sess_test[i] <= sess_test[i + 1] for i in range(len(sess_test) - 1)):
            print('WAAAAAARRRNIIIIING: test sessions not sorted by id')
            test.sort_values(['SessionId', 'Time'], inplace=True)
            print(' -- corrected the order')

        test.SessionId.unique()

    else:
        print('data check not possible due to individual column names')


def rename_cols(df):
    names = {}
    names['item_id'] = 'ItemId'
    names['sessionId'] = 'SessionId'
    names['user_id'] = 'UserId'
    names['created_at'] = 'Time'

    names['itemId'] = 'ItemId'
    names['session_id'] = 'SessionId'
    names['userId'] = 'UserId'
    names['eventdate'] = 'Time'

    names['itemid'] = 'ItemId'
    names['session_id'] = 'SessionId'
    names['visitorid'] = 'UserId'
    names['timestamp'] = 'Time'

    names['product_id'] = 'ItemId'
    names['user_session'] = 'SessionId'
    names['user_id'] = 'UserId'
    names['event_time'] = 'Time'

    for col in list(df.columns):
        if col in names:
            df[names[col]] = df[col]
            del df[col]

    return df


def count_repetitions(path, file, rows_train=None, rows_test=None, slice_num=None, density=1, train_eval=False):
    '''
    Loads a tuple of training and test set with the given parameters.

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file.
        This option will automatically filter the test set to only retain items included in the training set.
    rows_test : int or None
        Number of rows to load from the test set file.
    slice_num :
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1).
        The result is cached for the execution of multiple experiments.
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)

    '''

    print('START load data')
    st = time.time()
    sc = time.clock()

    split = ''
    if (slice_num != None and isinstance(slice_num, int)):
        split = '.' + str(slice_num)

    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'

    density_appendix = ''
    if (density < 1):  # create sample

        if not os.path.isfile(path + file + train_appendix + split + '.txt.' + str(density)):
            train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})
            test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId': np.int64})

            sessions = train.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            train = train[~train.SessionId.isin(drop_sessions)]
            train.to_csv(path + file + train_appendix + split + '.txt.' + str(density), sep='\t', index=False)

            sessions = test.SessionId.unique()
            drop_n = round(len(sessions) - (len(sessions) * density))
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            test = test[~test.SessionId.isin(drop_sessions)]
            test = test[np.in1d(test.ItemId, train.ItemId)]
            test.to_csv(path + file + test_appendix + split + '.txt.' + str(density), sep='\t', index=False)

        density_appendix = '.' + str(density)

    if (rows_train == None):
        train = pd.read_csv(path + file + train_appendix + split + '.txt' + density_appendix, sep='\t',
                            dtype={'ItemId': np.int64})
    else:
        train = pd.read_csv(path + file + train_appendix + split + '.txt' + density_appendix, sep='\t',
                            dtype={'ItemId': np.int64}, nrows=rows_train)

    if (rows_test == None):
        test = pd.read_csv(path + file + test_appendix + split + '.txt' + density_appendix, sep='\t',
                           dtype={'ItemId': np.int64})
    else:
        test = pd.read_csv(path + file + test_appendix + split + '.txt' + density_appendix, sep='\t',
                           dtype={'ItemId': np.int64}, nrows=rows_test)

    test = test[np.in1d(test.ItemId, train.ItemId)]

    session_lengths = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, session_lengths[session_lengths > 1].index)]

    test.sort_values('SessionId', inplace=True)
    train.sort_values('SessionId', inplace=True)

    # output
    data_start = datetime.fromtimestamp(train.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)

    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data_start = datetime.fromtimestamp(test.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(test.Time.max(), timezone.utc)

    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    check_data(train, test)

    df_out = train[train.duplicated(subset=['SessionId', 'ItemId'], keep=False)] \
        .groupby('SessionId')['ItemId'] \
        .agg({'ItemId': 'nunique'}) \
        .rename(columns={'ItemId': 'Duplicates'})

    print(df_out.reset_index())
    print("Number of sessions: " + str(df_out.shape[0]))
    print("More than 1 repetition: " + str(df_out[df_out['Duplicates'] > 1].count()))

    print('END load data ', (time.clock() - sc), 'c / ', (time.time() - st), 's')

    return (train, test)
