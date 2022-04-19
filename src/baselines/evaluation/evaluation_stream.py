import time
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def evaluate_sessions(pr, metrics, test_data, train_data, items=None, session_key='SessionId',
                                 user_key='UserId', item_key='ItemId', time_key='Time'):
    """
    Evaluates the HGRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Concatenates train sessions to test sessions to bootstrap the hidden states of the HGRU.
    The number of the last sessions of each user that are used in the bootstrapping is controlled by `bootstrap_length`.

    Parameters
    --------
    pr : gru4rec.HGRU4Rec
        A trained instance of the HGRU4Rec network.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs,
        one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to.
        If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Default value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation.
         If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties,
        which can mess up the evaluation.
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates;
        (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    bootstrap_length: int
        Number of sessions in train data used to bootstrap the hidden state of the predictor,
        starting from the last training session of each user.
        If -1, consider all sessions. (default: -1)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock();
    st = time.time();

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset();

    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    # pr.predict = None

    items_to_predict = train_data[item_key].unique()

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()  # this will be used only in algorithms with "predict_with_training_data() == True", e.g., hgru4rec

    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    if pr.support_users():  # e.g. hgru4rec
        if pr.predict_with_training_data():
            test_data = pd.concat([train_data, test_data])

    test_data.sort_values([user_key, session_key, time_key], inplace=True)
    test_data = test_data.reset_index(drop=True)

    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    length_session = np.zeros(test_data[session_key].nunique(), dtype=np.int32)
    offset_sessions[1:] = test_data.groupby([user_key, session_key]).size().cumsum() # offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    length_session[0:] = test_data.groupby([user_key, session_key]).size() # length_session[0:] = test_data.groupby(session_key).size()

    current_session_idx = 0
    # pos: to iterate over test data to retrieve the current session and it's first interaction
    pos = offset_sessions[current_session_idx] # index of the first element of the current session in the test data
    position = 0  # position (index) of the current element in the current session
    finished = False

    prev_sid = -1
    while not finished:

        if count % 1000 == 0:
            print('    eval process: ', count, ' of ', len(test_data), ' actions: ', (count / len(test_data) * 100.0), ' % in',
                  (time.time() - st), 's')


        crs = time.clock();
        trs = time.time();

        current_item = test_data[item_key][pos]
        current_session = test_data[session_key][pos]
        current_user = test_data[user_key][pos] # current_user = test_data[user_key][pos] if user_key is not None else -1
        ts = test_data[time_key][pos]
        rest = test_data[item_key][
               pos + 1:offset_sessions[current_session_idx] + length_session[current_session_idx]].values

        if prev_sid != current_session:
            prev_sid = current_session
            if hasattr(pr, 'predict_for_extended_model'):
                past_items = pr.predict_for_extended_model(current_user)
                for past_item in past_items:
                    pr.predict_next(current_session, past_item, current_user, items_to_predict)  # to update the state for the current session, we do not need the predictions

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict(pr)

        if pr.support_users():  # session-aware (e.g. hgru4rec)
            preds = pr.predict_next(current_session, current_item, current_user, items_to_predict, timestamp=ts)
        else:  # session-based (e.g. sknn)
            preds = pr.predict_next(current_session, current_item, items_to_predict, timestamp=ts)  # without user_id

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict(pr)

        preds[np.isnan(preds)] = 0
         #  preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
        preds.sort_values(ascending=False, inplace=True)

        time_sum_clock += time.clock() - crs
        time_sum += time.time() - trs
        time_count += 1

        count += 1

        if test_data['in_eval'][pos] == True:
            for m in metrics:
                if hasattr(m, 'add_multiple'):
                    m.add_multiple(preds, rest, for_item=current_item, session=current_session, position=position)
                elif hasattr(m, 'add'):
                    m.add(preds, rest[0], for_item=current_item, session=current_session, position=position)

        pos += 1
        position += 1

        # check if we make prediction for all items of the current session (except the last one)
        if pos + 1 == offset_sessions[current_session_idx] + length_session[current_session_idx]:
            current_session_idx += 1 # start the next session

            if current_session_idx == test_data[session_key].nunique(): # if we check all sessions of the test data
                finished = True # finish the evaluation

            # retrieve the index of the first interaction of the next session we want to iterate over
            pos = offset_sessions[current_session_idx]
            position = 0 # reset the first position of the first interaction in the session
            # increment count because of the last item of the session (which we do not make prediction for)
            count += 1


    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
    print('    time count ', (time_count), 'count/', (time_sum), ' sum')

    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock / time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append(m.result())

    return res
