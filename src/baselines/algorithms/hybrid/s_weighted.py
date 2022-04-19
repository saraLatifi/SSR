class SUWeightedHybrid:
    '''
    WeightedHybrid(algorithms, weights)

    Parameters
    --------
    algorithms : list
        List of algorithms to combine weighted.
    weights : float
        Proper list of weights. Must have the same length as algorithms.
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?

    '''

    def __init__(self, algorithms, weights, fit=True, clearFlag=True):
        self.algorithms = algorithms
        self.weights = weights
        self.run_fit = fit
        self.clearFlag = clearFlag
        
    def init(self, train, test=None, slice=None):
        for a in self.algorithms: 
            if hasattr(a, 'init'):
                a.init( train, test, slice )

    def fit(self, data, test=None, stream_data_total=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data or new stream from the previous candidate
            It contains the transactions of the sessions.
            It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps) or their order.
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        stream_data_total: pandas.DataFrame
            Training data and all previous candidate sets. It contains the all transactions of the sessions so far.
        '''
        if self.run_fit:
            for a in self.algorithms:
                a.fit(data, stream_data_total=stream_data_total)

    # streaming recommendation
    def online_learning(self, new_data, total_data, items=None):
        '''
        Update the predictor.

        Parameters
        --------
        new_data: pandas.DataFrame
            The previous candidate sets. It contains the new transactions of the sessions.
            It must have a header.
        total_data: pandas.DataFrame
            Training data and all previous candidate sets. It contains the all transactions of the sessions so far.
            It must have a header.
        '''
        self.fit(new_data, stream_data_total=total_data)  # fit(self, data, items=None, stream_data_total=None)

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids, skip = False, mode_type = 'view', timestamp = 0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        predictions = []
        for a in self.algorithms:
            predictions.append(a.predict_next(session_id, input_item_id, input_user_id, predict_for_item_ids, skip=skip, mode_type=mode_type))

        if skip:
            return

        final = predictions[0] * self.weights[0]
        i = 1
        while i < len(predictions):
            final += (predictions[i] * self.weights[i])
            i += 1

        return final

    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()

    def support_users(self):
        '''
            whether it is a session-based or session-aware algorithm
            (if returns True, method "predict_with_training_data" must be defined as well)

            Parameters
            --------

            Returns
            --------
            True : if it is session-aware
            False : if it is session-based
        '''
        return True

    def predict_with_training_data(self):
        '''
            (this method must be defined if "support_users is True")
            whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

            Parameters
            --------

            Returns
            --------
            True : e.g. hgru4rec
            False : e.g. uvsknn
            '''
        return False