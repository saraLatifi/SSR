class SUMixedHybrid:
    '''
      MixedHybrid(algorithms, lengths)

      Use different algorithms depending on positions of predicted next items of the current session.

      Parameters
      --------
      algorithms : list
          List of algorithms to combine with a mixed strategy to make a recommendation list consisting (up to down):
          lengths[0] items from top of recommendation list of the algorithms[0]
          ...
          lengths[k] items from top of recommendation list of the algorithms[k]
          ...
          rest of the recommendation list will be from top of recommendation list of the algorithms[n]
      recomLengths : float
          Proper list of desire length of the recommendation list of each algorithm to be added to the hybrid's recommendation list.
          len(lengths) = len(algorithms) - 1
          For [10,15]
              1st algorithm is applied for first 10 recommended items,
              2nd algorithms for the next 15 recommended items,
              and 3rd algorithms for the rest of the recommendation list's items.
      fit: bool
          Should the fit call be passed through to the algorithms or are they already trained?

      '''

    def __init__(self, algorithms, recomLengths, fit=False, clearFlag=True):
        self.algorithms = algorithms
        self.recomLengths = recomLengths
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

        self.session = -1
        self.session_items = []

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

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids, skip=False, timestamp=0):
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

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()

        self.session_items.append(input_item_id)

        predictions = []
        for a in self.algorithms:
            predictions.append(a.predict_next(session_id, input_item_id, input_user_id, predict_for_item_ids, skip))

        for i, prediction in list(enumerate(predictions)):
            prediction.sort_values(ascending=False, inplace=True)
            prediction = prediction[prediction > 0]
            if i == 0:
                # first algorithm
                if len(prediction) >= self.recomLengths[i]: #handling the case which prediction list is less than defined recommendation list
                    final = prediction[:self.recomLengths[i]]
                else:
                    final = prediction
            else :
                if i == (len(predictions)-1) :
                    #last algorithm
                    for idx, pre in list(enumerate(final)):
                        if final.index[idx] in prediction.index:
                            prediction = prediction.drop(final.index[idx]) #pre2.drop(final.index[idx], inplace=True)
                    final = final.append(prediction)
                    break
                else:
                    # all algorithms except first & last ones
                    for idx, pre in list(enumerate(final)):
                        if final.index[idx] in prediction.index:
                            prediction = prediction.drop(final.index[idx])
                    if len(prediction) >= self.recomLengths[i]: #handling the case which prediction list is less than defined recommendation list
                        final = final.append(prediction[:self.recomLengths[i]])
                    else:
                        final = final.append(prediction)
            for itemId, confidenceValue in final.iteritems():
                final.at[itemId] = confidenceValue+1;

        final.sort_values(ascending=False, inplace=True)

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