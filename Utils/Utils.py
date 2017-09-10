class Utils():
    # def __init__(self):

    def load_data(self):
        """ Load data sets. To generalize for other data this can be changed.
        """
        import pandas as pd
        import numpy as np

        read_path = './Data/'
        self.data = np.load(read_path + 'datacube.npy')
        self.labels = pd.read_csv(read_path + 'new_labels.csv', engine='c', low_memory=False)
        self.labels = np.array(self.labels.label)
        print('------------')
        print('Data Loaded.')
        print('------------')

    def randomize(data, labels):
        """ This function shuffles the patient info and labels together since
        they are currently stored in separate matrices.

        :param data: X
        :param labels: y
        :return: X and y shuffled together.
        """
        import numpy as np

        permutation = np.random.permutation(data.shape[0])
        shuffled_data = data[permutation, :]
        shuffled_labels = labels[permutation]

        return(shuffled_data, shuffled_labels)

    def aggregate(self, window_size=1):
        """ The data cube is extremely sparse. This function takes the non-zero
        average of ranges of data by patient.

        :param window_size: Determines how many hours to group the data by.
        :return: A compressed data cube that consists of pool summaries.
        """
        import numpy as np
        import math
        print('\n--------------------------')
        print('Averaging in groups of %i.' % window_size)
        print('--------------------------')
        # convert 0's to nans for nanmean
        self.data[self.data == 0] = np.nan

        # compute nanmean for each patient's slice
        upper_lim = math.ceil(self.data.shape[1]/window_size)
        new_cube = np.empty(shape=(self.data.shape[0], upper_lim, self.data.shape[2]))
        index = count = 0
        while count < self.data.shape[1]:
            new_cube[:, index, :] = np.nanmean(self.data[:, count:(count + window_size), :], axis=1)
            count += window_size
            index += 1

        # convert nans to 0's for models
        self.data = np.nan_to_num(new_cube)

    def data_split(self, proportion_training, val_set):
        """
        Split data into training/testing set.

        :param proportion_training: Determines the proportion of data to be used
        as training
        :param val_set: Set to True if you want to break the data further into 3
        sets with the same training proportions
        :return: Split data, either training/validation or training/testing/validation
        """

        import numpy as np
        from itertools import compress

        np.random.seed(np.random.randint(low = 0, high = 10000))

        self.data, self.labels = Utils.randomize(self.data, self.labels)

        msk = np.random.rand(self.data.shape[0]) < proportion_training

        self.training_data = self.data[msk, :, :]
        self.training_labels = np.array(list(compress(self.labels, msk)))
        self.validation_data = self.data[~msk, :, :]
        self.validation_labels = np.array(list(compress(self.labels, ~msk)))

        if val_set:
            msk = np.random.rand(self.training_data.shape[0]) < proportion_training

            self.holdout_data, self.holdout_labels = self.validation_data, self.validation_labels
            self.training_data, self.validation_data = self.training_data[msk, :, :], self.training_data[~msk, :, :]
            self.training_labels, self.validation_labels = np.array(list(compress(self.training_labels, msk))), np.array(list(compress(self.training_labels, ~msk)))

            print('------ Data Generated ------')
            print('Training Set: %s\nValidation Set:%s\nHoldout Set: %s'
                  % ((self.training_data.shape), (self.testing_data.shape),
                     (self.validation_data.shape) ))
            print('----------------------------\n')

        elif ~val_set:
            print('------ Data Generated ------')
            print('Training Set: %s \nHoldout Set: %s'
                  % ((self.training_data.shape), (self.validation_data.shape) ))
            print('----------------------------\n')
    def set_model_parameters(self, epochs, batch_size, modelType, nodes = None,
                             filters = None, kernel_size = None):
        """
        Define the hyperparameters for training and testing these models.

        :param epochs: Max number of epochs to allow models to train for
        :param batch_size: Batch Size
        :param nodes: List of nodes per layer to pass into LSTM
        :param filters: Number of filters to use in CNN
        :param kernel_size: Kernel size for CNN
        :param modelType: String indicating the type of model to train
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.nodeList = nodes
        self.filters = filters
        self.kernel_size = kernel_size
        self.modelType = modelType

    def run(self, training_data=None, training_labels=None,
            validation_data=None, validation_labels=None):
        """
        Build and train the model, then generate predicted labels for testing data.
        If no further parameters are passed, model is trained on current
        training_data and validation_data variables. Inputting new values to this
        function allows for cross validation.

        :return: training history, predicted values for the given validation_data
        """
        from Models import Models
        from keras.callbacks import EarlyStopping

        # 1. Compile the appropriate model based on the given data
        if self.modelType == 'LSTM':

            model = Models.LSTM(training_data=self.training_data,
                                training_labels=self.training_labels,
                                validation_data=self.validation_data,
                                validation_labels=self.validation_labels)

            model.setParams(batch_size=self.batch_size,
                            nodeList=self.nodeList)

            model = model.build(time_distributed=False)

        elif self.modelType == 'CNN':
            model = Models.CNN()
            model.setParams()
            model = model.build()

        else:
            raise ValueError('modelType not LSTM or CNN')
        # 2. Compile and run the model

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # model.summary()
        earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0,
                                      mode='auto')
        self.history = model.fit(x=self.training_data, y=self.training_labels,
                                 epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=1,
                                 validation_data=(self.validation_data,
                                                  self.validation_labels),
                                 callbacks=[earlyStopping])

        # send the predicted values through the trained network and make predictions

        self.y_pred = model.predict(self.validation_data)

    def cross_validate(self, k):
        """
        Cross validate over the training set with k folds, record the predictions
        for each fold.

        :param k: Number of cross validation folds
        :return: Average evaluation metrics over all k folds
        """

        import numpy as np
        from sklearn.model_selection import KFold

        print('Cross Validating with %i folds.' % k)

        kf = KFold(n_splits=k, shuffle=False)
        metric_list = []
        print(self.modelType)
        for train_index, test_index in kf.split(X=self.training_data):
            X_train, X_test = self.training_data[train_index], self.training_data[test_index]
            y_train, y_test = self.training_labels[train_index], self.training_labels[test_index]
            Utils.run(self, training_data=X_train, training_labels=y_train,
                      validation_data=X_test, validation_labels=y_test)
            metric_list.append(Utils.evaluate(self))

        self.evaluation_metrics = np.mean(metric_list, axis = 0)

    def evaluate(self):
        """
        :return: List of evaluation metrics computed between predicted
        and true labels.
        """
        import sklearn.metrics

        self.y_pred = [1 if i > 0.5 else 0 for i in self.y_pred]

        auc = sklearn.metrics.roc_auc_score(self.validation_labels, self.y_pred)
        f1 = sklearn.metrics.f1_score(self.validation_labels, self.y_pred)
        prec = sklearn.metrics.precision_score(self.validation_labels, self.y_pred)
        recall = sklearn.metrics.recall_score(self.validation_labels, self.y_pred)
        accuracy = sklearn.metrics.accuracy_score(self.validation_labels, self.y_pred)

        self.evaluation_metrics = [auc, f1, prec, recall, accuracy]

        # return(evaluation_metrics)

    def file_writer(self, filename, experimental_summary):
        """
        :param filename: Name of the file to be written
        :param experimental_summary: A list of the current experiment's
        hyperparameters.
        :return: Append row to existing csv
        """
        import csv

        output_row = experimental_summary + self.evaluation_metrics

        with open(filename, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(output_row)