"""
In this file, we will add more Deep Learning models. To add a model, say a different
method for RNN's, there are 3 steps:

1. Create the __init__() method and pass in the training/validation data.
2. Set the model-specific parameters that will be passed in from the Utils .run()
   function.
3. Define a build() method that uses the training/testing data and then

"""

class LSTM():
    """
    Pre-defined architecture for LSTM Recurrent Neural Networks.
    Number of layers, nodes per layer, and activation types can be altered.
    """
    def __init__(self, training_data, training_labels, validation_data,
                 validation_labels):
        """
        Pass the training and validation data into the LSTM object.
        """
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def setParams(self, batch_size, nodeList):
        """
        Define LSTM-specific hyper-parameters. Some hyperparameters do not cross
        between models, such as nodes per layer or number of filters. Model
        hyper-parameters must be passed into this new object. Maybe some sort of
        inheritance could be added.

        :param nodeList: List of nodes per layer. For instance, [5 2] describes a
        2-hidden-layer LSTM with 5 nodes in the first hidden layer, then 2 nodes in
        the second hidden layer.
        """
        self.nodeList = nodeList
        self.batch_size = batch_size

    def build(self, time_distributed):
        """
        Build the LSTM RNN model based on the pre-defined parameters. This is where we
        can update the RNN model itself, add layers or try different techniques.

        :param time_distributed: Boolean variable. If True, compute an output node
        at each time step. This will be useful for early predictions.
        :return: Keras model object to be trained using a run() function.
        """
        from keras.models import Sequential
        from keras.layers import Dense, Masking, LSTM, TimeDistributed

        model = Sequential()
        model.add(Masking(mask_value = 0.0,
                          batch_input_shape=(self.batch_size,
                                             self.training_data.shape[1],
                                             self.training_data.shape[2])))

        if len(self.nodeList) > 1:
            for nodes in self.nodeList[:-1]:
                model.add(LSTM(nodes,
                               activation='tanh', return_sequences=True))

            model.add(LSTM(self.nodeList[-1],
                           activation='tanh'))
        else:
            if time_distributed:
                model.add(LSTM(self.nodeList[0], activation='tanh',
                               return_sequences=True))
            else:
                model.add(LSTM(self.nodeList[0], activation='tanh',
                               return_sequences=False))

        if time_distributed:
            model.add(TimeDistributed(Dense(10, activation='sigmoid')))
        else:
            model.add(Dense(1, activation='sigmoid'))

        return(model)


class CNN():
    """
    Currently unsupported.
    """

    def __init__(self, training_data, training_labels, validation_data,
                 validation_labels):
        """
        Pass the training and validation data into the LSTM object.
        """
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels


    def setParams(self, filters, kernel_size, batch_size):
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size

    def build(self):
        from keras.models import Sequential
        from keras.layers.convolutional import Conv2D
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers import Dense, Dropout, Flatten

        model = Sequential()

        model.add(Conv2D(self.filters[0], kernel_size=self.kernel_size[0], activation=self.activation,
                         input_shape=(self.train.shape[1], self.train.shape[2], 1), padding='valid'))
        # model.add(Dropout(.2))
        #
        # model.add(MaxPooling2D(pool_size=(25, 1)))
        # model.add(Dropout(.4))
        #
        # model.add(Conv2D(self.filters, self.kernel_size,
        #                  activation=self.activation, padding='valid'))
        # model.add(Dropout(.2))
        #
        # model.add(MaxPooling2D(pool_size=(4, 1)))
        # model.add(Dropout(.4))

        # assume that self.filters and self.kernel_size are filters

        counter = 0
        for filt, ks in zip(self.filters[1:], self.kernel_size[1:]):
            model.add(Conv2D(filt, ks, activation = 'tanh', padding = 'valid'))
            model.add(Dropout(.2))
            if counter < 1:
                model.add(MaxPooling2D(pool_size=(25, 1)))
                model.add(Dropout(.4))
            counter += 1

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))
        return(model)
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])
        # print(model.summary())
        # history = model.fit(train, train_labels,
        #                     epochs=self.epochs,
        #                     batch_size=self.batch_size,
        #                     validation_data=(validation, validation_labels))
        # return(history)
        #
        #
        #
