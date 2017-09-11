class LSTM():
    """
    Builds a model for predicting whether or not a patient will get sick.
    Remembers patients and looks for common patterns.
    """

    def __init__(self, datacube, labels):
        self.datacube = datacube
        self.labels = labels
#        self.nodes = 2
        self.activation = 'tanh'
#        self.epochs = 2
        self.batch_size = 32

    def setParams(self, activation, epochs, batch_size, nodes):
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.nodes = nodes

    def formatting(self):
        import numpy as np
        from itertools import compress

        def randomize(data, labels):
            permutation = np.random.permutation(self.datacube.shape[0])
            shuffled_data = data[permutation, :]
            shuffled_labels = self.labels[permutation]
            return(shuffled_data, shuffled_labels)
        self.datacube, self.labels = randomize(self.datacube, self.labels)

        msk = np.random.rand(self.datacube.shape[0]) < 0.8

        train = self.datacube[msk, :, :]
        train_labels = np.array(list(compress(self.labels, msk)))
        validation = self.datacube[~msk, :, :]
        validation_labels = np.array(list(compress(self.labels, ~msk)))

        return(train, train_labels, validation, validation_labels)

    def run(self, train, train_labels, validation, validation_labels):
        from keras.models import Sequential
        from keras.layers import Dense, Masking, LSTM, SimpleRNN
        model = Sequential()
        model.add(LSTM(self.nodes, batch_input_shape=(self.batch_size,
                                                      train.shape[1], train.shape[2]), activation=self.activation))
        # if we want to add 2nd layer make return_sequences = true i think fixes error
        # model.add(LSTM(self.nodes,activation=self.activation))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(train, train_labels,
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(validation, validation_labels), verbose=1)

        return(history)
