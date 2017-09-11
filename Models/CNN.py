class CNN():
    """
    Builds a model for predicting whether or not a patient will get sick.
    Treats the data as an image.
    """

    def __init__(self, datacube, labels):
        self.datacube = datacube
        self.labels = labels
        # self.activation = 'tanh'
        # self.epochs = 1
        # self.batch_size = 32


    def setParams(self, filters, kernel_size, activation, epochs, batch_size):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

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

        train = train.reshape(
            train.shape[0], train.shape[1], train.shape[2], 1).astype('float32')
        validation = validation.reshape(
            validation.shape[0], validation.shape[1], validation.shape[2], 1).astype('float32')
        return(train, train_labels, validation, validation_labels)

    def run(self, train, train_labels, validation, validation_labels):
        from keras.models import Sequential
        from keras.layers.convolutional import Conv2D
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers import Dense, Dropout, Flatten

        model = Sequential()

        model.add(Conv2D(self.filters[0], kernel_size=self.kernel_size[0], activation=self.activation,
                         input_shape=(train.shape[1], train.shape[2], 1), padding='valid'))
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

        for filt, ks in zip(self.filters[1:], self.kernel_size[1:]):
            model.add(Conv2D(filt, ks, activation = self.activation, padding = 'valid'))
            model.add(Dropout(.2))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        history = model.fit(train, train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(validation, validation_labels))
        return(history)
