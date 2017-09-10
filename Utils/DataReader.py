class DataReader():
    def read():
        import pandas as pd
        import numpy as np

        read_path = './Data/'
        data = np.load(read_path + 'datacube.npy')
        labels = pd.read_csv(read_path + 'new_labels.csv', engine='c', low_memory=False)
        labels = np.array(labels.label)

        return(data, labels)