class Dataformatting():
    def __init__(self, data):
        import numpy as np
        self.data = data

    def normalize(self):
        import numpy as np
        minimum = np.min(self.data.value)
        maximum = np.max(self.data.value)
        vector = np.array(self.data.value)
        normalized = (vector - minimum) / (maximum - minimum)
        self.data.value = normalized

    def timeUpdate(self):
        import numpy as np

        for patient in self.data.hadm_id.unique():
            temp_df = self.data[self.data.hadm_id == patient]
            global_day_min = np.min(temp_df.Day)
            temp_df.Day = temp_df.Day - global_day_min
            temp_df.Hour = temp_df.Hour + 24 * temp_df.Day
            self.data.loc[temp_df.index] = temp_df

        self.data.Hour = self.data.Hour - 1

    def pad(df, max_len):
        """Does not currenctly support left/right
        """
        import numpy as np
        data_vector = df.value  # only the values from one patient
        zero_vector = np.zeros(max_len)
        # zero_vector.fill(-1)
        #zero_vector[0:len(data_vector)] = data_vector
        # fills the full length vector with values from current patient
        zero_vector[df.Hour] = data_vector

        return(zero_vector)

    def cubify(self):
        import numpy as np
        counter = []
        for patient in self.data.hadm_id.unique():
            temp_df = self.data[self.data.hadm_id == patient]
            count = []
            for test in self.data.itemid.unique():
                t_df = temp_df[temp_df['itemid'] == test]
                count.append(t_df.shape[0])

            counter.append(max(count))

        max_len = int(np.max(counter)) + 1

        dataCube = list()
        counter = 0

        for patient in self.data.hadm_id.unique():

            # Gets df for each patient
            patientdf = self.data[self.data['hadm_id'] == patient]

            # Just so you dont append a blank list at the start
            if counter >= 1: # this appends the last patient to the cube?
                dataCube.append(paddedData)

            # appends array for each test of length 743 per patient
            paddedData = list()

            for test in self.data.itemid.unique():
                # gets df for one patient and each test
                patient_and_test_df = patientdf[patientdf['itemid'] == test]
                paddedData.append(Dataformatting.pad(
                    patient_and_test_df, max_len))

            if counter == 0:
                dataCube.append(paddedData)

            counter += 1

        datacube = np.array(dataCube)
        datacube = np.transpose(datacube, (0, 2, 1))
        return(datacube)
