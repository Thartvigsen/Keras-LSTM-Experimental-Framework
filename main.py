from Utils.Utils import Utils
import numpy as np
# import time
import argparse

np.random.seed(np.random.randint(10000))


def run_experiments(arguments):
    import csv

    ###############################################################################

    # initialize utility class - could read in experiment type here.. would require
    # many updates right now
    processor = Utils()

    # load data into the processor object
    processor.load_data()

    processor.set_model_parameters(epochs=arguments.epochs,
                                   batch_size=arguments.batch_size,
                                   nodes=arguments.nodeList,
                                   modelType=arguments.modelType)

    experimental_summary = [arguments.modelType, arguments.window_size, arguments.epochs, arguments.nodeList]
    
#    processor.data = processor.data[:, 0:1500, :]    
    for i in range(processor.data.shape[0]):
        processor.data[i] = np.where(np.isnan(processor.data[i]), np.ma.array(processor.data[i], mask=np.isnan(processor.data[i])).mean(axis=0), processor.data[i])

    processor.data = processor.data[:, 0:1000, :]
    if arguments.window_size is not False:
        processor.aggregate(window_size=int(arguments.window_size))
    # split the data into testing/training sets
#    processor.data = processor.data[:, 0:5, :]
    # take variable-wise non-zero means
#    for i in range(processor.data.shape[0]):
#        processor.data[i] = np.where(np.isnan(processor.data[i]), np.ma.array(processor.data[i], mask=np.isnan(processor.data[i])).mean(axis=0), processor.data[i])
#    processor.data = processor.data[:, 0:5, :]
    if arguments.experiment_type == 'train_test':
        print('\n------ Experimental Summary ------')
        print('1. Model trained on training data.\n'
              '2. Model tested on holdout data.')
        print('----------------------------------\n')

        processor.data_split(proportion_training=0.8, val_set=False)
        processor.run()
        processor.evaluate()
        print("Proportion positive: %0.2f" % processor.label_prop)
    if arguments.experiment_type == 'CVtrain_test':
        print('\n------------------ Experimental Summary ------------------')
        print('1. Model selected by cross-validation on the training data.\n'
              '2. Model validated using holdout data.')
        print('-----------------------------------------------------------\n')
        processor.data_split(proportion_training=0.8, val_set=False)
        processor.cross_validate(k=10)
        processor.evaluate()

    if arguments.experiment_type == 'train_validate_test':
        print('\n--------- Experimental Summary ---------')
        print('1. Model trained on training data.\n'
              '2. Model selected using validation data.\n'
              '3. Model validated using holdout data.')
        print('----------------------------------------\n')

        processor.data_split(proportion_training=0.8, val_set=True)
        processor.run()
        processor.evaluate()

        processor.run(training_data=processor.training_data,
                      training_labels=processor.training_labels,
                      validation_data=processor.holdout_data,
                      validation_labels=processor.holdout_labels)
        processor.evaluate()
        

    # output_row = experimental_summary + metric_list

    processor.file_writer(arguments.filename, experimental_summary)
    # with open(arguments.filename, 'a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(output_row)

        # processor.data_split(proportion_training=0.8, val_set=False)
        # processor.run()

    # processor.data_split(proportion_training=0.8, val_set=arguments.validation_set)

    # set the model parameters - these will change by shell loop




    # build and run the model

    # If cv_folds argument is passed, then use the cross validation function,
    # which wraps around the .run() function. Otherwise, use the preset training
    # and validation set.

    # if arguments.cv_folds is not 0:
    #     processor.cross_validate(k=arguments.cv_folds)
    # else:
    #     processor.run()

    # processor.config(experiment_type=arguments.experiment_type)

    # evaluate predictions made using preset metrics

    # metric_list = processor.evaluate()

    # experimental_summary = [arguments.modelType, arguments.epochs, arguments.nodeList]

    # combine the summary values with the evaluation metrics
    # output_row = experimental_summary + metric_list
    #
    # with open(arguments.filename, 'a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(output_row)

    ###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # convert shell arguments to python variables. Not all must be set.
    parser.add_argument('-model', action='store', dest='modelType',
                        default='None',type=str)
    parser.add_argument('-nodes', nargs='+', dest='nodeList', type=int)
    parser.add_argument('-filter', nargs='+', dest='filterList', type=int)
    parser.add_argument('-stride', action='store', dest='stride', default=1)
    parser.add_argument('-ks', action='append', dest='kernel_size', default=[])
    parser.add_argument('-epoch', action='store', dest='epochs', default=10, type=int)
    parser.add_argument('-bs', action='store', dest='batch_size',
                        default=64, type=int)
    parser.add_argument('-file', action='store', dest='filename', type=str)
    parser.add_argument('-ws', action='store', dest='window_size',
                        type=int, default=False)
    parser.add_argument('-cv', action='store', dest='cv_folds',
                        type=int, default=0)
    parser.add_argument('-exp', action='store', dest='experiment_type',
                        type=str)

    arguments = parser.parse_args()

    run_experiments(arguments)
