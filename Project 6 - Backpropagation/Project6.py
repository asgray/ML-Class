# Anthony Gray
# Intro to Machine Learning Project 6
# Main Project File

# import libraries
import pandas as pd
from time import sleep as sleep
import csv

# project files
import utils as u
import BackProp

# method to split dataset into fifths, then try each split as the test set agains the combination of the others as trainging sets 
def five_fold_validation(data_set, data_name, n_layers, n_neurons, demo = False):
    # split the datasets into fifths
    splits = u.five_fold_split(data_set)
    errors = []
    export = True
    # for each fifth of the dataset
    for split in splits:
        test_set = None
        training_set = pd.DataFrame(columns=data_set.columns.values)
        # check each fifth
        for s in splits:   
            # if fifth in question
            if s == split:
                # this fifth is test set
                test_set = splits[s]
            # all others are training sets
            else:
                training_set = training_set.append(splits[s], sort=False)
        
        # train network
        model = BackProp.build_network(training_set, n_layers, n_neurons, 0.25)
        # classify test set
        classified = BackProp.classify(model, test_set)
        
        # find and append the classification error
        err = u.classification_error(classified)
        errors.append(err)
        
        # print results of first split
        if demo:
            print('Sample Training Data\n', training_set.head())
            print('\nNetwork')
            print(model[0])
            print('\nClassified Test Set\n',test_set)
            break
        # remove Guess column to prevent errors in future fold tests
        test_set.drop(['Guess'], axis=1, inplace=True)
    # retrn average error
    return sum(errors)/len(errors)
# end five_fold_validation() ---------------------------------------------------------------------------------------------------

# method for finding the best combination of layers and neurons for each dataset -----------------------------------------------
def tune(name):
    print(name, 'tuning')
    data = u.get_data(name)
    max_nodes = data.shape[1] - 1
    min_nodes = int(max_nodes/2)
    for i in range(0,3):
        for j in range(max_nodes+1):
            # data setup
            test_data = data.sample(45)
            sets = u.split_to_train_test_sets(test_data)
            training_set = sets['Training_Set']
            test_set = sets['Test_Set']
            error = 'oops'
            try:
            # training and testing
                model = BackProp.build_network(training_set,i,j,0.25)
                classified = BackProp.classify(model, test_set)
                error = u.classification_error(classified)
                error = round(error,4)*100
            except:
                pass
            # record to file
            row = [i,j,error]
            with open(name+'_tuning.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            file.close()
# end tune() --------------------------------------------------------------------------------------------------------------------

# DATA COLELCTION
# name = 'iris'
# layers = 1
# neurons = 4
# # tune(name)
# data = u.get_data(name)
# print(name, layers, neurons, 'eta 0.25 700 epochs ')
# errs = []
# for i in range(10):
#     errs.append(five_fold_validation(data, name, layers, neurons))
# print(errs)


# DEMONSTRATE both algorithms on a subset of each dataset
datasets = {'iris':4,'glass':8,'house':11,'cancer':12,'soybean':12}
results = {}
print('\n')
for ds in datasets:
    n_neurons = datasets[ds]
    print(f' Running {ds.capitalize()} Dataset, using {n_neurons} Neurons:')
    sleep(2)
    data = u.get_data(ds)
    errors = {0:0,1:0,2:0}
    for i in range(3):
        demo_set = data.sample(45)
        print(f'\tDemonstrating With {i} Hidden Layers')
        err = five_fold_validation(demo_set, ds, i, n_neurons, demo=True)*100
        errors[i] = err
        print(f' > Classification Accuracy: {err}%\n')
        sleep(2)
    print(f'{ds.capitalize()} Layer Performance:\n  0 Layers: {errors[0]}%\n 1 Layer: {errors[1]}%\n 2 Layers: {errors[2]}%')
    best_layers = max(errors, key=errors.get)
    print(f'Best Number of Layers: {best_layers}\n')
    results[ds] = [best_layers, errors[best_layers]]
    sleep(4)
print(f'The best number of layers for each dataset is:')
for ds in results:
    print(f'\t{ds.capitalize()} Datset: {results[ds][0]} Layers for {results[ds][1]}% Accuracy')
print('\n\n')
