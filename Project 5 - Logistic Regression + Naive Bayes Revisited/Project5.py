# Anthony Gray
# Intro to Machine Learning Project 5
# Main Project File

# import libraries
import pandas as pd
import random
from pprint import pprint
from time import sleep as sleep

# project files
import utils as u
import Naive_Bayes
import Logistic_Regression

# method to split dataset into fifths, then try each split as the test set agains the combination of the others as trainging sets 
def five_fold_validation(data_set, data_name, eta=None, demo = False):
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
        # only export and demonstrate one of the folds
        if split != 1:
            export = False
        else:
            export = True 
        
        # if eta is supplied, perform Linear Regression
        if eta:
            model = Logistic_Regression.learn_models(training_set, eta, data_name, export=export)
            Logistic_Regression.classify(test_set, model)
        # of no eta is supplied, perform Naive Bayes
        else:
            model = Naive_Bayes.learn(training_set, data_name, export=export)
            Naive_Bayes.classify(test_set, model)

        # find and append the classification error
        err = u.classification_error(test_set)
        errors.append(err)
        
        # print results of first split
        if demo:
            print('Sample Training Data\n', training_set.head())
            print('\nWeight Vectors')
            for m in model:
               print(m, model[m])
            print('\nClassified Test Set\n',test_set)
            break
        # remove Guess column to prevent errors in future fold tests
        test_set.drop(['Guess'], axis=1, inplace=True)
    # retrn average error
    return sum(errors)/len(errors)
# end five_fold_validation() ---------------------------------------------------------------------------------------------------

# DATA COLLECTION
datasets = ['iris', 'house', 'glass', 'cancer', 'soybean']
eta = 0.1
# for ds in datasets:
#     print(ds)
#     data = u.get_data(ds)
#     for i in range(10):
#         print('LR',five_fold_validation(data, ds, eta=eta))
#         print('NB',five_fold_validation(data, ds))

# DEMONSTRATE both algorithms on a subset of each dataset
for ds in datasets:
    print(f'{ds.capitalize()} Dataset:')
    data = u.get_data(ds).sample(30)
    print('Demonstrating Naive Bayes')
    sleep(1)
    nb_err = five_fold_validation(data, ds, demo=True)*100
    print(f' > Classification Accuracy: {nb_err}%\n')
    print('Demonstrating Logistic Regression')
    lr_err = five_fold_validation(data, ds, eta=eta, demo=True)*100
    print(f' > Classification Accuracy: {lr_err}%\n')
    print(f'{ds.capitalize()} Naive Bayes v Logistic Regression: {round(nb_err, 2)}% v {round(lr_err,2)}%\n')
    sleep(2)
