# Anthony Gray
# Intro to Machine Learning Project 4
# Main Project File

# import libraries
import pandas as pd # data processing
import numpy as np # math functions
from pprint import pprint # displying large dictionaries
from time import sleep as sleep

# import project files
import utils as u
import decision_tree as dt

# method to split dataset into fifths, then try each split as the test set agains the combination of the others as traingin sets 
def five_fold_validation(data_set):
    # split the datasets into fifths
    splits = u.five_fold_split(data_set)
    errors = []
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

        # construct tree with training set
        tree = dt.build_tree(training_set)
        pprint(tree)
        print(test_set)
        # number of values in test set
        total_tests = test_set.shape[0]
        correct_guesses = 0
        for i in range(total_tests):
            # test each item
            test_item = test_set.iloc[i,:]
            prediction = dt.make_prediction(test_item, tree)
            actual_val = test_item[-1]
            # compare prediction and real value
            if prediction == actual_val:
                correct_guesses += 1
        errors.append(correct_guesses/total_tests)
    # retrn average error
    return sum(errors)/len(errors)
# end five_fold_validation() ---------------------------------------------------------------------------------------------------

# method for removing class values with less than 5 examples from abalone set -------------------------------------------------
def abalone_filter(abalone_dat):
    remove_classes = ['1','2','24','25','26','27','28','29']
    abalone_dat = abalone_dat[~abalone_dat.Rings.isin(remove_classes)]
    return abalone_dat
# end abalone_filter() --------------------------------------------------------------------------------------------------------

# method for demonstrating project --------------------------------------------------------------------------------------------
def proj_demo(name):
    # sample subset of dataset, for readable tree and reasonable runtime
    dat = u.get_data(name).sample(120)
    # split to simple train/test split
    sets = u.split_to_train_test_sets(dat)
    training_set = sets['Training_Set']
    test_set = sets['Test_Set']
    # build tree
    tree = dt.build_tree(training_set)
    # show data subsets and tree
    print('Training Data Sample: \n ', training_set.head())
    print('Test Data Sample : \n',test_set.head())
    pprint(tree)
    sleep(5)
    # classify each item in training set
    total_tests = test_set.shape[0]
    correct_guesses = 0
    print('Sample Classifications:')
    for i in range(total_tests):
        # test each item
        test_item = test_set.iloc[i,:]
        prediction = dt.make_prediction(test_item, tree)
        actual_val = test_item[-1]
        correct = prediction == actual_val
        if i < 5:
            print(f'Correct: {correct} \t Actual Value: {actual_val} \t Predicted Value: {prediction}')
        # compare prediction and real value
        if correct:
            correct_guesses += 1
    # show classification accuracy
    print(f'Accuracy: {round(correct_guesses/total_tests,2)} \n')
# end proj_demo() ---------------------------------------------------------------------------------------------------

# run demo for each dataset
sets = ['car','image','abalone']
for s in sets:
    print(f'Demonstrating {s}')
    proj_demo(s)
    sleep(5)

# ds = ['abalone', 'car', 'image']
# print(ds)
# # print('filterd')
# dat = u.get_data(ds)
# # dat = abalone_filter(dat)
# errors = []
# for i in range(10):
#     try:
#         test = dat.sample(500)
#         err = five_fold_validation(test)
#         errors.append(err)
#     except:
#         pass
# print(errors)

