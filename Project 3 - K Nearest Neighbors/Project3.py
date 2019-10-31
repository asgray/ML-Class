# Anthony Gray
# Intro to Machine Learning Project 3
# Main Project File

# import libraries
import pandas as pd

# import project files
import utils as u
import knn

# method accepts regression function and filename, repeats function and adds results to csv ----------------------------------------------------------------------
def collect_regression(func):
    def wrapper(data,distance, name):
        path = "C:\\Users\\Anthony\\Dropbox\\Serious\\School\\Classes\\current\\Intro to Machine Leanring - EN.605.649\\Projects\\Project 3\\"
        results = pd. DataFrame(columns=['K','Error'])
        for i in range(25):
            k = func(data,distance, name)
            results.loc[len(results)] = k
        results.to_csv(path+name+".csv", index=None, header=1)
    return wrapper
# end collect_regression() ----------------------------------------------------------------------------------------------------------------------------------

# method accepts classification function and filename, repeats function and adds results to csv ----------------------------------------------------------------------
def collect_classify(func):
    def wrapper(data,distance, stratified, condensed, name):
        path = "C:\\Users\\Anthony\\Dropbox\\Serious\\School\\Classes\\current\\Intro to Machine Leanring - EN.605.649\\Projects\\Project 3\\"
        results = pd. DataFrame(columns=['K','Error'])
        for i in range(25):
            k = func(data,distance, stratified, condensed, name)
            results.loc[len(results)] = k
        results.to_csv(path+name+".csv", index=None, header=1)
    return wrapper
# end collect_classify() ----------------------------------------------------------------------------------------------------------------------------------

# method for running five-fold validations ---------------------------------------------------------------------------------------------------
# accepts full dataset and parameters for each method
#   split method: stratified or not
#   knn menthod: classification or regression
#   k: value for knn
#   distance measure: euclidean or manhattan
#   error measure: classification error or mean squared error, as appropriate
def five_fold_validation(data_set, split_method, knn_method, k, distance_measure, error_measure, condensed=False):
    # split the datasets into fifths
    splits = split_method(data_set)
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
        if condensed:
            # print('Condensing Data')
            training_set = knn.k_neareast_condenser(training_set, distance_measure)
        # run tests
        tested_data = knn_method(training_set, test_set, k, distance_measure)
        error = error_measure(tested_data)
        show_data = tested_data.sample(10)
        # strip guess column for next run
        tested_data.drop(['guess'], axis=1, inplace=True)
        errors.append(error)
    return [sum(errors)/len(errors), show_data]
# end five_fold_validation() -------------------------------------------------------------------------------------------------------------------

# method accepts parameters specifying how to run k-neareast neighbors classification, tests k ranges 1-15 to find best accuracy ------------------
#    data is whole dataset
#    distance: either euclidean or manhattan, from utils file
#    stratified or random splits, from utils file
#    training set condensed before testing or not
@collect_classify
def tune_k_classify(data, distance, stratified, condensed, name):
    print(f'Tuning K for Classification...')
    # split methods
    if stratified:
        print('Stratifying Data by Class')       
        split = u.five_fold_sample_by_class
    else:
        print('Splitting Data Randomly')
        split = u.five_fold_split
    # distance methods
    if distance == 'E':
        print('Using Euclidean Distance...')
        measure = u.euclid_dist
    elif distance == 'M':
        print('Using Manhattan Distance...')
        measure = u.manhat_dist
    # test for best accuracy
    best_accuracy = 0
    best_k = -1
    for i in range(15):
        accuracy = five_fold_validation(data, split, knn.k_nearest_neighbors_classify, i, measure, u.classification_error, condensed=condensed)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = i
    return [best_k, accuracy]
# end tune_k_classify() ---------------------------------------------------------------------------------------------------------------------------

# method accepts parameters specifying how to run k-neareast neighbors regression, tests k ranges 1-15 to minimize mean squared error ------------------
#    data is whole dataset
#    distance: either euclidean or manhattan, from utils file
@collect_regression
def tune_k_regression(data, distance, name):
    print(f'Tuning K for Regression...')
    # distance methods
    if distance == 'E':
        print('Using Euclidean Distance...')
        measure = u.euclid_dist
    elif distance == 'M':
        print('Using Manhattan Distance...')
        measure = u.manhat_dist
    # test for best accuracy
    min_mse = 9999999999
    best_k = -1
    for i in range(1,5):
        mse = five_fold_validation(data, u.five_fold_split, knn.k_nearest_neighbors_regression, i, measure, u.mean_squared_error)
        if mse < min_mse:
            min_mse = mse
            best_k = i
    return [best_k, mse]
# end tune_k_regression() -----------------------------------------------------------------------------------------------------------------------------

# DATA VARIABLES ----------------------------------------------------------
ecoli = u.get_data('ecoli')
image = u.get_data('image')
hardware = u.get_data('hardware')
fires = u.get_data('fires')
# --------------------------------------------------------------------------

'''
# ~~~~~~~~~~ TUNING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classification
tune_k_classify(ecoli, 'M', False, False, 'EcoliManhat')
tune_k_classify(ecoli, 'M', True, True, 'EcoliManhatStratCondensed')
tune_k_classify(ecoli, 'M', True, False, 'EcoliManhatStrat')
tune_k_classify(ecoli, 'M', False, True, 'EcoliManhatCondensed')

tune_k_classify(ecoli, 'E', False, False, 'EcoliEuclid')
tune_k_classify(ecoli, 'E', True, True, 'EcoliEuclidStratCondensed')
tune_k_classify(ecoli, 'E', True, False, 'EcoliEuclidStrat')
tune_k_classify(ecoli, 'E', False, True, 'EcoliEuclidCondensed')

tune_k_classify(image, 'M', False, False, 'ImageManhat')
tune_k_classify(image, 'M', True, True, 'ImageManhatStratCondensed')
tune_k_classify(image, 'M', True, False, 'ImageManhatStrat')
tune_k_classify(image, 'M', False, True, 'ImageManhatCondensed')

tune_k_classify(image, 'E', False, False, 'ImageEuclid')
tune_k_classify(image, 'E', True, True, 'ImageEuclidStratCondensed')
tune_k_classify(image, 'E', True, False, 'ImageEuclidStrat')
tune_k_classify(image, 'E', False, True, 'ImageEuclidCondensed')

# Regression
tune_k_regression(hardware, 'M','HardwareManhat')
tune_k_regression(hardware, 'E', 'HardwareEuclid')
tune_k_regression(fires, 'M', 'FiresManhat')
tune_k_regression(fires, 'E', 'FiresEuclid')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
'''
# ~~~~~~~~~~~ Find Accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hardware
print('testing hardware performance')
hardware_performance = []
for i in range(20):
    res = five_fold_validation(hardware, u.five_fold_split, knn.k_nearest_neighbors_regression, 2, u.manhat_dist, u.mean_squared_error)
    hardware_performance.append(res[0])
print(hardware_performance)
print(sum(hardware_performance)/len(hardware_performance))


# fires
print('testing fires performance')
fires_performance = []
for i in range(20):
    res = five_fold_validation(fires, u.five_fold_split, knn.k_nearest_neighbors_regression, 4, u.euclid_dist, u.mean_squared_error)
    fires_performance.append(res[0])
print(fires_performance)
print(sum(fires_performance)/len(fires_performance))

# image
print('testing image performance')
image_performance = []
for i in range(20):
    res = five_fold_validation(image, u.five_fold_sample_by_class, knn.k_nearest_neighbors_classify, 1, u.manhat_dist, u.classification_error, condensed=False)
    image_performance.append(res[0])
print(image_performance)
print(sum(image_performance)/len(image_performance))

# ecoli
print('testing ecoli performance')
ecoli_performance = []
for i in range(20):
    res = five_fold_validation(ecoli, u.five_fold_sample_by_class, knn.k_nearest_neighbors_classify, 8, u.manhat_dist, u.classification_error, condensed=False)
    ecoli_performance.append(res[0])
print(ecoli_performance)
print(sum(ecoli_performance)/len(ecoli_performance))
'''


# ~~~~~~~~~~~~ DEMO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nK-Nearest Neighbors Demonstration:')
print('Regression Sets')
# Hardware
print('Hardware Dataset Regression \n K = 2 \n Using Manhattan Distance')
res_hardware = five_fold_validation(hardware, u.five_fold_split, knn.k_nearest_neighbors_regression, 2, u.manhat_dist, u.mean_squared_error)
print(f'MSE = {res_hardware[0]}')
print(res_hardware[1])
print('\n')
    
# Fires
print('Fires Dataset Regression \n K = 4 \n Using Manhattan Distance')
res_fires = five_fold_validation(fires, u.five_fold_split, knn.k_nearest_neighbors_regression, 4, u.manhat_dist, u.mean_squared_error)
print(f'MSE = {res_fires[0]}')
print(res_fires[1])
print('\n')

print('Classification Sets')
# Ecoli
print('Ecoli Dataset Regression \n K = 8 \n Using Manhattan Distance \n and Stratified Splitting')
res_ecoli = five_fold_validation(ecoli, u.five_fold_sample_by_class, knn.k_nearest_neighbors_classify, 8, u.manhat_dist, u.classification_error, condensed=False)
print(f'Accuracy = {res_ecoli[0]}')
print(res_ecoli[1])
print('\n')
    
# Image
print('Image Dataset Regression \n K = 1 \n Using Manhattan Distance \n and Stratified Splitting')
res_image = five_fold_validation(image, u.five_fold_sample_by_class, knn.k_nearest_neighbors_classify, 1, u.manhat_dist, u.classification_error, condensed=False)
print(f'Accuracy = {res_image[0]}')
print(res_image[1])
print('\n')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


