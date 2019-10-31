# Anthony Gray
# Intro to Machine Learning Project 3
# Generic Data Processing methods

import pandas as pd
import numpy as np
import random

# Method to load and clean data from CSV files ----------------------------------------------------------------------------------------------
# goals where applicable:
    # have all learnable attributes in numeric form
    # have Class labels in the last column of the table
    # have Class lables in non-numeric form
    # replace missing values with column average
    # remove non-learnable attributes such as ID  or sample number 
def get_data(set):
    path = "C:\\Users\\Anthony\\Dropbox\\Serious\\School\\Classes\\current\\Intro to Machine Leanring - EN.605.649\\Projects\\Datasets\\"
    dat = 'Dataset Not Found'
    # ecoli data ---------------------
    if set == 'ecoli':
        colnames = ['seq_name','mcg','gvh','lip','chg','aac','alm1','alm2','class']
        dat = pd.read_csv(path + 'ecoli.data', header=None, names=colnames, delim_whitespace=True)
        # data is clean per NAMES file
        # dropping sequence name, as it is not learnable
        dat.drop(['seq_name'], axis=1, inplace=True)

    # image data ---------------------
    if set == 'image':
        dat = pd.read_csv(path + 'segmentation.data', header=2)
        # moving class vlaues from row lables to last column
        dat['class'] = dat.index
        dat.reset_index(drop=True, inplace=True)
        # data is clean per NAMES file
        # all atteributes are learnable

    # hardware data ---------------------
    if set == 'hardware':
        colnames = ['vendor','model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
        dat = pd.read_csv(path + 'machine.data', header=None, names=colnames)
        # data is clean per NAMES file
        # dropping vendor and model as they are not learnable attributes
        dat.drop(['vendor', 'model'], axis=1, inplace=True)

    # fires data ---------------------
    if set == 'fires':
        weekdays = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
        months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        dat = pd.read_csv(path + 'forestfires.data', header=0)
        # replace month strings with numeric months
        dat['month'] = dat['month'].map(months)
        # replace day strings with numeric days of the week
        dat['day'] = dat['day'].map(weekdays)
        # no missing data per NAMES file
        # all atteributes are learnable

    return dat
# End get_data() -------------------------------------------------------------------------------------------------------------------------------

# medhod calculates Euclidean Distance between two n-dimensional points -------------------------------------------------------------------------
def euclid_dist(x_i, mu_j):
    ns = []
    # iterate n times
    for i in range(len(x_i)):
        p_i = x_i[i]
        q_i = mu_j[i]
        # append square of difference
        ns.append((p_i - q_i)**2)
    # return square root of the sum of squared differences
    return sum(ns)**(1/2.0)
# end euclid_dist() ----------------------------------------------------------------------------------------------------------------------------

# medhod calculates Manhattan Distance between two n-dimensional points -------------------------------------------------------------------------
def manhat_dist(x_i, mu_j):
    ns = []
    # iterate n times
    for i in range(len(x_i)):
        p_i = x_i[i]
        q_i = mu_j[i]
        # append absolute value of difference
        ns.append(abs(p_i - q_i))
    # return sum of absolute values differences
    return sum(ns)
# end manhat_dist() ----------------------------------------------------------------------------------------------------------------------------

# method to return descriptive stats for each class in dataset -----------------------------------------------------------------------------
def describe_classes(dataframe, target_col):
    # print('Generating class descriptions')
    classes = dataframe[target_col].unique()
    response = {}
    for cl in classes:
        c = dataframe.iloc[:,-1]==cl
        sub_df = dataframe[c]
        sub_desc = sub_df.describe()
        # remove Class from summary stats if it's there
        if sub_desc.iloc[:,-1].dtype != np.float64:
            sub_desc.drop(sub_desc.columns[-1], axis=1, inplace=True)
        response[cl] = sub_desc
    return response
# end describe_classes() ---------------------------------------------------------------------------------------------------------------------

# method splits the dataset into 5 equal subsets ---------------------------------------------------------------------------------------------
def five_fold_split(data):
    groups = {1:None, 2:None, 3:None, 4:None, 5:None}
    group_size = int(data.shape[0]/5)
    # add 1/5th of remainder to each group
    for group in groups:
        # randomly choose 1/5th of remainder
        one_fifth = data.sample(group_size)
        # add to group
        groups[group] = one_fifth
        # subtract from dataset
        data = data[~data.isin(one_fifth)].dropna()
    remainder = data.shape[0]
    # note how many rows are not used
    # print(f'{remainder} Instances left over')
    return groups
# end five_fold_split() ---------------------------------------------------------------------------------------------------------------------

# Method that splits a dataset into 5 equal parts, with equal representation of each class --------------------------------------------------
def five_fold_sample_by_class(data):
    # find summary stats about each class
    class_summary = describe_classes(data, 'class')
    # find numbers of each class 
    class_dist = {}
    for cl in class_summary:
        class_dist[cl] = class_summary[cl].iloc[0,0]

    groups = {1:None, 2:None, 3:None, 4:None, 5:None}
    # initialize each group
    for group in groups:
        # build dataset of 1/5th of each class
        one_fifth_subset = pd.DataFrame()
        for cl in class_dist:
            # randomly choose 1/5th of current class
            one_fifth_class = data.loc[data['class']==cl].sample(int(class_dist[cl]/5))
            # add 1/5th of current class to overall 1/5th subset
            one_fifth_subset = one_fifth_subset.append(one_fifth_class)
        # remove selected rows from dataset
        data = data[~data.isin(one_fifth_subset)].dropna()
        # assign subset to a group
        groups[group] = one_fifth_subset
    # count unassigned rows
    remainder = data.shape[0]
    if remainder:
        # add 1/5th of remainder to each group
        for group in groups:
            # randomly choose 1/5th of remainder
            one_fifth_remainder = data.sample(int(remainder/5))
            # add to group
            groups[group] = groups[group].append(one_fifth_remainder)
            # subtract from dataset
            data = data[~data.isin(one_fifth_remainder)].dropna()
        # recalculate remainder
        remainder = data.shape[0]
    # note how many rows are not used
    # print(f'{remainder} Instances left over')
    return groups
# end five_fold_sample_by_class() ------------------------------------------------------------------------------------------------------------

# method returns proportion of correctly classified examples ---------------------------------------------------------------------------------
def classification_error(data):
    # finds number of rows
    total = data.shape[0]
    correct = 0
    for row in data.itertuples():
        # assume last column is the guess, 2nd to last is correct class
        if row[-2] == row[-1]:
            correct += 1
    return correct/total
# end classification_error() -----------------------------------------------------------------------------------------------------------------

# method returns for finding mean squared error values ---------------------------------------------------------------------------------------
def mean_squared_error(data):
    squared_diffs = []
    for row in data.itertuples():
        # assume last column is regression guess, 2nd to last is correct answer
        squared_diffs.append((row[-1]-row[-2])**2)
    return sum(squared_diffs)/len(squared_diffs)
# end mean_squared_error() -------------------------------------------------------------------------------------------------------------------

#~
#~~~~~~~~~~~~~~~~~~~~ Unused Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~

# # method converts all values to boolean values by splitting based on the column mean --------------------------------------------------------
# def binaryize(dataframe):
#     # find mean of each column
#     means = dataframe.describe().iloc[1]
#     # iterate over all columns except last
#     for column in dataframe.iloc[:,:-1].columns:
#         # retrieve mean of column
#         avg = means[column]
#         # update values as either 1 or 0
#         dataframe.loc[:, column] = (dataframe[column] > avg).astype(int)
#     return dataframe
# # end binaryize() --------------------------------------------------------------------------------------------------------------------------

# # Method to subset full dataset into training and test sets ------------------------------------------------------------------------------------
# def split_to_train_test_sets(full_set):
#     # print('Generating training and test sets')
#     # sample 1/3rd of total set to be used as test set
#     test_df = full_set.sample(frac=0.3)
#     # remove test set from full set to produce training set
#     train_df = full_set.drop(test_df.index)
#     return {'Test_Set': test_df, 'Training_Set': train_df}
# # End split_to_train_test_sets() ----------------------------------------------------------------------------------------------------------------

# # method to calulate silhouette coefficient of dataset with assigned clusters ------------------------------------------------------------------
# def silhouette_score(data):
#     # find number of each instance in each cluster
#     cluster_counts = dict(data['cluster'].value_counts())
#     sub_sets = {}
#     # subset dataframe by cluster
#     for c in cluster_counts:
#         sub_sets[c] = data.loc[data['cluster']==c]
    
#     s_is = []
#     # for each point, check average distance between it and all other points by cluster
#     for x_i in data.itertuples(): #---------------------------------------------------
#         assigned_cluster = x_i[-1]
#         average_distances = {'a_i': 0, 'b_i': 999999}
#         # iterate over clusters
#         for s in sub_sets: #----------------------------------------------------------
#             # determine if cluster is own cluster
#             calc_a_i = False
#             if s == assigned_cluster:
#                 calc_a_i = True
#             # find distance between point and all points in cluster
#             set_dists = []
#             for x_k in sub_sets[s].itertuples(): # -----------------------------------
#                 # disregard distance to self
#                 if not x_i[1:-1]== x_k[1:-1]:
#                     set_dists.append(euclid_dist(x_i[1:-1], x_k[1:-1]))
            
#             # find average distance for cluster set
#             if set_dists:
#                 set_avg = sum(set_dists)/len(set_dists)
#             else: 
#                 set_avg = 0
#             # save a_i or smallest b_i found
#             if calc_a_i:
#                 average_distances['a_i'] = set_avg
#             else:
#                 if set_avg < average_distances['b_i']:
#                     average_distances['b_i'] = set_avg
#         # calculate and save s_i for x_i
#         s_is.append((average_distances['b_i'] - average_distances['a_i'])/max(average_distances.values()))
#     # return average s_i
#     return sum(s_is)/len(s_is)
# # end silhouette_score() ----------------------------------------------------------------------------------------------------------------------