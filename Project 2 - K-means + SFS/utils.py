# Anthony Gray
# Intro to Machine Learning Project 2
# Generic Data Processing methods

import pandas as pd
from numpy import nan as NaN
import numpy as np

# Method to load and clean data from CSV files ----------------------------------------------------------------------------------------------
# goals where applicable:
    # have all learnable attributes in numeric form
    # have Class labels in the last column of the table
    # have Class lables in non-numeric form
    # replace missing values with column average
    # remove non-learnable attributes such as ID  or sample number 
def get_data(set):
    path = "D:\\Documents\\Class Materials\\Intro to Machine Learning\\Datasets\\"

    # iris data-----------------------------------------------------------------------------------------------------------
    if set == 'iris':
        iris_colnames = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
        dat = pd.read_csv(path + "iris.data", header=None, names=iris_colnames)
        # data is clean per NAMES file

    # glass data--------------------------------------------------------------------------------------------------------
    if set == 'glass':
        glass_colnames = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
        dat = pd.read_csv(path + "glass.data", header=None, names=glass_colnames)
        # removing ID column as it is not a learnable attribute
        dat.drop(dat.columns[0], axis=1, inplace=True)
        # Change Class values to strings so they will be ignored by summary stats
        dat['Class'] = dat.Class.astype(str)
        # data is clean per NAMES file

    # spambase data ----------------------------------------------------------------------------------------------------
    if set == 'spam':
        # change 'word_freq' to 'wf', 'char_frew' to 'cf' and 'capital_run_length' to 'crl'
        spam_colnames = ['wf_make', 'wf_address', 'wf_all', 'wf_3d', 'wf_our', 'wf_over', 'wf_remove', 'wf_internet', 'wf_order', 'wf_mail', 'wf_receive', 'wf_will', 'wf_people', 
                        'wf_report', 'wf_addresses', 'wf_free', 'wf_business', 'wf_email', 'wf_you', 'wf_credit', 'wf_your', 'wf_font', 'wf_000', 'wf_money', 'wf_hp', 'wf_hpl', 
                        'wf_george', 'wf_650', 'wf_lab', 'wf_labs', 'wf_telnet', 'wf_857', 'wf_data', 'wf_415', 'wf_85', 'wf_technology', 'wf_1999', 'wf_parts', 'wf_pm', 'wf_direct', 
                        'wf_cs', 'wf_meeting', 'wf_original', 'wf_project', 'wf_re', 'wf_edu', 'wf_table', 'wf_conference', 'cf_;',  'cf_(',  'cf_[', 'cf_!', 'cf_$', 'cf_#', 
                        'crl_average', 'crl_longest', 'crl_total', 'Class']
        dat = pd.read_csv(path + 'spambase.data', header=None, names=spam_colnames)
        dat['Class'] = dat.Class.astype(str)
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

# method to calulate silhouette coefficient of dataset with assigned clusters ------------------------------------------------------------------
def silhouette_score(data):
    # find number of each instance in each cluster
    cluster_counts = dict(data['cluster'].value_counts())
    sub_sets = {}
    # subset dataframe by cluster
    for c in cluster_counts:
        sub_sets[c] = data.loc[data['cluster']==c]
    
    s_is = []
    # for each point, check average distance between it and all other points by cluster
    for x_i in data.itertuples(): #---------------------------------------------------
        assigned_cluster = x_i[-1]
        average_distances = {'a_i': 0, 'b_i': 999999}
        # iterate over clusters
        for s in sub_sets: #----------------------------------------------------------
            # determine if cluster is own cluster
            calc_a_i = False
            if s == assigned_cluster:
                calc_a_i = True
            # find distance between point and all points in cluster
            set_dists = []
            for x_k in sub_sets[s].itertuples(): # -----------------------------------
                # disregard distance to self
                if not x_i[1:-1]== x_k[1:-1]:
                    set_dists.append(euclid_dist(x_i[1:-1], x_k[1:-1]))
            
            # find average distance for cluster set
            if set_dists:
                set_avg = sum(set_dists)/len(set_dists)
            else: 
                set_avg = 0
            # save a_i or smallest b_i found
            if calc_a_i:
                average_distances['a_i'] = set_avg
            else:
                if set_avg < average_distances['b_i']:
                    average_distances['b_i'] = set_avg
        # calculate and save s_i for x_i
        s_is.append((average_distances['b_i'] - average_distances['a_i'])/max(average_distances.values()))
    # return average s_i
    return sum(s_is)/len(s_is)
# end silhouette_score() ----------------------------------------------------------------------------------------------------------------------

# method to return descriptive stats for each class in dataset -----------------------------------------------------------------------------
def describe_classes(dataframe):
    print('Generating class descriptions')
    classes = dataframe.cluster.unique()
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


#~~~~~~~~~~~~~~~~~~~~ Unused Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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