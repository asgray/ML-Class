# Anthony Gray
# Intro to Machine Learning Project 6
# Generic Data Processing methods

import pandas as pd
from numpy import NaN as NaN
import numpy as np
import random
# smallest value, to prevent X/0 errors
epsilon = np.finfo(float).eps

# Method to load and clean data from CSV files ----------------------------------------------------------------------------------------------------
# goals where applicable:
    # have all learnable attributes in numeric form
    # have Class labels in the last column of the table
    # have Class lables in non-numeric form
    # replace missing values with column average
    # remove non-learnable attributes such as ID  or sample number 
    # encode data by one-hot-encoding
def get_data(set):
    path = "C:\\Users\\Anthony\\Dropbox\\Serious\\School\\Classes\\current\\Intro to Machine Leanring - EN.605.649\\Projects\\Datasets\\"
    dat = 'Dataset Not Found'
        # house-votes-84 data ---------------------------------------------------------------------------------------------------------------------
    if set == 'house':
        house_colnames = ['Class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 
            'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 
            'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 
            'duty-free-exports', 'export-administration-act-south-africa']
        dat = pd.read_csv(path + "house-votes-84.data", header=None, names=house_colnames)

        # All other datasets have Class as last column: moving Party to last column for consistency
            # list of column names must be in original order for import
        # list of  colnames without the first
        new_col_names = house_colnames[1::] 
        # adding Party-Class to end of list
        new_col_names.append('Class')
        # reorder dataframe columns
        dat = dat[new_col_names]

        #CLEANING
        # switching y/n strings for 1/0 boolean
        dat.replace(to_replace=['y','n', '?'], value=[1,0, NaN], inplace=True)
        # replacing NaN with random 0 or 1
        dat.fillna(random.randint(0,1), inplace=True)

    # iris data-----------------------------------------------------------------------------------------------------------
    if set == 'iris':
        iris_colnames = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
        dat = pd.read_csv(path + "iris.data", header=None, names=iris_colnames)
        # data is clean per NAMES file
        normalize_0_to_1(dat)


    # glass data--------------------------------------------------------------------------------------------------------
    if set == 'glass':
        glass_colnames = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
        dat = pd.read_csv(path + "glass.data", header=None, names=glass_colnames)
        # removing ID column as it is not a learnable attribute
        dat.drop(dat.columns[0], axis=1, inplace=True)
        # Change Class values to strings so they will be ignored by summary stats
        dat['Class'] = dat.Class.astype(str)
        # data is clean per NAMES file
        normalize_0_to_1(dat)


    # breast-cancer-wisconsin data---------------------------------------------------------------------------------------
    if set == 'cancer':
        cancer_colnames = ['Sample#','Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
            'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nuclei', 'Mitoses', 'Class']
        dat = pd.read_csv(path + "breast-cancer-wisconsin.data", header=None, names=cancer_colnames, dtype=str)
        # removing ID column as it is not a learnable attribute
        dat.drop(dat.columns[0], axis=1, inplace=True)
        # Change Class values to strings so they will be ignored by summary stats
        # dat['Class'] = dat.Class.astype(str)
        # CLEANING
        # remove rows with missing values
        dat.replace(to_replace='?', value=NaN, inplace=True)
        dat.dropna(axis=0, inplace=True)
        # remove columns with the same values
        remove_uniform_rows(dat)
        # one-hot-encoding
        dat = one_hot_encode_categorical(dat)

    # soybean-small data------------------------------------------------------------------------------------------------------
    if set == 'soybean':
        soybean_colnames = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
            'leafspots-marg', 'leafspot-size', 'leaf-shread', 'lead-malf', 'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external-decay', 'mycelium',
            'int-discolor', 'sclerotia', 'fruit-pods', 'fruit-spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots', 'Class']
        dat = pd.read_csv(path + "soybean-small.data", header=None, names=soybean_colnames, dtype=str)
        # data is clean per NAMES file
        # one-hot-encoding
        dat = one_hot_encode_categorical(dat)
        
    return dat
# End get_data() -------------------------------------------------------------------------------------------------------------------------------

# method for changing categorical variables to OHE ---------------------------------------------------------------------------------------------
def one_hot_encode_categorical(dat):
    # build list of attribute names
    colnames = list(dat.columns[:-1])
    encoded_colnames = []
    for col in colnames:
        # find unique values in each column
        uniques = dat[col].unique()
        for val in uniques:
            # create new column names as "ATTRIBUTE_VALUE"
            encoded_colnames.append(col+'_'+val)
    # new dataframe with new OHE column names
    encoded_df = pd.DataFrame(columns=encoded_colnames)
    for i, row in dat.iterrows():
        for val in row.iteritems():
            # ignore Class column 
            if val[0] != 'Class':
                # find column name from atrribute and value
                colname = val[0]+'_'+val[1]
                # appropriate cell = 1
                encoded_df.loc[i, colname] = 1
    # replace all NaNs with 0 for binary values
    encoded_df.replace(to_replace=NaN, value=0, inplace=True)
    # add unchanged class column to new dataframe
    encoded_df['Class'] = dat['Class']
    return encoded_df
# end one_hot_encode_categorical() --------------------------------------------------------------------------------------------------------------

# method normalizes continuous data to between 0 and 1 ------------------------------------------------------------------------------------------
def normalize_0_to_1(dat):
    # chech each column in dataframe
    for col in dat:
        # don't normalize Class column
        if col == 'Class':
            pass
        else:
            # fin attribute min and max
            col_max = dat[col].max()
            col_min = dat[col].min()
            # calc normalizing denominator, epsilon prevents division errors and is rounded out later
            denom = col_max - col_min + epsilon
            # build new column
            norm_col = []
            for val in dat[col]:
                norm_val = (val-col_min)/denom
                norm_col.append(round(norm_val,5))
            # replace column with normalized column
            dat[col] = norm_col
# end normalize_0_to_1() -----------------------------------------------------------------------------------------------------------------------

# method removes columns that only contain one value -------------------------------------------------------------------------------------------
def remove_uniform_rows(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
# end remove_uniform_rows() --------------------------------------------------------------------------------------------------------------------

# method bins continuous data based on the quintile it falls into ------------------------------------------------------------------------------
def bin_by_quintile(dat):
    # build list of attribute names
    colnames = list(dat.columns[:-1])
    for col in colnames:
        dat[col] = pd.cut(dat[col],5, labels=['1/5','2/5','3/5','4/5','5/5'], duplicates='drop')
# end bin_by_quintile() ------------------------------------------------------------------------------------------------------------------------

# method splits the dataset into 5 equal subsets -----------------------------------------------------------------------------------------------
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
# end five_fold_split() ----------------------------------------------------------------------------------------------------------------------

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

# Method to subset full dataset into training and test sets ------------------------------------------------------------------------------------
def split_to_train_test_sets(full_set):
    # print('Generating training and test sets')
    # sample 1/3rd of total set to be used as test set
    test_df = full_set.sample(frac=0.3)
    # remove test set from full set to produce training set
    train_df = full_set.drop(test_df.index)
    return {'Test_Set': test_df, 'Training_Set': train_df}
# End split_to_train_test_sets() ----------------------------------------------------------------------------------------------------------------

