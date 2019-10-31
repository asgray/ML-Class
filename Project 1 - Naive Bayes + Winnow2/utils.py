# Anthony Gray
# Intro to Machine Learning Project 1
# Data Processing methods

import pandas as pd
from numpy import nan as NaN
import numpy as np

# method converts all values to boolean values by splitting based on the column mean --------------------------------------------------------
def binaryize(dataframe):
    # find mean of each column
    means = dataframe.describe().iloc[1]
    # iterate over all columns except last
    for column in dataframe.iloc[:,:-1].columns:
        # retrieve mean of column
        avg = means[column]
        # update values as either 1 or 0
        dataframe.loc[:, column] = (dataframe[column] > avg).astype(int)
    return dataframe
# end binaryize() --------------------------------------------------------------------------------------------------------------------------

# Method to load and clean data from CSV files ----------------------------------------------------------------------------------------------
# goals where applicable:
    # have all learnable attributes in numeric form
    # have Class labels in the last column of the table
    # have Class lables in non-numeric form
    # replace missing values with column average
    # remove non-learnable attributes such as ID  or sample number 
def get_data(set):
    path = "D:\\Documents\\Class Materials\\Intro to Machine Learning\\Datasets\\Project 1 Data\\"
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
        # replacing NaN with column average rounded to 0 or 1
        dat.fillna(dat.mean().round(decimals=0), inplace=True)

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

    # breast-cancer-wisconsin data---------------------------------------------------------------------------------------
    if set == 'cancer':
        cancer_colnames = ['Sample#','Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
            'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nuclei', 'Mitoses', 'Class']
        dat = pd.read_csv(path + "breast-cancer-wisconsin.data", header=None, names=cancer_colnames)
        # removing ID column as it is not a learnable attribute
        dat.drop(dat.columns[0], axis=1, inplace=True)
        # Change Class values to strings so they will be ignored by summary stats
        dat['Class'] = dat.Class.astype(str)
        # CLEANING
        dat.replace(to_replace='?', value=NaN, inplace=True)
        dat['Bare_Nuclei'] = dat.Bare_Nuclei.astype(float)
        dat.fillna(dat.mean().round(decimals=1), inplace=True)

    # soybean-small data------------------------------------------------------------------------------------------------------
    if set == 'soybean':
        soybean_colnames = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
            'leafspots-marg', 'leafspot-size', 'leaf-shread', 'lead-malf', 'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external-decay', 'mycelium',
            'int-discolor', 'sclerotia', 'fruit-pods', 'fruit-spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots', 'Class']
        dat = pd.read_csv(path + "soybean-small.data", header=None, names=soybean_colnames)
       # data is clean per NAMES file

    return binaryize(dat)
# End get_data() -------------------------------------------------------------------------------------------------------------------------------

# Method to subset full dataset into training and test sets ------------------------------------------------------------------------------------
def split_to_train_test_sets(full_set):
    # print('Generating training and test sets')
    # sample 1/3rd of total set to be used as test set
    test_df = full_set.sample(frac=0.3)
    # remove test set from full set to produce training set
    train_df = full_set.drop(test_df.index)
    
    return {'Test_Set': test_df, 'Training_Set': train_df}
# End split_to_train_test_sets() ----------------------------------------------------------------------------------------------------------------

# method to return descriptive stats for each class in dataset
def describe_classes(dataframe):
    print('Generating class descriptions')
    classes = dataframe.Class.unique()
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