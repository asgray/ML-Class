# Anthony Gray
# Intro to Machine Learning Project 4
# ID3 Methods

# numpy imports
import numpy as np
from numpy import log2 as log2
from numpy import mean as mean
# smallest value, to prevent X/0 errors
epsilon = np.finfo(float).eps

# ~~~~~~~~~~~~~~~~~ Calculation Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# method for finding entropy of dataset I ------------------------------------------
def find_general_entropy(dat):
    # total number of examples
    total_instances = dat.shape[0]
    # dictionary of counts by class
    class_dist = dict(dat.iloc[:,-1].value_counts())
    node_info = 0
    # sum entropy I
    for cl in class_dist:
        frac = class_dist[cl]/total_instances
        node_info += -frac*log2(frac)
    return node_info
# end find_general_entropy() ------------------------------------------------------

# method for finding feature entropy E --------------------------------------------
def find_categorical_entropy(dat):
    # name of feature being calculated
    target = dat.keys()[0]
    # list values feature can take
    feature_vals = dat.iloc[:,0].unique()
    # total instances
    instance_count = dat.shape[0]
    E = 0
    for j in feature_vals:
        # subset by value
        sub_dat = dat[dat[target] == j]
        # count for value
        var_count = sub_dat.shape[0]
        # sum entropy for feature
        E += var_count/instance_count* find_general_entropy(sub_dat)
    return E
# find_categorical_entropy() -----------------------------------------------------

# method for finding intrinsic information IV ------------------------------------
def find_IV(dat):
    # total number of examples
    total_instances = dat.shape[0]
    # counts for each subpartition
    feature_dist = dict(dat.iloc[:,0].value_counts())
    # sum IV values for each of i subpartion
    intrinsic_value = 0
    for i in feature_dist:
        frac = (feature_dist[i]/total_instances)+epsilon
        intrinsic_value += -frac*log2(frac)
    return intrinsic_value
# end find_IV() -------------------------------------------------------------------

# method for finding the feature with the best information gain ratio I*E/IV ------
def find_best_feature(dat):
    # find entropy of set I
    I = find_general_entropy(dat)
    # list attributes
    attributes = dat.columns[:-1]
    # store name of class feature
    class_col = dat.keys()[-1]
    best_attr = None
    best_gain_ratio = 0
    best_split = None
    for attr in attributes:
        feature = dat[[attr, class_col]].copy(deep =True)
        is_continuous = np.issubdtype(dat[attr].dtype, np.number)
        split = None
        if is_continuous:
            split = binary_bin(feature)
        #else:
        # calc entropy of feature E
        E = find_categorical_entropy(feature)
        # find info gain of feature
        gain = (I - E)+epsilon
        # find gain ratio
        gain_ratio = gain / find_IV(feature)
        # save feature with best gain ratio and split, if applicable
        if gain_ratio > best_gain_ratio and gain_ratio > 0:
            best_gain_ratio = gain_ratio
            best_attr = attr
            best_split = split
    return [best_attr, best_split]
# end find_best_feature() ------------------------------------------------------
# ~~~~~~ End Calculation Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~ Data Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# method to bin continuous data based on most informative split ----------------
def binary_bin(dat):
    # find entropy of feature I
    I = find_general_entropy(dat)

    intrinsic_value = find_IV(dat)
    # list attributes
    attributes = dat.keys()[:-1]
    # store name of class feature
    class_col = dat.keys()[-1]
    for attr in attributes:
        # extract feature and class labels
        feature = dat[[attr, class_col]]
        # check for categorical data
        is_continuous = np.issubdtype(dat[attr].dtype, np.number)
        # if not categorical
        if is_continuous:
            # sort values 
            feature = feature.sort_values(by=attr)
            # find all valid splits
            splits = find_splits(feature)
            # find spit with best gain ratio
            best_ratio = 0
            best_split = 0
            for split in splits:
                # find gain ratio for each split
                split_gain_ratio = test_split(feature, split, attr, I, intrinsic_value)
                # test against previous best
                if split_gain_ratio > best_ratio:
                    best_ratio = split_gain_ratio
                    best_split = split
            # update dataframe based on best split
            dat.loc[:, attr] = (dat[attr] > best_split).astype(str) 
            return best_split
# end binary_bin() -------------------------------------------------------------

# helper method that finds all valid splits within a continuous variable -------
def find_splits(feature):
    # total number of examples
    total_instances = feature.shape[0]
    splits = []
    # iterate over consecutive pairs of examples
    for i in range(total_instances-1):
        # skip pairs with the same class
        if feature.iloc[i,1] != feature.iloc[i+1,1]:
            val = feature.iloc[i,0]
            next_val = feature.iloc[i+1,0]
            # skip pairs with equal values
            if val != next_val:
                # append mean/midpoint of the pair
                splits.append((val + next_val)/2)
    return splits
# end find_splits() -------------------------------------------------------------

# method to find the gain ratio for given split in continuous data --------------
def test_split(feature, split, attr, I, intrinsic_value):
    # create dummy copy of feature data
    test_dat = feature.copy(deep=True)
    # assign 'False' to values lower than split. 'True' to values higher
    test_dat.loc[:, attr] = (test_dat[attr] > split).astype(str)
    # calc entropy of split E
    E = find_categorical_entropy(test_dat)
    # find info gain of split
    gain = I - E
    # find gain ratio of plit
    gain_ratio = gain / intrinsic_value
    return gain_ratio
# end test_split() ---------------------------------------------------------------
# ~~~~~~ End Data Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~  Tree Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# method for recursivly building a decision tree using gain ratio --------------
def build_tree(dat, tree=None):
    # store name of class feature
    class_col = dat.keys()[-1]
    # find feature with best gain ratio
    feature = find_best_feature(dat)
    # extract feature
    node_feature = feature[0]
    if node_feature:
        # if feature came with a split
        if feature[1]:
            # extract split
            split_val = feature[1]
            val_str = str(split_val)
            # NOT IDEAL
            # replace numeric values with string showing thier relationship to split
            dat[node_feature] = dat[node_feature].apply(lambda x: '>'+val_str if x>split_val else '<'+val_str) 

        # list values feature can take
        feature_values = dat[node_feature].unique()
        # instantiate new node
        if not tree:
            tree = {}
            tree[node_feature] = {}
        # partition data by feature value
        for val in feature_values:
            partition = dat[dat[node_feature]==val]#.copy(deep=True)
            # list classes present in partition
            class_dist = partition[class_col].unique()
            # if pure partition
            if len(class_dist) == 1:
                # assign class label leaf node
                tree[node_feature][val] = class_dist[0]
            # else recusive call to build new tree from each partition
            else:
                tree[node_feature][val] = build_tree(partition)
    return tree
# end build_tree() --------------------------------------------------------------

# method uses values from test instance -----------------------------------------
def make_prediction(instance, tree):
    for node in tree:
        # current splitting variable
        target_value = instance[node]
        # subtree associated with splitting variable
        sub_tree = tree[node]

        # check for categorical data
        if type(target_value) != str:
            # if numeric value, find split value from node branches
            split = float(list(sub_tree.keys())[0][1:])
            # reassign target value to appropriate branch
            if target_value > split:
                target_value = '>'+str(split)
            else:
                target_value='<'+str(split)

        # check if target value is a node in the tree
        if target_value in sub_tree:
            # proceed through tree if possible
            sub_tree = tree[node][target_value]
        else:
            # return unknown class if value is not in tree
            sub_tree = 'unknown'
        # check if more recursion is required
        if type(sub_tree) is dict:
            prediction = make_prediction(instance, sub_tree)
        # if at leaf, find value and return it
        else:
            prediction = sub_tree
    return prediction
# make_prediction() --------------------------------------------------------------
# ~~~~~~ End Tree Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~