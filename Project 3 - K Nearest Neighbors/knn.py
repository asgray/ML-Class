# Anthony Gray
# Intro to Machine Learning Project 3
# K-Nearest Neighbors methods

from numpy import NaN as NaN
import pandas as pd

# method classifies test data based on nearest neighbors in training data----------------------------------------------------
def k_nearest_neighbors_classify(training_data, test_data, k, distance_measure):
    # add colum to hold classification
    test_data['guess'] = NaN
    # iterate over each row to classify
    # iterate over each example
    for row in test_data.itertuples():
        # only pass learnable attributes to distance function
        row_attr = row[1:-2]
        # find k nearest points to current row from training data
        nearest_points = find_nearest_k(row_attr, training_data, k, distance_measure)

        # get class values of nearest points from training data
        nearest_classes = list(training_data.loc[nearest_points,'class'])
        # find highest count for each class
        max_vote = 0
        guess = ''
        for cl in set(nearest_classes):
            cl_vote = nearest_classes.count(cl)
            # save class with highest count
            if cl_vote > max_vote:
                max_vote = cl_vote
                guess = cl
        # assign most common class amoung nearest points
        test_data.loc[row[0], 'guess'] = guess
    return test_data
# end k_nearest_neighbors_classify() -------------------------------------------------------------------------------------

# method finds agerage of nearest points for regression ------------------------------------------------------------------
def k_nearest_neighbors_regression(training_data, test_data, k, distance_measure):
    test_data['guess'] = NaN
    # iterate over each row to classify
    for row in test_data.itertuples():
        # only pass learnable attributes to distance function
        row_attr = row[1:-2]
        # find k nearest points to current row from training data
        nearest_points = find_nearest_k(row_attr, training_data, k, distance_measure)
        # add identified nearest rows to new subset
        nearest_instances = pd.DataFrame(columns=training_data.columns.values)
        for point in nearest_points:
            point_index = training_data.loc[point]
            nearest_instances = nearest_instances.append(point_index)
        # find average values of subset
        nearest_summary = nearest_instances.describe()
        # cast row mean to tuple to preserve order, extract average regression stat
        guess = tuple(nearest_summary.iloc[1,:])[-1]
        # assign regression value to row
        test_data.loc[row[0], 'guess'] = guess
    return test_data
# end k_nearest_neighbors_regression() ------------------------------------------------------------------------------------

# helper method to find k nearest points -----------------------------------------------------------------------------
def find_nearest_k(target_row, training_data, k, distance_measure):
    # find distances between target row and all training data
    distances = []
    for row in training_data.itertuples():
        # trim row to only include learnable parameters
        test_row = row[1:-1]
        # build list of tuples, first item is traing row index, second is distance
        distances.append((row[0], distance_measure(target_row, test_row)))
    # sort list by distance
    distances.sort(key= lambda x: x[1])
    # find k lowest distance values
    k_nearest = []
    for i in range(k):
        # return row indices of nearest neighbors
        k_nearest.append(distances[i][0])
    return k_nearest
# end find_all_distances() -------------------------------------------------------------------------------------------

# method selects representative subset of training set ---------------------------------------------------------------
def k_neareast_condenser(training_data, distance_measure):
    # initialize Z to empty, columns from training data are passed to preserve order
    Z = pd.DataFrame(columns=training_data.columns.values)
    cur_len = -1
    # shuffle dataset
    shuffled_data = training_data.sample(frac=1)
    # check to see if Z has changed
    while cur_len != Z.shape[0]:
        # update size of Z
        cur_len = Z.shape[0]
        # iterate over each shuffled row
        for row in shuffled_data.itertuples():
            # first item is random
            if Z.shape[0] == 0:
                # always add random first point to Z
                first_point = shuffled_data.loc[row[0],:]
                Z = Z.append(first_point)
            else:
                # find class of current row
                row_class = row[-1]
                # only pass learnable attributes to distance function
                row_attr = row[1:-2]
                # find closest point in dataset
                closest_point = find_nearest_k(row_attr, Z, 1, distance_measure)
                # find class of closest point
                nearest_class = tuple(Z.loc[closest_point,'class'])[0]
                # if not the same class, add to Z
                if nearest_class != row_class:
                    point = shuffled_data.loc[row[0],:]
                    Z = Z.append(point)
        # subtract Z from shuffled dataset
        shuffled_data = shuffled_data[~shuffled_data.isin(Z)].dropna()
    return Z
# end k_neareast_condenser() ------------------------------------------------------------------------------------------
