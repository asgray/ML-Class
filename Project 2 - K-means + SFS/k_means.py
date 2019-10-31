# Anthony Gray
# Intro to Machine Learning Project 2
# K-Means Clustering methods

import utils as u
import random

# main K-Means method -----------------------------------------------------------------
# accepts data number of clusters k, assigns each instance to one of k clusters
def K_Means(Data, k):
    # initialise mu_1 ... mu_k randomly
    centroids = {}
    crit_stats = Data.iloc[:,:-1].describe()
    for i in range(k):
        centroids[i+1] = random_centroid(crit_stats)
    # repeat
    convergence = False
    while not convergence:
        # iterate over each instance
        # exclude cluster column from calculation
        for x_i in Data.itertuples():
            min_dist = 999999
            c = -1
            # c = argmin_mu d(x_i, mu_j) d() is distance between x_i and mu_j
            for c_i in centroids:
                # trim values of x_i to exclude index and cluster for calculations
                dist = u.euclid_dist(x_i[1:-1], centroids[c_i])
                if dist < min_dist:
                    min_dist = dist
                    c = c_i
            # assign x_i to cluster c
            Data.at[x_i[0],'cluster'] = c
        # recalcualte all mu_j based on new clusters
        new_mu = recalc_mu(Data)
        # until no change in all mu
        convergence = True
        for mu in new_mu:
            # update mu values that have changed, prevent loop from ending
            if not centroids[mu] == new_mu[mu]:
                centroids[mu] = new_mu[mu]
                convergence = False
    return Data
# end K_means() --------------------------------------------------------------------

# method generates a random centroid using summary stats
# random range is (min-2*std):(max+2*std) for each dimension -----------------------
def random_centroid(crit_stats):
    centroid = []
    # iterate over each attribute
    attr_num = crit_stats.shape[1]
    for i in range(attr_num):
        # extract summary stats for attribute
        min_i = crit_stats.iloc[3, i]
        max_i = crit_stats.iloc[7, i]
        std_i = crit_stats.iloc[2, i]
        range_min = min_i - (2*std_i)
        range_max = max_i + (2*std_i)
        # generate random number based on summary stats
        centroid.append(random.uniform(range_min, range_max))
    return centroid
# end random_centroid() --------------------------------------------------------------

# method uses pandas .describe() to find the mean point by cluster -------------------
def recalc_mu(dataframe):
    # find all unique clusters
    clusters = dataframe.cluster.unique()
    response = {}
    for cl in clusters:
        # subset dataframe by cluster
        c = dataframe.iloc[:,-1]==cl
        sub_df = dataframe[c]
        # decribe returns table of summary statistics for each column
        sub_desc = sub_df.describe()
        # extract the means of each attribute, cast to list
        sub_desc = sub_desc.iloc[1,:-1].tolist()
        response[cl] = sub_desc
    return response
# end describe_clusters() ------------------------------------------------------------