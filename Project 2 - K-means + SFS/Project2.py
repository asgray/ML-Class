# Anthony Gray
# Intro to Machine Learning Project 2
# Main Project File
# import libraries
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from collections import OrderedDict

# import project files
import utils as u
from sfs import SFS
from k_means import K_Means

# names of datasets with associated k values
data_sets = {'iris': 3, 'glass': 7, 'spam': 2}

# method to import dataset, remove Class and add Cluster column -------------------------------
def prep_data(set_name):
    data = u.get_data(set_name)
    data = data.iloc[:,:-1]
    data['cluster'] = -1
    return data
# end prep_data() -----------------------------------------------------------------------------

# method to run SFS/K-Means on each dataset and print results to console ----------------------
def project_demo(data_sets):
    print('\n')
    for name in data_sets:
        print(f'Finding best features for {name} dataset...')
        data = prep_data(name)
        if name == 'spam':
            data = data.sample(175)
        sub_features = SFS(data, K_Means, u.silhouette_score, data_sets[name])
        for f in sub_features:
            print(f, sub_features[f])
        print('\n')
# end project_demo() --------------------------------------------------------------------------

# method runs SFS/K-Means 50 times on each dataset, records the SC and feature sets returned --
def find_general_SC(data_sets):
    SCs = {}
    sig_features = {}
    for name in data_sets:
        data = prep_data(name)
        features = {}
        # reduce Spambase for time
        if name == 'spam':
            data = data.sample(175)
        for i in range(50):
            sub_features = SFS(data, K_Means, u.silhouette_score, data_sets[name])
            # round to 1 decimal place for binning
            sc = round(sub_features['Average Silhouette Coefficient: '], 1)
            feat = sub_features['Selected Features: ']
            if sc in SCs:
                SCs[sc] += 1
            else:
                SCs[sc] = 1
            for f in feat:
                if f in features:
                    features[f] += 1
                else:
                    features[f] = 1
        sig_features[name] = features
    print(SCs, sig_features)
# end find_general_SC() ------------------------------------------------------------------------

# method runs SFS/K-Means on each dataset 50 times, records SCs for each set -------------------
def find_SC_by_set(data_sets):
    all_sets = {}
    for name in data_sets:
        SCs = {}
        data = prep_data(name)
        if name == 'spam':
            data = data.sample(175)
        for i in range(50):
            sub_features = SFS(data, K_Means, u.silhouette_score, data_sets[name])
            # round to 1 decimal place for binning
            sc = round(sub_features['Average Silhouette Coefficient: '], 1)
            if sc in SCs:
                SCs[sc] += 1
            else:
                SCs[sc] = 1
        all_sets[name] = SCs
    print(all_sets)
# end find_SC_by_set() -------------------------------------------------------------------------

# method prints out information regarding the range of eacha attribute for each dataset --------
def describe_attr_range(data_sets):
    for s in data_sets:
        print(s)
        data = prep_data(s)
        stats = u.describe_classes(data)
        for stat in stats:
            p = stats[stat]
        # data minimums
        mi = p.iloc[3]
        # data maximums
        ma = p.iloc[7]
        # data ranges
        dif = ma-mi
        print(dif)
        print(sum(dif)/len(dif))
        print(max(dif))
# end describe_attr_range() --------------------------------------------------------------------

# method to plot two columns against each other, color by cluster
def plot_w_SC(set_name, x_val, y_val, k):
    data = prep_data(set_name)
    data = K_Means(data[[x_val, y_val, 'cluster']], k)
    if set_name == 'spam':
        data = data.sample(175)
    sc = u.silhouette_score(data)
    cmap = cm.get_cmap('Spectral')
    plt.xlabel(x_val)
    plt.ylabel(y_val)
    plt.title(f'{set_name.capitalize()} Data (SC: {round(sc, 5)})')
    plt.scatter(data[x_val], data[y_val], c=data['cluster'], cmap=cmap)
    plt.show()
# end plot_w_SC() ------------------------------------------------------------------------------
 
project_demo(data_sets)
plot_w_SC('iris', 'Petal_Length', 'Petal_Width', 3)
# plot_w_SC('glass', 'RI', 'Ca', 7)
# plot_w_SC('spam', 'crl_average', 'crl_total', 2)


# find_general_SC(data_sets)
# find_SC_by_set(data_sets)
# describe_attr_range(data_sets)
