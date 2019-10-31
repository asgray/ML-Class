# Anthony Gray
# Intro to Machine Learning Project 2
# Stepwise Forward Selection methods

from random import shuffle

# SFS wrapper method ------------------------------------------------------------------------
def SFS(Data, cluster, evaluate, k):
    features = list(Data.iloc[:,:-1])
    shuffle(features)
    selected_features = []
    base_performance = 0
    ret_dat = None
    while features:
        best_performance = 0
        best_feature = ''
        # iterate over all attributes except Class
        for f in features:
            # copy selected features to test vector
            test_features = selected_features[:]
            # add new feature to selected features
            test_features.append(f)
            # add cluster back into the dataset for training
            test_features.append('cluster')
            # cluster data
            clustered_data = cluster(Data[test_features],k)
            # evaluate subset model
            current_performance = evaluate(clustered_data)
            # remove cluster from list again
            test_features.pop()
            # if new best performance found
        if current_performance > best_performance:
            # update performance
            best_performance = current_performance
            # update best new feature
            best_feature = f
        # if an overall performance increase occurs
        if best_performance > base_performance:
            # update overall performance increase
            base_performance = best_performance
            # save best feature
            selected_features.append(best_feature)
            # remove chosen feature from remaining pool
            features.remove(best_feature)
            ret_dat = clustered_data.sample(5)
        # if no performance increase occurs, end loop
        else:
            break
    return {'Selected Features: ': selected_features, 'Average Silhouette Coefficient: ': base_performance, '5 Sample Instances: \n': ret_dat}
# end SFS() -----------------------------------------------------------------------