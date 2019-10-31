# Anthony Gray
# Intro to Machine Learning Project 5
# Logistic Regression File

from numpy import NaN as NaN
import numpy as np

# method calls learn_class for each class in dataset to generate complete set of models ------------------------------------------------
def learn_models(dat, eta, data_set, export=False):
    classes = dat['Class'].unique()
    models = {}
    for cl in classes:
        models[cl] = learn_class(dat, eta, cl)
    if export:
        export_model(models, data_set)
    return models
# end learn_models() ------------------------------------------------------------------------------------------------------------------

# method uses the logistic funtion to generate a vector of weights for identifying a target class -------------------------------------
def learn_class(dat, eta, target_class):
    # initialize random weight vector
    W = list(np.random.uniform(-0.01,0.01, [1,dat.shape[1]])[0])
    count = 0
    convergence = False
    while not convergence:
        delta_W = [0]*len(W)
        # iterate over each instance
        for i in range(dat.shape[0]):
            o = 0
            instance = list(dat.iloc[i,:])
            # attributes X, W_0 is 1
            X = [1] + instance[:-1]
            # multiply Xj*Wj
            for j in range(len(X)):
                o += X[j]*W[j]
            # use sigmoid function
            y = 1/(1+ np.exp(-o))
            # assign r based on class
            r = 0
            if target_class == instance[-1]:
                r = 1
            # update delta_W
            for j in range(len(delta_W)):
                delta_W[j] += (r-y)*X[j]
        # update W
        new_W = [0]*len(W)
        for j in range(len(W)):
            new_W[j] = W[j] + delta_W[j]*eta
        # check for convergence
        test_new_W = [round(elem, 3) for elem in new_W]
        test_W = [round(elem, 3) for elem in W]
        # end if the weights have not significantly changed, or if 200o iterations have occurred
        if test_new_W == test_W or count > 2000:
            convergence = True
        W = new_W
        count += 1
    return W
# end learn_class() -----------------------------------------------------------------------------------------------------------------

# method adds a class guess to test set based on models provided from the learing algorithm ----------------------------------------- 
def classify(dat, models):
    dat['Guess'] = NaN
    # iterate over each instance
    for i in range(dat.shape[0]):
        o = 0
        instance = list(dat.iloc[i,:])
        # attributes X, W_0 is 1
        X = [1] + instance[:-2]
        best_y = 0
        best_class = ''
        # test each model to find best score
        for cl in models:
            # class specific weight vector
            W = models[cl]
            # multiply Xj*Wj
            o = 0
            for j in range(len(X)):
                o += X[j]*W[j]
            # use sigmoid function
            y = 1/(1+ np.exp(-o))
            # save highest y
            if y > best_y:
                best_y = y
                best_class = cl
        # assign guess column
        dat.iloc[i,-1] = best_class
# end classify() ------------------------------------------------------------------------------------------------------------------

# writes model to file ------------------------------------------------------------------------------------------------------------
def export_model(model, data_set):
    with open('Gray_Proj5_models.txt', 'a') as the_file:
        the_file.write(f'{data_set.capitalize()} Logistic Regression Model \n')
        for cl in model:
            the_file.write(f'{cl} {model[cl]}\n')
        the_file.write('\n')
# end export_model() --------------------------------------------------------------------------------------------------------------
