# Anthony Gray
# Intro to Machine Learning Project 1
# Naive Bayes methods

import utils as u
from numpy import prod as product

# Method uses training set to generate conditional probabilities for each attribute
def learn(training_set, data_set):
    # retrieve classes and respective counts from dataframe 
    classes = training_set.Class.value_counts()
    class_info = {}
    total_instances = training_set.shape[0]
    
    for cl in classes.index:
        # refactor pandas table to dictionary for easier use
        # calculate P(C)
        class_info[cl] = {'P(C)': classes[cl]/total_instances}
        # subset data by class
        sub_df = training_set.loc[training_set.Class == cl]
        # get count of instances by class
        class_instances = sub_df.shape[0]
        # array of P(Fi|C)
        attr_probs = []
        # iterate over columns Fi, except Class Column
        for col in sub_df.iloc[:,:-1].columns:
            # P(Fi|C)
            # includes m-estimate
            prob = (sub_df[col].sum() + 0.001)/(class_instances + 1)
            attr_probs.append(prob)
        # add array to class dictionary
        class_info[cl]['P(Fi=1|C)'] = attr_probs
    export_model(class_info, data_set)
    return class_info
# end learn() -----------------------------------------------------------------------

# method uses attribute probabilities from learn() to guess the classes of instances in datset ---------------------
def classify(test_set, class_info):
    print('Testing Naive Bayes')
    # gives number of attributes per instance
    attr_num = test_set.shape[1]-1

    # Creates list of classes from dataset
    classes = test_set.Class.unique()

    # counters to measure success
    total_tests = 0
    correct_classifications = 0
    for index, row in test_set.iterrows():
        # print(row)
        # pandas rows are hard to iterate over, copy weight vector
        test_row = [None]*attr_num
        # add each row attribute to new weight vector
        for attr in range(0, attr_num):
            test_row[attr] = row[attr]

        # calculate probability for each class 
        class_weights = {}
        for cl in class_info:
            # probabiliity  of class
            prob = predict_class(test_row, class_info[cl])
            # store each class and it's weight
            class_weights[cl] = prob
        # highest probability
        max_weight = max(class_weights.values()) 
        # retrieve key class corresponding to max_weight`
        best_class = [k for k, v in class_weights.items() if v == max_weight] 
        # test for correct classification
        if best_class[0] == row[-1]:
            correct_classifications += 1
        total_tests += 1
    # return success rate
    success_rate = correct_classifications/total_tests
    return success_rate
# end classify() -------------------------------------------------------------------------------------------------------------

# method supports classify() by containing calculating the product of an instances probabilities ----------------------------------
def predict_class(instance_vector, class_info):
    # store P(Fi|C)
    pFi = class_info['P(Fi=1|C)']
    # store P(C)
    pC = class_info['P(C)']

    multiplication_vector = [pC]
    for Fi in range(len(instance_vector)):
        # generate and append P(Fi)
        if instance_vector[Fi] == 1:
            # instance vector contains P(Fi=1|C)
            multiplication_vector.append(pFi[Fi])
        else:
            # find P(Fi=0|C)
            multiplication_vector.append(1-pFi[Fi])
    return product(multiplication_vector)
# end predict_class() -------------------------------------------------------------------------------------------------------------

# writes model to file ------------------------------------------------------------------------------------------------------------
def export_model(model, data_set):
    with open('Proj1_Models.txt', 'a') as the_file:
        the_file.write(f'{data_set.capitalize()} Naive Bayes Model \n')
        for cl in model:
            the_file.write(f'{cl} {model[cl]}\n')
        the_file.write('\n')
# end export_model() --------------------------------------------------------------------------------------------------------------