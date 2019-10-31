# Anthony Gray
# Intro to Machine Learning Project 1
# Winnow2 methods

from numpy import mean as mean 

# demote() supports learn() ---------------------------------------------------
def demote(instance, weight_vector, alpha):
    new_weights = [w for w in weight_vector]
    for i in range(len(instance)):
        if instance[i] == 1:
            # if attribute is 1, divide by alpha
            new_weights[i] = weight_vector[i]/alpha
    return new_weights
# end demote() ----------------------------------------------------------------

# promote() supports learn() --------------------------------------------------
def promote(instance, weight_vector, alpha):
    new_weights = [w for w in weight_vector]
    for i in range(len(instance)):
        if instance[i] == 1:
            # if attribute is 1, multiply by alpha
            new_weights[i] = weight_vector[i]*alpha
    return new_weights
# end promote() ---------------------------------------------------------------

# method learns weight vector from training set -------------------------------
def learn(training_set, target_class, other_classes, theta, alpha):
    # gives number of attributes per instance
    attr_num = training_set.shape[1]-1

    # instantiate vector of 1s with same length as the number of attributes
    initial_weights = [1]*attr_num
    weights = initial_weights
    convergence = False
    
    while not convergence:
        # iterate over all rows in the training set
        for index, row in training_set.iterrows():
            # get current weight vector
            test_weights = weights

            # copy instance
            test_instance = [None]*attr_num
            # add each row attribute to new row
            for attr in range(0, attr_num):
                test_instance[attr] = row[attr]

            # weight all values
            weighted_instance = test_instance
            for attr in range(0, attr_num):
                weighted_instance[attr] = test_instance[attr]*test_weights[attr]

            # f(x) > theta
            if sum(weighted_instance) > theta:
                # guess class is 1 or greater
                predict_class = target_class
                guess = 1
            else:
                # f(x) < theta
                # guess class is 0
                predict_class = other_classes
                guess = 0
            
            # retrieve Class of instance
            actual_class = row[attr_num]
            # correct answer, do nothing
            if actual_class in predict_class:
                pass
            # if classified wrong
            else:
                # guessed 1 when it was 0: Demote
                if guess == 0:
                    test_weights = demote(test_instance, test_weights, alpha)
                # guessed 0 when it was 1: Promote
                else:
                    test_weights = promote(test_instance, test_weights, alpha)
            # test for convergence
            if test_weights == weights:
                convergence = True
            weights = test_weights
            # reset weights for next pass
    return weights
# end learn()--------------------------------------------------------------------------------------------------

# train() calls learn() for each class in the dataset, returns a dictionary of the classes and thier weight vectors ---------------
def train(training_set, data_set, theta, alpha):
    # Creates list of classes from dataset
    classes = list(training_set.Class.unique())
    class_vectors = {}
    # run learn() for each class
    for cl in classes:
        other_classes = [c for c in classes if c != cl]
        class_vectors[cl] = learn(training_set, cl, other_classes, theta, alpha)
    # return weight vectors for learning each class
    export_model(class_vectors, data_set)
    return class_vectors
# end train() -----------------------------------------------------------------------------------------------------------------------

# runs each classes respective weight vector on the test set ------------------------------------------------------------------------
def test(training_set, weight_vectors, theta):
    print('Testing Winnow2')
    res = []
    for cl in weight_vectors:
        success_rate = classify_by_best_guess(training_set, weight_vectors, theta)
        res.append([cl,success_rate])
    return res
# end test() ------------------------------------------------------------------------------------------------------------------------

# Method to test learned Winnow2 model on test set, returns % success ----------------------------------------
def classify_by_best_guess(test_set, weight_vectors, theta):
    # gives number of attributes per instance
    attr_num = test_set.shape[1]-1

    # counters to measure success
    total_guesses = 0
    correct_classifications = 0
    total_tests = 0
    no_guesses = 0

    # iterate over all rows in the training set
    for index, row in test_set.iterrows():
        total_tests += 1
        # retrieve Class of instance
        actual_class = row[attr_num]
        
        possible_classifications = {}
        # check each classes weight vector
        for cl in weight_vectors:
            # copy instance
            test_instance = [None]*attr_num
            # add each row attribute to new row
            for attr in range(0, attr_num):
                test_instance[attr] = row[attr]
            # weight all values
            for attr in range(0, attr_num):
                test_instance[attr] = test_instance[attr]*weight_vectors[cl][attr]
            f_of_x = sum(test_instance)
            if f_of_x > theta:
                possible_classifications[cl] = f_of_x
        predict_class = ''
        if possible_classifications:
            # find highest weighted sum
            max_f_of_x = max(possible_classifications.values())
            # find class associated with highest weighted sum
            predict_class = [k for k, v in possible_classifications.items() if v == max_f_of_x] 
            total_guesses += 1
        else:
            no_guesses += 1

        if actual_class == predict_class :
            correct_classifications += 1
    # return % of guesses attempted and % success
    res = [no_guesses/total_tests]
    if total_guesses > 0:
        res.append(correct_classifications/total_guesses)
    return res
# end classify() ----------------------------------------------------------------------------------------------

# writes model to file ------------------------------------------------------------------------------------------------------------
def export_model(model, data_set):
    with open('Proj1_Models.txt', 'a') as the_file:
        the_file.write(f'{data_set.capitalize()} Winnow Model \n')
        for cl in model:
            the_file.write(f'{cl} {model[cl]}\n')
        the_file.write('\n')
# end export_model() --------------------------------------------------------------------------------------------------------------