# Anthony Gray
# Intro to Machine Learning Project 6
# Backpropagation Methods

import numpy as np
from numpy import NaN as NaN
from random import random
from math import exp

# method feeds instance through network, assigns class label based on output ------------------------------------
def classify(model, test_set):
    # break down model
    network = model[0]
    classes = model[1]
    # add guess column
    test_set['Guess'] = NaN
    # iterate over test examples
    for i in range(test_set.shape[0]):
        # extract instance values
        instance = list(test_set.iloc[i,:])[:-2]
        # instance = instance[:-2]
        # run through network
        outputs = forward_propagate(network, instance)
        # identify best node
        max_node = outputs.index(max(outputs))
        # assign class based on classes vector
        class_guess = classes[max_node]
        test_set.iloc[i, -1] = class_guess
    return test_set
# end classify() -----------------------------------------------------------------------------------------------

# method randomly initalizes network, then trains it ---------------------------------------------------------------
def build_network(training_set, n_hidden_layers, n_neurons_per, eta):
    # number of inputs given by number of attributes in dataset
    n_inputs = training_set.shape[1] - 1
    # indentify possible classes, sort list
    classes = list(training_set['Class'].unique())
    classes.sort()
    # number of outputs given by number of classes in dataset
    n_outputs = len(classes)
    # initialize network
    net = initialize_network(n_inputs, n_hidden_layers, n_neurons_per, n_outputs)
    # train network
    train_network(net,training_set, eta, classes)
    # return trained network and ordered list of classes to use for training
    return [net, classes]
# end build_network() ---------------------------------------------------------------------------------------------

# method generates a randomly initialized complete neural network -------------------------------------------------
# n_inputs = number of attributes in dataset
# n_nidden layers is tunable number of hidden layers generated
# n_nerons_per is number of neurons in each hidden layer
# n_outputs is number of classes in dataset
def initialize_network(n_inputs, n_hidden_layers, n_neurons_per, n_outputs):
    network = []
    if n_hidden_layers > 0:
        # generate hidden layers
        for n in range(n_hidden_layers):
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for j in range(n_neurons_per)]
            network.append(hidden_layer)
        # add output layer
        output_layer = [{'weights':[random() for i in range(n_neurons_per + 1)]} for j in range(n_outputs)]
        network.append(output_layer)
    else:
        output_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for j in range(n_outputs)]
        network.append(output_layer)
    return network
# end initialize_network() ---------------------------------------------------------------------------------------

# method trains a network until convergence (error falls below a threshold) -------------------------------------
def train_network(network, training_set, eta, classes):
    # indentify possible classes, sort list
    # classes = list(training_set['Class'].unique())
    # classes.sort()
    # repeat until convergence
    converged = False
    epochs = 0
    while not converged:
        error = 0
        # shuffle training set
        training_set = training_set.sample(frac=1)
        # iterate over instances
        for row in training_set.itertuples():
            vals = list(row)
            # extract attribute values
            row = vals[1:-1]
            # extract class
            target = vals[-1]
            # identify which output neuron should be greatest
            target_index = classes.index(target)
            outputs = forward_propagate(network, row)
            # generate expected output vector,s et to 0
            expected = [0 for i in range(len(classes))]
            # set corresponding expected output to 1
            expected[target_index] = 1
            # backpropagate and update
            back_propagation(network, expected)
            update_weights(network, row, eta)
            # calculate error
            error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        epochs += 1
        # test for convergence
        if error < 1 or epochs > 500:
            converged = True
# end train_network() -------------------------------------------------------------------------------------------

# method calcualtes neuron activation for an input ---------------------------------------------------------------
def calculate_activation(weights, inputs):

    # reassign bias as activation
    activation = weights[-1]
    # for i in range(len(weights)-1):
    for i in range(len(inputs)):
        # sum weight*input with bias
        activation += weights[i] * inputs[i]
    # return sigma(z)
    return 1.0 / (1.0 + exp(-activation))
# calculate_activation() ----------------------------------------------------------------------------------------

# method forward propagates an input to an output --------------------------------------------------------------
def forward_propagate(network, row):
    # propagation variable
    inputs = row
    for layer in network:
        new_inputs = []
        # recalcualte activation values based on weights
        for neuron in layer:
            neuron['activation'] = calculate_activation(neuron['weights'], inputs)
            new_inputs.append(neuron['activation'])
        # reassign propagation variable
        inputs = new_inputs
    # after all layers, inputs holds actvation values of output layer
    return inputs
# end forward_propagate() ---------------------------------------------------------------------------------------

# method to backpropagate error ---------------------------------------------------------------------------------
def back_propagation(network, expected):
    # iterate from output layer back
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        # if not output layer
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                # calculate error for weights leading to next layer
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['updates'])
                errors.append(error)
        else:
            # if output layer 
            for j in range(len(layer)):
                neuron = layer[j]
                # compare outputs to expected values
                errors.append(expected[j] - neuron['activation'])
        # update delta
        for j in range(len(layer)):
            neuron = layer[j]
            output = neuron['activation']
            neuron['updates'] = errors[j] * output * (1.0 - output)
# end back_propagation() ---------------------------------------------------------------------------------------

# method updates network weights based on delta ----------------------------------------------------------------
def update_weights(network, row, eta):
    # update layer by layer
    for i in range(len(network)):
        # first layer inputs are data instance
        inputs = row[:-1]
        # otherwise...
        if i != 0:
            # reassign weights in first layer
            inputs = [neuron['activation'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                # update each weight
                neuron['weights'][j] += eta * neuron['updates'] * inputs[j]
            # update bias
            neuron['weights'][-1] += eta * neuron['updates']
# end update_weights() ------------------------------------------------------------------------------------------
