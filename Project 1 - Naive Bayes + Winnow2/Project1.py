# Anthony Gray
# Intro to Machine Learning Project 1
# Main Project File

import utils as u
import winnow2
import Naive_Bayes

# Method loads data, randomly splits it into test and training sets, then runs one of two algorithms on the sets ---------------- 
def learn_data(algorithm, set_name, theta=None, alpha=None):
    data = u.get_data(set_name)
    print('Generating Test and Training Sets')
    data_split = u.split_to_train_test_sets(data)
    print('Learning weights on Training Set')
    if algorithm == 'winnow':
        print('Verifying weight vector on Test Set')
        weight_vector = winnow2.train(data_split['Training_Set'], set_name, theta=theta, alpha=alpha)
        success_rate = winnow2.test(data_split['Test_Set'], weight_vectors=weight_vector, theta=theta)
    if algorithm == 'nbayes':
        print('Verifying weight vector on Test Set')
        probabilities = Naive_Bayes.learn(data_split['Training_Set'], set_name)
        success_rate = Naive_Bayes.classify(data_split['Test_Set'], probabilities)
    return {'Data Set': set_name, 'Success Rate': success_rate}
# end learn_data() -----------------------------------------------------------------------------------------------------------------

# method parses response from winnow algorithm and prints results ------------------------------------------------------------------
def parse_winnow(res):
    data_set = res['Data Set'].capitalize()
    rates = res['Success Rate'][0][1]
    attempts = rates[0]*100
    successes = rates[1]*100
    print(f'Data Set: {data_set} \nWinnow Results: \n  Percent of Instances Classified: {round(attempts, 2)}% \n'
    f'  Percent of Correct Classifications: {round(successes,2)}%')
# end parse_winnow() ---------------------------------------------------------------------------------------------------------------

# method runs both Naive Bayes and Winnow2 on the same dataset, then prints the results --------------------------------------------
def demo(data_set, theta, alpha):
    print(f'Retrieving {data_set.capitalize()} data')
    win_res = learn_data('winnow', data_set, theta, alpha)
    nabyes_res = learn_data('nbayes', data_set)
    bayes_success = round(nabyes_res['Success Rate'], 4)
    parse_winnow(win_res)
    print(f'Naive Bayes Results:\n Percent of Successful Classifications: {bayes_success*100}%\n')
# end demo()-------------------------------------------------------------------------------------------------------------------------

# Demonstrate Project on each Data Set:
demo('iris', 3, 1.5)
demo( 'glass', 4.5, 1.5)
demo('house', 6, 1.5)
demo('cancer', 5, 1.5)
demo('soybean', 2, 1.5)
