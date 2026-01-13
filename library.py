import math
import random
import numpy as np

'''
x_data: 2D array of inputs (each row representing a single data point)
    if 1D array of inputs is passed, the function will automatically convert it into 2D where each row is its own 1D array of size 1
y_data: 1D array of outputs
alpha: used for learning rate of regression model
epochs: number of simulation runs
params: list of 1+n scalars used to get prediction value (y-hat) where n is the # of columns
regularization: if not blank, set it to "L1" or "L2"
lambda_const: constant value to use for regularization penalty

returns the refined parameters & r^2
'''
def linear_regression(x_data, y_data, alpha = 0.02, epochs = 100, params = [], regularization = '', lambda_const = 1):

    # prepping data here
    # converts x_data from 1D to 2D if necessary
    if not isinstance(x_data[0], list):
        x_data = [[row] for row in x_data]

    # if no parameters were passed, set them equal to 0
    if len(params) == 0:
        params = [0 for i in range(len(x_data[0]))]
        params.append(0) # adding extra 0 for the bias

    n = len(x_data)

    # returns y-hat
    def prediction(input_list):
        result = params[0] # the bias

        for i in range(len(input_list)):
            result += params[i + 1] * input_list[i] # coefficient * input
        
        return result # y-hat
    
    # returns sum of residuals
    def sum_residuals(multiplier_index = -1):
        
        multipliers = [1 for i in range(len(x_data))] if multiplier_index == -1 else [row[multiplier_index] for row in x_data]

        sum_error = 0

        for i in range(len(x_data)):
            sum_error += (y_data[i] - prediction(x_data[i]))*multipliers[i]

        return sum_error



    # running simulations and updating parameters here
    for epoch in range(epochs):
        params[0] += 2 * alpha * sum_residuals() / n # updating bias term

        # updating the coefficient terms
        for i in range(1, len(params)):
            reg_penalty = 0

            if regularization.upper() == 'L1': # lasso regression
                reg_penalty = alpha * lambda_const * np.sign(params[i])
            elif regularization.upper() == 'L2': # ridge
                reg_penalty = 2 * alpha * lambda_const * params[i]
            
            params[i] += (2 * alpha * sum_residuals(i - 1) / n) - reg_penalty

    # # defining r^2
    # sse, sst = 0, 0
    # y_average = sum(y_data) / n

    # for i in range(n):
    #     sse += (y_data[i] - prediction(x_data[i]))**2
    #     sst += (y_data[i] - y_average)**2

    # r2 = 1 - (sse/sst)

    return params




'''
x_data: 2D array of inputs (each row representing a single data point)
    if 1D array of inputs is passed, the function will automatically convert it into 2D where each row is its own 1D array of size 1
y_data: 1D array of outputs
alpha: used for learning rate of regression model
epochs: number of simulation runs
params: list of 1+n scalars used to get prediction value (y-hat) where n is the # of columns

returns the refined parameters & r^2
'''
def logistic_regression(x_data, y_data, alpha = 0.02, epochs = 100, params = []):

    # prepping data here
    # converts x_data from 1D to 2D if necessary
    if not isinstance(x_data[0], list):
        x_data = [[row] for row in x_data]

    # if no parameters were passed, set them equal to 0
    if len(params) == 0:
        params = [0 for i in range(len(x_data[0]))]
        params.append(0) # adding extra 0 for the bias

    n = len(x_data)

    # return y-hat
    def prediction(input_list):
        polynomial = params[0]

        for i in range(len(input_list)):
            polynomial += params[i + 1] * input_list[i] # coefficient * input

        return 1 / (1 + (math.e**(-1 * polynomial))) # logistic function
    
    # return sum of residuals
    def sum_residuals(multiplier_index = -1):
        multipliers = [1 for i in range(len(x_data))] if multiplier_index == -1 else [row[multiplier_index] for row in x_data]

        sum_error = 0

        for i in range(len(x_data)):
            sum_error += (y_data[i] - prediction(x_data[i]))*multipliers[i]

        return sum_error
    
    # running simulations and updating parameters here
    for epoch in range(epochs):
        params[0] += alpha * sum_residuals() / n # updating bias term

        # updating the coefficient terms
        for i in range(1, len(params)):
            params[i] += alpha * sum_residuals(i - 1) / n

    return params




'''
x_data: 2D array of inputs (each row representing a single data point)
    if 1D array of inputs is passed, the function will automatically convert it into 2D where each row is its own 1D array of size 1
y_data: 1D array of outputs
sample_rate: float in the range (0, 1) that determines the proportion of data to be sampled
alpha: used for learning rate of regression model
epochs: number of simulation runs
params: list of 1+n scalars used to get prediction value (y-hat) where n is the # of columns
setting: determines if mini-batch will used linear or logistic regression model

returns the refined parameters & r^2
'''
def mini_batch_gradient_descent(x_data, y_data, sample_rate, alpha = 0.02, epochs = 100, params = [], setting = 'linear'):
    # first combine x & y dataset
    combined = [(x_data[i], y_data[i]) for i in range(len(x_data))]
    n = math.floor(len(x_data) * sample_rate) # the number of samples

    for i in range(epochs):
        curr_sample = random.sample(combined, n) # take n random sample of the combined dataset
        curr_x_sample = [curr_sample[i][0] for i in range(n)]
        curr_y_sample = [curr_sample[i][1] for i in range(n)]

        # learn from current sample
        if setting == 'linear':
            params = linear_regression(curr_x_sample, curr_y_sample, alpha, epochs=1, params=params)
        elif setting == 'logistic':
            params = logistic_regression(curr_x_sample, curr_y_sample, alpha, epochs=1, params=params)

    return params