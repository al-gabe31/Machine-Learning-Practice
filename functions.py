from math import sqrt

def linear_regression(dependent: float, independent: list, alpha = 0.002):
    # contains information about the multivariable linear equation:
    # p0 + p1*x1 + p2*x2 + p3*x3 + ... + pn*xn
    parameters = [0 for i in range(len(independent) + 1)]

    # the cost function to be minimized
    def calculate_cost(curr_parameters = parameters):
        total_cost = 0

        # goes through each data point
        for i in range(len(dependent)):
            curr_prediction = curr_parameters[0]

            # calculates the current prediction based on inputs
            for j in range(len(independent)):
                curr_prediction += curr_parameters[j + 1] * independent[j][i]
            
            total_cost += (dependent[i] - curr_prediction)**2 # adds the R squared regression to the cost

        return total_cost # returns the cost (sum of squared regressions)

    MAX_ITERS = 1000

    # changes the parameters based on which one reduces the cost function the most
    for i in range(MAX_ITERS):
        parameter_set = []

        # generates all possible partial derivatives
        for j in range(len(parameters)):
            curr_set = parameters[:]

            curr_set[j] += alpha
            parameter_set.append(curr_set[:])

            curr_set[j] -= alpha * 2
            parameter_set.append(curr_set[:])
        
        # goes through each partial derivative and find the gradient (largest descent)
            
        curr_min_cost = calculate_cost()
        params_changed = False
        j_index = 0

        # print(f"current parameters {parameters} has cost {curr_min_cost}")
        for j in range(len(parameter_set)):
            curr_cost = calculate_cost(parameter_set[j])
            # print(f"set {parameter_set[j]} has cost {curr_cost}")

            # new lowest cost parameter set found
            if curr_cost < curr_min_cost:
                # print(f"New min cost found!")
                curr_min_cost = curr_cost
                params_changed = True
                j_index = j
        # print()
        
        

        if params_changed == False: # we're currently at the local minimum
            # print(f"Ended at iteration {i}")
            break
        elif j_index % 2 == 0: # gradient is increase in parameters
            parameters[j_index // 2] += alpha * (calculate_cost() - curr_min_cost)
            # parameters[j // 2] += alpha
        else: # gradient is decrease in parameters
            parameters[j_index // 2] -= alpha * (calculate_cost() - curr_min_cost)
            # parameters[j // 2] -= alpha
    
    return parameters