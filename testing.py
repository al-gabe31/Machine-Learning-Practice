parameter_set = []
parameters = [3, 4, 5, 6, 7, 8]
alpha = 0.2

for i in range(len(parameters)):
    curr_set = parameters[:]

    curr_set[i] += alpha
    parameter_set.append(curr_set[:])

    curr_set[i] -= alpha * 2
    parameter_set.append(curr_set[:])

for set in parameter_set:
    print(set)