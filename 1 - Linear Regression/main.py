import csv
import matplotlib.pyplot as plt
from functions import *

MAX_ROWS = 1000
file = 'C:/Users/alfon/OneDrive/Desktop/Python_Code/machine_learning_practice/1 - linear regression/insurance.csv'

age = [] # index 0
sex = [] # index 1  male = 0 | female = 1
bmi = [] # index 2
children = [] # index 3
smoker = [] # index 4   no = 0 | yes = 1
region = [] # index 5   NE = 0 | NW = 1 | SW = 2 | SE = 3
charges = [] # index 6

# reads the csv file on insurance
with open(file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    header = next(csv_reader)
    # print(header)

    curr_row = 1
    for row in csv_reader:
        age.append(int(row[0]))
        if row[1] == 'male':
            sex.append(0) # male
        else:
            sex.append(1) # female
        bmi.append(float(row[2]))
        children.append(int(row[3]))
        if row[4] == 'no':
            smoker.append(0) # no
        else:
            smoker.append(1) # yes
        if row[5] == 'northeast':
            region.append(0) # northeast
        elif row[5] == 'northwest':
            region.append(1) # northwest
        elif row[5] == 'southwest':
            region.append(2) # southwest
        else:
            region.append(3) # southeast
        charges.append(float(row[6]))
        # print(row)

        curr_row += 1
        # if curr_row > MAX_ROWS:
        #     break



# curr_independent = region

# age, sex, bmi, children, smoker, region
parameters = linear_regression(charges, [age, sex, bmi, children, smoker, region], 0.0002)
print(parameters)

# linear_regression_values = [parameters[0] + parameters[1] * x for x in curr_independent]



# plt.style.use('fivethirtyeight')
# plt.scatter(curr_independent, charges)
# plt.plot(curr_independent, linear_regression_values, color='red')
# plt.xlabel('BMI Index')
# plt.ylabel('Charges')
# plt.show()
        


# f = lambda a, b, c, d, e, f : 0 + 241.732*a + 0*b + 126.043*c + 0*d + 329.527*e + 0*f
# g = lambda a, b, c, d, e, f : 13270.423 + 107.386*a + 0*b + -116.618*c + 0*d + 291.894*e + 0*f


# f_score = 0
# g_score = 0
# for i in range(len(charges)):
#     f_error = abs(f(age[i], sex[i], bmi[i], children[i], smoker[i], region[i]) - charges[i])
#     g_error = abs(g(age[i], sex[i], bmi[i], children[i], smoker[i], region[i]) - charges[i])

#     if f_error <= g_error:
#         f_score += 1
#     else:
#         g_score += 1

# print(f"f_score = {f_score}")
# print(f"g_score = {g_score}")