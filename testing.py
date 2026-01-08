from library import *

import matplotlib.pyplot as plt
import numpy as np
import random

mu = 0
sigma = 2.3
num_samples = 100

rng = np.random.default_rng()
sample_differences = rng.normal(loc=mu, scale=sigma, size=num_samples)




actual_a = np.random.normal(loc=5, scale=1.2, size=1)
actual_b = np.random.normal(loc=0, scale=2.3, size=1)
actual = lambda x: actual_a[0] + (actual_b[0] * x)

print(f'actual function: y = {float(actual_a)} + {float(actual_b)}x')

df_x = [random.uniform(0, 10) for i in range(num_samples)]
df_y = [actual(df_x[i]) + sample_differences[i] for i in range(num_samples)]

params, r2 = linear_regression(df_x, df_y)
print(params)



x_list = [x/100 for x in range(0, 1000)]
y_predict = [params[0] + (x * params[1]) for x in x_list]

plt.scatter(df_x, df_y, color='blue')
plt.plot(x_list, y_predict, color='red', label=f'y = {round(params[0], 2)} + {round(params[1], 2)}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'r^2 = {r2}')
plt.legend(loc='upper right')

plt.show()