import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def Problem05(x):
     return-(1.4 - 3*x)* math.sin(18*x)


def train_gaussian_process(x_train,y_train):
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 5, normalize_y = False,random_state=11)
    gpr.fit(x_train,y_train)
    return gpr
def expected_improvement(x, y_pred, std_dev, f_min):
    arg = f_min - y_pred
    EI_exploit = arg * norm.cdf(arg / std_dev)
    EI_explor = std_dev * norm.pdf(arg / std_dev)
    EI = EI_exploit + EI_explor
    return EI


def maximize_acquisition_function(gpr, x_train, y_train, f_min, x_low, x_high):
    n_grid_points = 5000
    x_grid = np.linspace(x_low, x_high, n_grid_points)
    EI_max = -1
    x_next = x_grid[0]

    for x_val in x_grid:
        y_pred, y_std = gpr.predict(x_val.reshape(-1, 1), return_std=True, return_cov=False)
        EI_val = expected_improvement(x_train, y_pred[0], y_std[0], f_min)
        if EI_val > EI_max:
            EI_max = EI_val
            x_next = x_val

    return x_next


def initial_sample(n_init, x_low, x_high):
    x_0 = random.uniform(x_low, x_high)
    y_0 = Problem05(x_0)
    x_train = np.array([[x_0]])
    y_train = np.array([[y_0]])
    i = 1
    while i < n_init:
        x_i = random.uniform(x_low, x_high)
        y_i = Problem05(x_i)
        x_train = np.vstack((x_train, x_i))
        y_train = np.vstack((y_train, y_i))
        i = i + 1
    return x_train, y_train


def evaluate_x_next(x_next):
    y_next = Problem05(x_next)
    return y_next


'''
Use the four equally spaced points from part a
x_train and y_train 

'''

x_low = 0
x_high = 1.2
n_init = 4
x_train, y_train = initial_sample(n_init, x_low, x_high)
f_min = np.min(y_train)
iteration_max = 10
iteration = 0
y_best = min(y_train)

while iteration < iteration_max:
    gpr = train_gaussian_process(x_train, y_train)
    x_next = maximize_acquisition_function(gpr, x_train, y_train, f_min, x_low, x_high)
    y_next = evaluate_x_next(x_next)

    if y_next < y_best:
        y_best = y_next
        x_best = x_next

    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))
    iteration = iteration + 1
    print(iteration, x_next, y_next)
print(f"Best point found: x = {x_best}, f(x) = {y_best}")
