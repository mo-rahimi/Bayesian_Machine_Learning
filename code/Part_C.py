import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def Problem05(x):
    return -(1.4 - 3 * x) * math.sin(18 * x)

# Generate X and Y train
x_plot = np.linspace(0,1.2,1000)
test_f = np.vectorize(Problem05)
y_plot = test_f(x_plot)
x_low = 0
x_high = 1.2
n= 3  #there will be 3 intervals between 4 points, as I need four equally spaced points.
x1= x_low
x2= x1 + x_high/n
x3=x2 + x_high/n
x4=x_high
x_train=np.array([[x1, x2, x3,x4]]).T
y_train=np.array([[Problem05(x1), Problem05(x2), Problem05(x3),Problem05(x4)]]).T

def evaluate_x_next(x_next):
    y_next = Problem05(x_next)
    return y_next

def upper_confidence_bound(x, y_pred, std_dev, k):
    UCB = y_pred - k * std_dev
    return UCB

def maximize_acquisition_function_ucb(gpr, x_train, y_train, k, x_low, x_high):
    n_grid_points = 2000
    x_grid = np.linspace(x_low, x_high, n_grid_points)
    UCB_max = -np.inf
    x_next = x_grid[0]

    for x_val in x_grid:
        y_pred, y_std = gpr.predict(x_val.reshape(-1, 1), return_std=True, return_cov=False)
        UCB_val = upper_confidence_bound(x_val, y_pred[0], y_std[0], k)
        if UCB_val > UCB_max:
            UCB_max = UCB_val
            x_next = x_val

    return x_next

def train_gaussian_process(x_train,y_train):
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 5, normalize_y = False,random_state=11)
    gpr.fit(x_train,y_train)
    return gpr

gpr = train_gaussian_process(x_train, y_train)
k_values = [0.5, 3, "diminishing"]

for k_index, k in enumerate(k_values):
    print(f"Experiment with k = {k}")
    x_train_init = x_train.copy()
    y_train_init = y_train.copy()
    iter_max = 8
    iteration = 0
    y_best = min(y_train_init)

    while iteration < iter_max:
        if k == "diminishing":
            k_iter = 4 / (iteration + 1)
        else:
            k_iter = k

        x_next = maximize_acquisition_function_ucb(gpr, x_train_init, y_train_init, k_iter, x_low, x_high)
        y_next = evaluate_x_next(x_next)

        if y_next < y_best:
            y_best = y_next
            x_best = x_next

        x_train_init = np.vstack((x_train_init, x_next))
        y_train_init = np.vstack((y_train_init, y_next))
        iteration = iteration + 1
        print(f"Iteration {iteration}, x_next: {x_next}, y_next: {y_next}")

    print(f"Best point found: x = {x_best}, f(x) = {y_best}")
    print("\n")