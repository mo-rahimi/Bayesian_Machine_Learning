# Bayesian Machine Learning:
- `(Gaussian Process regression, Bayesian optimization)`
  
This repository is created  for different ML topics, such as Graph ML, Bayesian ML, and etc. for those who are interested to learn these topics. 

## Project Overview:

We can start with a function and sample that at FOUR equally spaced points between its lower and upper bounds (the two bounds should be included in the four sampled points).

A. Then we use this sampled data to train a `Gaussian Process regression model`, by using `RBF kernel`.

B. Then we try to optimize the function using Bayesian optimization:

C. We will try a different acquisition function, called **Upper Confidence Bound (UCB)**, to use within Bayesian optimization and to minimize the function. 
- UCB is defined as:
           `𝑈𝐶𝐵(𝑥) = 𝑦)(𝑥) − 𝜅𝑠(𝑥)`
          

<img src="GIF/animation.gif" width="800" height="500"/>

### Part A:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import imageio
import math

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

def train_gaussian_process(x_train, y_train):
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False, random_state=11)
    gpr.fit(x_train, y_train)
    return gpr


gpr = train_gaussian_process(x_train, y_train)

x_plot = np.linspace(0, 1.2, 1000)
y_pred, y_std = gpr.predict(x_plot.reshape(-1, 1), return_std=True, return_cov=False)

images = []
for i in range(len(x_plot)):
    plt.figure()
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.xlim([0, 1.2])

    plt.plot(x_plot[:i], y_plot[:i], label="True function")
    plt.plot(x_plot[:i], y_pred[:i], label="GP prediction")
    plt.scatter(x_train, y_train, label="Training points")
    plt.fill_between(x_plot[:i].ravel(), y_pred[:i].ravel() - y_std[:i], y_pred[:i].ravel() + y_std[:i], alpha=0.3,
                     label="Confidence interval")
    plt.legend(loc="upper left")

    plt.gca().set_ylim([-2, 3])
    plt.draw()
    plt.savefig('temp_plot.png', dpi=300)
    plt.close()
    images.append(imageio.imread('temp_plot.png'))

imageio.mimsave('animation.gif', images, duration=33.3, loop=0)
```
## Part B

```python
def expected_improvement(x, y_pred, std_dev, f_min):
    arg = f_min - y_pred
    EI_exploit = arg * norm.cdf(arg / std_dev)
    EI_explor = std_dev * norm.pdf(arg / std_dev)
    EI = EI_exploit + EI_explor
    return EI
```

```python
def maximize_acquisition_function(gpr,x_train,y_train,f_min,x_low,x_high):

    n_grid_points = 5000
    x_grid = np.linspace(x_low,x_high,n_grid_points)
    EI_max = -1
    x_next = x_grid[0]
    
    for x_val in x_grid:
        y_pred, y_std = gpr.predict(x_val.reshape(-1, 1), return_std=True, return_cov=False)
        EI_val = expected_improvement(x_train,y_pred[0],y_std[0],f_min)
        if EI_val > EI_max:
            EI_max = EI_val
            x_next = x_val
            
    return x_next

```

```python
def initial_sample(n_init,x_low,x_high):
    x_0 = random.uniform(x_low,x_high)
    y_0 = Problem05(x_0)
    x_train = np.array([[x_0]])
    y_train = np.array([[y_0]])
    i = 1
    while i < n_init:
        x_i = random.uniform(x_low,x_high)
        y_i = Problem05(x_i)
        x_train = np.vstack((x_train,x_i))
        y_train = np.vstack((y_train,y_i))
        i = i+1
    return x_train, y_train
```

```python
def evaluate_x_next(x_next):
    y_next = Problem05(x_next)
    return y_next
```

```python
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
    print(iteration,x_next,y_next)
print(f"Best point found: x = {x_best}, f(x) = {y_best}")
```

`1 0.17043408681736347 -0.06550784312569946`

`2 0.06409281856371274 -1.1041681612899061`

` 8 0.07945589117823564 -1.1501710270005965`

` 9 0.07945589117823564 -1.1501710270005965`

`10 0.07945589117823564 -1.1501710270005965`

**Best point found:** `x = 0.07945589117823564, f(x) = -1.1501710270005965`



## Part C
**To implement the Upper Confidence Bound (UCB) as an alternative acquisition function for Bayesian Optimization**, we'll follow these steps:
- Define the UCB acquisition function.
- Modify the maximize_acquisition_function to work with UCB.
- Perform three experiments with different values of kappa (0.5, 3, and n!).
- Use the same four initial points as in part (a) and sample eight more points for evaluation.
- Print the value of x selected by the UCB and the value of the function evaluated at x.
- Compare the results and comment on which value of kappa worked best for the optimization of the function.
  
`Let's start by defining the UCB acquisition function:`
To implement the Upper Confidence Bound (UCB) acquisition function, I'll follow these steps:
- Define a function upper_confidence_bound that takes x, y_pred, std_dev, and k as inputs and calculates the UCB value.
- Modify the maximize_acquisition_function to accept an additional parameter k and use the UCB function instead of the Expected Improvement function.
- Perform Bayesian optimization using the `UCB acquisition function` for the three experiments with different values of `k`.


### Conclusion
  Based on the results, the `diminishing k (k = 4/n) worked best` for the optimization of the function. In the first two experiments with k = 0.5 and k = 3, the optimizer got stuck at a single point and did not explore more. However, with the diminishing k, the optimizer was able to explore more points and found the best point at x = 0.9382691345672836 with f(x) = -1.308613809062413.
Comparing this to the Expected Improvement (EI) results from part (b), the best point found using EI was x = 0.9661932386477294 with f(x) = -1.4890696850503946. The performance of the UCB with diminishing k is very close to the performance of EI, but the EI slightly was better than UCB in this case, but if I try another value of k, UCB might gives better result.
