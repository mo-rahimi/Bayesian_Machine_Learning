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
