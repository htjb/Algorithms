import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

"""
Following the tutorial here https://towardsdatascience.com/gibbs-sampling-8e4844560ae5
"""

# generating the joint probability for x an y as the example to
# approximate with gibbs sampling...
f = lambda x, y: np.exp(-(x**2*y**2+x**2+y**2-8*x-8*y)/2)

xx = np.linspace(-1, 8, 100)
yy = np.linspace(-1, 8, 100)
xg, yg = np.meshgrid(xx, yy)
z = f(xg.ravel(), yg.ravel())
zg = z.reshape(xg.shape)

# knowing the conditional probabilities P(y|x) and p(x|y) are normal distributions...
# initialise samples, number of iterations and define sigma and mu of
# conditional probabilities
# conditionals come from P(y|x) = P(y, x)/p(x)
# gibbs sampling is designed to give samples on p(y, x) given knowledge in the conditionals
N = 500
x = np.zeros(N+1)
y = np.zeros(N+1)
y[0] = 6
x[0] = 3
sigma = lambda z, i:np.sqrt(1/(1+z[i]**2))
mu = lambda z, i: 4/(1+z[i]**2)

# the gibbs sampling algorithm...
for i in range(1, N, 2):
    # get x[i] given y[i-1] from conditional equation
    # and set y[i] as y[i-1]
    sigma_x = sigma(y, i-1)
    mu_x = mu(y, i-1)
    x[i] = np.random.normal(mu_x, sigma_x)
    y[i] = y[i-1]

    # get y[i+1] given x[i] from above
    # and set x[i+1] as x[i]
    sigma_y = sigma(x, i)
    mu_y = mu(x, i)
    y[i+1] = np.random.normal(mu_y, sigma_y)
    x[i+1] = x[i]

plt.contourf(xg, yg, zg, cmap='inferno', label='True Distribution')
plt.scatter(x, y, marker='.', c='r', alpha=0.3, label='Gibbs Samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('gibbs_sampling_tutorial.png', dpi=300)
plt.show()
