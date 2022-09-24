import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
Working from this tutorial...
https://peterroelants.github.io/posts/gaussian-process-tutorial/
"""

def GP(X1, y1, X2, kernel_func, sigma_noise=0):
    """
    X1 and y1 are the data points and X2 is the range you want to
    perform the regression over so y2 is the fitted output.

    can think of 2 as the posterior and the kernel as a prior
    """
    sigma11 = kernel_func(X1, X1) + ((sigma_noise ** 2) * np.eye(len(X1)))
    sigma12 = kernel_func(X1, X2)
    sigma22 = kernel_func(X2, X2)
    solved = scipy.linalg.solve(sigma11, sigma12, assume_a='pos').T
    mu2 = solved @ y1 # technically this is the mean of 2 given the data 1 and assuming mu2=0
    sigma2 = sigma22 - (solved @ sigma12)
    return mu2, sigma2


def exponentiated_quadratic_kernel(xa, xb):
    matrix = []
    for i in range(len(xa)):
        row = []
        for j in range(len(xb)):
            row.append(np.exp(-1/2*(xa[i] - xb[j])**2))
        matrix.append(row)
    return np.array(matrix)

X1 = np.random.uniform(-4, 4, 200)
X1 = np.sort(X1)
sn = 1
y1 = np.sin(X1) + np.random.normal(0, sn**2, len(X1))

X2 = np.linspace(-6, 6, 100)

mu2, sigma2 = GP(X1, y1, X2, exponentiated_quadratic_kernel, sigma_noise=sn)

s2 = np.sqrt(np.diag(sigma2))
y2 = np.random.multivariate_normal(mean=mu2, cov=sigma2)

plt.plot(X2, np.sin(X2), label='Sin Wave')
plt.axhline(0, ls=':', c='gray', alpha=0.4)
plt.plot(X2, y2, c='k', label='Model (X2, Y2)')
plt.plot(X1, y1, ls='', marker='*', c='k', label='Data (X1, Y1)')
plt.fill_between(X2, y2-2*s2, y2+2*s2, color='r', alpha=0.2, label=r'$2\sigma$')
plt.fill_between(X2, y2-s2, y2+s2, color='r', alpha=0.5, label=r'$1\sigma$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('basic_gp_sine_wave.png', dpi=300)
plt.show()
