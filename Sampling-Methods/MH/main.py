import numpy as np
import matplotlib.pyplot as plt
import random

def fit(likelihood, mins, maxs, niter=1000):

    """
    A version of the metropolis hastings algorithm.

    Using old_likelihood/new_likelihood and accepting if ratio is greater than
    some randomly picked threshold between 0 and 1.

    e.g x_new > x_old then the ratio > 1 so we always accept but if x_new is
    smaller than x_old then the ratio is < 1 and we accept it sometimes
    based on the value of u (supposed to allow the algorithm the oportunity to
    look for additional modes I think).
    """

    def gaussian_proposal(x):
        return np.random.normal(x, 1, 3)

    tol = (maxs - mins)/100
    x, i = [], 0
    x.append(np.random.uniform(mins, maxs))
    while i < niter:
        x_old = x[i].copy()
        x_new =  gaussian_proposal(x_old)
        if np.any(x_new < mins) or np.any(x_new > maxs):
            pass
        else:
            u = np.random.uniform(0, 1)
            A =  likelihood(x[i])/likelihood(x_new)
            if u <= A:
                x.append(x_new)
                i += 1
    x = np.array(x)
    return x


def model_function(params):
    return params[0]*np.exp(-(x-params[1])**2/(2*params[2]**2))

def likelihood(theta):
    noise = 1
    model = model_function(theta)
    loglikelihood = (-0.5 *((data - model) / noise)**2).sum()
    return loglikelihood

x = np.linspace(10, 20, 100)
A, mu, sigma = -50, 14, 1
experiment_noise = np.random.normal(0, 1, len(x))
data = A*np.exp(-(x-mu)**2/(2*sigma**2)) + experiment_noise

mins = np.array([-100, 12, 0])
maxs = np.array([-30, 18, 2])

points = fit(likelihood, mins, maxs, niter=100000)

likes = np.array([likelihood(points[i]) for i in range(len(points))])
max_arg = np.where(likes == likes.max())[0][0]

from anesthetic.samples import MCMCSamples

names = ['A', r'$\mu$', r'$\sigma$']
samples = MCMCSamples(data=points, columns=names)
print(samples)

fig, axes = samples.plot_2d(names)
axes.iloc[0, 0].axvline(A, ls='--', c='k')
axes.iloc[1, 1].axvline(mu, ls='--', c='k')
axes.iloc[2, 2].axvline(sigma, ls='--', c='k')
plt.savefig('basic_MH_sampling_signal_posterior.png', dpi=300)
plt.show()

signals = [model_function(points[i]) for i in range(len(points))]

plt.plot(x, data, label='Data')
plt.plot(x, signals[max_arg], label='Fitted Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('basic_MH_sampling_signal.png', dpi=300)
plt.show()
