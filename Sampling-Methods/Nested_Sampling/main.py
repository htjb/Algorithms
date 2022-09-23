import numpy as np
import matplotlib.pyplot as plt

class ns():
    def __init__(self, nlive, prior, likelihood, ndims, niter):
        self.nlive = nlive
        self.ndims = ndims
        self.niter = niter
        self.prior = prior
        self.likelihood = likelihood

    def sampling(self):
        t = np.exp(-1/self.nlive) #compression
        X = 1 # initial volume
        Z = 0 # initial evidence

        initial_points = self.prior(self.nlive, self.ndims)
        l = np.array([self.likelihood(initial_points[i]) for i in range(len(initial_points))])
        dead_points = []
        for i in range(self.niter):
            args = np.argsort(l)
            l = l[args]
            initial_points = initial_points[args]
            lstar = l[0]
            lnew = lstar-100
            while lnew <= lstar:
                new_point = self.prior(1, self.ndims)[0]
                lnew = likelihood(new_point)
            dead_points.append(initial_points[0])
            initial_points[0] = new_point
            l[0] = lnew
            Z = Z + lstar*(1-t)*X
            X = t*X
        Z = Z + np.average(l)*X
        return Z, np.vstack([np.array(dead_points), initial_points])

def prior(nlive, ndims):
    theta = np.zeros([nlive, ndims])
    theta[:, 0] = np.random.uniform(-100, -30, nlive)
    theta[:, 1] = np.random.uniform(12, 18, nlive)
    theta[:, 2] = np.random.uniform(0, 2, nlive)
    return theta

def model_function(params):
    return params[0]*np.exp(-(x-params[1])**2/(2*params[2]**2))

def likelihood(theta):
    noise = 1
    model = model_function(theta)
    loglikelihood = (-0.5*np.log(2*np.pi*(noise**2))-0.5 \
        *((data - model) / noise)**2).sum()
    return loglikelihood

x = np.linspace(10, 20, 100)
A, mu, sigma = -50, 14, 1
experiment_noise = np.random.normal(0, 1, len(x))
data = A*np.exp(-(x-mu)**2/(2*sigma**2)) + experiment_noise

Z, points = ns(500, prior, likelihood, 3, 2000).sampling()

from anesthetic.samples import MCMCSamples

names = ['A', r'$\mu$', r'$\sigma$']
samples = MCMCSamples(data=points, columns=names)

fig, axes = samples.plot_2d(names)
plt.savefig('basic_nested_sampling_signal_posterior.png')
plt.show()

signals = [model_function(points[i]) for i in range(len(points))]

plt.plot(x, data, label='Data')
plt.plot(x, np.average(signals, axis=0), label='Fitted Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('basic_nested_sampling_signal.png')
plt.show()
