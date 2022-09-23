import numpy as np
from globalemu.eval import evaluate
from anesthetic.samples import MCMCSamples
import matplotlib.pyplot as plt

"""
Approximate Bayesian Computation algorithm. I'm not sure I've implemented this
properly. I have used a chi^2 as my distance metric and set a fairly
random value for the threshold epsilon.

ABC is supposed to be a form of LFI but by using a chi^2 theshold I feel like
I am just using a Gaussian likelihood function...

The example relies heavily on the 21-cm signal emulator globalemu
(https://github.com/htjb/globalemu, just a feed
forward neural network).
"""

def threshold(ypri, y):
    # usign a chi^2 for the threshold although I'm not sure
    # this is a good idea?
    return np.sum((ypri-y)**2)/len(y)

def prior(N):
    return np.random.uniform(mins, maxs, (N, 7))

def sampling(N):
    """
    Generate N samples from the prior, simulate the corresponding signals,
    calculate their chi^2 (or some other distance metric) and compare the
    corresponding result with a threshold value epsilon. If the outcome
    is less than that threshold append the sample to the posterior otherwise
    throw it away...
    """
    samples = prior(N)
    simulation, _ = predictor(samples)
    posterior = []
    for i in range(len(simulation)):
        if threshold(simulation[i], data) < epsilon:
            posterior.append(samples[i])
    return posterior

mins = np.loadtxt('/Users/harry/Documents/globalemu/T_release/data_mins.txt')
maxs = np.loadtxt('/Users/harry/Documents/globalemu/T_release/data_maxs.txt')

N = 30000
predictor = evaluate(base_dir='/Users/harry/Documents/globalemu/T_release/', logs=[])

true_params = prior(1)[0]
true_signal, z = predictor(true_params)
noise = np.random.normal(0, 5, len(true_signal))

data = true_signal + noise

epsilon = 50

post = sampling(N)

names = ['fstar', 'vc', 'fx', 'nu_min', 'tau', 'alpha', 'R_mfp']

samples = MCMCSamples(data=post, columns=names)
print(samples)

fig, axes = samples.plot_2d(names, types={'lower': 'kde', 'diagonal':'kde'})
[axes.iloc[i, i].axvline(true_params[i], color='red') for i in range(len(true_params))]
plt.savefig('ABC_posterior.png', dpi=300)
plt.show()

plt.plot(z, true_signal+noise, label='True Signal')
plt.plot(z, np.average(predictor(post)[0], axis=0), label='Fitted Model')
plt.legend()
plt.savefig('ABC_signal_fit.png', dpi=300)
plt.show()
