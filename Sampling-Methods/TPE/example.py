# Following the tutorial at
# https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


def rmse(m, b):
    """
    Consumes coefficients of the linear model and returns RMSE.
    Basically goodness-of-fit metric

    :param m: slope
    :param b: intercept
    :return: RMSE
    """
    preds = m * x + b
    return np.sqrt(np.mean((y - preds) ** 2))


class UniformPrior:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def rvs(self, n):
        return np.random.uniform(self.low, self.high, n)


def sample_priors(space, n):

    seed = np.array([space[hp].rvs(n) for hp in space])

    seed_rmse = np.array([rmse(m, b) for m, b in seed.T])

    data = np.stack([seed[0], seed[1], seed_rmse]).T
    trials = pd.DataFrame(data, columns=["m", "b", "rmse"])
    return trials


def segment_distributions(trials, gamma):
    """
    Splits samples into l(x) and g(x) dists based on qunatile cutoff gamma

    return KDE for each
    """

    cut = np.quantile(trials["rmse"], gamma)
    l_x = trials[trials["rmse"] <= cut][["m", "b"]]
    g_x = trials[~trials.isin(l_x)][["m", "b"]].dropna()
    lx_mixtures = len(l_x.values)//5
    gx_mixtures = len(g_x.values)//5
    if lx_mixtures == 0:
        lx_mixtures = 1
    if gx_mixtures == 0:
        gx_mixtures = 1
    print('Number of Lx mixtures:', lx_mixtures)
    print('Number of Gx mixtures:', gx_mixtures)
    l_gmm = GaussianMixture(n_components=lx_mixtures).fit(l_x.values)
    g_gmm = GaussianMixture(n_components=gx_mixtures).fit(g_x.values)
    return l_gmm, g_gmm


def choose_next_hps(l_gmm, g_gmm, n_samples):
    samples = l_gmm.sample(n_samples)[0]

    l_score = l_gmm.score_samples(samples)
    g_score = g_gmm.score_samples(samples)

    hps = samples[np.argmax(l_score - g_score)]
    return hps


def tpe(space, n_seed, n_total, gamma):
    """
    Consumes a hyperparameter search space, number of iterations for seeding
    and total number of iterations and performs Bayesian Optimization. TPE
    can be sensitive to choice of quantile cutoff, which we control with gamma.
    """

    # Seed priors
    trials = sample_priors(space, n_seed)

    for i in range(n_seed, n_total):
        print('Iteration:', i)

        # Segment trials into l and g distributions
        l_gmm, g_gmm = segment_distributions(trials, gamma)

        if i % 10 == 0:
            g_gmm_samples = g_gmm.sample(100)[0]
            plt.scatter(g_gmm_samples[:, 0],
                        g_gmm_samples[:, 1], marker='x')
            l_gmm_samples = l_gmm.sample(100)[0]
            plt.scatter(l_gmm_samples[:, 0],
                        l_gmm_samples[:, 1])
            plt.xlabel("m")
            plt.ylabel("b")
            plt.title(f"Trial {i}")
            plt.savefig(base_dir + f"scatter_{i}.png")
            plt.close()

        # Determine next pair of hyperparameters to test
        hps = choose_next_hps(l_gmm, g_gmm, 100)

        # Evaluate with rmse and add to trials
        result = np.concatenate([hps, [rmse(hps[0], hps[1])]])

        trials = trials._append(
            {col: result[i] for i, col in enumerate(trials.columns)},
            ignore_index=True
        )

        print('-------------------')

    return trials


import os
import shutil
base_dir = 'figs/'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir)


np.random.seed(42)
# raw data
x = np.linspace(0, 100, 1000)
m = np.random.randint(0, 10)
b = np.random.randint(-100, 100)
y = m * x + b + np.random.randn(1000) * 20
print("True m:", m)
print("True b:", b)

# define the prior
search_space = {"m": UniformPrior(1, 10), "b": UniformPrior(-200, 100)}

# quantile threshold to split the data
# group top 20% of models into a good
# distribution and the rest in a bad distribution
gamma = 0.5

trails = tpe(search_space, 10, 400, gamma)

plt.plot(trails["rmse"])
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("TPE Optimization")
plt.show()

plt.scatter(trails["m"], trails["b"], c=trails["rmse"], cmap="viridis")
plt.colorbar()
plt.axvline(m, color="r", linestyle="--")
plt.axhline(b, color="r", linestyle="--")
plt.xlabel("m")
plt.ylabel("b")
plt.title("TPE Optimization")
plt.show()

plt.plot(x, y, label="True")
best = trails.loc[trails["rmse"].idxmin()]
plt.plot(x, best["m"] * x + best["b"], label="Best")
plt.legend()
plt.show()

from anesthetic import MCMCSamples

samples = MCMCSamples(data=trails)
ax = samples.plot_2d(['m', 'b'])
ax.axlines({'m': m, 'b':b}, color='r', linestyle='--')
plt.show()