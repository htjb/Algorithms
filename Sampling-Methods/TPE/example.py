# Following the tutorial at
# https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity


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
    l_kde = KernelDensity(kernel="gaussian", bandwidth=5).fit(l_x.values)
    g_kde = KernelDensity(kernel="gaussian", bandwidth=5).fit(g_x.values)
    return l_kde, g_kde


def choose_next_hps(l_kde, g_kde, n_samples):
    samples = l_kde.sample(n_samples)

    l_score = l_kde.score_samples(samples)
    g_score = g_kde.score_samples(samples)

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

        # Segment trials into l and g distributions
        l_kde, g_kde = segment_distributions(trials, gamma)

        if i % 10 == 0:
            # contour plot of l_kde
            plt.scatter(trials["m"], trials["b"],
                        c=trials["rmse"], cmap="viridis")
            plt.colorbar()
            plt.xlabel("m")
            plt.ylabel("b")
            plt.title(f"Trial {i}")
            plt.savefig(f"figs/scatter_{i}.png")
            plt.close()

        # Determine next pair of hyperparameters to test
        hps = choose_next_hps(l_kde, g_kde, 100)

        # Evaluate with rmse and add to trials
        result = np.concatenate([hps, [rmse(hps[0], hps[1])]])

        trials = trials._append(
            {col: result[i] for i, col in enumerate(trials.columns)},
            ignore_index=True
        )

    return trials


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
gamma = 0.2

trails = tpe(search_space, 10, 500, gamma)

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
