import jax
import jax.numpy as jnp
from tqdm import tqdm
import time
import matplotlib.pyplot as plt 

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} took ' +
            f'{end_time - start_time:.4f} seconds')
        return result
    return wrapper

key = jax.random.PRNGKey(0)

@jax.jit
def gaussian_proposal(x, key):
        return jax.random.normal(key, (len(x),)) * 0.1 + x


def update_step(x_old, key, mins, maxs, likelihood):
    key, subkey = jax.random.split(key)
    x_new = gaussian_proposal(x_old, subkey)
    accept = jnp.logical_and(jnp.all(x_new > mins), jnp.all(x_new < maxs))
    u = jnp.log(jax.random.uniform(subkey))
    A = likelihood(x_new) - likelihood(x_old)
    accept = jnp.logical_and(accept, u <= A)
    return jnp.where(accept, x_new, x_old), key

@timeit
def fit(likelihood, mins, maxs, niter, ndims, key):
    x = jax.random.uniform(key, (ndims,)) * (maxs - mins) + mins
    xs = [x]
    for _ in tqdm(range(niter)):
        x, key = update_step(x, key, mins, maxs, likelihood)
        xs.append(x)
    return jnp.array(xs)

x = jnp.linspace(10, 20, 100)
A, mu, sigma = -50, 14, 1
experiment_noise = jax.random.normal(key, (len(x),))
data = A*jnp.exp(-(x-mu)**2/(2*sigma**2)) + experiment_noise

@jax.jit
def model_function(params):
    return params[0]*jnp.exp(-(x-params[1])**2/(2*params[2]**2))

@jax.jit
def likelihood(theta):
    noise = 1
    model = model_function(theta)
    loglikelihood = (-0.5 *((data - model) / noise)**2).sum()
    return loglikelihood

mins = jnp.array([-100, 12, 0])
maxs = jnp.array([-30, 18, 2])


points = fit(likelihood, mins, maxs, 10000, 3, key)

from anesthetic.samples import MCMCSamples

names = ['A', r'$\mu$', r'$\sigma$']
samples = MCMCSamples(data=points, columns=names)
print(samples)

axes = samples.plot_2d(names)
axes.iloc[0, 0].axvline(A, ls='--', c='k')
axes.iloc[1, 1].axvline(mu, ls='--', c='k')
axes.iloc[2, 2].axvline(sigma, ls='--', c='k')
plt.savefig('basic_MH_sampling_signal_posterior.png', dpi=300)
plt.show()

signals = [model_function(points[i]) for i in range(len(points))]

[plt.plot(x, signals[i]) for i in range(0, len(signals), 100)]
plt.plot(x, data, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('basic_MH_sampling_signal.png', dpi=300)
plt.show()
