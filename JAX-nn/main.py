"""
From the following tutorial:
https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
"""

import jax.numpy as jax
from jax import grad, jit, vmap
from jax import random

# function to initialise a dense layer
def random_layer_parameters(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), \
        scale * random.normal(b_key, (n,))

# initialise all the layers in the network
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_parameters(m, n, k) 
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))
"""for i, p in enumerate(params):
    print(f"Layer {i} has weights of shape {p[0].shape}" +
          f"and bias of shape {p[1].shape}")"""

from jax.special import logsumexp

def relu(x):
    return jax.maximum(0, x)

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jax.vdot(w, activations) + b
        activations = relu(outputs)
    finalw, finalb = params[-1]
    logits = jax.vdot(finalw, activations) + finalb
    return logits - logsumexp(logits)

random_flattened_images = random.normal(random.key(1), (28*28,))
print(predict(params, random_flattened_images))