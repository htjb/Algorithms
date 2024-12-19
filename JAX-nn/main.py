"""
From the following tutorial:
https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

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

from jax.scipy.special import logsumexp

def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    finalw, finalb = params[-1]
    logits = jnp.dot(finalw, activations) + finalb
    return logits - logsumexp(logits)

batch_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batch_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batch_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'

mnidt_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnidt_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
num_lables = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c

train_images, train_labels = train_data['image'], train_data['label']
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, num_lables)

test_images, test_labels = test_data['image'], test_data['label']
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, num_lables)


print('Train data shape: ', train_images.shape, train_labels.shape)
print('Test data shape: ', test_images.shape, test_labels.shape)

import time

def get_train_batches():
    ds = tfds.load(name='mnist', split='train', data_dir=data_dir, as_supervised=True)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)

for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in get_train_batches():
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot(y, num_lables)
        params = update(params, x, y)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} in {epoch_time:.2f} sec")

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print(f"Train acc: {train_acc}, Test acc: {test_acc}")