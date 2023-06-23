import numpy as np
import matplotlib.pyplot as plt

def slice_sampling(x0, x, f):
    fx0 = np.interp(x0, x, f)
    y = np.random.uniform(0, fx0)
    x_rel = x[f > y]
    return np.random.choice(x_rel)


x = np.linspace(-4, 4, 1000)

f = 1/np.sqrt(2*np.pi)*np.exp(-1/2*(x - -2)**2) + \
    1/np.sqrt(2*np.pi*0.2**2)*np.exp(-1/2*(x - 1)**2/0.2**2)

x0 = np.random.choice(x)
samples = [x0]
for i in range(100):
    x0 = slice_sampling(x0, x, f)
    samples.append(x0)

plt.plot(x, f)
plt.scatter(samples, [0]*len(samples))
plt.savefig('slice_sampling_1d_example.png')
plt.show()