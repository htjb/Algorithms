import numpy as np
import matplotlib.pyplot as plt

def f(x, target, root):
    return x**root-target

def f_pri(x, root):
    return root*x**(root-1)

def NR(x0, target, func, derivative, epsilon=1e-3, root=2):

    x = []
    x.append(x0)
    precision = 1
    while precision > epsilon:
        x_new = x0 - func(x0, target, root)/derivative(x0, root)
        precision = np.abs(x0 - x_new)
        x.append(x_new)
        x0 = x_new
    return x[-1], x

print(NR(1.5, 27, f, f_pri, root=3)[0])
print(NR(100, 100, f, f_pri)[0])
print(NR(1.5, 16, f, f_pri, root=5)[0])
print(NR(1.5, 100, f, f_pri, root=5)[0])

sol, iters = NR(1, 100, f, f_pri, root=2)

plt.plot(iters, label='Approx.')
plt.axhline(10, ls='--', label='True')
plt.xlabel('Iteration')
plt.ylabel(r'$\sqrt{100}$')
plt.savefig('root_solver_example.png', dpi=300)
plt.show()
