import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

dataset_x = np.array([-1, 0, 1, 2])
dataset_y = np.array([0, 0, 1, 1])
X = np.transpose(np.stack([np.ones(len(dataset_x)), dataset_x]))
theta_LS = np.linalg.lstsq(X, dataset_y, rcond=None)[0]
print(theta_LS)

M = np.transpose(np.stack([np.ones(len(dataset_x)), dataset_x, dataset_y]))
u_1, s_1, vh_1 = np.linalg.svd(X, full_matrices=False)
u_2, s_2, vh_2 = np.linalg.svd(M, full_matrices=False)
min_singular_M = s_2[2]
theta_TLS = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) - (min_singular_M**2)*np.identity(2)), np.dot(np.transpose(X), dataset_y))
print(theta_TLS)

print(s_1)
print(s_2)
print(s_2**2)

axes()
_ = plt.plot(dataset_x[0], dataset_y[0], 'o', label='Data point 1', markersize=4)
_ = plt.plot(dataset_x[1], dataset_y[1], 'o', label='Data point 2', markersize=4)
_ = plt.plot(dataset_x[2], dataset_y[2], 'o', label='Data point 3', markersize=4)
_ = plt.plot(dataset_x[3], dataset_y[3], 'o', label='Data point 4', markersize=4)
_ = plt.plot(dataset_x, theta_LS[0] + theta_LS[1]*dataset_x, label='Least-squares line')
_ = plt.plot(dataset_x, theta_TLS[0] + theta_TLS[1]*dataset_x, label='Total least-squares line')
_ = plt.legend()
plt.show()

x = np.arange(-5.0, 4.0, 0.01)
axes()
plt.plot(x, x+1, label='y=x+1')
plt.plot(x, 0*x, label='y=0')
plt.fill_between(x, 0, x+1, color='grey', label='{(x, y): y(x-y+1) >= 0}')
plt.legend()
plt.show()