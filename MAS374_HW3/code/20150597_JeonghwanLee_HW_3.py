import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

#### ---- code for part (a) ---- ####

def my_lstsq(A, y):
    m = A.shape[0]
    n = A.shape[1]
    r = np.linalg.matrix_rank(A)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    pseudo_smat = np.zeros((n, m))
    for i in range(r):
        pseudo_smat[i, i] = 1/s[i]
    pseudo_inv_A = np.dot(np.transpose(vh), np.dot(pseudo_smat, np.transpose(u)))
    theta = np.dot(pseudo_inv_A, y)
    return theta

#### ---- code for part (b) ---- ####

def label(a, b):     # label a given point in \mathbb{R}^2 according to the rule (4)
    if a**2 + b**2 <= 1:
        return 1
    else:
        return -1

def generate_data_matrix(first_coordinate, second_coordinate):     # construct a proper data matrix (2-dimensional array) of shape length \times 6
    input_length = first_coordinate.shape[0]
    data_matrix = np.zeros((length, 6))
    for i in range(input_length):
        data_matrix[i, 0] = 1
        data_matrix[i, 1] = first_coordinate[i]
        data_matrix[i, 2] = second_coordinate[i]
        data_matrix[i, 3] = first_coordinate[i]**2
        data_matrix[i, 4] = first_coordinate[i]*second_coordinate[i]
        data_matrix[i, 5] = second_coordinate[i]**2
    return data_matrix

def conic_section_discriminant(A, B, C, D, E, F):     # discriminate the conic section Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    discriminant = B**2 - 4*A*C
    if A == C and B == 0:
        return "Circle"
    elif discriminant < 0:
        return "Ellipse"
    elif discriminant == 0:
        return "Parabola"
    else:
        return "Hyperbola"

length = 250     # number of samples
first_coordinate_1 = np.dot(4, np.random.rand(length)) - np.dot(2, np.ones(length))     # the first coordinates of 250 samples chosen uniformly at random from [-2, 2] \times [-2, 2]
second_coordinate_1 = np.dot(4, np.random.rand(length)) - np.dot(2, np.ones(length))     # the second coordinates of 250 samples chosen uniformly at random from [-2, 2] \times [-2, 2]
y_1 = np.zeros(length)     # initialize vector of labels for part (b)
num_1 = 0
for i in range(length):
    y_1[i] = label(first_coordinate_1[i], second_coordinate_1[i])     # label the 250 random sample points according to the rule (4)
    if y_1[i] > 0:
        num_1 = num_1 + 1

A_1 = generate_data_matrix(first_coordinate_1, second_coordinate_1)
theta_1 = my_lstsq(A_1, y_1)     # compute the optimal order-2 polynomial that minimizes (3)
print(theta_1)
print(conic_section_discriminant(theta_1[3], theta_1[4], theta_1[5], theta_1[1], theta_1[2], theta_1[0]))     # discriminate the conic section formed as the zero set of the optimal order-2 polynomial that minimizes (3) (= the decision boundary)

x = np.linspace(-5, 5, 4000)
y = np.linspace(-5, 5, 4000)
x, y = np.meshgrid(x, y)
axes()
plt.contour(x, y, (theta_1[3]*x**2 + theta_1[4]*x*y + theta_1[5]*y**2 + theta_1[1]*x + theta_1[2]*y + theta_1[0]), [0], colors='k')
plt.show()

#### ---- code for part (c) ---- ####

length = 250     # number of samples
first_coordinate_2 = np.dot(2, np.random.rand(length))     # the first coordinates of 250 samples chosen uniformly at random from [0, 2] \times [0, 2]
second_coordinate_2 = np.dot(2, np.random.rand(length))     # the second coordinates of 250 samples chosen uniformly at random from [0, 2] \times [0, 2]
y_2 = np.zeros(length)     # initialize vector of labels for part (c)
num_2 = 0
for i in range(length):
    y_2[i] = label(first_coordinate_2[i], second_coordinate_2[i])     # label the 250 random sample points according to the rule (4)
    if y_2[i] > 0:
        num_2 = num_2 + 1

A_2 = generate_data_matrix(first_coordinate_2, second_coordinate_2)
theta_2 = my_lstsq(A_2, y_2)     # compute the optimal order-2 polynomial that minimizes (3)
print(theta_2)
print(conic_section_discriminant(theta_2[3], theta_2[4], theta_2[5], theta_2[1], theta_2[2], theta_2[0]))     # discriminate the conic section formed as the zero set of the optimal order-2 polynomial that minimizes (3) (= the decision boundary)

x = np.linspace(-5, 5, 4000)
y = np.linspace(-5, 5, 4000)
x, y = np.meshgrid(x, y)
axes()
plt.contour(x, y, (theta_2[3]*x**2 + theta_2[4]*x*y + theta_2[5]*y**2 + theta_2[1]*x + theta_2[2]*y + theta_2[0]), [0], colors='k')
plt.show()

print(np.allclose(theta_1, np.linalg.lstsq(A_1, y_1, rcond=None)[0]))
print(np.linalg.lstsq(A_1, y_1, rcond=None)[0])
print(np.allclose(theta_2, np.linalg.lstsq(A_2, y_2, rcond=None)[0]))
print(np.linalg.lstsq(A_2, y_2, rcond=None)[0])