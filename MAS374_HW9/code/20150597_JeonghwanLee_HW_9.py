import numpy as np
import math

#### ---- Problem 2(a) ---- ####

def dual_proj(l):
    dim = l.shape[0]
    proj_l = np.zeros(dim)
    for i in range(dim):
        if l[i] >= 0:
            proj_l[i] = l[i]
        else:
            proj_l[i] = 0
    return proj_l

#### ---- Problem 2(b) ---- ####

def dual_grad(l, x, A, b):
    grad = np.dot(np.dot(A, np.transpose(A)), l) - (np.dot(A, x) - b)
    return grad


#### ---- Problem 2(c) ---- ####

def solve_dual(x, A, b):
    tol = 2 ** -40
    dim = A.shape[0]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    L1 = s[0] ** 2
    learning_rate = 1/L1
    next_itr = np.zeros(dim)
    distance = math.inf
    while distance > tol:
        current_itr = next_itr
        next_itr = dual_proj(current_itr - (learning_rate * dual_grad(current_itr, x, A, b)))
        distance = np.linalg.norm(np.dot(np.transpose(A), next_itr) - np.dot(np.transpose(A), current_itr), 2)
    return next_itr


#### ---- Problem 3(a) ---- ####

def prim_proj(x, A, b):
    dual_opt = solve_dual(x, A, b)
    return x - np.dot(np.transpose(A),dual_opt)


#### ---- Problem 3(b) ---- ####

def grad_f0(x, H, c):
    return np.dot(H, x) + c

def f0(x, H, c):
    return 1/2 * np.dot(np.transpose(x), np.dot(H, x)) + np.dot(np.transpose(c), x)


#### --  A helper function which prints the results in a given format -- ####

def print_results(x_opt, H, c):
    np.set_printoptions(floatmode="unique")  # print with full precision
    print("optimal value p* =")
    print("", f0(x_opt, H, c), sep="\t")
    print("\noptimal solution x* =")
    for coord in x_opt:
        print("", coord, sep='\t')
    return

# first example in page 3 of the document,
# written for you so you can test your code.

H = np.array([[6, 4],
              [4, 14]])
c = np.array([-1, -19])

A = np.array([[-3, 2],
              [-2, -1],
              [1, 0]])
b = np.array([-2, 0, 4])

#### ---- Problem 3(c) ---- ####

def solve_prim(H, c, A, b):
    eps = 2 ** -40
    dim = H.shape[0]
    u, s, vh = np.linalg.svd(H, full_matrices=True)
    L2 = s[0]
    learning_rate = 1/L2
    next_itr = np.zeros(dim)
    distance = math.inf
    while distance > eps:
        current_itr = next_itr
        next_itr = prim_proj(current_itr - (learning_rate * grad_f0(current_itr, H, c)), A, b)
        distance = np.linalg.norm(next_itr - current_itr, 2)
    return next_itr

x_opt = solve_prim(H, c, A, b)

# printing the results
print_results(x_opt, H, c)