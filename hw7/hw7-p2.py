import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')
y = sp.Symbol('y')

# Define our differential equation
F = sp.Matrix([1 - x - 4*x*y/(1 + x**2), x*(1 - y/(1 + x**2))])
X = sp.Matrix([x, y])

# Get the symbolic jacobian, bind it to a function
J = F.jacobian(X)
jac = sp.lambdify((x, y), J)


def g_jac(x, y, h): return np.identity(2) - h*jac(x, y)
# What we need to invert in the iterative process


x0 = 0
y0 = 2
t0 = 0
tf = 20
h = 0.1


def NextNewton(f_fn, df_fn, xi):
    # No checking for f'(x0) ≈ 0
    return xi - f_fn(xi) / df_fn(xi)


def MultiBackwardEuler(g, dg, h, tol, y0, t0, t_end):
    t_grid = np.arange(t0, t_end, h)
    pts = np.shape(t_grid)[0] + 1

    y_soln = np.zeros(pts)
    y_k = y0
    y_soln[0] = y_k

    # Bind new functions now that we know h
    def g_euler(y_k, y_k1, t_k1): return g(y_k, y_k1, t_k1, h)
    def dg_euler(y_k1): return dg(y_k1, h)

    for i, t_val in enumerate(t_grid):

        # Bind new function now that we know t as well
        # Use y_soln[i-1] as y_k and y_k1 change as we iterate
        def g_newt(y_k1): return g_euler(y_soln[i], y_k1, t_val)

        k = 0
        y_k1 = NextNewton(g_newt, dg_euler, y_k)
        while abs(y_k1 - y_k) > tol:
            y_k = y_k1
            y_k1 = NextNewton(g_newt, dg_euler, y_k)

            k = k + 1
            if k >= 50:
                print('NR not converging, stopped after 50 iterations')
                break

        y_soln[i+1] = y_k1
        # print('i: ' + str(i) + ', k: ' + str(k))

    return y_soln
