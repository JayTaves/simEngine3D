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

F_temp = sp.lambdify((x, y), F)
def G(Y_k, Y_k1, h): return Y_k - Y_k1 + h*F_temp(Y_k1[0, 0], Y_k1[0, 1]).T
def g_jac(Y, h): return -np.identity(2) + h*jac(Y[0, 0], Y[0, 1])
# What we need to invert in the iterative process


x0 = 0
y0 = 2
Y0 = np.array([[0, 2]])
t0 = 0
tf = 20
h = 0.1
tol = 1e-12
max_iters = 500


def NextNewton(f_fn, df_fn, xi):
    # No checking for f'(x0) ≈ 0
    # print(df_fn(xi))
    # print(f_fn(xi))
    # print(np.linalg.inv(df_fn(xi)))
    return xi - f_fn(xi) @ np.linalg.inv(df_fn(xi))


def MultiBackwardEuler(g, dg, h, tol, y0, t0, t_end):
    t_grid = np.arange(t0, t_end, h)
    pts = np.shape(t_grid)[0] + 1

    y_soln = np.zeros((pts, 2))
    y_k = y0
    y_soln[0, :] = y_k

    # Bind new functions now that we know h
    def g_euler(y_k, y_k1): return g(y_k, y_k1, h)
    def dg_euler(y_k1): return dg(y_k1, h)

    for i, _ in enumerate(t_grid):

        # Bind new function now that we know t as well
        # Use y_soln[i-1] as y_k and y_k1 change as we iterate
        def g_newt(y_k1): return g_euler(y_soln[i, :], y_k1)

        k = 0
        y_k1 = NextNewton(g_newt, dg_euler, y_k)
        while np.linalg.norm(y_k1 - y_k) > tol:
            y_k = y_k1
            y_k1 = NextNewton(g_newt, dg_euler, y_k)

            k = k + 1
            if k >= max_iters:
                print('NR not converging, stopped after ' +
                      str(max_iters) + ' iterations')
                break

        # print(str(1 + y_k1[0, 0]**2 - y_k1[0, 1]))

        y_soln[i+1, :] = y_k1
        # if (i < 100):
        # print('i: ' + str(i) + ', k: ' + str(k))

    return y_soln


# Plot xy plane
fig = plt.figure()
Y = MultiBackwardEuler(G, g_jac, h, tol, Y0, t0, tf)
ts = np.linspace(t0, tf, int((tf - t0) / h) + 1)
plt.plot(Y[:, 0], Y[:, 1], 'r.')
fig.suptitle('Flow of points in the xy plane, h = 0.1', fontsize=20)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
fig.savefig('hw7/p2-plane.jpg')

fig = plt.figure()
Y = MultiBackwardEuler(G, g_jac, h, tol, Y0, t0, tf)
ts = np.linspace(t0, tf, int((tf - t0) / h) + 1)
plt.plot(ts, Y[:, 0], 'b.')
fig.suptitle('x value of solution, h = 0.1', fontsize=20)
plt.xlabel('t', fontsize=18)
plt.ylabel('x', fontsize=18)
fig.savefig('hw7/p2-x.jpg')

fig = plt.figure()
Y = MultiBackwardEuler(G, g_jac, h, tol, Y0, t0, tf)
ts = np.linspace(t0, tf, int((tf - t0) / h) + 1)
plt.plot(ts, Y[:, 1], 'g.')
fig.suptitle('x value of solution, h = 0.1', fontsize=20)
plt.xlabel('t', fontsize=18)
plt.ylabel('y', fontsize=18)
fig.savefig('hw7/p2-y.jpg')
