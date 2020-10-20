import numpy as np
import matplotlib.pyplot as plt


def f(y, t): return -y**2 - 1/t**4
def g_euler(y_k, y_k1, t_k1, h): return y_k - y_k1 + h*f(y_k1, t_k1)
def dg_euler(y_k1, h): return -1 - 2*h*y_k1


def g_bdf(y_k, y_k1, y_k2, y_k3, y_k4, t_k4, h): return 25*y_k4 - \
    48*y_k3 + 36*y_k2 - 16*y_k1 + 3*y_k - 12*h*f(y_k4, t_k4)


def dg_bdf(y_k4, h): return 25 + 24*h*y_k4


def soln(t): return 1/t + (1/t**2) * np.tan(1/t + np.pi - 1)


y_start = 1
t_start = 1
t_end = 10
h = 0.5
tol = 1e-12


def NextNewton(f_fn, df_fn, xi):
    # No checking for f'(x0) ≈ 0
    return xi - f_fn(xi) / df_fn(xi)


def BackwardEuler(g, dg, h, tol, y0, t0, t_end):
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


def BDF4(g, dg, h, tol, seed_pts, t0, t_end):
    t_grid = np.arange(t0, t_end + h, h)
    pts = np.shape(t_grid)[0]

    y_soln = np.zeros(pts)
    y_soln[0:4] = seed_pts
    y_k = y_soln[3]

    # Bind new functions now that we know h
    def g_bdf(y_k, y_k1, y_k2, y_k3, y_k4, t_k4): return g(
        y_k, y_k1, y_k2, y_k3, y_k4, t_k4, h)

    def dg_bdf(y_k4): return dg(y_k4, h)

    for i, t_val in enumerate(t_grid):
        # Skip the loop if we're still setting up our function
        if i < 4:
            continue

        # Bind new function now that we know t and y_kn+[1:3] as well
        def g_newt(y_k4): return g_bdf(
            y_soln[i-4], y_soln[i-3], y_soln[i-2], y_soln[i-1], y_k4, t_val)

        k = 0
        y_k4 = NextNewton(g_newt, dg_bdf, y_k)
        while abs(y_k4 - y_k) > tol:
            y_k = y_k4
            y_k4 = NextNewton(g_newt, dg_bdf, y_k)

            k = k + 1
            if k >= 50:
                print('NR not converging, stopped after 50 iterations')
                break

        y_soln[i] = y_k4
        # print('i: ' + str(i) + ', k: ' + str(k))

    return y_soln


def ConvEuler(g, dg, exact, h_list, tol, y0, t0, t_end):
    # Note: we enforce that t_sample always be at t_end
    y_exact = exact(t_end)

    y_ests = [BackwardEuler(g, dg, h, tol, y0, t0, t_end)[-1] for h in h_list]
    errs = [np.abs(y_exact - y_est) for y_est in y_ests]

    # Picked these values to get h = 0.01...
    slope_h = np.log10(h_list[6] / h_list[-1])
    slope_err = np.log10(errs[6] / errs[-1])
    print('Backward Euler slope: ' + str(slope_err / slope_h))

    plt.loglog(h_list, errs, 'bx', label='Backward Euler')


def ConvBDF(g, dg, exact, h_list, tol, y0, t0, t_end):
    # Note: we enforce that t_sample always be at t_end
    y_exact = exact(t_end)
    y_ests = np.zeros(np.shape(h_list))

    for i, h in enumerate(h_list):
        seed_pts = np.array([soln(t) for t in np.linspace(
            t_start, t_start + 3*h, 4, endpoint=True)])
        y_ests[i] = BDF4(g, dg, h, tol, seed_pts, t0, t_end)[-1]

    errs = [np.abs(y_exact - y_est) for y_est in y_ests]

    # Picked these values to get h = 0.01...
    slope_h = np.log10(h_list[6] / h_list[-1])
    slope_err = np.log10(errs[6] / errs[-1])
    print('BDF4 slope: ' + str(slope_err / slope_h))

    plt.loglog(h_list, errs, 'rx', label='BDF4')


fig = plt.figure()
fig.suptitle('Convergence Analysis', fontsize=20)
ConvEuler(g_euler, dg_euler, soln, [1, 0.5, 0.2, 0.1, 0.05, 0.02,
                                    0.01, 0.005, 0.002, 0.001], tol, y_start, t_start, t_end)
ConvBDF(g_bdf, dg_bdf, soln, [1, 0.5, 0.2, 0.1, 0.05, 0.02,
                              0.01, 0.005, 0.002, 0.001], tol, y_start, t_start, t_end)
plt.xlabel('h', fontsize=18)
plt.ylabel('error', fontsize=18)
plt.legend(loc='lower right')
fig.savefig('hw7/p3-conv.jpg')
