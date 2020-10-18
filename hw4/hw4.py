from sympy import *

x = Symbol('x')
f = x**6 - x - 1

tolerance = 1e-6
a_check = 1.13472413840152

x0 = 2
x0_2nd = -1


def getLambdas(f, x):
    # Take in a symbolic function f and return f(x) and f'(x) that are proper python functions
    df = f.diff(x)
    f_fn = lambdify(x, f)
    df_fn = lambdify(x, df)

    return (f_fn, df_fn)


def nextNewton(f_fn, df_fn, xi):
    # No checking for f'(x0) ≈ 0
    return xi - f_fn(xi) / df_fn(xi)


def iterNewton(f, x, x0, tol):
    (f_fn, df_fn) = getLambdas(f, x)

    k = 0

    xi = x0
    a = nextNewton(f_fn, df_fn, xi)
    while abs(a - xi) > tol:
        xi = a
        a = nextNewton(f_fn, df_fn, xi)
        k = k + 1

    return a


def iterNewtonTable(f, x, x0, tol):
    (f_fn, df_fn) = getLambdas(f, x)

    k = 0

    xi = x0
    a = nextNewton(f_fn, df_fn, xi)

    f_xk = f_fn(xi)
    xk_a = xi - a_check
    xk1_xk = a - xi

    print('{:d} {:f} {:f} {:f} {:f}'.format(k, xi, f_xk, xk_a, xk1_xk))

    while abs(a - xi) > tol:
        xi = a
        a = nextNewton(f_fn, df_fn, xi)
        k = k + 1

        f_xk = f_fn(xi)
        xk_a = xi - a_check
        xk1_xk = a - xi

        print('{:d} {:f} {:f} {:f} {:f}'.format(k, xi, f_xk, xk_a, xk1_xk))

    return a


a = iterNewtonTable(f, x, x0, 10e-7)
print(a)
print('')
a_2nd = iterNewtonTable(f, x, x0_2nd, 10e-7)
print(a_2nd)
