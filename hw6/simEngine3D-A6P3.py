from gcons import *
import sympy as sp

# Parameters
L = 2
t0 = 0

# Set up driving constraint
t = sp.symbols('t')
f = sp.pi/4 * sp.cos(2*t)
df = sp.diff(f, t)
ddf = sp.diff(df, t)

# Read from file, set up bodies + constraints
(bodies, constraints) = ReadModelFile('hw6/hw6.mdl')

pendulum = Body(bodies[0])
ground = Body({}, True)

dp1 = CreateConstraint(constraints[0], pendulum, ground)
# Manually add these functions rather than have the parser do it
dp1.f = sp.lambdify(t, f)
dp1.df = sp.lambdify(t, df)
dp1.ddf = sp.lambdify(t, ddf, 'numpy')


# Compute values
print("  Φ : ", dp1.GetPhi(t0))
print("  ν : ", dp1.GetNu(t0))
print("  γ : ", dp1.GetGamma(t0))
print("Φ_r : ", dp1.GetPhiR(t0))
print("Φ_p : ", dp1.GetPhiP(t0))
