from gcons import *
import sympy as sp

# Set up driving constraint
t = sp.symbols('t')
f = sp.pi/4 * sp.cos(2*t)
df = sp.diff(f, t)
ddf = sp.diff(df, t)

# Read from file, set up bodies + constraints
(bodies, constraints) = ReadModelFile('hw6/revJoint.mdl')

pendulum = Body(bodies[0])
ground = Body({}, True)

dp1 = CreateConstraint(constraints[0], pendulum, ground)

# Manually add these functions rather than have the parser do it
dp1.f = sp.lambdify(t, f)
dp1.df = sp.lambdify(t, df)
dp1.ddf = sp.lambdify(t, ddf, 'numpy')

# Parameters
L = 2
t0 = 0
θ0 = dp1.f(t0)
r0 = np.array([[L], [0], [0]])
z_axis = np.array([[0], [0], [1]])
p0 = RotAxis(z_axis, θ0)
print(p0)

pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * E(p0).T @ (dp1.df(t0) * z_axis)

print(pendulum.dp)

# Compute values
print("  Φ : ", dp1.GetPhi(t0))
print("  ν : ", dp1.GetNu(t0))
print("  γ : ", dp1.GetGamma(t0))
print("Φ_r : ", dp1.GetPhiR(t0))
print("Φ_p : ", dp1.GetPhiP(t0))
