from gcons import *
import sympy as sp

# Parameters
L = 2
t0 = 0
Δt = 10e-3

# Set up driving constraint
t = sp.symbols('t')
f_sym = sp.pi/4 * sp.cos(2*t)
df_sym = sp.diff(f_sym, t)
ddf_sym = sp.diff(df_sym, t)

f = sp.lambdify(t, f_sym)
df = sp.lambdify(t, df_sym)
ddf = sp.lambdify(t, ddf_sym)

# Read from file, set up bodies + constraints
(bodies, constraints) = ReadModelFile('hw6/revJoint.mdl')

pendulum = Body(bodies[0])
ground = Body({}, True)

# Driving DP1 constraint
dp1_drive = CreateConstraint(constraints[0], pendulum, ground)

# Manually add these functions rather than have the parser do it
dp1_drive.f = f
dp1_drive.df = df
dp1_drive.ddf = ddf

# DP1 Constraints
dp1_zy = CreateConstraint(constraints[1], pendulum, ground)
dp1_zz = CreateConstraint(constraints[2], pendulum, ground)

# CD i Constraint
cd_i = CreateConstraint(constraints[3], pendulum, ground)
cd_i.c = lambda t: np.array([[0], [L * np.sin(f(t))], [-L * np.cos(f(t))]])

# CD j/k Constraints
cd_j = CreateConstraint(constraints[4], pendulum, ground)
cd_k = CreateConstraint(constraints[5], pendulum, ground)

# Euler Parameter Constraint
e_param = EulerCon(pendulum)

# Derived initializations
θ0 = dp1_drive.f(t0)
r0 = np.array([[L], [0], [0]])
z_axis = np.array([[0], [0], [1]])
p0 = RotAxis(z_axis, θ0)

# Didn't read these in from the file, so set them now
pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * E(p0).T @ (dp1_drive.df(t0) * z_axis)

# Group our constraints together
g_cons = ConGroup([dp1_drive, dp1_zy, dp1_zz, cd_i, cd_j, cd_k, e_param])
print(g_cons.GetPhi(0))

# # Compute values
# Φ = dp1_drive.GetPhi(t0)
# nu = dp1_drive.GetNu(t0)  # Darn, ν ≈ v in my font
# γ = dp1_drive.GetGamma(t0)
# Φ_r = dp1_drive.GetPhiR(t0)
# Φ_p = dp1_drive.GetPhiP(t0)

# print(Φ)
# print(Φ_r)
# print(Φ_p)

# Φ_q = np.concatenate((Φ_r, Φ_p), axis=1)
# print(Φ_q)

# q_dot = np.linalg.solve(Φ_q, nu)
# print(q_dot)
