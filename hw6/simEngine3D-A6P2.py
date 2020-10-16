from gcons import *
import sympy as sp

# Parameters
L = 2
t0 = 0
Δt = 10e-3

# Set up driving constraint
t = sp.symbols('t')
θ_sym = sp.pi/4 * sp.cos(2*t)
f_sym = sp.cos(θ_sym)
df_sym = sp.diff(f_sym, t)
ddf_sym = sp.diff(df_sym, t)

θ = sp.lambdify(t, θ_sym)
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

# CD ijk Constraints
cd_i = CreateConstraint(constraints[3], pendulum, ground)
cd_j = CreateConstraint(constraints[4], pendulum, ground)
cd_k = CreateConstraint(constraints[5], pendulum, ground)

# Signs flipped here from what you might intuitively expect
# because of the direction we take the difference (0 - a) vs. (a - 0)
CD_con_j = -L * sp.sin(θ_sym)
CD_con_k = L * sp.cos(θ_sym)

cd_j.f = sp.lambdify(t, CD_con_j)
cd_j.df = sp.lambdify(t, sp.diff(CD_con_j, t))
cd_j.ddf = sp.lambdify(t, sp.diff(CD_con_j, t, t))

cd_k.f = sp.lambdify(t, CD_con_k)
cd_k.df = sp.lambdify(t, sp.diff(CD_con_k, t))
cd_k.ddf = sp.lambdify(t, sp.diff(CD_con_k, t, t))

# Euler Parameter Constraint
e_param = EulerCon(pendulum)

# Derived initializations
θ0 = θ(t0)
r0 = np.array([[0], [L * np.sin(θ0)], [-L * np.cos(θ0)]])
z_axis = np.array([[0], [0], [1]])
y_axis = np.array([[0], [1], [0]])
p0 = (RotAxis(y_axis, np.pi/2) * RotAxis(z_axis, θ0)).arr

# Didn't read these in from the file, so set them now
pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * E(p0).T @ (dp1_drive.df(t0) * z_axis)

# Group our constraints together
g_cons = ConGroup([dp1_drive, dp1_zy, dp1_zz, cd_i, cd_j, cd_k, e_param])

# # Compute values
Φ = g_cons.GetPhi(t0)
nu = g_cons.GetNu(t0)  # Darn, ν ≈ v in my font
γ = g_cons.GetGamma(t0)
Φ_r = g_cons.GetPhiR(t0)
Φ_p = g_cons.GetPhiP(t0)
Φ_q = g_cons.GetPhiQ(t0)

# print(Φ)
# print(nu)
# print(γ)
# print(Φ_r)
# print(Φ_p)
# print(Φ_q)

# print(r0)
# print(p0)

# q = np.concatenate((r0, p0), axis=0) - np.linalg.inv(Φ_q) @ Φ
# print(q)
