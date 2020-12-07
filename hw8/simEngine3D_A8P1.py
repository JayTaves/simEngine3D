from gcons import *
from collections import namedtuple
import sympy as sp
import matplotlib.pyplot as plt

# Physical constants
L = 2                                   # [m] - length of the bar
t_start = 0                             # [s] - simulation start time
h = 1e-2                                # [s] - time step size
t_end = 10                              # [s] - simulation end time
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar
g_acc = np.array([[0], [0], [-9.81]])   # [m/s^2] - gravity (global frame)

# Derived constants
V = 2*L * w**2                          # [m^3] - bar volume
m = ρ * V                               # [kg] - bar mass
M = m * np.identity(3)                  # [kg] - mass moment tensor
J_xx = 1/6 * m * w**2                   # [kg m^2] - inertia xx component
J_yz = 1/12 * m * (w**2 + (2*L)**2)     # [kg m^2] - inertia yy = zz component
J = np.diag([J_xx, J_yz, J_yz])         # [kg m^2] - inertia tensor
Fg = m * g_acc                          # [N] - force of gravity on the body

# Simulation parameters
tol = 1e-2                         # Convergence threshold for Newton-Raphson
max_iters = 50                      # Iterations to abort after for Newton-Raphson

# Set up driving constraint
t = sp.symbols('t')
θ_sym = sp.pi/2 + sp.pi/4 * sp.cos(2*t)
f_sym = sp.cos(θ_sym)
df_sym = sp.diff(f_sym, t)
ddf_sym = sp.diff(df_sym, t)

θ = sp.lambdify(t, θ_sym)
f = sp.lambdify(t, f_sym)
df = sp.lambdify(t, df_sym)
ddf = sp.lambdify(t, ddf_sym)

# Read from file, set up bodies + constraints
(bodies, constraints) = ReadModelFile('hw8/revJointSingle.mdl')

pendulum = Body(bodies[0])
ground = Body({}, True)

nb = len(bodies)  # [-] - number of bodies

# Driving DP1 constraint - only used to get initial change in orientation (dp)
dp1_drive = CreateConstraint(constraints[0], pendulum, ground)

# Manually add these functions rather than have the parser do it
dp1_drive.f = f
dp1_drive.df = df
dp1_drive.ddf = ddf

# DP1 Constraints
dp1_xx = CreateConstraint(constraints[1], pendulum, ground)
dp1_yx = CreateConstraint(constraints[2], pendulum, ground)

# CD ijk Constraints
cd_i = CreateConstraint(constraints[3], pendulum, ground)
cd_j = CreateConstraint(constraints[4], pendulum, ground)
cd_k = CreateConstraint(constraints[5], pendulum, ground)

# Manually add these so that we don't have to hard-code '2' in the .mdl file
cd_i.si = np.array([[-L], [0], [0]])
cd_j.si = np.array([[-L], [0], [0]])
cd_k.si = np.array([[-L], [0], [0]])

# Euler Parameter Constraint
e_param = EulerCon(pendulum)

# Get the initial quaternion by rotating around two axes
θ0 = θ(t_start)
z_axis = np.array([[0], [0], [1]])
y_axis = np.array([[0], [1], [0]])

# p0 = (RotAxis(y_axis, np.pi/2) * RotAxis(z_axis, θ0 - np.pi/2)).arr
# r0 = np.array([[0], [L * np.sin(θ0)], [L * np.cos(θ0)]])
p0 = np.array([[0.6533], [0.2706], [0.6533], [0.2706]])
r0 = np.array([[0], [1.4142], [-1.4142]])
q0 = np.concatenate((r0, p0), axis=0)

# Didn't read these in from the file, so set them now
pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * pendulum.E().T @ (dp1_drive.df(t_start) * z_axis)

# Group our constraints together. Don't attach the Euler param constraint or the old driving constraint
g_cons = ConGroup([cd_i, cd_j, cd_k, dp1_xx, dp1_yx])
nc = g_cons.nc

# Compute initial values
Φ = g_cons.GetPhi(t_start)
nu = g_cons.GetNu(t_start)  # Darn, ν looks like v in my font
γ = g_cons.GetGamma(t_start)
Φ_q = g_cons.GetPhiQ(t_start)
Φ_r = Φ_q[:, 0:3]
Φ_p = Φ_q[:, 3:7]

t_steps = int(t_end/h)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)

# Create arrays to hold O data
O_pos = np.zeros((t_steps, 3))
O_vel = np.zeros((t_steps, 3))
O_acc = np.zeros((t_steps, 3))

# Velocity constraint violation data
vel_con_norm = np.zeros((t_steps, 1))

# Angular velocity data
omega = np.zeros((t_steps, 3))

# Create arrays to hold Fr and nr data
Fr = np.zeros((t_steps, 3))
nr = np.zeros((t_steps, 3, 6))

# Put in the initial position/velocity/acceleration values
O_pos[0, :] = q0[0:3, 0].T

pendulum.r = r0
pendulum.p = p0

G = pendulum.G()
Jp = 4*G.T @ J @ G
# M is constant, defined above

# Quantities for the right-hand side
γ = g_cons.GetGamma(t_start)[0:nc+1, :]
γp = -2*pendulum.dp.T @ pendulum.dp
τ = 8*pendulum.dG().T @ J @ G @ pendulum.dp
# Fg is constant, defined above

# Here we solve the larger system and redundantly retrieve r̈ and p̈
LHS_mat = np.block([[M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T], [np.zeros((4*nb, 3*nb)), Jp, 2*pendulum.p, Φ_p.T], [
    np.zeros((nb, 3*nb)), 2*pendulum.p.T, 0, np.zeros((nb, nc))], [Φ_r, Φ_p, np.zeros((nc, nb)), np.zeros((nc, nc))]])
RHS_mat = np.block([[Fg], [τ], [γp], [γ]])

z = np.linalg.solve(LHS_mat, RHS_mat)
ddr = np.array(z[0:3])
ddp = z[3:7]
λp = z[7]
λ = z[8:]
assert nc == np.shape(λ)[0], "λ has incorrect length"

pendulum.ddr = ddr
pendulum.ddp = ddp
O_acc[0, :] = pendulum.ddr.T

# So we can use these the first time around
r_prev = pendulum.r
p_prev = pendulum.p

dr_prev = pendulum.dr
dp_prev = pendulum.dp

# Setup BDF values
BDFVals = namedtuple('BDFVals', ['β', 'α'])
bdf1 = BDFVals(β=1, α=[-1, 1, 0])
bdf2 = BDFVals(β=2/3, α=[-1, 4/3, -1/3])

for i, t in enumerate(t_grid):
    if i == 0:
        continue

    bdf = bdf1 if i == 1 else bdf2
    C_dr = bdf.α[1]*pendulum.dr + bdf.α[2]*dr_prev
    C_r = bdf.α[1]*pendulum.r + bdf.α[2]*r_prev + bdf.β*h*C_dr

    C_dp = bdf.α[1]*pendulum.dp + bdf.α[2]*dp_prev
    C_p = bdf.α[1]*pendulum.p + bdf.α[2]*p_prev + bdf.β*h*C_dp

    Ψ0 = np.concatenate(
        (M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T), axis=1)
    Ψ1 = np.concatenate((np.zeros((4*nb, 3*nb)), Jp,
                         2*pendulum.p, Φ_p.T), axis=1)
    Ψ2 = np.concatenate((np.zeros((nb, 3*nb)), 2*pendulum.p.T,
                         np.zeros((nb, nb)), np.zeros((nb, nc))), axis=1)
    Ψ3 = np.concatenate((Φ_r, Φ_p, np.zeros((nc, nb)),
                         np.zeros((nc, nc))), axis=1)
    Ψ = np.block([[Ψ0], [Ψ1], [Ψ2], [Ψ3]])
    Ψ_inv = np.linalg.inv(Ψ)

    r_prev = pendulum.r
    p_prev = pendulum.p

    dr_prev = pendulum.dr
    dp_prev = pendulum.dp

    # Setup and do Newton-Raphson Iteration
    k = 0
    while True:
        pendulum.r = C_r + bdf.β**2 * h**2 * pendulum.ddr
        pendulum.p = C_p + bdf.β**2 * h**2 * pendulum.ddp

        pendulum.dr = C_dr + bdf.β*h*pendulum.ddr
        pendulum.dp = C_dp + bdf.β*h*pendulum.ddp

        # Compute values needed for the g matrix
        # We can't move this outside the loop since the g_cons
        #   use e.g. body.p in their computations and body.p gets updated as we iterate
        Φ = g_cons.GetPhi(t)
        Φ_q = g_cons.GetPhiQ(t)
        Φ_r = Φ_q[:, 0:3]
        Φ_p = Φ_q[:, 3:7]
        G = pendulum.G()
        dG = pendulum.dG()
        Jp = 4*G.T @ J @ G
        τ = 8*dG.T @ J @ dG @ pendulum.p
        # M is constant, defined above
        # Fg is constant, defined above

        # Form g matrix
        g0 = M @ pendulum.ddr + Φ_r.T @ λ - Fg
        g1 = Jp @ pendulum.ddp + Φ_p.T @ λ + 2*pendulum.p * λp - τ
        g2 = 1/(bdf.β**2 * h**2) * e_param.GetPhi(t)
        g3 = 1/(bdf.β**2 * h**2) * Φ
        g = np.block([[g0], [g1], [g2], [g3]])

        Δz = Ψ_inv @ -g
        z = z + Δz

        pendulum.ddr = z[0:3]
        pendulum.ddp = z[3:7]
        λp = z[7]
        λ = z[8:8+nc]

        print('i: ' + str(i) + ', k: ' + str(k) +
              ', norm: ' + str(np.linalg.norm(Δz)))

        if np.linalg.norm(Δz) < tol:
            break

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

    # Compute violation of velocity kinematic constraint
    vel_con = Φ_r @ pendulum.dr + Φ_p @ pendulum.dp - g_cons.GetNu(t)
    vel_con_norm[i] = np.linalg.norm(vel_con)

    # Compute body angular velocity
    omega[i, :] = (2*pendulum.G() @ pendulum.dp).T

    O_pos[i, :] = pendulum.r.T
    O_vel[i, :] = pendulum.dr.T
    O_acc[i, :] = pendulum.ddr.T

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# O′ - position
ax1.plot(t_grid, O_pos[:, 0])
ax1.plot(t_grid, O_pos[:, 1])
ax1.plot(t_grid, O_pos[:, 2])
ax1.set_title('Position of point O′')
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position [m]')

# O′ - velocity
ax2.plot(t_grid, O_vel[:, 0])
ax2.plot(t_grid, O_vel[:, 1])
ax2.plot(t_grid, O_vel[:, 2])
ax2.set_title('Velocity of point O′')
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Velocity [m/s]')

# O′ - acceleration
ax3.plot(t_grid, O_acc[:, 0])
ax3.plot(t_grid, O_acc[:, 1])
ax3.plot(t_grid, O_acc[:, 2])
ax3.set_title('Acceleration of point O′')
ax3.set_xlabel('t [s]')
ax3.set_ylabel('Acceleration [m/s²]')

plt.show()

# Rather than run the full inverse dynamics at the 0th time step,
#   just copy the value here so the graph looks nicer...
nr[0, :] = nr[1, :]

fig_nr = plt.figure()
fig_nr.suptitle('Reaction Torque on pendulum')
plt.plot(t_grid, nr[:, 0, 5])
plt.plot(t_grid, nr[:, 1, 5])
plt.plot(t_grid, nr[:, 2, 5])
plt.xlabel('t', fontsize=18)
plt.ylabel('Reaction Torque', fontsize=18)
fig_nr.savefig('hw7/ReactionTorque.jpg')

plt.show()

fig_Fr = plt.figure()
fig_Fr.suptitle('Reaction Force on Pendulum')
plt.plot(t_grid, Fr[:, 0])
plt.plot(t_grid, Fr[:, 1])
plt.plot(t_grid, Fr[:, 2])
plt.xlabel('t', fontsize=18)
plt.ylabel('Reaction Force', fontsize=18)

# plt.show()

# fig = plt.figure()
# plt.plot(Q_pos[:, 0], Q_pos[:, 1], 'x')
# fig.suptitle('Position of point Q', fontsize=20)
# axes = plt.gca()
# axes.set_xlim([-4, 4])
# axes.set_ylim([-4, 4])
# plt.xlabel('y', fontsize=18)
# plt.ylabel('z', fontsize=18)
# fig.savefig('hw6/pointQ.jpg')

# plt.show()
