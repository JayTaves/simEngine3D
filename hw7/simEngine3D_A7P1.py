from gcons import *
import sympy as sp
import matplotlib.pyplot as plt

# Physical constants
L = 2                               # [m] - length of the bar
t_start = 0                         # [s] - simulation start time
Δt = 10e-3                          # [s] - time step size
t_end = 10                          # [s] - simulation end time
w = 0.05                            # [m] - side length of bar
ρ = 7800                            # [kg/m^3] - density of the bar
g = np.array([[0], [0], [-9.81]])   # [m/s^2] - gravity (global frame)

# Derived constants
V = L * w**3                        # [m^3] - bar volume
m = ρ * V                           # [kg] - bar mass
M = m * np.identity(3)              # [kg] - mass moment tensor
J_xx = 1/6 * m * w**2               # [kg m^2] - inertia xx component
J_yz = 1/12 * m * (w**2 + (2*L)**2)     # [kg m^2] - inertia yy = zz component
J = np.diag([J_xx, J_yz, J_yz])     # [kg m^2] - inertia tensor
Fg = m * g                          # [N] - force of gravity on the body

# Simulation parameters
tol = 1e-12                          # Convergence threshold for Newton-Raphson
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

p0 = (RotAxis(y_axis, np.pi/2) * RotAxis(z_axis, θ0 - np.pi/2)).arr
r0 = np.array([[0], [L * np.sin(θ0)], [L * np.cos(θ0)]])
q0 = np.concatenate((r0, p0), axis=0)

# Didn't read these in from the file, so set them now
pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * pendulum.E().T @ (dp1_drive.df(t_start) * z_axis)

# Group our constraints together
g_cons = ConGroup([cd_i, cd_j, cd_k, dp1_xx, dp1_yx, dp1_drive, e_param])

# Compute initial values (HW6)
Φ = g_cons.GetPhi(t_start)
nu = g_cons.GetNu(t_start)  # Darn, ν looks like v in my font
γ = g_cons.GetGamma(t_start)
Φ_r = g_cons.GetPhiR(t_start)
Φ_p = g_cons.GetPhiP(t_start)
Φ_q = g_cons.GetPhiQ(t_start)

t_steps = int(t_end/Δt)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)

# Get arrays to hold O and Q data
O_pos = np.zeros((t_steps, 3))
O_vel = np.zeros((t_steps, 3))
O_acc = np.zeros((t_steps, 3))

Q_pos = np.zeros((t_steps, 3))
Q_vel = np.zeros((t_steps, 3))
Q_acc = np.zeros((t_steps, 3))

# Get arrays to hold Fr and nr data
Fr = np.zeros((t_steps, 3))
nr = np.zeros((t_steps, 4))

# Put in the initial position/velocity/acceleration values
Q0 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
O_pos[0, :] = q0[0:3, 0].T
Q_pos[0, :] = Q0[0:3, 0].T

pendulum.r = q0[0:3]
pendulum.p = q0[3:]

inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t_start))

O_vel[0, :] = (inv_phi_q @ g_cons.GetNu(t_start))[0:3, 0].T
O_acc[0, :] = (inv_phi_q @ g_cons.GetGamma(t_start))[0:3, 0].T

# Set initial conditions
q_k = q0

for i, t in enumerate(t_grid):
    if i == 0:
        continue

    # Only get (inverse of the) Jacobian once per time step
    inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))
    phi = g_cons.GetPhi(t)

    # Setup and do Newton-Raphson Iteration
    k = 0
    q_k1 = q_k - inv_phi_q @ phi
    while np.linalg.norm(q_k1 - q_k) > tol:
        q_k = q_k1

        # Trying with the bodies updated with each q_k guess
        pendulum.r = q_k[0:3]
        pendulum.p = q_k[3:]
        inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))
        phi = g_cons.GetPhi(t)

        q_k1 = q_k - inv_phi_q @ phi

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

        # print('i: ' + str(i) + ', k: ' + str(k) +
        #       ', norm: ' + str(np.linalg.norm(q_k1 - q_k)))

    pendulum.r = q_k1[0:3]
    pendulum.p = q_k1[3:]

    # Once the position converges, we can compute velocity...
    inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))

    dq_k = inv_phi_q @ g_cons.GetNu(t)

    pendulum.dr = dq_k[0:3]
    pendulum.dp = dq_k[3:]

    # ... and acceleration
    ddq_k = inv_phi_q @ g_cons.GetGamma(t)

    # With quantities computed, fill them into our output arrays
    Q_k1 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
    O_pos[i, :] = q_k1[0:3, 0].T
    Q_pos[i, :] = Q_k1[0:3, 0].T

    O_vel[i, :] = dq_k[0:3, 0].T
    O_acc[i, :] = ddq_k[0:3, 0].T

    # Inverse dynamics
    # Quantities for the left-hand side
    #   0:6 removes euler parameter constraint
    Φ_r_mod = g_cons.GetPhiR(t)[0:6, :]
    Φ_p_mod = g_cons.GetPhiP(t)[0:6, :]
    G = pendulum.G()
    Jp = 4*G.T @ J @ G
    # M is constant, defined above

    # Quantities for the right-hand side
    γ = g_cons.GetGamma(t)[0:6, :]
    γp = -2*pendulum.dp.T @ pendulum.dp
    τ = -8*pendulum.dG().T @ J @ G @ pendulum.dp
    # Fg is constant, defined above

    # We could also solve this smaller system instead and get the same result
    # LHS_mat = np.block([[Φ_r_mod.T, np.zeros((3, 1))],
    #                     [Φ_p_mod.T, pendulum.p]])
    # RHS_mat = np.block([[Fg - M @ ddq_k[0:3]], [τ - J @ ddq_k[3:7]]])

    # Here we solve the larger system and redundantly retrieve r̈ and p̈
    LHS_mat = np.block([[M, np.zeros((3, 4)), np.zeros((3, 1)), Φ_r_mod.T], [np.zeros((4, 3)), Jp, 2*pendulum.p, Φ_p_mod.T], [
        np.zeros((1, 3)), 2*pendulum.p.T, 0, np.zeros((1, 6))], [Φ_r_mod, Φ_p_mod, np.zeros((6, 1)), np.zeros((6, 6))]])
    RHS_mat = np.block([[Fg], [τ], [γp], [γ]])

    rank = np.linalg.matrix_rank(LHS_mat)
    rows, cols = np.shape(LHS_mat)
    assert rows == cols and rows == rank, "Deficient LHS matrix"

    x = np.linalg.solve(LHS_mat, RHS_mat)
    assert np.abs(np.amax(ddq_k[0:3] - x[0:3])) < tol, "Invalid r̈"
    assert np.abs(np.amax(ddq_k[3:7] - x[3:7])) < tol, "Invalid p̈"
    λp = x[7]
    λ = x[8:]

    Fr[i, :] = (-Φ_r_mod.T @ λ).T
    nr[i, :] = (-Φ_p_mod.T @ λ).T

# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# # O′ - position
# ax1.plot(t_grid, O_pos[:, 0])
# ax1.plot(t_grid, O_pos[:, 1])
# ax1.plot(t_grid, O_pos[:, 2])
# ax1.set_title('Position of point O′')
# ax1.set_xlabel('t [s]')
# ax1.set_ylabel('Position [m]')

# # O′ - velocity
# ax2.plot(t_grid, O_vel[:, 0])
# ax2.plot(t_grid, O_vel[:, 1])
# ax2.plot(t_grid, O_vel[:, 2])
# ax2.set_title('Velocity of point O′')
# ax2.set_xlabel('t [s]')
# ax2.set_ylabel('Velocity [m/s]')

# # O′ - acceleration
# ax3.plot(t_grid, O_acc[:, 0])
# ax3.plot(t_grid, O_acc[:, 1])
# ax3.plot(t_grid, O_acc[:, 2])
# ax3.set_title('Acceleration of point O′')
# ax3.set_xlabel('t [s]')
# ax3.set_ylabel('Acceleration [m/s²]')

# plt.show()

# Rather than run the full inverse dynamics at the 0th time step,
#   just copy the value here so the graph looks nicer...
nr[0, :] = nr[1, :]

fig_nr = plt.figure()
fig_nr.suptitle('Reaction Torque on pendulum')
plt.plot(t_grid, nr[:, 0])
plt.plot(t_grid, nr[:, 1])
# plt.plot(t_grid, nr[:, 2])    # Same values as 0
# plt.plot(t_grid, nr[:, 3])    # Same values as 1
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
