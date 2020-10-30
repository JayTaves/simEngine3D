from gcons import *
import sympy as sp
import matplotlib.pyplot as plt

# Physical constants
L = 2                                   # [m] - length of the bar
t_start = 0                             # [s] - simulation start time
h = 1e-3                                # [s] - time step size
t_end = 10                              # [s] - simulation end time
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar
g_acc = np.array([[0], [0], [-9.81]])   # [m/s^2] - gravity (global frame)

# Read from file, set up bodies + constraints
(file_bodies, constraints) = ReadModelFile('hw8/revJoint.mdl')

bodies = [Body(file_body) for file_body in file_bodies]

pend_1 = bodies[0]
pend_2 = bodies[1]
ground = Body({}, True)

nb = len(bodies)  # [-] - number of bodies

# Derived constants
pend_1.V = 2*L * w**2                        # [m^3] - first bar volume
pend_2.V = L * w**2                          # [m^3] - second bar volume

pend_1.m = ρ * pend_1.V                           # [kg] - first bar mass
pend_2.m = ρ * pend_2.V                           # [kg] - second bar mass

pend_1.F = pend_1.m * g_acc
pend_2.F = pend_2.m * g_acc
M = np.diag([body.m for body in bodies])

J_xx_1 = 1/6 * pend_1.m * w**2
J_yz_1 = 1/12 * pend_1.m * (w**2 + (2*L)**2)
J_xx_2 = 1/6 * pend_2.m * w**2
J_yz_2 = 1/12 * pend_2.m * (w**2 + L**2)
pend_1.J = np.diag([J_xx_1, J_yz_1, J_yz_1])
pend_2.J = np.diag([J_xx_2, J_yz_2, J_yz_2])

# Simulation parameters
tol = 1e-2                         # Convergence threshold for Newton-Raphson
max_iters = 50                      # Iterations to abort after for Newton-Raphson


# DP1 Constraints
dp1_xx = CreateConstraint(constraints[1], pend_1, ground)
dp1_yx = CreateConstraint(constraints[2], pend_1, ground)

# CD ijk Constraints
cd_i = CreateConstraint(constraints[3], pend_1, ground)
cd_j = CreateConstraint(constraints[4], pend_1, ground)
cd_k = CreateConstraint(constraints[5], pend_1, ground)

# Manually add these so that we don't have to hard-code '2' in the .mdl file
cd_i.si = np.array([[-L], [0], [0]])
cd_j.si = np.array([[-L], [0], [0]])
cd_k.si = np.array([[-L], [0], [0]])

# Euler Parameter Constraint
euler_cons = [EulerCon(body) for body in bodies]

# Get the initial quaternion by rotating around two axes
z_axis = np.array([[0], [0], [1]])
y_axis = np.array([[0], [1], [0]])

# Set initial conditions for each pendulum
pend_1.r = np.array([[0], [L], [0]])
pend_1.p = (RotAxis(y_axis, np.pi/2) * RotAxis(z_axis, np.pi/2)).arr

pend_2.r = np.array([[0], [L], [-L/2]])
pend_2.p = RotAxis(y_axis, np.pi/2).arr

# Group our constraints together. Don't attach the Euler parameter constraints or the old driving constraint
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

# Create arrays to hold Fr and nr data
Fr = np.zeros((t_steps, 3))
nr = np.zeros((t_steps, 3, 6))

# Put in the initial position/velocity/acceleration values
O_pos[0, :] = pend_1.r.T

G = pend_1.G()
Jp = 4*G.T @ J @ G
# M is constant, defined above

# Quantities for the right-hand side
γ = g_cons.GetGamma(t_start)[0:nc+1, :]
γp = -2*pend_1.dp.T @ pend_1.dp
τ = 8*pend_1.dG().T @ J @ G @ pend_1.dp
# Fg is constant, defined above

# Here we solve the larger system and redundantly retrieve r̈ and p̈
LHS_mat = np.block([[M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T], [np.zeros((4*nb, 3*nb)), Jp, pend_1.p, Φ_p.T], [
    np.zeros((nb, 3*nb)), pend_1.p.T, 0, np.zeros((nb, nc))], [Φ_r, Φ_p, np.zeros((nc, nb)), np.zeros((nc, nc))]])
RHS_mat = np.block([[Fg], [τ], [γp], [γ]])

z = np.linalg.solve(LHS_mat, RHS_mat)
ddr = np.array(z[0:3])
ddp = z[3:7]
λp = z[7]
λ = z[8:]
assert nc == np.shape(λ)[0], "λ has incorrect length"

pend_1.ddr = ddr
pend_1.ddp = ddp
O_acc[0, :] = pend_1.ddr.T

# Set BDF values
BDF1_β = 1
BDF1_α0 = 1
BDF1_α1 = -1
BDF1_α2 = 0

BDF2_β = 2/3
BDF2_α0 = 1
BDF2_α1 = -4/3
BDF2_α2 = 1/3

for i, t in enumerate(t_grid):
    if i == 0:
        continue

    if i == 1:
        β = BDF1_β
        α0 = -BDF1_α0
        α1 = -BDF1_α1
        α2 = -BDF1_α2

        C_r = α1*pend_1.r
        C_dr = α1*pend_1.dr

        C_p = α1*pend_1.p
        C_dp = α1*pend_1.dp
    else:
        β = BDF2_β
        α0 = -BDF2_α0
        α1 = -BDF2_α1
        α2 = -BDF2_α2

        C_r = α1*pend_1.r + α2*r_prev
        C_dr = α1*pend_1.dr + α2*dr_prev

        C_p = α1*pend_1.p + α2*p_prev
        C_dp = α1*pend_1.dp + α2*dp_prev

    Ψ0 = np.concatenate(
        (M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T), axis=1)
    Ψ1 = np.concatenate(
        (np.zeros((4*nb, 3*nb)), Jp, pend_1.p, Φ_p.T), axis=1)
    Ψ2 = np.concatenate(
        (np.zeros((nb, 3*nb)), pend_1.p.T, np.zeros((nb, nb)), np.zeros((nb, nc))), axis=1)
    Ψ3 = np.concatenate((Φ_r, Φ_p, np.zeros((nc, nb)),
                         np.zeros((nc, nc))), axis=1)
    Ψ = np.block([[Ψ0], [Ψ1], [Ψ2], [Ψ3]])
    Ψ_inv = np.linalg.inv(Ψ)

    r_prev = pend_1.r
    p_prev = pend_1.p

    dr_prev = pend_1.dr
    dp_prev = pend_1.dp

    # Setup and do Newton-Raphson Iteration
    k = 0
    while True:
        # Compute values needed for the g matrix
        # We can't move this outside the loop since the g_cons
        #   use e.g. body.p in their computations and body.p gets updated as we iterate
        Φ = g_cons.GetPhi(t)
        Φ_q = g_cons.GetPhiQ(t)
        Φ_r = Φ_q[:, 0:3]
        Φ_p = Φ_q[:, 3:7]
        G = pend_1.G()
        Jp = 4*G.T @ J @ G
        # M is constant, defined above

        τ = 8*pend_1.dG().T @ J @ G @ pend_1.dp
        # Fg is constant, defined above

        # Form g matrix
        g0 = M @ pend_1.ddr + Φ_r.T @ λ - Fg
        g1 = Jp @ pend_1.ddp + Φ_p.T @ λ + pend_1.p * λp - τ
        g2 = 1/(β**2 * h**2) * \
            np.array([[e_con.GetPhi(t) for e_con in euler_cons]])
        g3 = 1/(β**2 * h**2) * Φ
        g = np.block([[g0], [g1], [g2], [g3]])

        # Δz = np.linalg.solve(Ψ, -g)
        Δz = -Ψ_inv @ g
        z = z + Δz

        pend_1.ddr = z[0:3]
        pend_1.ddp = z[3:7]
        λp = z[7]
        λ = z[8:8+nc]

        pend_1.r = C_r + β**2 * h**2 * pend_1.ddr
        pend_1.p = C_p + β**2 * h**2 * pend_1.ddp

        pend_1.dr = C_dr + β*h*pend_1.ddr
        pend_1.dp = C_dp + β*h*pend_1.ddp

        print('i: ' + str(i) + ', k: ' + str(k) +
              ', norm: ' + str(np.linalg.norm(Δz)))
        print(C_r)
        print(np.amax(β**2 * h**2 * pend_1.ddr))

        if np.linalg.norm(Δz) < tol:
            break

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

    O_pos[i, :] = pend_1.r.T
    O_vel[i, :] = pend_1.dr.T
    O_acc[i, :] = pend_1.ddr.T

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
