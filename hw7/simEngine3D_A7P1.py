from gcons import *
import sympy as sp
import matplotlib.pyplot as plt

# Parameters
L = 2
t0 = 0
Δt = 10e-3
t_end = 10

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

cd_i.si = np.array([[-L], [0], [0]])
cd_j.si = np.array([[-L], [0], [0]])
cd_k.si = np.array([[-L], [0], [0]])

# Signs flipped here from what you might intuitively expect
# because of the direction we take the difference (0 - a) vs. (a - 0)
# CD_con_j = -L * sp.sin(θ_sym)
# CD_con_k = L * sp.cos(θ_sym)

# cd_j.f = sp.lambdify(t, CD_con_j)
# cd_j.df = sp.lambdify(t, sp.diff(CD_con_j, t))
# cd_j.ddf = sp.lambdify(t, sp.diff(CD_con_j, t, t))

# cd_k.f = sp.lambdify(t, CD_con_k)
# cd_k.df = sp.lambdify(t, sp.diff(CD_con_k, t))
# cd_k.ddf = sp.lambdify(t, sp.diff(CD_con_k, t, t))

# Euler Parameter Constraint
e_param = EulerCon(pendulum)

# Derived initializations
θ0 = θ(t0)
r0 = np.array([[0], [L * np.sin(θ0)], [L * np.cos(θ0)]])
z_axis = np.array([[0], [0], [1]])
y_axis = np.array([[0], [1], [0]])
p0 = (RotAxis(y_axis, np.pi/2) * RotAxis(z_axis, θ0 - np.pi/2)).arr

# Didn't read these in from the file, so set them now
pendulum.r = r0
pendulum.p = p0
pendulum.dp = 1/2 * E(p0).T @ (dp1_drive.df(t0) * z_axis)

# Group our constraints together
g_cons = ConGroup([cd_i, cd_j, cd_k, dp1_xx, dp1_yx, dp1_drive, e_param])

# # Compute values
Φ = g_cons.GetPhi(t0)
nu = g_cons.GetNu(t0)  # Darn, ν ≈ v in my font
γ = g_cons.GetGamma(t0)
Φ_r = g_cons.GetPhiR(t0)
Φ_p = g_cons.GetPhiP(t0)
Φ_q = g_cons.GetPhiQ(t0)

print(Φ)
# print(nu)
print(γ)
# print(Φ_r)
# print(Φ_p)
# print(Φ_q)

# print(r0)
# print(p0)

q0 = np.concatenate((r0, p0), axis=0)


def NextNewton(phi, inv_phi_q, q_k):
    # Caller takes care of the inversion
    return q_k - inv_phi_q @ phi


t_start = 0
tol = 1e-5
t_steps = int(t_end/Δt)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)
max_iters = 500

M = np

# Get arrays to hold O and Q data
O_pos = np.zeros((t_steps, 3))
O_vel = np.zeros((t_steps, 3))
O_acc = np.zeros((t_steps, 3))

Q_pos = np.zeros((t_steps, 3))
Q_vel = np.zeros((t_steps, 3))
Q_acc = np.zeros((t_steps, 3))

# Put in the initial values
Q0 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
O_pos[0, :] = q0[0:3, 0].T
Q_pos[0, :] = Q0[0:3, 0].T

pendulum.r = q0[0:3]
pendulum.p = q0[3:]

inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t_start))

O_vel[0, :] = (inv_phi_q @ g_cons.GetNu(t_start))[0:3, 0].T
O_acc[0, :] = (inv_phi_q @ g_cons.GetGamma(t_start))[0:3, 0].T

print(q0)

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
    q_k1 = NextNewton(phi, inv_phi_q, q_k)
    while np.linalg.norm(q_k1 - q_k) > tol:
        q_k = q_k1

        # Trying with the bodies updated with each q_k guess
        pendulum.r = q_k[0:3]
        pendulum.p = q_k[3:]
        inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))
        phi = g_cons.GetPhi(t)

        # Uh-oh, do we need to update the bodies as we iterate? So that phi(q, t) is updated?
        q_k1 = NextNewton(phi, inv_phi_q, q_k)

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

        # print('i: ' + str(i) + ', k: ' + str(k) +
        #       ', norm: ' + str(np.linalg.norm(q_k1 - q_k)))

    pendulum.r = q_k1[0:3]
    pendulum.p = q_k1[3:]

    # Once the position converges, we can compute velocity and acceleration
    inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))

    print(g_cons.GetGamma(t)[5, 0])

    dq_k = inv_phi_q @ g_cons.GetNu(t)
    ddq_k = inv_phi_q @ g_cons.GetGamma(t)

    # With quantities computed, fill them into our output arrays
    Q_k1 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
    O_pos[i, :] = q_k1[0:3, 0].T
    Q_pos[i, :] = Q_k1[0:3, 0].T

    O_vel[i, :] = dq_k[0:3, 0].T
    O_acc[i, :] = ddq_k[0:3, 0].T

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

# For velocity solve using nu

# For acceleration solve using gamma
