from simEngine3D_A6P2 import *
import matplotlib.pyplot as plt


def NextNewton(phi, inv_phi_q, q_k):
    # Caller takes care of the inversion
    return q_k - inv_phi_q @ phi


t_start = 0
tol = 1e-5
t_steps = int(t_end/Δt)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)
max_iters = 500

# q0[5, 0] = 0.5

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

    print(g_cons.GetGamma(t))

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
