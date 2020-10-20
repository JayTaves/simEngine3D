from simEngine3D_A6P2 import *
import matplotlib.pyplot as plt


def NextNewton(phi, inv_phi_q, q_k):
    # Caller takes care of the inversion
    return q_k - inv_phi_q @ phi


t_start = 0
tol = 1e-6
t_steps = int(t_end/Δt)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)
max_iters = 500

# Get arrays to hold our y and z positions
O_pos = np.zeros((t_steps, 2))
Q_pos = np.zeros((t_steps, 2))

# Put in the initial values
Q0 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
O_pos[0, :] = q0[1:2, 0].T
Q_pos[0, :] = Q0[1:2, 0].T

pendulum.r = q0[0:3]
pendulum.p = q0[3:]


# Set initial conditions
q_k = q0

for i, t in enumerate(t_grid):
    if i == 0:
        continue
    if i > 78:
        break

    # Only get (inverse of the) Jacobian once per time step
    inv_phi_q = np.linalg.inv(g_cons.GetPhiQ(t))
    phi = g_cons.GetPhi(t)
    # print('Inverse Jac: ')
    # print(np.shape(inv_phi_q))
    # print('Φ:')
    # print(np.shape(phi))

    # Setup and do Newton-Raphson Iteration
    k = 0
    q_k1 = NextNewton(phi, inv_phi_q, q_k)
    while np.linalg.norm(q_k1 - q_k) > tol:
        q_k = q_k1

        # Trying with the bodies updated with each q_k guess
        pendulum.r = q_k[0:3]
        pendulum.p = q_k[3:]
        phi = g_cons.GetPhi(t)

        # Uh-oh, do we need to update the bodies as we iterate? So that phi(q, t) is updated?
        q_k1 = NextNewton(phi, inv_phi_q, q_k)

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

        print('i: ' + str(i) + ', k: ' + str(k) +
              ', norm: ' + str(np.linalg.norm(q_k1 - q_k)))

    pendulum.r = q_k1[0:3]
    pendulum.p = q_k1[3:]

    Q_k1 = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
    O_pos[0, :] = q_k1[1:2, 0].T
    Q_pos[0, :] = Q_k1[1:2, 0].T

fig = plt.figure()
plt.plot(O_pos[:, 0], O_pos[:, 1], 'o')
fig.suptitle('Position of point O′', fontsize=20)
axes = plt.gca()
axes.set_xlim([-2, 2])
axes.set_ylim([-2, 2])
plt.xlabel('y', fontsize=18)
plt.ylabel('z', fontsize=18)
fig.savefig('hw6/pointO.jpg')

plt.show()

fig = plt.figure()
plt.plot(Q_pos[:, 0], Q_pos[:, 1], 'x')
fig.suptitle('Position of point Q', fontsize=20)
axes = plt.gca()
axes.set_xlim([-2, 2])
axes.set_ylim([-2, 2])
plt.xlabel('y', fontsize=18)
plt.ylabel('z', fontsize=18)
fig.savefig('hw6/pointQ.jpg')

plt.show()

# For velocity solve using nu

# For acceleration solve using gamma
