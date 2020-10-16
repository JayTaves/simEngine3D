from simEngine3D_A6P2 import *
import matplotlib.pyplot as plt

y_O = [q0[1, 0]]
z_O = [q0[2, 0]]

pos_Q = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
y_Q = [pos_Q[1, 0]]
z_Q = [pos_Q[2, 0]]

qi = q0 - np.linalg.inv(g_cons.GetPhiQ(t0)) @ g_cons.GetPhi(t0 + Δt)
pendulum.r = q0[0:3]
pendulum.p = q0[3:]
ti = t0 + Δt

for k in range(0, int(t_end/Δt)):
    y_O.append(qi[1, 0])
    z_O.append(qi[2, 0])

    pos_Q = A(pendulum.p) @ (-L * np.array([[1], [0], [0]]))
    y_Q.append(pos_Q[1, 0])
    z_Q.append(pos_Q[2, 0])

    qi = qi - np.linalg.inv(g_cons.GetPhiQ(t0)) @ g_cons.GetPhi(ti + Δt)
    pendulum.r = qi[0:3]
    pendulum.p = qi[3:]
    ti = ti + Δt

fig = plt.figure()
plt.plot(y_O, z_O, 'o')
fig.suptitle('Position of point O′', fontsize=20)
axes = plt.gca()
axes.set_xlim([-2, 2])
axes.set_ylim([-2, 2])
plt.xlabel('y', fontsize=18)
plt.ylabel('z', fontsize=18)
fig.savefig('hw6/pointO.jpg')

plt.show()

fig = plt.figure()
plt.plot(y_Q, z_Q, 'x')
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
