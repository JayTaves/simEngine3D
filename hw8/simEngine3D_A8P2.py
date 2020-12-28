from gcons_2body import *
import sympy as sp
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
from pstats import SortKey

profiler = cProfile.Profile()

# Physical constants
L = 2                                   # [m] - length of the bar
t_start = 0                             # [s] - simulation start time
h = 1e-3                                # [s] - time step size
t_end = 10                              # [s] - simulation end time
w = 0.05                                # [m] - side length of bar
ρ = 7800                                # [kg/m^3] - density of the bar
g_acc = np.array([[0], [0], [-9.81]])   # [m/s^2] - gravity (global frame)

# Read from file, set up bodies + constraints
(file_bodies, constraints) = ReadModelFile('revJoint.mdl')

bodies = [Body(file_body) for file_body in file_bodies]

pend1 = bodies[0]
pend2 = bodies[1]
ground = Body({}, True)

# bodies = [bodies[0]]

# Derived constants
pend1.V = 2*L * w**2                        # [m^3] - first bar volume
pend2.V = L * w**2                          # [m^3] - second bar volume

pend1.m = ρ * pend1.V                           # [kg] - first bar mass
pend2.m = ρ * pend2.V                           # [kg] - second bar mass

pend1.F = pend1.m * g_acc
pend2.F = pend2.m * g_acc
M = np.diagflat([[body.m] * 3 for body in bodies])
Fg = np.vstack([body.F for body in bodies])

J_xx_1 = 1/6 * pend1.m * w**2
J_yz_1 = 1/12 * pend1.m * (w**2 + (2*L)**2)
J_xx_2 = 1/6 * pend2.m * w**2
J_yz_2 = 1/12 * pend2.m * (w**2 + L**2)
pend1.J = np.diag([J_xx_1, J_yz_1, J_yz_1])
pend2.J = np.diag([J_xx_2, J_yz_2, J_yz_2])

# Simulation parameters
tol = 1e-2                         # Convergence threshold for Newton-Raphson
max_iters = 50                      # Iterations to abort after for Newton-Raphson

# DP1 Constraints
dp1_xx = CreateConstraint(constraints[1], pend1, ground)
dp1_yx = CreateConstraint(constraints[2], pend1, ground)

dp1_xx_2 = CreateConstraint(constraints[9], pend2, ground)
dp1_yx_2 = CreateConstraint(constraints[10], pend2, ground)

# CD ijk Constraints
cd_i = CreateConstraint(constraints[3], pend1, ground)
cd_j = CreateConstraint(constraints[4], pend1, ground)
cd_k = CreateConstraint(constraints[5], pend1, ground)
cd_i2 = CreateConstraint(constraints[6], pend1, pend2)
cd_j2 = CreateConstraint(constraints[7], pend1, pend2)
cd_k2 = CreateConstraint(constraints[8], pend1, pend2)

# Manually add these so that we don't have to hard-code '2' in the .mdl file
cd_i.si = -L * X_AXIS
cd_j.si = -L * X_AXIS
cd_k.si = -L * X_AXIS

cd_i2.si = L * X_AXIS
cd_j2.si = L * X_AXIS
cd_k2.si = L * X_AXIS

cd_i2.sj = -L/2 * Z_AXIS
cd_j2.sj = -L/2 * Z_AXIS
cd_k2.sj = -L/2 * Z_AXIS

# Euler Parameter Constraint
euler_cons = [EulerCon(body) for body in bodies]

t = sp.symbols('t')
θ_sym = sp.pi/2 + sp.pi/4 * sp.cos(2*t)
θ = sp.lambdify(t, θ_sym)
θ0 = θ(t_start)

p0 = (RotAxis(Y_AXIS, np.pi/2) * RotAxis(Z_AXIS, θ0 - np.pi/2)).arr
r0 = np.array([[0], [L * np.sin(θ0)], [L * np.cos(θ0)]])

# Set initial conditions for each pendulum
pend1.r = L * Y_AXIS
pend1.p = (RotAxis(Y_AXIS, np.pi/2) * RotAxis(Z_AXIS, np.pi/2)).arr

# pend1.r = r0
# pend1.p = p0

pend2.r = L*Y_AXIS + -(L/2)*Z_AXIS
pend2.p = RotAxis(Y_AXIS, np.pi/2).arr

# Group our constraints together. Don't attach the Euler parameter constraints or the old driving constraint
g_cons = ConGroup([cd_i, cd_j, cd_k, dp1_xx, dp1_yx,
                   cd_i2, cd_j2, cd_k2, dp1_xx_2, dp1_yx_2])
# g_cons = ConGroup([cd_i, cd_j, cd_k, dp1_xx, dp1_yx])
nc = g_cons.nc
nb = g_cons.nb

# Compute initial values
Φ = g_cons.GetPhi(t_start)
nu = g_cons.GetNu(t_start)  # Darn, ν looks like v in my font
γ = g_cons.GetGamma(t_start)
Φ_r = g_cons.GetPhiR(t_start)
Φ_p = g_cons.GetPhiP(t_start)

t_steps = int(t_end/h)
t_grid = np.linspace(t_start, t_end, t_steps, endpoint=True)

vel_con_norm = np.zeros((t_steps, 1))
omega = [np.zeros((t_steps, 3)) for body in bodies]

Jp = BlockMat([body.getJ() for body in bodies])

# Quantities for the right-hand side
γ = g_cons.GetGamma(t_start)
γp = np.vstack([-2*body.dp.T @ body.dp for body in bodies])
τ = np.vstack([body.getTau() for body in bodies])
# Fg is constant, defined above

P = BlockMat([body.p for body in bodies])

# Here we solve the larger system and redundantly retrieve r̈ and p̈
Ψ = np.block([[M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T], [np.zeros((4*nb, 3*nb)), Jp, P, Φ_p.T], [
    np.zeros((nb, 3*nb)), P.T, np.zeros((nb, nb)), np.zeros((nb, nc))], [Φ_r, Φ_p, np.zeros((nc, nb)), np.zeros((nc, nc))]])
g = np.block([[Fg], [τ], [γp], [γ]])

z = np.linalg.solve(Ψ, g)

ddr = [z[3*i:3*(i+1)] for i, _ in enumerate(bodies)]
ddp = [z[3*nb + 4*i:3*nb + 4*(i+1)] for i, _ in enumerate(bodies)]
λp = np.concatenate(tuple([z[7*nb + i:7*nb + i+1]
                           for i, _ in enumerate(bodies)]), axis=0)
λ = z[8*nb:8*nb + nc]

for i, body in enumerate(bodies):
    body.ddr = ddr[i]
    body.ddp = ddp[i]

for body in bodies:
    body.CacheRPValues()

num_iters = [0] * t_steps

profiler.enable()
for i, t in enumerate(t_grid):
    if i == 0:
        continue

    bdf = bdf1 if i == 1 else bdf2
    for body in bodies:
        body.UpdateBDFCoeffs(bdf, h)

    P = BlockMat([2*body.p for body in bodies])

    Ψ0 = np.concatenate(
        (M, np.zeros((3*nb, 4*nb)), np.zeros((3*nb, nb)), Φ_r.T), axis=1)
    Ψ1 = np.concatenate(
        (np.zeros((4*nb, 3*nb)), Jp, P, Φ_p.T), axis=1)
    Ψ2 = np.concatenate(
        (np.zeros((nb, 3*nb)), P.T, np.zeros((nb, nb)), np.zeros((nb, nc))), axis=1)
    Ψ3 = np.concatenate((Φ_r, Φ_p, np.zeros((nc, nb)),
                         np.zeros((nc, nc))), axis=1)
    Ψ = np.block([[Ψ0], [Ψ1], [Ψ2], [Ψ3]])
    Ψ_inv = np.linalg.inv(Ψ)

    for body in bodies:
        body.CacheRPValues()

    # Setup and do Newton-Raphson Iteration
    k = 0
    while True:
        for body in bodies:
            body.r = body.C_r + bdf.β**2 * h**2 * body.ddr
            body.p = body.C_p + bdf.β**2 * h**2 * body.ddp
            body.dr = body.C_dr + bdf.β*h*body.ddr
            body.dp = body.C_dp + bdf.β*h*body.ddp

        # Compute values needed for the g matrix
        # We can't move this outside the loop since the g_cons
        #   use e.g. body.p in their computations and body.p gets updated as we iterate
        Φ = g_cons.GetPhi(t)
        Φ_r = g_cons.GetPhiR(t)
        Φ_p = g_cons.GetPhiP(t)
        Jp = BlockMat([body.getJ() for body in bodies])
        τ = np.vstack([body.getTau() for body in bodies])

        P = BlockMat([2*body.p for body in bodies])

        ddr = np.vstack([body.ddr for body in bodies])
        ddp = np.vstack([body.ddp for body in bodies])

        # Form g matrix
        g0 = M @ ddr + Φ_r.T @ λ - Fg
        g1 = Jp @ ddp + Φ_p.T @ λ + P @ λp - τ
        g2 = 1/(bdf.β**2 * h**2) * \
            np.vstack([e_con.GetPhi(t) for e_con in euler_cons])
        g3 = 1/(bdf.β**2 * h**2) * Φ
        g = np.block([[g0], [g1], [g2], [g3]])

        Δz = Ψ_inv @ -g
        z = z + Δz

        ddr = [z[3*j:3*(j+1)] for j, _ in enumerate(bodies)]
        ddp = [z[3*nb + 4*j:3*nb + 4*(j+1)] for j, _ in enumerate(bodies)]
        λp = np.concatenate(tuple([z[7*nb + j:7*nb + j+1]
                                   for j, _ in enumerate(bodies)]), axis=0)
        λ = z[8*nb:8*nb + nc]

        for j, body in enumerate(bodies):
            body.ddr = ddr[j]
            body.ddp = ddp[j]

        print('i: ' + str(i) + ', k: ' + str(k) +
              ', norm: ' + str(np.linalg.norm(Δz)))

        if np.linalg.norm(Δz) < tol:
            break

        k = k + 1
        if k >= max_iters:
            print('NR not converging, stopped after ' +
                  str(max_iters) + ' iterations')
            break

    num_iters[i] = k

    # Compute violation of velocity kinematic constraint
    # Use the index 5: because the constraints for the 2nd revolute joint are all at the end
    dr = np.concatenate(tuple([body.dr for body in bodies]))
    dp = np.concatenate(tuple([body.dp for body in bodies]))
    # vel_con = Φ_r[5:, :] @ dr + Φ_p[5:, :] @ dp - g_cons.GetNu(t)[5:, :]
    # vel_con_norm[i] = np.linalg.norm(vel_con)

    # Compute the angular velocity of the bodies
    # I think this might be in the wrong frame
    for j, body in enumerate(bodies):
        omega[j][i, :] = (2*body.G() @ body.dp).T
profiler.disable()


def PrintProfiling(profiler):
    """
    Prints out profiling information, based on suggestions here https://docs.python.org/3/library/profile.html#module-cProfile
    """
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def PlotKinematicsAnalysis(grid, position, velocity, acceleration):
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # O′ - position
    ax1.plot(grid, position[:, 0])
    ax1.plot(grid, position[:, 1])
    ax1.plot(grid, position[:, 2])
    ax1.set_title('Position of point O′')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('Position [m]')

    # O′ - velocity
    ax2.plot(grid, velocity[:, 0])
    ax2.plot(grid, velocity[:, 1])
    ax2.plot(grid, velocity[:, 2])
    ax2.set_title('Velocity of point O′')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Velocity [m/s]')

    # O′ - acceleration
    ax3.plot(grid, acceleration[:, 0])
    ax3.plot(grid, acceleration[:, 1])
    ax3.plot(grid, acceleration[:, 2])
    ax3.set_title('Acceleration of point O′')
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('Acceleration [m/s²]')

    plt.show()


print('Avg. Iterations: ' + str(np.mean(num_iters)))

PrintProfiling(profiler)
PlotKinematicsAnalysis(t_grid, O_pos, O_vel, O_acc)
