import numpy as np
from collections import namedtuple
import json as js
from enum import Enum

I3 = np.identity(3)

X_AXIS = np.array([[1], [0], [0]])  # x-out of page
Y_AXIS = np.array([[0], [1], [0]])  # y-right
Z_AXIS = np.array([[0], [0], [1]])  # z-up

# Setup BDF values
BDFVals = namedtuple('BDFVals', ['β', 'α'])
bdf1 = BDFVals(β=1, α=[-1, 1, 0])
bdf2 = BDFVals(β=2/3, α=[-1, 4/3, -1/3])


class Constraints(Enum):
    DP1 = 0
    CD = 1
    DP2 = 2
    D = 3
    Euler = 4


def BlockMat(lst):
    """
    Takes a list (of len k) of (m x n) matrices and assembles a (km x kn) block matrix with the matrices on the diagonal
    """

    m, n = np.shape(lst[0])
    k = len(lst)

    ret = np.zeros((m*k, n*k))
    for i, mat in enumerate(lst):
        ret[m*i:m*(i+1), n*i:n*(i+1)] = mat

    return ret


def CheckVector(v, n):
    if v.shape == (1, n):
        print('Warning: Input was a row vector, automatically transposed it')
        v = v.T
    elif v.shape != (n, 1):
        raise ValueError('Input vector v did not have dimension ' + str(n))

    return v


class Quaternion:
    """
    Weird, couldn't find a NumPy/SciPy package for this...
    """

    def __init__(self, v):
        v = CheckVector(v, 4)
        self.r = v[0, 0]
        self.i = v[1, 0]
        self.j = v[2, 0]
        self.k = v[3, 0]

        self.arr = v

    def __mul__(self, other):
        """
        Maybe the matrix definition of multiplication is cleaner? idk
        """
        r = self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k
        i = self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j
        j = self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i
        k = self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r

        return Quaternion(np.array([[r], [i], [j], [k]]))


def RotAxis(v, θ):
    """
    Gets the quaternion representing a rotation of θ radians about the v axis
    """
    v = CheckVector(v, 3)

    e0 = np.array([[np.cos(θ/2)]])
    e = v * np.sin(θ/2)

    return Quaternion(np.concatenate((e0, e), axis=0))


def GetCross(v):
    """
    Computes the n by n cross product matrix ṽ for a given vector of dimensions n. Expects a column vector but will
    noisily transpose a row vector

    ṽ satisfies ṽa = v x a - where x is the cross-product operator
    """
    v = CheckVector(v, 3)

    return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])


def A(p):
    """
    Computes a rotation matrix (A) from a given orientation vector (unit quaternion) p. Expects a column vector but will
    noisily transpose a row vector
    """
    p = CheckVector(p, 4)

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = GetCross(e)

    return (e0**2 - e.T @ e) * I3 + 2*(e @ e.T + e0 * ẽ)


def B(p, a):
    """
    Computes the B matrix from a given orientation vector (unit quaternion) and position vector. Expects column vectors
    but will noisily transpose row vectors.

    TODO: Write down what B actually means...
    """
    p = CheckVector(p, 4)
    a = CheckVector(a, 3)

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = GetCross(e)

    c1 = (e0 * I3 + ẽ) @ a
    c2 = e @ a.T - (e0 * I3 + ẽ) @ GetCross(a)

    return 2 * np.concatenate((c1, c2), axis=1)


def CreateConstraint(json_cons, body_i, body_j):
    con_type = Constraints[json_cons["type"]]
    if con_type == Constraints.DP1:
        con = DP1(json_cons, body_i, body_j)
    elif con_type == Constraints.CD:
        con = CD(json_cons, body_i, body_j)
    elif con_type == Constraints.DP2:
        raise ValueError('DP2 constraint not implemented yet')
    elif con_type == Constraints.D:
        raise ValueError('D constraint not implemented yet')
    else:
        raise ValueError('Unmapped enum value')

    return con


def InitTwoBodyModel(file_name):
    (bodies, constraints) = ReadModelFile(file_name)

    body_i = Body(bodies[0])
    body_j = Body(bodies[1])

    cons = [CreateConstraint(con, body_i, body_j) for con in constraints]

    return cons


def ReadModelFile(file_name):
    with open(file_name) as model_file:
        model_data = js.load(model_file)

        model_bodies = model_data['bodies']
        model_constraints = model_data['constraints']

    return (model_bodies, model_constraints)


class Body:
    def __init__(self, dict, is_ground=False):
        self.is_ground = is_ground
        if is_ground:
            self.id = "ground"

            self.r = np.zeros((3, 1))
            self.p = np.array([[1, 0, 0, 0]]).T
            self.dp = np.zeros((4, 1))
            self.dr = np.zeros((3, 1))
        else:
            self.id = dict['id']

            self.r = np.array([dict['r']]).T
            self.dr = np.array([dict['dr']]).T
            p = np.array([dict['p']]).T
            self.p = p / np.linalg.norm(p)

            self.dp = np.array([dict['dp']]).T

            # Give ourselves some properties to use later
            self.J = None
            self.F = None
            self.m = None
            self.V = None

        self.r_prev = self.r
        self.p_prev = self.p
        self.dr_prev = self.dr
        self.dp_prev = self.dp

        self.C_r = 0
        self.C_p = 0
        self.C_dr = 0
        self.C_dp = 0

    def CacheRPValues(self):
        self.r_prev = self.r
        self.p_prev = self.p
        self.dr_prev = self.dr
        self.dp_prev = self.dp

    def UpdateBDFCoeffs(self, bdf, h):
        self.C_dr = bdf.α[1]*self.dr + bdf.α[2]*self.dr_prev
        self.C_r = bdf.α[1]*self.r + bdf.α[2]*self.r_prev + bdf.β*h*self.C_dr

        self.C_dp = bdf.α[1]*self.dp + bdf.α[2]*self.dp_prev
        self.C_p = bdf.α[1]*self.p + bdf.α[2]*self.p_prev + bdf.β*h*self.C_dp

    def dG(self):
        """
        Computes the Ġ matrix Ġ(p) = d/dt[-e, -ẽ + e_0 I]
        """

        e = self.dp[1:, ...]
        e0 = self.dp[0, 0]
        ẽ = GetCross(e)

        return np.concatenate((-e, -ẽ + e0 * I3), axis=1)

    def G(self):
        """
        Computes the G matrix G(p) = [-e, -ẽ + e_0 I]
        """

        e = self.p[1:, ...]
        e0 = self.p[0, 0]
        ẽ = GetCross(e)

        return np.concatenate((-e, -ẽ + e0 * I3), axis=1)

    def E(self):
        """
        Computes the E matrix E(p) = [-e, ẽ + e_0 I]
        """

        e = self.p[1:, ...]
        e0 = self.p[0, 0]
        ẽ = GetCross(e)

        return np.concatenate((-e, ẽ + e0 * I3), axis=1)

    def getJ(self):
        G = self.G()

        return 4*G.T @ self.J @ G

    def getTau(self):
        return 8*self.dG().T @ self.J @ self.G() @ self.dp

    @property
    def dω(self):
        return 2*self.G() @ self.ddp

    @property
    def ω(self):
        return 2*self.G() @ self.dp

    @property
    def A(self):
        return A(self.p)


class CD:
    cons_type = Constraints.CD

    def __init__(self, dict, body_i, body_j):
        si = np.array([dict['si']]).T
        sj = np.array([dict['sj']]).T
        c = np.array([dict['c']]).T

        self.Initialize(body_i, body_j, si, sj, c,
                        dict['f'], dict['df'], dict['ddf'])

    def Initialize(self, body_i, body_j, si, sj, c, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.si = si
        self.sj = sj

        self.c = c

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    def GetPhi(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        return self.c.T @ (self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si) - self.f(t)

    def GetGamma(self, t):
        term1 = B(self.body_i.dp, self.si) @ self.body_i.dp
        term2 = B(self.body_j.dp, self.sj) @ self.body_j.dp

        return self.c.T @ (term1 - term2) + self.ddf(t)

    def GetNu(self, t):
        return [self.df(t)]

    def GetPhiR(self, t):
        if self.body_i.is_ground:
            return [(self.body_j.id, self.c.T)]
        if self.body_j.is_ground:
            return [(self.body_i.id, -self.c.T)]
        return [(self.body_i.id, -self.c.T), (self.body_j.id, self.c.T)]

    def GetPhiP(self, t):
        Bpj = (self.body_j.id, self.c.T @ B(self.body_j.p, self.sj))
        Bpi = (self.body_i.id, -self.c.T @ B(self.body_i.p, self.si))

        if self.body_i.is_ground:
            return [Bpj]
        if self.body_j.is_ground:
            return [Bpi]

        return [Bpi, Bpj]


class DP1:
    cons_type = Constraints.DP1

    def __init__(self, dict, body_i, body_j):
        ai = np.array([dict['ai']]).T
        aj = np.array([dict['aj']]).T

        self.Initialize(
            body_i, body_j, ai, aj, dict['f'], dict['df'], dict['ddf'])

    def Initialize(self, body_i, body_j, ai, aj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.ai = ai
        self.aj = aj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    def GetPhi(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        return self.ai.T @ Ai.T @ Aj @ self.aj - self.f(t)

    def GetGamma(self, t):
        B_dpj = B(self.body_j.dp, self.aj)
        B_dpi = B(self.body_i.dp, self.ai)

        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        aiT = self.ai.T @ Ai.T
        ajT = self.aj.T @ Aj.T

        ai_dot = B(self.body_i.p, self.ai) @ self.body_i.dp
        aj_dot = B(self.body_j.p, self.aj) @ self.body_j.dp

        γ = -aiT @ B_dpj @ self.body_j.dp - ajT @ B_dpi @ self.body_i.dp - \
            2*ai_dot.T @ aj_dot + self.ddf(t)

        return γ

    def GetNu(self, t):
        return [self.df(t)]

    def GetPhiR(self, t):
        return []

    def GetPhiP(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        term_i = (self.body_i.id, (B(self.body_i.p, self.ai).T @ Aj @ self.aj).T)
        term_j = (self.body_j.id, (self.ai.T @ Ai.T @
                                   B(self.body_j.p, self.aj)).T)

        if self.body_i.is_ground:
            return [term_j]
        if self.body_j.is_ground:
            return [term_i]

        return [term_i, term_j]


class DP2:
    cons_type = Constraints.DP2

    def __init__(self, dict, body_i, body_j):
        ai = np.array([dict['ai']]).T
        aj = np.array([dict['aj']]).T

        si = np.array([dict['si']]).T
        sj = np.array([dict['sj']]).T

        self.Initialize(
            body_i, body_j, ai, aj, si, sj, dict['f'], dict['df'], dict['ddf'])

    def Initialize(self, body_i, body_j, ai, aj, si, sj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.ai = ai
        self.aj = aj

        self.si = si
        self.sj = sj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    def GetPhi(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        return self.ai.T @ Ai.T @ (self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si) - self.f(t)

    def GetGamma(self, t):
        B_dpi = B(self.body_i.dp, self.si)
        B_dpj = B(self.body_j.dp, self.sj)
        B_dpi_ai = B(self.body_i.dp, self.ai)

        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        aiT = self.ai.T @ Ai.T
        # ajT = self.aj.T @ Aj.T

        ai_dot = B(self.body_i.p, self.ai) @ self.body_i.dp
        # aj_dot = B(self.body_j.p, self.aj) @ self.body_j.dp

        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si
        dij_dot = self.body_j.dr + B(self.body_j.p, self.sj) @ self.body_j.dp - \
            self.body_i.dr - B(self.body_i.p, self.si) @ self.body_i.dp

        # I match what is on the slides there, but I think there is a typo...
        γ = -aiT @ B_dpj @ self.body_j.dp - \
            aiT @ B_dpi @ self.body_i.dp - \
            dij.T @ B_dpi_ai @ self.body_i.dp - \
            2 * ai_dot.T @ dij_dot + self.ddf(t)

        return γ

    def GetNu(self, t):
        return [[self.df(t)]]

    def GetPhiR(self, t):
        ai = self.ai.T @ A(self.body_i.p).T
        if self.body_i.is_ground:
            return ai
        if self.body_j.is_ground:
            return -ai
        return np.concatenate((-ai, ai), axis=1)

    def GetPhiP(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si

        term_i = B(self.body_i.p, self.ai).T @ dij - \
            self.ai.T @ Ai.T @ B(self.body_i.p, self.si)
        term_j = self.ai.T @ Ai.T @ B(self.body_j.p, self.sj)

        if self.body_i.is_ground:
            return term_j.T
        if self.body_j.is_ground:
            return term_i.T

        # To be more technically correct we could compute our B matrices with respect to [p_i 0 0 0 0] and [0 0 0 0 p_j]
        # but it is just easier to concatenate instead
        return np.concatenate((term_i.T, term_j.T), axis=1).T


class D:
    cons_type = Constraints.D

    def __init__(self, dict, body_i, body_j):
        ai = np.array([dict['ai']]).T
        aj = np.array([dict['aj']]).T

        si = np.array([dict['si']]).T
        sj = np.array([dict['sj']]).T

        self.Initialize(
            body_i, body_j, ai, aj, si, sj, dict['f'], dict['df'], dict['ddf'])

    def Initialize(self, body_i, body_j, ai, aj, si, sj, f, df, ddf):
        self.body_i = body_i
        self.body_j = body_j

        if body_i.is_ground and body_j.is_ground:
            raise ValueError('Both bodies cannot be ground')

        self.ai = ai
        self.aj = aj

        self.si = si
        self.sj = sj

        self.f = lambda t: 0
        self.df = lambda t: 0
        self.ddf = lambda t: 0

    def GetPhi(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si

        return dij.T @ dij - self.f(t)

    def GetGamma(self, t):
        B_dpi = B(self.body_i.dp, self.si)
        B_dpj = B(self.body_j.dp, self.sj)

        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si
        dij_dot = self.body_j.dr + B(self.body_j.p, self.sj) @ self.body_j.dp - \
            self.body_i.dr - B(self.body_i.p, self.si) @ self.body_i.dp

        γ = -2 * dij.T @ B_dpj @ self.body_j.dp + \
            2 * dij.T @ B_dpi @ self.body_i.dp - \
            2 * dij_dot.T @ dij_dot + self.ddf(t)

        return γ

    def GetNu(self, t):
        return [[self.df(t)]]

    def GetPhiR(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)
        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si

        if self.body_i.is_ground:
            return 2*dij
        if self.body_j.is_ground:
            return -2*dij
        return np.concatenate((-2*dij, 2*dij), axis=1)

    def GetPhiP(self, t):
        Ai = A(self.body_i.p)
        Aj = A(self.body_j.p)

        dij = self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si

        term_i = -2 * B(self.body_i.p, self.si) @ dij
        term_j = 2 * B(self.body_j.p, self.sj) @ dij

        if self.body_i.is_ground:
            return term_j.T
        if self.body_j.is_ground:
            return term_i.T

        # To be more technically correct we could compute our B matrices with respect to [p_i 0 0 0 0] and [0 0 0 0 p_j]
        # but it is just easier to concatenate instead
        return np.concatenate((term_i.T, term_j.T), axis=1).T


class EulerCon:
    cons_type = Constraints.Euler

    def __init__(self, body):
        self.body = body

    def GetPhi(self, t):
        return self.body.p.T @ self.body.p - 1

    def GetGamma(self, t):
        return -2 * (self.body.dp.T @ self.body.dp)

    def GetNu(self, t):
        return [[0]]

    def GetPhiR(self, t):
        return np.zeros((1, 3))

    def GetPhiP(self, t):
        return 2*self.body.p.T

    def GetPhiQ(self, t):
        return np.concatenate((self.GetPhiR(t), self.GetPhiP(t)), axis=1)


class ConGroup:
    def __init__(self, con_list):
        self.cons = con_list
        self.nc = len(self.cons)
        self.nb = len(set([b_id for con in con_list for b_id in (
            con.body_i.id, con.body_j.id)])) - 1

        self.Φ = np.zeros((self.nc, 1))
        self.Φr = np.zeros((self.nc, 3*self.nb))
        self.Φp = np.zeros((self.nc, 4*self.nb))
        self.γ = np.zeros((self.nc, 1))
        self.nu = np.zeros((self.nc, 1))

    def GetPhi(self, t):
        for i, con in enumerate(self.cons):
            self.Φ[i, 0] = con.GetPhi(t)
        return self.Φ

    def GetGamma(self, t):
        for i, con in enumerate(self.cons):
            self.γ[i] = con.GetGamma(t)
        return self.γ

    def GetNu(self, t):
        for i, con in enumerate(self.cons):
            self.nu[i] = con.GetNu(t)
        return self.nu

    def GetPhiR(self, t):
        for i, con in enumerate(self.cons):
            for b_id, phiR in con.GetPhiR(t):
                self.Φr[i, 3*b_id:3*(b_id + 1)] = phiR
        return self.Φr

    def GetPhiP(self, t):
        for i, con in enumerate(self.cons):
            for b_id, phiP in con.GetPhiP(t):
                self.Φp[i, 4*b_id:4*(b_id + 1)] = phiP
        return self.Φp

    def GetPhiQ(self, t):
        return np.concatenate((self.GetPhiR(t), self.GetPhiP(t)), axis=1)
