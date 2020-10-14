import numpy as np
import json as js
from enum import Enum

I3 = np.identity(3)


class Constraints(Enum):
    DP1 = 0
    CD = 1
    DP2 = 2
    D = 3


def GetCross(v):
    """
    Computes the n by n cross product matrix ṽ for a given vector of dimensions n. Expects a column vector but will
    noisily transpose a row vector

    ṽ satisfies ṽa = v x a - where x is the cross-product operator
    """
    if v.shape == (1, 3):
        print('Warning: Input to GetCross was a row vector, automatically transposed it')
        v = v.T
    elif v.shape != (3, 1):
        raise ValueError('Input vector v was not a column vector')
    return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])


def GetAMatrix(p):
    """
    Computes a rotation matrix (A) from a given orientation vector (unit quaternion) p. Expects a column vector but will
    noisily transpose a row vector
    """
    if p.shape == (1, 4):
        print('Warning: Input to GetAMatrix was a row vector, automatically transposed it')
        p = p.T
    elif p.shape != (4, 1):
        raise ValueError('Input vector p was not in R^4')

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = GetCross(e)

    return (e0**2 - e.T @ e) * I3 + 2*(e @ e.T + e0 * ẽ)


def GetBMatrix(p, a):
    """
    Computes the B matrix from a given orientation vector (unit quaternion) and position vector. Expects column vectors
    but will noisily transpose row vectors.

    TODO: Write down what B actually means...
    """
    if p.shape == (1, 4):
        print('Warning: Input to GetBMatrix was a row vector, automatically transposed it')
        p = p.T
    elif p.shape != (4, 1):
        raise ValueError('Input vector p was not in R^4')
    if a.shape == (1, 3):
        print('Warning: Input to GetBMatrix was a row vector, automatically transposed it')
        a = a.T
    elif a.shape != (3, 1):
        raise ValueError('Input vector a was not in R^3')

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
            self.r = np.array([[0, 0, 0]]).T
            self.p = np.array([[1, 0, 0, 0]]).T
            self.dp = np.array([[0, 0, 0, 0]]).T
        else:
            self.id = dict['id']

            self.r = np.array([dict['r']]).T
            p = np.array([dict['p']]).T
            self.p = p / np.linalg.norm(p)

            # NOTE: Mysterious normalization step only needed to exactly match example values
            # Could just do
            # self.dp = np.array([dict['dp]]).T
            dp = np.array([dict['dp']]).T
            dp[3, 0] = -np.dot(dp[0:3, 0], p[0:3, 0]) / p[3, 0]
            self.dp = dp / np.linalg.norm(dp)


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
        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        return self.c.T @ (self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si) - self.f(t)

    def GetGamma(self, t):
        term1 = GetBMatrix(self.body_i.dp, self.si) @ self.body_i.dp
        term2 = GetBMatrix(self.body_j.dp, self.sj) @ self.body_j.dp

        return self.c.T @ (term1 - term2) + self.ddf(t)

    def GetNu(self, t):
        return [[self.df(t)]]

    def GetPhiR(self, t):
        if self.body_i.is_ground:
            return self.c.T
        if self.body_j.is_ground:
            return -self.c.T
        return np.concatenate((-self.c.T, self.c.T), axis=1)

    def GetPhiP(self, t):
        Bpj = self.c.T @ GetBMatrix(self.body_j.p, self.sj)
        Bpi = -self.c.T @ GetBMatrix(self.body_i.p, self.si)

        if self.body_i.is_ground:
            return Bpj
        if self.body_j.is_ground:
            return Bpi

        return np.concatenate((Bpi, Bpj), axis=1)


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
        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        return self.ai.T @ Ai.T @ Aj @ self.aj - self.f(t)

    def GetGamma(self, t):
        B_dpj = GetBMatrix(self.body_j.dp, self.aj)
        B_dpi = GetBMatrix(self.body_i.dp, self.ai)

        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        aiT = self.ai.T @ Ai.T
        ajT = self.aj.T @ Aj.T

        ai_dot = GetBMatrix(self.body_i.p, self.ai) @ self.body_i.dp
        aj_dot = GetBMatrix(self.body_j.p, self.aj) @ self.body_j.dp

        γ = -aiT @ B_dpj @ self.body_j.dp - ajT @ B_dpi @ self.body_i.dp - \
            2*ai_dot.T @ aj_dot + self.ddf(t)

        return γ

    def GetNu(self, t):
        return [[self.df(t)]]

    def GetPhiR(self, t):
        if self.body_i.is_ground or self.body_j.is_ground:
            return np.zeros((1, 3))
        return np.zeros((1, 6))

    def GetPhiP(self, t):
        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        term_i = np.transpose(GetBMatrix(
            self.body_i.p, self.ai)) @ Aj @ self.aj
        term_j = np.transpose(
            self.ai) @ Ai.T @ GetBMatrix(self.body_j.p, self.aj)

        if self.body_i.is_ground:
            return term_j
        if self.body_j.is_ground:
            return term_i

        # To be more technically correct we could compute our B matrices with respect to [p_i 0 0 0 0] and [0 0 0 0 p_j]
        # but it is just easier to concatenate instead
        return np.concatenate((term_i.T, term_j), axis=1)
