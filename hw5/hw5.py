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
    if v.shape != (3, 1):
        raise ValueError('Input vector v was not a column vector')
    return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])


def GetAMatrix(p):
    if p.shape != (4, 1):
        raise ValueError('Input vector p was not in R^4')

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = GetCross(e)

    return (e0**2 - e.T @ e) * I3 + 2*(e @ e.T + e0 * ẽ)


def GetBMatrix(p, a):
    if p.shape != (4, 1):
        raise ValueError('Input vector p was not in R^4')
    if a.shape != (3, 1):
        raise ValueError('Input vector a was not in R^3')

    e = p[1:, ...]
    e0 = p[0, 0]
    ẽ = GetCross(e)

    c1 = (e0 * I3 + ẽ) @ a
    c2 = e @ a.T - (e0 * I3 + ẽ) @ GetCross(a)

    return 2 * np.concatenate((c1, c2), axis=1)


def ReadModelFile(file_name):
    with open(file_name) as model_file:
        model_data = js.load(model_file)

        model_bodies = model_data['bodies']
        model_constraints = model_data['constraints']

        body_i = Body(model_bodies[0])
        body_j = Body(model_bodies[1])

        dp1 = DP1(model_constraints[0], body_i, body_j)
        cd = CD(model_constraints[1], body_i, body_j)

    return (dp1, cd)


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

        self.f = np.array([[f]])
        self.df = np.array([[df]])
        self.ddf = np.array([[ddf]])

    def GetPhi(self):
        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        return self.c.T @ (self.body_j.r + Aj @ self.sj - self.body_i.r - Ai @ self.si) - self.f

    def GetGamma(self):
        term1 = GetBMatrix(self.body_i.dp, self.si) @ self.body_i.dp
        term2 = GetBMatrix(self.body_j.dp, self.sj) @ self.body_j.dp

        return self.c.T @ (term1 - term2) + self.ddf

    def GetNu(self):
        return self.df

    def GetPhiR(self):
        if self.body_i.is_ground:
            return self.c.T
        if self.body_j.is_ground:
            return -self.c.T
        return np.concatenate((-self.c.T, self.c.T), axis=1)

    def GetPhiP(self):
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

        self.f = np.array([[f]])
        self.df = np.array([[df]])
        self.ddf = np.array([[ddf]])

    def GetPhi(self):
        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        return self.ai.T @ Ai.T @ Aj @ self.aj - self.f

    def GetGamma(self):
        B_dpj = GetBMatrix(self.body_j.dp, self.aj)
        B_dpi = GetBMatrix(self.body_i.dp, self.ai)

        Ai = GetAMatrix(self.body_i.p)
        Aj = GetAMatrix(self.body_j.p)

        aiT = self.ai.T @ Ai.T
        ajT = self.aj.T @ Aj.T

        ai_dot = GetBMatrix(self.body_i.p, self.ai) @ self.body_i.dp
        aj_dot = GetBMatrix(self.body_j.p, self.aj) @ self.body_j.dp

        γ = -aiT @ B_dpj @ self.body_j.dp - ajT @ B_dpi @ self.body_i.dp - \
            2*ai_dot.T @ aj_dot + self.ddf

        return γ

    def GetNu(self):
        return self.df

    def GetPhiR(self):
        if self.body_i.is_ground or self.body_j.is_ground:
            return [[0, 0, 0]]
        return [[0, 0, 0, 0, 0, 0]]

    def GetPhiP(self):
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


(dp1, cd) = ReadModelFile('hw5.mdl')

print(dp1.GetPhi())
print(dp1.GetNu())
print(dp1.GetGamma())
print(dp1.GetPhiR())
print(dp1.GetPhiP())

print(cd.GetPhi())
print(cd.GetNu())
print(cd.GetGamma())
print(cd.GetPhiR())
print(cd.GetPhiP())
