"""Leading-order multipole strain of two point masses (Thorne 1980 eq. 4.12'): used by the
event validation scripts to check orbit conventions against each model's h_+, h_x."""

import numpy as np

eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1


def multipole_strain(x1, x2, m1, m2, dt, N):
    """r * h_ij (G=c=1) of two point masses, positions sampled uniformly with step dt."""

    def d(a, n):
        for _ in range(n):
            a = np.gradient(a, dt, axis=0, edge_order=2)
        return a

    I2 = m1 * np.einsum("ti,tj->tij", x1, x1) + m2 * np.einsum("ti,tj->tij", x2, x2)
    I3 = m1 * np.einsum("ti,tj,tk->tijk", x1, x1, x1) + m2 * np.einsum(
        "ti,tj,tk->tijk", x2, x2, x2
    )
    v1, v2 = d(x1, 1), d(x2, 1)
    # current quadrupole S_ij = STF sum m x_i (x cross v)_j
    S = m1 * np.einsum("ti,tj->tij", x1, np.cross(x1, v1)) + m2 * np.einsum(
        "ti,tj->tij", x2, np.cross(x2, v2)
    )
    S = (S + S.transpose(0, 2, 1)) / 2
    S -= np.einsum("tkk->t", S)[:, None, None] * np.eye(3) / 3

    h = 2 * d(I2, 2)
    h += 2 / 3 * np.einsum("tijk,k->tij", d(I3, 3), N)
    A = np.einsum("pqi,tjp,q->tij", eps, d(S, 2), N)
    h += 8 / 3 * (A + A.transpose(0, 2, 1)) / 2
    return h


def project(h, P, Q):
    hp = 0.5 * (np.einsum("tij,i,j->t", h, P, P) - np.einsum("tij,i,j->t", h, Q, Q))
    hc = 0.5 * (np.einsum("tij,i,j->t", h, P, Q) + np.einsum("tij,i,j->t", h, Q, P))
    return hp, hc
