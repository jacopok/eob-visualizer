"""Labels of the two BHs by orbit clustering, in the spirit of arXiv:2409.07519 but with a
trajectory-based distance (see `gw250114/relabel.py` for the study on the posterior).

Swapping the labels of the two BHs describes the same binary; the m_1 >= m_2 convention
picks one representative per sample. Here instead "BH 1" is the body whose orbit agrees
with that of BH 1 in the other samples: with u_i(t) the direction of BH 1 from BH 2 in
sample i, a relabeling flips u_i -> -u_i, and with
    C_ij = sum_t w(t) u_i(t) . u_j(t),
the signs s_i maximize sum_ij s_i s_j C_ij = sum_t w |sum_i s_i u_i|^2
(Z2 synchronization): spectral initialization (leading eigenvector of C),
then Lloyd iterations s_i = sign(sum_t w u_i . mu), mu = sum_j s_j u_j.
"""

import numpy as np


def synchronize_signs(u, weight, max_iter=100):
    """Signs s maximizing sum_ij s_i s_j C_ij for u of shape (samples, times, 3) (NaN
    where undefined), and the per-sample agreement
    score_i = sum_t w s_i u_i . mu / sqrt(sum_t w |u_i|^2 sum_t w |mu|^2) in [-1, 1].
    The overall sign is fixed so that most samples keep their labels."""
    from scipy.sparse.linalg import svds

    v = np.nan_to_num(u) * np.sqrt(weight).astype(u.dtype)[None, :, None]
    V = v.reshape(len(v), -1).astype(np.float32)
    # leading eigenvector of C = V V^T, without forming C
    left, _, _ = svds(V, k=1)
    s = np.where(left[:, 0] >= 0, 1, -1)
    for _ in range(max_iter):
        mu = s @ V
        new = np.where(V @ mu >= 0, 1, -1)
        if np.array_equal(new, s):
            break
        s = new
    if np.count_nonzero(s < 0) > len(s) / 2:
        s = -s
    mu = s @ V
    score = s * (V @ mu) / np.sqrt(np.sum(V**2, axis=1) * (mu @ mu))
    return s, score
