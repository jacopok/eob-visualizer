import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from functools import partial

def get_j_grouping(data_array):
    """Return a dictionary in the form {j: list of indices},
    where the indices are those of all the elements with approximately 
    that value of j.
    """

    sorting_indices = np.argsort(data_array)

    threshold_indices = [0]

    for i, j in enumerate(data_array[sorting_indices]):
        if not np.isclose(j, data_array[sorting_indices][threshold_indices[-1]]):
            threshold_indices.append(i)
    threshold_indices.append(len(data_array)-1)
    
    j_grouping = {}
    for ti_previous, ti_next in zip(threshold_indices, threshold_indices[1:]):
        j_grouping[data_array[sorting_indices][ti_previous]] = []
        for ind in range(ti_previous, ti_next):
            j_grouping[data_array[sorting_indices][ti_previous]].append(
                sorting_indices[ind]
            )
    return j_grouping

def get_limits(E, N, j_grouping):

    lower_limits = {}
    upper_limits = {}

    for j, indices in j_grouping.items():
        sort_E = np.argsort(E[indices])[::-1]
        
        for index, n_encounters in enumerate(N[indices][sort_E]):
            if n_encounters == 2:
                if j > 5 and E[indices][sort_E][index] < 1.01:
                    # violentissimo, c'è un punto in cui per carenza di
                    # statistica non si è esplorata la regione N=2 ad alta E
                    break
                lower_limits[j] = E[indices][sort_E][index]
                upper_limits[j] = E[indices][sort_E][index-1]
                break
    
    return lower_limits, upper_limits

def function(j, alpha, beta, gamma):
    return j**2*alpha + j*beta + gamma

def log_likelihood(x, lower, upper):
    ll = 0
    for j, lim in lower.items():
        E_est = function(j, *x)
        if E_est < lim:
            ll += (E_est - lim)**2

    for j, lim in upper.items():
        E_est = function(j, *x)
        if E_est > lim:
            ll += (E_est - lim)**2
    
    return ll

def fit_12_separation(fname):
    data = np.load(fname, allow_pickle=True)
    
    j_grouping = get_j_grouping(data['j'])

    E = data['E']
    N = data['N']
    lower_limits, upper_limits = get_limits(E, N, j_grouping)

    llike = partial(log_likelihood, lower=lower_limits, upper=upper_limits)
    res = minimize(llike, x0=(0, 1, 0))

    js = np.array(list(lower_limits.keys()))
    lower = np.array(list(lower_limits.values()))
    upper = np.array(list(upper_limits.values()))
    estimates = np.array([function(j, *res.x) for j in js])

    # plt.plot(js, lower)
    # plt.plot(js, estimates)
    # plt.plot(js, upper)

    plt.title(f'q={data["q"]}, chi1={data["chi1"]}, chi2={data["chi2"]}')
    plt.plot(js, lower-estimates)
    plt.plot(js, upper-estimates)
    plt.xlabel('j')
    plt.ylabel('$\\Delta E = \\alpha j^2 + \\beta j + \\gamma - E$')
    plt.ylim(-2e-4, 2e-4)
    plt.show()

    print(res.fun)
    print(res.x)
    return res


if __name__ == '__main__':
    fnames = [
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q001_A-050_B-050.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q002_A-050_B-050.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q004_A-050_B-050.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q008_A-050_B-050.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q001_A-025_B-025.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q002_A-025_B-025.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q004_A-025_B-025.npz',
        'span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q008_A-025_B-025.npz',
    ]
    for fname in fnames:
        fit_12_separation(fname)
