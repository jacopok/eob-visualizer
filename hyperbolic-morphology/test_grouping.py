from fit_n_encounters import get_j_grouping
import numpy as np

def test_grouping():
    data = np.load('span_dyn_data_spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28/q001_A-050_B-050.npz')

    j_grouping = get_j_grouping(data['j'])

    for key, indices in j_grouping.items():
        for ind in indices:
            assert np.isclose(key, data['j'][ind])
