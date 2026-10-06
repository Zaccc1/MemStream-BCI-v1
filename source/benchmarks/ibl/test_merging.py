import numpy as np

from merging import check_CCG, compute_CCG


def test_compute_ccg_handles_empty_and_short_spike_trains():
    K, T = compute_CCG([], [])
    assert K.shape == (1001,)
    assert np.all(K == 0)
    assert T == 0.0

    K, T = compute_CCG([0.1], [0.2, 0.3])
    assert K.shape == (1001,)
    assert np.all(K == 0)
    assert T == 0.0


def test_check_ccg_handles_empty_and_short_spike_trains():
    assert check_CCG([]) == (False, False, 1.0)
    assert check_CCG([0.1], [0.2, 0.3]) == (False, False, 1.0)


def test_compute_ccg_still_counts_valid_spike_trains():
    st1 = np.array([0.0, 0.01, 0.02], dtype=np.float64)
    st2 = np.array([0.0, 0.01, 0.02], dtype=np.float64)

    K, T = compute_CCG(st1, st2, tbin=0.001, nbins=10)

    assert K.shape == (21,)
    assert T > 0
    assert K.sum() > 0
