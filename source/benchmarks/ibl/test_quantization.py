import numpy as np

import config
from preprocessing import (
    clear_quantization_fixed_ranges,
    dequantize_uint8_codes,
    dequantize_signed_codes,
    FIRFilter,
    get_quantization_summary,
    prepare_quantized_weights,
    quantize_uint8_codes,
    quantize_weights,
    quantize_signed_codes,
    quantized_vmm,
    reset_quantization_diagnostics,
    set_quantization_fixed_ranges,
    start_quantization_range_collection,
    stop_quantization_range_collection,
    quantize_signal,
    WhiteningCovarianceAccumulator,
)
from template_matching import compute_score_matrix
from spike_clustering import _apply_lda
from spatial_aggregation import SpatialAggregator


def test_signed_code_ranges_for_configured_bit_widths():
    x = np.linspace(-2.0, 2.0, 17, dtype=np.float32)

    q8 = quantize_signed_codes(x, 8)
    assert q8.dtype == np.int8
    assert q8.min() >= -127
    assert q8.max() <= 127

    q6 = quantize_signed_codes(x, 6)
    assert q6.dtype == np.int8
    assert q6.min() >= -31
    assert q6.max() <= 31

    q4 = quantize_signed_codes(x, 4)
    assert q4.dtype == np.int8
    assert q4.min() >= -7
    assert q4.max() <= 7


def test_zero_array_quantization_is_finite():
    x = np.zeros((3, 5), dtype=np.float32)
    q, scale = quantize_signed_codes(x, 8, return_scale=True)
    y = dequantize_signed_codes(q, scale)
    assert scale == 0.0
    assert np.all(q == 0)
    assert np.all(np.isfinite(y))
    assert np.all(y == 0)


def test_quantized_vmm_int_accum_records_accumulator_stats():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "int_accum"
        config.QUANT_DIAGNOSTICS = True
        reset_quantization_diagnostics()

        x = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)
        w = np.array([[0.25, 0.5, -0.75]], dtype=np.float32)
        y = quantized_vmm(x, w, 8, 8, op_name="TEST")

        assert y.shape == (1, 1)
        assert np.all(np.isfinite(y))
        summary = get_quantization_summary()
        assert "TEST.output" in summary
        assert summary["TEST.output"]["acc_min"] <= summary["TEST.output"]["acc_max"]
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        reset_quantization_diagnostics()


def test_prepared_quantized_weights_match_uncached_vmm():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()

        x = np.linspace(-1.5, 2.5, 24, dtype=np.float32).reshape(6, 4)
        w = np.linspace(-0.75, 0.5, 12, dtype=np.float32).reshape(3, 4)
        y_plain = quantized_vmm(x, w, 8, 6, op_name="CACHE")
        y_cached = quantized_vmm(
            x, prepare_quantized_weights(w, 6), 8, 6,
            op_name="CACHE")

        np.testing.assert_array_equal(y_cached, y_plain)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_match_fast_int_accum_matches_explicit_window_vmm():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()
        set_quantization_fixed_ranges({
            "MATCH.input": {"min": -2.0, "max": 2.0},
            "MATCH_REF.input": {"min": -2.0, "max": 2.0},
            "MATCH.adc": {"min": -3.0, "max": 3.0},
            "MATCH_REF.adc": {"min": -3.0, "max": 3.0},
        })

        x = np.linspace(-1.5, 1.75, 60, dtype=np.float32).reshape(20, 3)
        wtemp = np.array([
            [0.2, -0.1, 0.4, 0.05, -0.2],
            [-0.3, 0.25, 0.1, -0.05, 0.15],
        ], dtype=np.float32)

        b_fast, pos_fast = compute_score_matrix(
            x, wtemp, stride=1, precision_bits=6)
        b_cached, pos_cached = compute_score_matrix(
            x, wtemp, stride=1, precision_bits=6)

        templates = quantize_weights(wtemp, 6)
        windows = np.lib.stride_tricks.sliding_window_view(
            x, wtemp.shape[1], axis=0)
        x_win = windows.reshape(windows.shape[0] * x.shape[1], wtemp.shape[1])
        b_ref = quantized_vmm(
            x_win, prepare_quantized_weights(templates, 6),
            config.PRECISION_DAC, 6, op_name="MATCH_REF")
        b_ref = b_ref.reshape(windows.shape[0], x.shape[1], wtemp.shape[0])
        b_ref = quantize_signal(
            b_ref, config.PRECISION_ADC, op_name="MATCH_REF.adc")

        np.testing.assert_array_equal(pos_fast, np.arange(2, 18))
        np.testing.assert_array_equal(pos_cached, pos_fast)
        np.testing.assert_allclose(b_fast, b_ref, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(b_cached, b_fast, rtol=0.0, atol=0.0)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_lda_weight_cache_matches_explicit_prepared_vmm():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()
        set_quantization_fixed_ranges({
            "LDA.input": {"min": -2.0, "max": 2.0},
            "LDA_REF.input": {"min": -2.0, "max": 2.0},
            "LDA.adc": {"min": -4.0, "max": 4.0},
            "LDA_REF.adc": {"min": -4.0, "max": 4.0},
        })

        features = np.linspace(-1.2, 1.4, 24, dtype=np.float32).reshape(6, 4)
        mean = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
        scalings = np.array([
            [0.3, -0.2],
            [-0.1, 0.4],
            [0.2, 0.1],
            [0.05, -0.35],
        ], dtype=np.float32)

        y_cached = _apply_lda(features, mean, scalings, precision_bits=8)
        centered = features - mean
        y_ref = quantized_vmm(
            centered, prepare_quantized_weights(scalings.T, 8),
            config.PRECISION_DAC, 8, op_name="LDA_REF")
        y_ref = quantize_signal(y_ref, config.PRECISION_ADC,
                                op_name="LDA_REF.adc")
        y_ref = quantize_weights(y_ref, 8)

        np.testing.assert_allclose(y_cached, y_ref, rtol=0.0, atol=1e-6)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_sa_sparse_batched_int_accum_matches_candidate_loop():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()
        set_quantization_fixed_ranges({
            "SA.sparse.input": {"min": -3.0, "max": 3.0},
            "SA_LOOP.input": {"min": -3.0, "max": 3.0},
            "SA.sparse.adc": {"min": -4.0, "max": 4.0},
            "SA_LOOP.adc": {"min": -4.0, "max": 4.0},
        })

        positions = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [40.0, 0.0],
            [60.0, 0.0],
        ], dtype=np.float32)
        sa = SpatialAggregator(
            positions, n_nearest=3, n_scales=2, precision_bits=8)
        B = np.linspace(-2.4, 2.8, 7 * 4 * config.MATCH_N_TEMPLATES,
                        dtype=np.float32).reshape(
                            7, 4, config.MATCH_N_TEMPLATES)
        candidates = np.array([
            [1, 0],
            [2, 1],
            [4, 1],
            [5, 3],
        ], dtype=np.int32)

        got = sa.sparse_aggregate(B, candidates)

        scores = np.zeros(len(candidates), dtype=np.float32)
        templates = np.zeros(len(candidates), dtype=np.int32)
        scales = np.zeros(len(candidates), dtype=np.int32)
        features = np.zeros(
            (len(candidates), sa.n_scales * config.MATCH_N_TEMPLATES),
            dtype=np.float32)
        for i, (t, ch) in enumerate(candidates):
            b = B[t, sa.neighbour_idx[ch], :]
            A = quantized_vmm(
                b.T, sa.G_prepared[ch], config.PRECISION_DAC,
                sa.precision_bits,
                op_name="SA_LOOP").T
            A = quantize_signal(A, config.PRECISION_ADC,
                                op_name="SA_LOOP.adc")
            features[i] = A.ravel()
            abs_A = np.abs(A)
            flat_idx = np.argmax(abs_A)
            s_idx, k_idx = np.unravel_index(flat_idx, A.shape)
            scores[i] = abs_A[s_idx, k_idx]
            templates[i] = k_idx
            scales[i] = s_idx

        np.testing.assert_allclose(got[0], scores, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(got[1], templates)
        np.testing.assert_array_equal(got[2], scales)
        np.testing.assert_allclose(got[3], features, rtol=0.0, atol=1e-6)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_fir_int_accum_fast_path_matches_window_vmm_reference():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()

        coeffs = np.array([0.2, -0.1, 0.4, 0.05, -0.2], dtype=np.float32)
        chunk = np.linspace(-2.0, 3.0, 32, dtype=np.float32).reshape(16, 2)

        filt = FIRFilter(coeffs, precision_bits=8)
        filt.reset(chunk.shape[1])
        y_fast = filt.process_chunk(chunk)

        extended = np.vstack([
            np.zeros((len(coeffs) - 1, chunk.shape[1]), dtype=np.float32),
            chunk,
        ])
        windows = np.lib.stride_tricks.sliding_window_view(
            extended, len(coeffs), axis=0)[:chunk.shape[0]]
        x_win = windows.reshape(chunk.shape[0] * chunk.shape[1], len(coeffs))
        w_fir = filt.coeffs[::-1][np.newaxis, :]
        y_ref = quantized_vmm(
            x_win, w_fir, config.PRECISION_DAC, filt.precision_bits,
            op_name="REF_FIR").reshape(chunk.shape)
        y_ref = quantize_signal(y_ref, config.PRECISION_ADC,
                                op_name="REF_FIR.adc")

        np.testing.assert_allclose(y_fast, y_ref, rtol=0.0, atol=1e-6)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_whitening_covariance_accumulator_matches_numpy_cov():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(37, 4)).astype(np.float32)
    groups = [
        {
            "input_channels": np.array([0, 2, 3]),
            "output_channels": np.array([0, 2, 3]),
        },
        {
            "input_channels": np.array([1, 3]),
            "output_channels": np.array([1, 3]),
        },
    ]

    acc = WhiteningCovarianceAccumulator(groups)
    acc.update(data[:11])
    acc.update(data[11:])
    covs = acc.covariances()

    np.testing.assert_allclose(
        covs[0], np.cov(data[:, [0, 2, 3]], rowvar=False).astype(np.float32),
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        covs[1], np.cov(data[:, [1, 3]], rowvar=False).astype(np.float32),
        rtol=1e-6, atol=1e-6)


def test_uint8_activation_quantization_uses_unsigned_codes():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    q, scale, zero_point = quantize_uint8_codes(
        x, 8, value_range=(-1.0, 1.0), return_params=True)
    y = dequantize_uint8_codes(q, scale, zero_point)

    assert q.dtype == np.uint8
    assert q.min() >= 0
    assert q.max() <= 255
    assert 0 <= zero_point <= 255
    assert np.all(np.isfinite(y))
    assert abs(y[1]) < 1e-2


def test_uint8_positive_value_range_does_not_saturate_half_range():
    x = np.linspace(2.0, 5.0, 7, dtype=np.float32)
    q, scale, zero_point = quantize_uint8_codes(
        x, 8, value_range=(2.0, 5.0), return_params=True)
    y = dequantize_uint8_codes(q, scale, zero_point)

    assert q.dtype == np.uint8
    assert zero_point < 0
    assert q[0] == 0
    assert q[-1] == 255
    assert float(np.max(np.abs(y - x))) < 0.01


def test_uint8_negative_value_range_does_not_saturate_half_range():
    x = np.linspace(-5.0, -2.0, 7, dtype=np.float32)
    q, scale, zero_point = quantize_uint8_codes(
        x, 8, value_range=(-5.0, -2.0), return_params=True)
    y = dequantize_uint8_codes(q, scale, zero_point)

    assert q.dtype == np.uint8
    assert zero_point > 255
    assert q[0] == 0
    assert q[-1] == 255
    assert float(np.max(np.abs(y - x))) < 0.01


def test_quantized_vmm_uint8_records_zero_point_and_fixed_range():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = True
        reset_quantization_diagnostics()
        set_quantization_fixed_ranges({
            "TEST.input": {
                "min": -1.0,
                "max": 1.0,
            }
        })

        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        w = np.array([[0.25, 0.5, -0.75]], dtype=np.float32)
        y = quantized_vmm(x, w, 8, 8, op_name="TEST")

        assert y.shape == (1, 1)
        assert np.all(np.isfinite(y))
        summary = get_quantization_summary()
        assert summary["TEST.input"]["q_expected_min"] == 0
        assert summary["TEST.input"]["q_expected_max"] == 255
        assert "zero_point_min" in summary["TEST.input"]
        assert summary["TEST.input"]["range_min_min"] == -1.0
        assert summary["TEST.input"]["range_max_max"] == 1.0
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_range_collection_freezes_percentile_range():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    old_overrides = config.QUANT_RANGE_STAGE_OVERRIDES
    old_override_path = config.QUANT_RANGE_OVERRIDE_PATH
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        config.QUANT_RANGE_STAGE_OVERRIDES = {}
        config.QUANT_RANGE_OVERRIDE_PATH = ""
        reset_quantization_diagnostics()
        start_quantization_range_collection(
            percentile=99.0,
            sample_limit=10_000,
            sample_per_update=10_000,
            symmetric=False,
        )

        x = np.linspace(-10.0, 10.0, 1000, dtype=np.float32)
        _ = quantize_signal(x, 8, op_name="RANGE.adc")
        payload = stop_quantization_range_collection(output_dir=None)

        spec = payload["ranges"]["RANGE.adc"]
        assert spec["observed_min"] == -10.0
        assert spec["observed_max"] == 10.0
        assert -10.0 < spec["min"] < -9.0
        assert 9.0 < spec["max"] < 10.0
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        config.QUANT_RANGE_STAGE_OVERRIDES = old_overrides
        config.QUANT_RANGE_OVERRIDE_PATH = old_override_path
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_range_collection_allows_manual_stage_value_range():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    old_overrides = config.QUANT_RANGE_STAGE_OVERRIDES
    old_override_path = config.QUANT_RANGE_OVERRIDE_PATH
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        config.QUANT_RANGE_OVERRIDE_PATH = ""
        config.QUANT_RANGE_STAGE_OVERRIDES = {
            "RANGE.*": {
                "min": -2.0,
                "max": 3.0,
            }
        }
        start_quantization_range_collection(
            percentile=99.0,
            sample_limit=10_000,
            sample_per_update=10_000,
            symmetric=False,
        )

        x = np.linspace(-10.0, 10.0, 1000, dtype=np.float32)
        _ = quantize_signal(x, 8, op_name="RANGE.adc")
        payload = stop_quantization_range_collection(output_dir=None)

        spec = payload["ranges"]["RANGE.adc"]
        assert spec["min"] == -2.0
        assert spec["max"] == 3.0
        assert spec["range_source"] == "manual"
        assert spec["percentile"] is None
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        config.QUANT_RANGE_STAGE_OVERRIDES = old_overrides
        config.QUANT_RANGE_OVERRIDE_PATH = old_override_path
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_range_collection_allows_stage_percentile_override():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    old_overrides = config.QUANT_RANGE_STAGE_OVERRIDES
    old_override_path = config.QUANT_RANGE_OVERRIDE_PATH
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        config.QUANT_RANGE_OVERRIDE_PATH = ""
        config.QUANT_RANGE_STAGE_OVERRIDES = {
            "RANGE.adc": {
                "percentile": 80.0,
            }
        }
        start_quantization_range_collection(
            percentile=99.0,
            sample_limit=10_000,
            sample_per_update=10_000,
            symmetric=False,
        )

        x = np.linspace(-10.0, 10.0, 1000, dtype=np.float32)
        _ = quantize_signal(x, 8, op_name="RANGE.adc")
        payload = stop_quantization_range_collection(output_dir=None)

        spec = payload["ranges"]["RANGE.adc"]
        assert spec["range_source"] == "percentile"
        assert spec["percentile"] == 80.0
        assert -8.5 < spec["min"] < -7.5
        assert 7.5 < spec["max"] < 8.5
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        config.QUANT_RANGE_STAGE_OVERRIDES = old_overrides
        config.QUANT_RANGE_OVERRIDE_PATH = old_override_path
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_range_collection_ties_adc_to_next_input_range():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    old_overrides = config.QUANT_RANGE_STAGE_OVERRIDES
    old_override_path = config.QUANT_RANGE_OVERRIDE_PATH
    old_tied_pairs = config.QUANT_RANGE_TIED_STAGE_PAIRS
    try:
        config.QUANTIZATION_BACKEND = "uint8_int_accum"
        config.QUANT_DIAGNOSTICS = False
        config.QUANT_RANGE_STAGE_OVERRIDES = {}
        config.QUANT_RANGE_OVERRIDE_PATH = ""
        config.QUANT_RANGE_TIED_STAGE_PAIRS = [("SRC.adc", "DST.input")]
        start_quantization_range_collection(
            percentile=99.0,
            sample_limit=10_000,
            sample_per_update=10_000,
            symmetric=False,
        )

        _ = quantize_signal(
            np.linspace(-10.0, 10.0, 1000, dtype=np.float32),
            8, op_name="SRC.adc")
        _ = quantize_signal(
            np.linspace(-1.0, 1.0, 1000, dtype=np.float32),
            8, op_name="DST.input")
        payload = stop_quantization_range_collection(output_dir=None)

        src = payload["ranges"]["SRC.adc"]
        dst = payload["ranges"]["DST.input"]
        assert dst["min"] == src["min"]
        assert dst["max"] == src["max"]
        assert dst["range_source"] == "tied_to:SRC.adc"
        assert dst["tied_from"] == "SRC.adc"
        assert dst["observed_min"] == -1.0
        assert dst["observed_max"] == 1.0
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
        config.QUANT_RANGE_STAGE_OVERRIDES = old_overrides
        config.QUANT_RANGE_OVERRIDE_PATH = old_override_path
        config.QUANT_RANGE_TIED_STAGE_PAIRS = old_tied_pairs
        clear_quantization_fixed_ranges()
        reset_quantization_diagnostics()


def test_sparse_nms_does_not_assume_kept_events_are_time_sorted():
    positions = np.column_stack([
        np.zeros(4, dtype=np.float32),
        np.arange(4, dtype=np.float32) * 20.0,
    ])
    sa = SpatialAggregator(
        positions, n_nearest=4, n_scales=1,
        sigma_um=[40.0], precision_bits=4)

    # Scores force accept order: t=100, then t=1000, then low-score t=1010,
    # then t=0.  Before the fix, reversed(kept_events) could hit t=0 first
    # and break before checking the nearby high-score t=1000 event.
    candidates = np.array([
        [100, 0],
        [1000, 0],
        [1010, 1],
        [0, 0],
    ], dtype=np.int32)
    scores = np.array([10.0, 9.0, 1.0, 2.0], dtype=np.float32)
    templates = np.zeros(len(scores), dtype=np.int32)
    scales = np.zeros(len(scores), dtype=np.int32)

    events = sa.spatiotemporal_nms(
        candidates, scores, templates, scales,
        nms_samples=20, nms_channels=4)

    assert 1010 not in events['times']
    assert set(events['times'].tolist()) == {0, 100, 1000}


def test_sparse_aggregate_uses_runtime_template_count():
    old_backend = config.QUANTIZATION_BACKEND
    old_diag = config.QUANT_DIAGNOSTICS
    try:
        config.QUANTIZATION_BACKEND = "fake"
        config.QUANT_DIAGNOSTICS = False

        positions = np.column_stack([
            np.zeros(4, dtype=np.float32),
            np.arange(4, dtype=np.float32) * 20.0,
        ])
        sa = SpatialAggregator(
            positions, n_nearest=4, n_scales=2,
            sigma_um=[20.0, 40.0], precision_bits=4)

        n_templates = int(config.MATCH_N_TEMPLATES) + 1
        B = np.linspace(
            -1.0, 1.0, 8 * 4 * n_templates,
            dtype=np.float32).reshape(8, 4, n_templates)
        candidates = np.array([[3, 0], [5, 2]], dtype=np.int32)

        scores, best_templates, best_scales, features = sa.sparse_aggregate(
            B, candidates)

        assert scores.shape == (2,)
        assert best_templates.max() < n_templates
        assert best_scales.max() < sa.n_scales
        assert features.shape == (2, sa.n_scales * n_templates)
    finally:
        config.QUANTIZATION_BACKEND = old_backend
        config.QUANT_DIAGNOSTICS = old_diag
