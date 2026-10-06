"""
Preprocessing Module
=====================
CAR (digital) -> FIR high-pass (crossbar) -> Local ZCA whitening (crossbar)

Signal quantization (DAC/ADC) is applied at every crossbar boundary.
Weight quantization is applied independently per crossbar stage.
"""

from fnmatch import fnmatch

import numpy as np
from scipy.signal import firwin, freqz
from scipy.spatial.distance import cdist

import config


# ==============================================================
# Quantization
# ==============================================================

_QUANT_DIAG_STATS = {}
_QUANT_DIAG_PRINT_COUNTS = {}
_QUANT_RANGE_COLLECTOR = None
_QUANT_FIXED_RANGES = {}


def quantization_backend():
    """Return the configured quantization backend name."""
    return str(getattr(config, "QUANTIZATION_BACKEND", "fake")).lower()


def using_int_accum_quantization():
    """True when crossbar operations should use integer-code accumulation."""
    return quantization_backend() in ("int_accum", "uint8_int_accum")


def using_uint8_input_quantization():
    """True when VMM input activations should be uint8 affine codes."""
    return quantization_backend() == "uint8_int_accum"


def _quant_diag_enabled():
    return bool(getattr(config, "QUANT_DIAGNOSTICS", False))


def _quant_diag_limit():
    return int(getattr(config, "QUANT_DIAG_MAX_PRINTS_PER_OP", 3))


def _signed_qrange(n_bits):
    if n_bits < 2:
        raise ValueError("signed quantization requires at least 2 bits")
    qmax = (1 << (n_bits - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def _signed_qdtype(n_bits):
    if n_bits <= 8:
        return np.int8
    if n_bits <= 16:
        return np.int16
    return np.int32


def _unsigned_qrange(n_bits):
    if n_bits < 1:
        raise ValueError("unsigned quantization requires at least 1 bit")
    return 0, (1 << int(n_bits)) - 1


def _unsigned_qdtype(n_bits):
    if n_bits <= 8:
        return np.uint8
    if n_bits <= 16:
        return np.uint16
    return np.uint32


def _quantize_signed_codes_with_stats(x, n_bits, scale=None):
    x = np.asarray(x, dtype=np.float32)
    if n_bits >= 32:
        stats = {
            "qmin": None, "qmax": None, "clip_fraction": 0.0,
            "scale": None,
        }
        return x.astype(np.float32), None, stats

    qmin, qmax = _signed_qrange(int(n_bits))
    dtype = _signed_qdtype(int(n_bits))
    if scale is None:
        finite = x[np.isfinite(x)]
        abs_max = float(np.max(np.abs(finite))) if finite.size else 0.0
        scale = 0.0 if abs_max < 1e-12 else abs_max / qmax
    else:
        scale = float(scale)

    if x.size == 0 or abs(scale) < 1e-12:
        q = np.zeros_like(x, dtype=dtype)
        clip_fraction = 0.0
    else:
        raw = np.round(x / scale)
        raw = np.nan_to_num(raw, nan=0.0,
                            posinf=float(qmax + 1),
                            neginf=float(qmin - 1))
        clipped = (raw < qmin) | (raw > qmax)
        clip_fraction = float(np.mean(clipped)) if clipped.size else 0.0
        q = np.clip(raw, qmin, qmax).astype(dtype)

    stats = {
        "qmin": qmin,
        "qmax": qmax,
        "clip_fraction": clip_fraction,
        "scale": scale,
    }
    return q, scale, stats


def quantize_signed_codes(x, n_bits, scale=None, return_scale=False):
    """Quantize values into signed integer codes.

    Uses a symmetric code range, e.g. 8-bit -> [-127, 127],
    6-bit -> [-31, 31], 4-bit -> [-7, 7].
    """
    q, used_scale, _ = _quantize_signed_codes_with_stats(x, n_bits, scale)
    if return_scale:
        return q, used_scale
    return q


def dequantize_signed_codes(q, scale):
    """Convert signed integer codes back to float32 values."""
    q = np.asarray(q)
    if scale is None:
        return q.astype(np.float32)
    if abs(float(scale)) < 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return (q.astype(np.float32) * float(scale)).astype(np.float32)


class PreparedQuantizedWeights:
    """Precomputed signed weight codes for repeated integer VMM calls."""

    def __init__(self, weights, n_bits):
        self.values = np.array(weights, dtype=np.float32, copy=True)
        self.n_bits = int(n_bits)
        self.q, self.scale, self.stats = _quantize_signed_codes_with_stats(
            self.values, self.n_bits)
        self.q_int32_t = self.q.astype(np.int32).T


def prepare_quantized_weights(weights, n_bits):
    """Precompute signed weight codes without changing quantized VMM math."""
    return PreparedQuantizedWeights(weights, n_bits)


def _as_float_pair(spec):
    if isinstance(spec, dict):
        if "range" in spec:
            spec = spec["range"]
        elif "min" in spec and "max" in spec:
            spec = (spec["min"], spec["max"])
        else:
            return None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo = float(spec[0])
        hi = float(spec[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
    return None


def _load_quant_range_override_mapping():
    import json
    import os

    merged = {}
    path = str(getattr(config, "QUANT_RANGE_OVERRIDE_PATH", "") or "").strip()
    if path:
        try:
            with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload = payload.get("overrides", payload.get("ranges", payload))
                if isinstance(payload, dict):
                    merged.update(payload)
        except Exception as exc:
            print(f"[QUANT-CALI] Override file skipped ({path}): {exc}")

    inline = getattr(config, "QUANT_RANGE_STAGE_OVERRIDES", {}) or {}
    if isinstance(inline, str):
        text = inline.strip()
        if text:
            try:
                inline = json.loads(text)
            except Exception as exc:
                print(f"[QUANT-CALI] Inline stage overrides skipped: {exc}")
                inline = {}
        else:
            inline = {}
    if isinstance(inline, dict):
        merged.update(inline)
    return merged


def _stage_override_for(name, overrides=None):
    overrides = _load_quant_range_override_mapping() if overrides is None else overrides
    if not isinstance(overrides, dict):
        return None
    if name in overrides:
        return overrides[name]
    for pattern, spec in overrides.items():
        if isinstance(pattern, str) and any(ch in pattern for ch in "*?[]"):
            if fnmatch(name, pattern):
                return spec
    return None


class QuantizationRangeCollector:
    """Reservoir-style sampler for calibration-time quantization ranges."""

    def __init__(self, percentile=None, sample_limit=None,
                 sample_per_update=None, symmetric=None, seed=None):
        self.percentile = float(
            percentile if percentile is not None
            else getattr(config, "QUANT_RANGE_PERCENTILE", 99.0))
        self.sample_limit = int(
            sample_limit if sample_limit is not None
            else getattr(config, "QUANT_RANGE_SAMPLE_LIMIT", 200_000))
        self.sample_per_update = int(
            sample_per_update if sample_per_update is not None
            else getattr(config, "QUANT_RANGE_SAMPLE_PER_UPDATE", 4096))
        self.symmetric = bool(
            symmetric if symmetric is not None
            else getattr(config, "QUANT_RANGE_SYMMETRIC", False))
        self.rng = np.random.default_rng(
            seed if seed is not None else getattr(config, "RANDOM_SEED", 42))
        self.records = {}
        self.stage_overrides = _load_quant_range_override_mapping()

    def record(self, name, values):
        if not name:
            return
        arr = np.asarray(values, dtype=np.float32).ravel()
        if arr.size == 0:
            return
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        rec = self.records.setdefault(name, {
            "count": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "samples": np.zeros(0, dtype=np.float32),
        })
        rec["count"] += int(finite.size)
        rec["min"] = min(float(rec["min"]), float(np.min(finite)))
        rec["max"] = max(float(rec["max"]), float(np.max(finite)))

        k = min(finite.size, self.sample_per_update)
        if finite.size > k:
            vals = finite[self.rng.choice(finite.size, size=k, replace=False)]
        else:
            vals = finite

        stored = rec["samples"]
        limit = self.sample_limit
        if stored.size < limit:
            room = limit - stored.size
            take = min(room, vals.size)
            if take:
                rec["samples"] = np.concatenate([stored, vals[:take]])
            vals = vals[take:]
            stored = rec["samples"]

        if vals.size and stored.size:
            seen = max(rec["count"], 1)
            replace_n = int(np.ceil(limit * vals.size / seen))
            replace_n = min(replace_n, vals.size, stored.size)
            if replace_n > 0:
                src = self.rng.choice(vals.size, size=replace_n, replace=False)
                dst = self.rng.choice(stored.size, size=replace_n, replace=False)
                stored[dst] = vals[src]

    def _range_from_samples(self, samples, percentile, symmetric):
        tail = max((100.0 - float(percentile)) / 2.0, 0.0)
        if symmetric:
            abs_range = float(np.percentile(np.abs(samples), percentile))
            return -abs_range, abs_range
        lo = float(np.percentile(samples, tail))
        hi = float(np.percentile(samples, 100.0 - tail))
        return lo, hi

    def compute_ranges(self):
        ranges = {}
        for name, rec in sorted(self.records.items()):
            samples = np.asarray(rec["samples"], dtype=np.float32)
            if samples.size == 0:
                continue
            override = _stage_override_for(name, self.stage_overrides)
            manual_range = _as_float_pair(override)
            if manual_range is not None:
                lo, hi = manual_range
                range_source = "manual"
                percentile = None
                symmetric = None
            else:
                override = override if isinstance(override, dict) else {}
                percentile = float(override.get("percentile", self.percentile))
                symmetric = bool(override.get("symmetric", self.symmetric))
                lo, hi = self._range_from_samples(samples, percentile, symmetric)
                range_source = "percentile"
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                center = float(np.mean(samples)) if samples.size else 0.0
                eps = max(abs(center) * 1e-6, 1e-6)
                lo, hi = center - eps, center + eps
                range_source = f"{range_source}_fallback"
            ranges[name] = {
                "min": lo,
                "max": hi,
                "observed_min": float(rec["min"]),
                "observed_max": float(rec["max"]),
                "count": int(rec["count"]),
                "n_samples": int(samples.size),
                "percentile": percentile,
                "symmetric": symmetric,
                "range_source": range_source,
            }
        for name, override in sorted(self.stage_overrides.items()):
            if name in ranges or any(ch in str(name) for ch in "*?[]"):
                continue
            manual_range = _as_float_pair(override)
            if manual_range is None:
                continue
            lo, hi = manual_range
            ranges[name] = {
                "min": lo,
                "max": hi,
                "observed_min": None,
                "observed_max": None,
                "count": 0,
                "n_samples": 0,
                "percentile": None,
                "symmetric": None,
                "range_source": "manual_unobserved",
            }
        self._tie_stage_boundary_ranges(ranges)
        return ranges

    def _tie_stage_boundary_ranges(self, ranges):
        pairs = getattr(config, "QUANT_RANGE_TIED_STAGE_PAIRS", ()) or ()
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            src, dst = str(pair[0]), str(pair[1])
            if src not in ranges:
                continue
            src_spec = ranges[src]
            if "min" not in src_spec or "max" not in src_spec:
                continue
            dst_existing = dict(ranges.get(dst, {}))
            tied = dict(dst_existing)
            tied["min"] = float(src_spec["min"])
            tied["max"] = float(src_spec["max"])
            tied.setdefault("observed_min", None)
            tied.setdefault("observed_max", None)
            tied.setdefault("count", 0)
            tied.setdefault("n_samples", 0)
            tied["percentile"] = src_spec.get("percentile")
            tied["symmetric"] = src_spec.get("symmetric")
            tied["range_source"] = f"tied_to:{src}"
            tied["tied_from"] = src
            tied["tied_source_min"] = float(src_spec["min"])
            tied["tied_source_max"] = float(src_spec["max"])
            ranges[dst] = tied
            print(
                f"[QUANT-CALI] Tied range: {dst} <- {src} "
                f"[{tied['min']:.6g}, {tied['max']:.6g}]")


def set_quantization_fixed_ranges(ranges):
    """Install fixed value ranges used by uint8 activation quantization."""
    global _QUANT_FIXED_RANGES
    _QUANT_FIXED_RANGES = {}
    for name, spec in (ranges or {}).items():
        if "min" not in spec or "max" not in spec:
            continue
        lo = float(spec["min"])
        hi = float(spec["max"])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            _QUANT_FIXED_RANGES[name] = dict(spec)


def get_quantization_fixed_ranges():
    """Return active fixed quantization ranges."""
    return {k: dict(v) for k, v in _QUANT_FIXED_RANGES.items()}


def clear_quantization_fixed_ranges():
    """Remove active fixed quantization ranges."""
    _QUANT_FIXED_RANGES.clear()


def start_quantization_range_collection(percentile=None, sample_limit=None,
                                        sample_per_update=None,
                                        symmetric=None):
    """Begin collecting calibration-time value ranges."""
    global _QUANT_RANGE_COLLECTOR
    _QUANT_RANGE_COLLECTOR = QuantizationRangeCollector(
        percentile=percentile,
        sample_limit=sample_limit,
        sample_per_update=sample_per_update,
        symmetric=symmetric,
    )
    clear_quantization_fixed_ranges()
    return _QUANT_RANGE_COLLECTOR


def stop_quantization_range_collection(output_dir=None):
    """Freeze collected ranges, save optional artifacts, and return payload."""
    global _QUANT_RANGE_COLLECTOR
    collector = _QUANT_RANGE_COLLECTOR
    _QUANT_RANGE_COLLECTOR = None
    if collector is None:
        return {"ranges": get_quantization_fixed_ranges(), "artifacts": {}}
    ranges = collector.compute_ranges()
    set_quantization_fixed_ranges(ranges)
    artifacts = {}
    if output_dir:
        artifacts = save_quantization_range_artifacts(
            collector, ranges, output_dir)
    return {
        "backend": quantization_backend(),
        "percentile": collector.percentile,
        "symmetric": collector.symmetric,
        "ranges": ranges,
        "artifacts": artifacts,
    }


def _record_quant_range(name, values):
    if _QUANT_RANGE_COLLECTOR is not None and name:
        _QUANT_RANGE_COLLECTOR.record(name, values)


def _fixed_range_for(name):
    spec = _QUANT_FIXED_RANGES.get(name)
    if spec is None:
        return None
    return float(spec["min"]), float(spec["max"])


def save_quantization_range_artifacts(collector, ranges, output_dir):
    """Save calibration range JSON and histogram visualizations."""
    import json
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "backend": quantization_backend(),
        "percentile": collector.percentile,
        "symmetric": collector.symmetric,
        "stage_overrides": collector.stage_overrides,
        "ranges": ranges,
    }
    json_path = os.path.join(output_dir, "quant_ranges.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    template = {}
    for name, spec in sorted(ranges.items()):
        template[name] = {
            "min": spec["min"],
            "max": spec["max"],
            "range_source": spec.get("range_source", "percentile"),
            "percentile": spec.get("percentile"),
            "symmetric": spec.get("symmetric"),
        }
    template_path = os.path.join(output_dir, "quant_range_overrides_template.json")
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    names = [name for name in sorted(ranges)
             if collector.records.get(name, {}).get("samples", np.array([])).size]
    if not names:
        print(f"[QUANT-CALI] Range JSON -> {json_path}")
        print(f"[QUANT-CALI] Override template -> {template_path}")
        return {"json": json_path, "override_template": template_path}

    n_cols = 3
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        samples = collector.records[name]["samples"]
        spec = ranges[name]
        ax.hist(samples, bins=120, color="#4c78a8", alpha=0.78)
        ax.axvline(spec["min"], color="#d62728", lw=1.4, ls="--")
        ax.axvline(spec["max"], color="#d62728", lw=1.4, ls="--")
        ax.axvline(spec["observed_min"], color="#555555", lw=0.8, ls=":")
        ax.axvline(spec["observed_max"], color="#555555", lw=0.8, ls=":")
        ax.set_title(
            f"{name}\n"
            f"fixed=[{spec['min']:.3g}, {spec['max']:.3g}], "
            f"obs=[{spec['observed_min']:.3g}, {spec['observed_max']:.3g}], "
            f"{spec.get('range_source', 'percentile')}",
            fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.18)
    for ax in axes[len(names):]:
        ax.set_axis_off()
    fig.suptitle(
        f"Quantization Calibration Ranges "
        f"({collector.percentile:.1f}% central mass)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = os.path.join(output_dir, "quant_range_histograms.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[QUANT-CALI] Range JSON -> {json_path}")
    print(f"[QUANT-CALI] Override template -> {template_path}")
    print(f"[QUANT-CALI] Range histograms -> {png_path}")
    return {
        "json": json_path,
        "override_template": template_path,
        "histograms": png_path,
    }


def _quantize_uint8_codes_with_stats(x, n_bits, value_range=None):
    x = np.asarray(x, dtype=np.float32)
    if n_bits >= 32:
        stats = {
            "qmin": None, "qmax": None, "clip_fraction": 0.0,
            "scale": None, "zero_point": None,
            "range_min": None, "range_max": None,
        }
        return x.astype(np.float32), None, None, stats

    qmin, qmax = _unsigned_qrange(int(n_bits))
    dtype = _unsigned_qdtype(int(n_bits))

    finite = x[np.isfinite(x)]
    if value_range is None:
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        else:
            lo, hi = 0.0, 0.0
    else:
        lo, hi = float(value_range[0]), float(value_range[1])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = 0.0 if not finite.size else float(np.mean(finite))
        eps = max(abs(center) * 1e-6, 1e-6)
        lo, hi = center - eps, center + eps

    scale = (hi - lo) / max(qmax - qmin, 1)
    zero_point = int(np.round(qmin - lo / max(scale, 1e-12)))

    if x.size == 0 or abs(scale) < 1e-12:
        q = np.full_like(x, zero_point, dtype=dtype)
        clip_fraction = 0.0
    else:
        raw = np.round(x / scale + zero_point)
        raw = np.nan_to_num(raw, nan=float(zero_point),
                            posinf=float(qmax + 1),
                            neginf=float(qmin - 1))
        clipped = (raw < qmin) | (raw > qmax)
        clip_fraction = float(np.mean(clipped)) if clipped.size else 0.0
        q = np.clip(raw, qmin, qmax).astype(dtype)

    stats = {
        "qmin": qmin,
        "qmax": qmax,
        "clip_fraction": clip_fraction,
        "scale": float(scale),
        "zero_point": zero_point,
        "range_min": float(lo),
        "range_max": float(hi),
    }
    return q, float(scale), zero_point, stats


def quantize_uint8_codes(x, n_bits=8, value_range=None, return_params=False):
    """Quantize values into unsigned activation codes."""
    q, scale, zero_point, _ = _quantize_uint8_codes_with_stats(
        x, n_bits, value_range=value_range)
    if return_params:
        return q, scale, zero_point
    return q


def dequantize_uint8_codes(q, scale, zero_point):
    """Convert uint activation codes back to float32 values."""
    q = np.asarray(q)
    if scale is None or zero_point is None:
        return q.astype(np.float32)
    if abs(float(scale)) < 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return ((q.astype(np.float32) - float(zero_point)) *
            float(scale)).astype(np.float32)


def reset_quantization_diagnostics():
    """Clear per-run quantization diagnostics."""
    _QUANT_DIAG_STATS.clear()
    _QUANT_DIAG_PRINT_COUNTS.clear()


def _merge_quant_stats(name, stats):
    entry = _QUANT_DIAG_STATS.setdefault(name, {"count": 0})
    entry["count"] += 1
    for key in ("max_abs_error", "mean_abs_error", "relative_rmse",
                "clip_fraction", "rail_fraction"):
        if key in stats:
            out_key = f"{key}_max"
            entry[out_key] = max(float(stats[key]),
                                 float(entry.get(out_key, 0.0)))
    if "scale" in stats and stats["scale"] is not None:
        scale = float(stats["scale"])
        entry["scale_min"] = min(scale, float(entry.get("scale_min", scale)))
        entry["scale_max"] = max(scale, float(entry.get("scale_max", scale)))
    for key in ("zero_point", "range_min", "range_max"):
        if key in stats and stats[key] is not None:
            val = float(stats[key])
            entry[f"{key}_min"] = min(val, float(entry.get(f"{key}_min", val)))
            entry[f"{key}_max"] = max(val, float(entry.get(f"{key}_max", val)))
    for key in ("q_observed_min", "q_expected_min", "acc_min"):
        if key in stats and stats[key] is not None:
            val = float(stats[key])
            entry[key] = min(val, float(entry.get(key, val)))
    for key in ("q_observed_max", "q_expected_max", "acc_max"):
        if key in stats and stats[key] is not None:
            val = float(stats[key])
            entry[key] = max(val, float(entry.get(key, val)))


def quantization_report(name, original, quantized_or_dequantized, q=None,
                        scale=None, q_range=None, clip_fraction=None,
                        accumulator=None, zero_point=None,
                        value_range=None):
    """Record and optionally print quantization diagnostics for one stage."""
    stats = {}

    if original is not None and quantized_or_dequantized is not None:
        orig = np.asarray(original, dtype=np.float32)
        out = np.asarray(quantized_or_dequantized, dtype=np.float32)
        if orig.size and out.size and orig.shape == out.shape:
            diff = out - orig
            abs_diff = np.abs(diff)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            ref = float(np.sqrt(np.mean(orig ** 2)))
            stats["max_abs_error"] = float(np.max(abs_diff))
            stats["mean_abs_error"] = float(np.mean(abs_diff))
            stats["relative_rmse"] = rmse / max(ref, 1e-12)

    if q is not None:
        q_arr = np.asarray(q)
        if q_arr.size:
            stats["q_observed_min"] = int(np.min(q_arr))
            stats["q_observed_max"] = int(np.max(q_arr))
            if q_range is not None and q_range[0] is not None:
                qmin, qmax = q_range
                rails = (q_arr <= qmin) | (q_arr >= qmax)
                stats["rail_fraction"] = float(np.mean(rails))

    if scale is not None:
        stats["scale"] = float(scale)
    if zero_point is not None:
        stats["zero_point"] = int(zero_point)
    if value_range is not None:
        stats["range_min"] = float(value_range[0])
        stats["range_max"] = float(value_range[1])
    if q_range is not None and q_range[0] is not None:
        stats["q_expected_min"] = int(q_range[0])
        stats["q_expected_max"] = int(q_range[1])
    if clip_fraction is not None:
        stats["clip_fraction"] = float(clip_fraction)
    if accumulator is not None:
        acc = np.asarray(accumulator)
        if acc.size:
            stats["acc_min"] = int(np.min(acc))
            stats["acc_max"] = int(np.max(acc))

    _merge_quant_stats(name, stats)

    if not _quant_diag_enabled():
        return stats
    count = _QUANT_DIAG_PRINT_COUNTS.get(name, 0)
    if count >= _quant_diag_limit():
        return stats
    _QUANT_DIAG_PRINT_COUNTS[name] = count + 1

    parts = []
    if "scale" in stats:
        parts.append(f"scale={stats['scale']:.4e}")
    if "zero_point" in stats:
        parts.append(f"zp={stats['zero_point']}")
    if "range_min" in stats:
        parts.append(
            f"range=[{stats['range_min']:.4e},{stats['range_max']:.4e}]")
    if "q_observed_min" in stats and "q_expected_min" in stats:
        parts.append(
            f"q=[{stats['q_observed_min']:.0f},{stats['q_observed_max']:.0f}]"
            f"/[{stats['q_expected_min']:.0f},{stats['q_expected_max']:.0f}]")
    if "clip_fraction" in stats:
        parts.append(f"clip={stats['clip_fraction'] * 100:.4f}%")
    if "rail_fraction" in stats:
        parts.append(f"rail={stats['rail_fraction'] * 100:.4f}%")
    if "max_abs_error" in stats:
        parts.append(f"max_err={stats['max_abs_error']:.4e}")
        parts.append(f"rel_rmse={stats['relative_rmse']:.4e}")
    if "acc_min" in stats:
        parts.append(f"acc=[{stats['acc_min']:.0f},{stats['acc_max']:.0f}]")
    if parts:
        print(f"[QUANT] {name}: " + ", ".join(parts))
    return stats


def get_quantization_summary():
    """Return collected quantization diagnostics."""
    return {k: dict(v) for k, v in _QUANT_DIAG_STATS.items()}


def format_quantization_summary(max_items=12):
    """Format a compact quantization diagnostics summary."""
    if not _QUANT_DIAG_STATS:
        return []
    warn_frac = float(getattr(config, "QUANT_CLIP_WARN_FRAC", 0.01))

    def rank_item(item):
        _, st = item
        return (
            float(st.get("clip_fraction_max", 0.0)),
            float(st.get("relative_rmse_max", 0.0)),
            float(st.get("max_abs_error_max", 0.0)),
        )

    lines = []
    for name, st in sorted(_QUANT_DIAG_STATS.items(),
                           key=rank_item, reverse=True)[:max_items]:
        pieces = [f"{name}"]
        if "scale_min" in st:
            pieces.append(f"scale={st['scale_min']:.3e}..{st['scale_max']:.3e}")
        if "zero_point_min" in st:
            pieces.append(
                f"zp={st['zero_point_min']:.0f}..{st['zero_point_max']:.0f}")
        if "range_min_min" in st:
            pieces.append(
                f"range=[{st['range_min_min']:.3e},{st['range_max_max']:.3e}]")
        if "q_observed_min" in st and "q_expected_min" in st:
            pieces.append(
                f"q=[{st['q_observed_min']:.0f},{st['q_observed_max']:.0f}]"
                f"/[{st['q_expected_min']:.0f},{st['q_expected_max']:.0f}]")
        if "clip_fraction_max" in st:
            pieces.append(f"clip_max={st['clip_fraction_max'] * 100:.4f}%")
        if "rail_fraction_max" in st:
            pieces.append(f"rail_max={st['rail_fraction_max'] * 100:.4f}%")
        if "max_abs_error_max" in st:
            pieces.append(f"max_err={st['max_abs_error_max']:.3e}")
        if "relative_rmse_max" in st:
            pieces.append(f"rel_rmse={st['relative_rmse_max']:.3e}")
        if "acc_min" in st:
            pieces.append(f"acc=[{st['acc_min']:.0f},{st['acc_max']:.0f}]")
        if float(st.get("clip_fraction_max", 0.0)) > warn_frac:
            pieces.append("WARNING clipping above threshold")
        lines.append("; ".join(pieces))
    return lines


def _should_sample_quant_details(name):
    if not _quant_diag_enabled():
        return False
    return _QUANT_DIAG_PRINT_COUNTS.get(name, 0) < _quant_diag_limit()


def quantized_vmm(x, w, x_bits, w_bits, op_name="VMM"):
    """Matrix multiply using integer activation/weight codes and int32 accum."""
    x = np.asarray(x, dtype=np.float32)
    prepared_w = w if isinstance(w, PreparedQuantizedWeights) else None
    w = prepared_w.values if prepared_w is not None else \
        np.asarray(w, dtype=np.float32)
    if not using_int_accum_quantization():
        return (x @ w.T).astype(np.float32)
    if int(x_bits) >= 32 or int(w_bits) >= 32:
        return (x @ w.T).astype(np.float32)
    if prepared_w is not None and prepared_w.n_bits != int(w_bits):
        raise ValueError(
            f"Prepared weights use {prepared_w.n_bits} bits, "
            f"but quantized_vmm was called with {int(w_bits)} bits")
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("quantized_vmm expects 2-D x and w arrays")
    if x.shape[1] != w.shape[1]:
        raise ValueError(
            f"quantized_vmm shape mismatch: x={x.shape}, w={w.shape}")
    if x.shape[0] == 0 or w.shape[0] == 0:
        return np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)

    input_name = f"{op_name}.input"
    _record_quant_range(input_name, x)

    if using_uint8_input_quantization():
        value_range = _fixed_range_for(input_name)
        x_q, x_scale, x_zero_point, x_stats = \
            _quantize_uint8_codes_with_stats(
                x, int(x_bits), value_range=value_range)
        x_centered = x_q.astype(np.int32) - int(x_zero_point or 0)
        x_range = (x_stats["qmin"], x_stats["qmax"])
    else:
        x_q, x_scale, x_stats = _quantize_signed_codes_with_stats(
            x, int(x_bits))
        x_zero_point = None
        x_centered = x_q.astype(np.int32)
        x_range = (x_stats["qmin"], x_stats["qmax"])

    if prepared_w is not None:
        w_q = prepared_w.q
        w_scale = prepared_w.scale
        w_stats = prepared_w.stats
        w_q_t = prepared_w.q_int32_t
    else:
        w_q, w_scale, w_stats = _quantize_signed_codes_with_stats(
            w, int(w_bits))
        w_q_t = w_q.astype(np.int32).T
    acc = x_centered @ w_q_t
    out = (acc.astype(np.float32) *
           float(x_scale or 0.0) * float(w_scale or 0.0))

    detail_name = f"{op_name}.output"
    _record_quant_range(detail_name, out)
    if _should_sample_quant_details(detail_name):
        if using_uint8_input_quantization():
            x_deq = dequantize_uint8_codes(x_q, x_scale, x_zero_point)
        else:
            x_deq = dequantize_signed_codes(x_q, x_scale)
        w_range = (w_stats["qmin"], w_stats["qmax"])
        quantization_report(
            input_name, x, x_deq,
            q=x_q, scale=x_scale, q_range=x_range,
            clip_fraction=x_stats["clip_fraction"],
            zero_point=x_zero_point,
            value_range=(x_stats.get("range_min"), x_stats.get("range_max"))
            if x_stats.get("range_min") is not None else None)
        quantization_report(
            f"{op_name}.weights", w,
            dequantize_signed_codes(w_q, w_scale),
            q=w_q, scale=w_scale, q_range=w_range,
            clip_fraction=w_stats["clip_fraction"])
        ref = (x @ w.T).astype(np.float32)
        quantization_report(detail_name, ref, out, accumulator=acc)
    elif _quant_diag_enabled():
        quantization_report(f"{op_name}.accumulator", None, None,
                            accumulator=acc)

    return out.astype(np.float32)


def quantize_weights(x, n_bits, scale=None, return_scale=False):
    """Symmetric uniform quantization for crossbar weights.

    In real memristor hardware, the quantization scale is determined once
    at calibration time and then frozen.  To support this:

    - At calibration: call with ``return_scale=True`` to obtain the scale.
    - During streaming (e.g. EMA center update): pass the saved ``scale``
      so that the same quantization grid is reused.

    Parameters
    ----------
    x : ndarray
        Weights to quantize.
    n_bits : int
        Bit width (>= 32 means no quantization).
    scale : float or None
        If provided, use this fixed scale instead of computing from data.
    return_scale : bool
        If True, return ``(quantized, scale)`` tuple.
    """
    if n_bits >= 32:
        out = x.astype(np.float32)
        if return_scale:
            return out, None
        return out
    n_levels = 2 ** n_bits
    if scale is None:
        abs_max = np.abs(x).max()
        if abs_max < 1e-12:
            if return_scale:
                return np.zeros_like(x, dtype=np.float32), 0.0
            return np.zeros_like(x, dtype=np.float32)
        scale = float(abs_max / (n_levels // 2 - 1))
    quantized = (np.round(x / scale) * scale).astype(np.float32)
    if return_scale:
        return quantized, scale
    return quantized


def quantize_signal(x, n_bits, op_name=None):
    """Quantize ADC/DAC signal conversion.

    The uint8 backend uses fixed calibration ranges for named operations.
    Other backends retain the original per-chunk symmetric quantization.
    """
    if op_name:
        _record_quant_range(op_name, x)
    if n_bits >= 32:
        out = x.astype(np.float32)
        if op_name:
            quantization_report(op_name, x, out)
        return out
    if using_uint8_input_quantization() and op_name:
        value_range = _fixed_range_for(op_name)
        q, scale, zero_point, stats = _quantize_uint8_codes_with_stats(
            x, int(n_bits), value_range=value_range)
        out = dequantize_uint8_codes(q, scale, zero_point)
        quantization_report(
            op_name, x, out, q=q, scale=scale,
            q_range=(stats["qmin"], stats["qmax"]),
            clip_fraction=stats["clip_fraction"],
            zero_point=zero_point,
            value_range=(stats["range_min"], stats["range_max"]))
        return out
    n_levels = 2 ** n_bits
    abs_max = np.abs(x).max()
    if abs_max < 1e-12:
        out = np.zeros_like(x, dtype=np.float32)
        if op_name:
            qmin, qmax = _signed_qrange(int(n_bits))
            q = np.zeros_like(x, dtype=_signed_qdtype(int(n_bits)))
            quantization_report(
                op_name, x, out, q=q, scale=0.0, q_range=(qmin, qmax),
                clip_fraction=0.0)
        return out
    scale = abs_max / (n_levels // 2 - 1)
    raw = np.round(x / scale)
    out = (raw * scale).astype(np.float32)
    if op_name:
        qmin, qmax = _signed_qrange(int(n_bits))
        clip_fraction = float(np.mean((raw < qmin) | (raw > qmax))) \
            if raw.size else 0.0
        q = np.clip(raw, qmin, qmax).astype(_signed_qdtype(int(n_bits)))
        quantization_report(
            op_name, x, out, q=q, scale=scale, q_range=(qmin, qmax),
            clip_fraction=clip_fraction)
    return out


# ==============================================================
# FIR High-Pass Filter
# ==============================================================

def design_fir_highpass(num_taps=None, cutoff_hz=None, fs=None, window=None):
    """Design FIR high-pass filter."""
    nt = num_taps or config.FIR_NUM_TAPS
    fc = cutoff_hz or config.FIR_HIGHPASS_FREQ
    sr = fs or config.SAMPLE_RATE
    win = window or config.FIR_WINDOW
    if nt % 2 == 0:
        nt += 1
    coeffs = firwin(nt, fc, pass_zero=False, fs=sr, window=win).astype(np.float32)
    print(f"[FIR] {nt} taps, cutoff={fc} Hz, fs={sr} Hz, window={win}")
    return coeffs


def fir_frequency_response(coeffs, fs=None, n_fft=4096):
    """Frequency response of FIR filter."""
    sr = fs or config.SAMPLE_RATE
    w, h = freqz(coeffs, worN=n_fft, fs=sr)
    return w, 20 * np.log10(np.abs(h) + 1e-12)


class FIRFilter:
    """FIR high-pass filter with overlap buffer for streaming.

    Signal path: DAC(input) -> weight-quantized convolution -> ADC(output)
    """

    def __init__(self, coeffs, chunk_samples=None, precision_bits=None):
        self.raw_coeffs = coeffs.copy()
        self.num_taps = len(coeffs)
        self.chunk_samples = chunk_samples or config.CHUNK_SAMPLES
        self.precision_bits = precision_bits or config.PRECISION_FIR
        self.coeffs = quantize_weights(coeffs, self.precision_bits)
        self._coeff_weight_cache = prepare_quantized_weights(
            self.coeffs[::-1][np.newaxis, :], self.precision_bits)
        self._overlap = None
        self._initialized = False
        err = np.abs(self.coeffs - self.raw_coeffs).max()
        print(f"[FIR] {self.precision_bits}-bit weights, max coeff quant error: {err:.2e}")

    def reset(self, n_channels=None):
        n_ch = n_channels or config.N_CHANNELS
        self._overlap = np.zeros((self.num_taps - 1, n_ch), dtype=np.float32)
        self._initialized = True

    def process_chunk(self, chunk):
        """Filter one chunk. Input is assumed already DAC-quantized by caller."""
        n_samples, n_ch = chunk.shape
        if not self._initialized:
            self.reset(n_ch)
        extended = np.vstack([self._overlap, chunk])
        if using_int_accum_quantization():
            filtered = self._process_chunk_int_accum(extended, n_samples, n_ch)
        else:
            from scipy.signal import fftconvolve
            full_conv = fftconvolve(
                extended, self.coeffs[:, np.newaxis], mode='full', axes=0)
            start = self.num_taps - 1
            filtered = full_conv[start:start + n_samples].astype(np.float32)
        self._overlap = chunk[-(self.num_taps - 1):].copy()
        # ADC quantization on output
        adc_name = "FIR.adc" if using_int_accum_quantization() else None
        filtered = quantize_signal(filtered, config.PRECISION_ADC,
                                   op_name=adc_name)
        return filtered

    def _process_chunk_int_accum(self, extended, n_samples, n_ch):
        """Integer FIR without materializing every sliding window.

        This is algebraically the same as quantized_vmm over the im2col
        sliding-window matrix, but it quantizes the extended chunk once and
        then performs one 1-D convolution per channel.  The output values are
        preserved while avoiding the very large temporary window matrix.
        """
        input_name = "FIR.input"
        _record_quant_range(input_name, extended)

        if using_uint8_input_quantization():
            value_range = _fixed_range_for(input_name)
            x_q, x_scale, x_zero_point, x_stats = \
                _quantize_uint8_codes_with_stats(
                    extended, int(config.PRECISION_DAC),
                    value_range=value_range)
            x_centered = x_q.astype(np.int32) - int(x_zero_point or 0)
            x_range = (x_stats["qmin"], x_stats["qmax"])
        else:
            x_q, x_scale, x_stats = _quantize_signed_codes_with_stats(
                extended, int(config.PRECISION_DAC))
            x_zero_point = None
            x_centered = x_q.astype(np.int32)
            x_range = (x_stats["qmin"], x_stats["qmax"])

        w_fir = self._coeff_weight_cache.values
        w_q = self._coeff_weight_cache.q
        w_scale = self._coeff_weight_cache.scale
        w_stats = self._coeff_weight_cache.stats
        kernel = w_q.ravel()[::-1].astype(np.int32)

        acc = np.empty((n_samples, n_ch), dtype=np.int32)
        start = self.num_taps - 1
        stop = start + n_samples
        for ch in range(n_ch):
            full = np.convolve(x_centered[:, ch], kernel, mode='full')
            acc[:, ch] = full[start:stop].astype(np.int32, copy=False)

        out = (acc.astype(np.float32) *
               float(x_scale or 0.0) * float(w_scale or 0.0))

        detail_name = "FIR.output"
        _record_quant_range(detail_name, out)
        if _should_sample_quant_details(detail_name):
            if using_uint8_input_quantization():
                x_deq = dequantize_uint8_codes(x_q, x_scale, x_zero_point)
            else:
                x_deq = dequantize_signed_codes(x_q, x_scale)
            w_range = (w_stats["qmin"], w_stats["qmax"])
            quantization_report(
                input_name, extended, x_deq,
                q=x_q, scale=x_scale, q_range=x_range,
                clip_fraction=x_stats["clip_fraction"],
                zero_point=x_zero_point,
                value_range=(x_stats.get("range_min"), x_stats.get("range_max"))
                if x_stats.get("range_min") is not None else None)
            quantization_report(
                "FIR.weights", w_fir,
                dequantize_signed_codes(w_q, w_scale),
                q=w_q, scale=w_scale, q_range=w_range,
                clip_fraction=w_stats["clip_fraction"])
            ref = np.empty_like(out, dtype=np.float32)
            ref_kernel = self.coeffs.astype(np.float32)
            for ch in range(n_ch):
                full = np.convolve(extended[:, ch], ref_kernel, mode='full')
                ref[:, ch] = full[start:stop].astype(np.float32, copy=False)
            quantization_report(detail_name, ref, out, accumulator=acc)
        elif _quant_diag_enabled():
            quantization_report("FIR.accumulator", None, None,
                                accumulator=acc)

        return out.astype(np.float32)


# ==============================================================
# CAR
# ==============================================================

def apply_car(data, method=None):
    """Common Average Reference (pure digital, no crossbar)."""
    m = method or config.CAR_METHOD
    if m == "median":
        ref = np.median(data, axis=1, keepdims=True)
    else:
        ref = np.mean(data, axis=1, keepdims=True)
    return (data - ref).astype(np.float32)


# ==============================================================
# Channel Grouping
# ==============================================================

def compute_channel_groups(channel_positions, mode=None, nrange=None, overlap=None):
    """Divide channels into local groups for whitening."""
    m = mode or config.WHITEN_MODE
    nr = nrange or config.WHITEN_NRANGE
    ov = overlap or config.WHITEN_OVERLAP
    n_ch = channel_positions.shape[0]
    sort_idx = np.lexsort((channel_positions[:, 0], channel_positions[:, 1]))

    if n_ch <= nr:
        ch = sort_idx[:n_ch]
        groups = [{
            "input_channels": ch,
            "output_channels": ch,
            "output_indices_in_group": np.arange(n_ch),
        }]
        print(f"[WHITEN] Single group: {n_ch} channels (< nrange={nr})")
        return groups

    if m == "non_overlap":
        return _groups_non_overlap(sort_idx, n_ch, nr)
    elif m == "overlap":
        return _groups_overlap(sort_idx, n_ch, nr, ov)
    elif m == "overlap_symmetric":
        return _groups_overlap_symmetric(sort_idx, n_ch, nr, ov)
    elif m == "sliding":
        return _groups_sliding(channel_positions, n_ch, nr)
    else:
        raise ValueError(f"Unknown mode: {m}")


def _groups_non_overlap(sort_idx, n_ch, nrange):
    groups = []
    for start in range(0, n_ch, nrange):
        end = min(start + nrange, n_ch)
        ch = sort_idx[start:end]
        groups.append({
            "input_channels": ch,
            "output_channels": ch,
            "output_indices_in_group": np.arange(len(ch)),
        })
    print(f"[WHITEN] Non-overlap: {len(groups)} groups x {nrange} ch")
    return groups


def _groups_overlap(sort_idx, n_ch, nrange, overlap):
    stride = nrange - overlap
    starts = list(range(0, n_ch - nrange + 1, stride))
    if starts[-1] + nrange < n_ch:
        starts.append(n_ch - nrange)

    groups = []
    assigned = set()

    for i, start in enumerate(starts):
        ch = sort_idx[start:start + nrange]
        out_ch = np.array([c for c in ch if c not in assigned])
        assigned.update(out_ch.tolist())
        ch_list = ch.tolist()
        out_indices = [ch_list.index(c) for c in out_ch]
        groups.append({
            "input_channels": ch,
            "output_channels": out_ch,
            "output_indices_in_group": np.array(out_indices),
        })

    total = sum(len(g["output_channels"]) for g in groups)
    missing = set(sort_idx.tolist()) - assigned
    if missing:
        for m in missing:
            pos_m = np.where(sort_idx == m)[0][0]
            best_group = min(range(len(groups)),
                           key=lambda gi: abs(np.mean(
                               [np.where(sort_idx == c)[0][0]
                                for c in groups[gi]["input_channels"]]) - pos_m))
            g = groups[best_group]
            g["output_channels"] = np.append(g["output_channels"], m)
            ch_list = g["input_channels"].tolist()
            if m in ch_list:
                g["output_indices_in_group"] = np.append(
                    g["output_indices_in_group"], ch_list.index(m))
            assigned.add(m)
        total = sum(len(g["output_channels"]) for g in groups)

    print(f"[WHITEN] Overlap: {len(groups)} groups, {nrange} ch, "
          f"overlap={overlap}, stride={stride}, total_out={total}/{n_ch}")
    assert total == n_ch, f"Coverage error: {total} != {n_ch}"
    return groups


def _groups_overlap_symmetric(sort_idx, n_ch, nrange, overlap):
    output_size = nrange - 2 * overlap
    if output_size <= 0:
        raise ValueError(f"nrange={nrange} too small for overlap={overlap}")
    stride = output_size
    starts = list(range(0, n_ch - nrange + 1, stride))
    if starts[-1] + nrange < n_ch:
        starts.append(n_ch - nrange)

    groups = []
    assigned = set()

    for i, start in enumerate(starts):
        ch = sort_idx[start:start + nrange]
        core_start = overlap
        core_end = nrange - overlap
        if i == 0:
            core_start = 0
        if i == len(starts) - 1:
            core_end = nrange
        out_ch = np.array([c for c in ch[core_start:core_end]
                           if c not in assigned])
        assigned.update(out_ch.tolist())
        ch_list = ch.tolist()
        out_indices = [ch_list.index(c) for c in out_ch]
        groups.append({
            "input_channels": ch,
            "output_channels": out_ch,
            "output_indices_in_group": np.array(out_indices),
        })

    total = sum(len(g["output_channels"]) for g in groups)
    missing = set(sort_idx.tolist()) - assigned
    if missing:
        for m in missing:
            pos_m = np.where(sort_idx == m)[0][0]
            best_group = min(range(len(groups)),
                           key=lambda gi: abs(np.mean(
                               [np.where(sort_idx == c)[0][0]
                                for c in groups[gi]["input_channels"]]) - pos_m))
            g = groups[best_group]
            g["output_channels"] = np.append(g["output_channels"], m)
            ch_list = g["input_channels"].tolist()
            if m in ch_list:
                g["output_indices_in_group"] = np.append(
                    g["output_indices_in_group"], ch_list.index(m))
            assigned.add(m)
        total = sum(len(g["output_channels"]) for g in groups)

    print(f"[WHITEN] Symmetric overlap: {len(groups)} groups, {nrange} ch, "
          f"overlap={overlap}, output_core={output_size}, total_out={total}/{n_ch}")
    assert total == n_ch, f"Coverage error: {total} != {n_ch}"
    return groups


def _groups_sliding(channel_positions, n_ch, nrange):
    dist = cdist(channel_positions, channel_positions)
    groups = []
    for ch in range(n_ch):
        nearest = np.argsort(dist[ch])[:nrange]
        out_idx = np.where(nearest == ch)[0]
        groups.append({
            "input_channels": nearest,
            "output_channels": np.array([ch]),
            "output_indices_in_group": out_idx,
        })
    print(f"[WHITEN] Sliding: {len(groups)} groups (1 per channel)")
    return groups


# ==============================================================
# ZCA Whitening
# ==============================================================

def compute_whitening_matrices(calibration_data, groups, epsilon=None,
                               precision_bits=None):
    """Compute local ZCA whitening: W = E @ diag((D+e)^{-1/2}) @ E^T"""
    eps = epsilon or config.WHITEN_EPSILON
    nb = precision_bits if precision_bits is not None else config.PRECISION_WHITEN

    whitening_ops = []

    for i, group in enumerate(groups):
        ch = group["input_channels"]
        local = calibration_data[:, ch]
        cov = np.cov(local, rowvar=False).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(cov)

        if i == 0:
            print(f"[WHITEN] Group 0 cov diagonal range: "
                  f"[{np.min(np.diag(cov)):.4e}, {np.max(np.diag(cov)):.4e}]")
            print(f"[WHITEN] Group 0 eigenvalue range: "
                  f"[{eigvals.min():.4e}, {eigvals.max():.4e}]")

        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
        W = (eigvecs @ D_inv_sqrt @ eigvecs.T).astype(np.float32)

        op = {
            "group_id": i,
            "input_channels": ch,
            "output_channels": group["output_channels"],
            "W_full": W,
            "cov": cov,
            "eigenvalues": eigvals.astype(np.float32),
        }
        if "output_indices_in_group" in group:
            op["output_indices_in_group"] = group["output_indices_in_group"]
        else:
            op["output_indices_in_group"] = np.arange(W.shape[0])

        W_q = quantize_weights(W, nb)
        op[f"W_{nb}bit"] = W_q
        if i == 0 and nb < 32:
            err = np.abs(W_q - W).max()
            print(f"[WHITEN] Group 0 {nb}-bit weights: max_err={err:.4e}")

        whitening_ops.append(op)

    print(f"[WHITEN] {len(whitening_ops)} matrices, size {W.shape}")
    return whitening_ops


class WhiteningCovarianceAccumulator:
    """Streaming covariance accumulator for local whitening groups."""

    def __init__(self, groups):
        self.groups = groups
        self.counts = [0 for _ in groups]
        self.sums = [
            np.zeros(len(group["input_channels"]), dtype=np.float64)
            for group in groups
        ]
        self.cross = [
            np.zeros((len(group["input_channels"]),
                      len(group["input_channels"])), dtype=np.float64)
            for group in groups
        ]

    def update(self, filtered_chunk):
        if filtered_chunk.size == 0:
            return
        for i, group in enumerate(self.groups):
            ch = group["input_channels"]
            local = np.asarray(filtered_chunk[:, ch], dtype=np.float64)
            self.counts[i] += int(local.shape[0])
            self.sums[i] += local.sum(axis=0)
            self.cross[i] += local.T @ local

    def covariances(self):
        covs = []
        for n, sums, cross in zip(self.counts, self.sums, self.cross):
            if n <= 1:
                cov = np.zeros_like(cross, dtype=np.float32)
            else:
                centered_cross = cross - np.outer(sums, sums) / float(n)
                cov = (centered_cross / float(n - 1)).astype(np.float32)
            covs.append(cov)
        return covs


def compute_whitening_matrices_from_covariances(covariances, groups,
                                                epsilon=None,
                                                precision_bits=None):
    """Compute local ZCA whitening matrices from pre-accumulated covariances."""
    eps = epsilon or config.WHITEN_EPSILON
    nb = precision_bits if precision_bits is not None else config.PRECISION_WHITEN

    whitening_ops = []
    W = None
    for i, (group, cov) in enumerate(zip(groups, covariances)):
        cov = np.asarray(cov, dtype=np.float32)
        eigvals, eigvecs = np.linalg.eigh(cov)

        if i == 0:
            print(f"[WHITEN] Group 0 cov diagonal range: "
                  f"[{np.min(np.diag(cov)):.4e}, {np.max(np.diag(cov)):.4e}]")
            print(f"[WHITEN] Group 0 eigenvalue range: "
                  f"[{eigvals.min():.4e}, {eigvals.max():.4e}]")

        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
        W = (eigvecs @ D_inv_sqrt @ eigvecs.T).astype(np.float32)

        op = {
            "group_id": i,
            "input_channels": group["input_channels"],
            "output_channels": group["output_channels"],
            "W_full": W,
            "cov": cov,
            "eigenvalues": eigvals.astype(np.float32),
        }
        if "output_indices_in_group" in group:
            op["output_indices_in_group"] = group["output_indices_in_group"]
        else:
            op["output_indices_in_group"] = np.arange(W.shape[0])

        W_q = quantize_weights(W, nb)
        op[f"W_{nb}bit"] = W_q
        if i == 0 and nb < 32:
            err = np.abs(W_q - W).max()
            print(f"[WHITEN] Group 0 {nb}-bit weights: max_err={err:.4e}")

        whitening_ops.append(op)

    if W is None:
        raise ValueError("No whitening covariances were provided")
    print(f"[WHITEN] {len(whitening_ops)} matrices, size {W.shape}")
    return whitening_ops


# ==============================================================
# Whitening Processor
# ==============================================================

class WhiteningProcessor:
    """Apply local ZCA whitening.

    Signal path per group: DAC(input) -> weight-quantized VMM -> ADC(output)
    """

    def __init__(self, whitening_ops, precision_bits=None):
        self.ops = whitening_ops
        self.precision_bits = precision_bits or config.PRECISION_WHITEN
        self.n_output_channels = sum(len(op["output_channels"]) for op in self.ops)
        self._output_order = np.concatenate([op["output_channels"] for op in self.ops])
        self._W_key = f"W_{self.precision_bits}bit"
        if self._W_key not in self.ops[0]:
            self._W_key = "W_full"
        self._W_cache_key = f"{self._W_key}_prepared"
        for op in self.ops:
            op[self._W_cache_key] = prepare_quantized_weights(
                op[self._W_key], self.precision_bits)
        self._needs_reorder = not np.array_equal(
            self._output_order, np.arange(self.n_output_channels))
        if self._needs_reorder:
            print(f"[WHITEN] Output channel order != raw order, will reorder")

    def process_chunk(self, filtered_chunk):
        """Whiten one chunk with DAC/ADC quantization at crossbar boundary."""
        n_samples = filtered_chunk.shape[0]
        whitened = np.zeros((n_samples, self.n_output_channels), dtype=np.float32)
        out_col = 0
        for op in self.ops:
            W = op[self._W_key]
            W_vmm = op.get(self._W_cache_key, W)
            in_ch = op["input_channels"]
            out_idx = op["output_indices_in_group"]
            n_out = len(out_idx)
            x = filtered_chunk[:, in_ch]
            if using_int_accum_quantization():
                y = quantized_vmm(
                    x, W_vmm, config.PRECISION_DAC, self.precision_bits,
                    op_name="WHITEN")
            else:
                # DAC quantization on crossbar input
                x = quantize_signal(x, config.PRECISION_DAC)
                y = x @ W.T
            whitened[:, out_col:out_col + n_out] = y[:, out_idx]
            out_col += n_out
        # Reorder columns to raw channel order
        if self._needs_reorder:
            reordered = np.empty_like(whitened)
            reordered[:, self._output_order] = whitened
            whitened = reordered
        # ADC quantization on crossbar output
        adc_name = "WHITEN.adc" if using_int_accum_quantization() else None
        whitened = quantize_signal(whitened, config.PRECISION_ADC,
                                   op_name=adc_name)
        return whitened

    @property
    def output_channel_order(self):
        return self._output_order


# ==============================================================
# Full Pipeline
# ==============================================================

class PreprocessingPipeline:
    """
    Calibration: raw -> CAR -> FIR design -> whitening matrices
    Streaming:   raw chunk -> CAR (digital) -> DAC -> FIR (crossbar) -> ADC
                 -> DAC -> whiten (crossbar) -> ADC
    """

    def __init__(self, channel_positions, precision_bits=None,
                 fir_bits=None, whiten_bits=None,
                 skip_car=False, skip_fir=False):
        self.channel_positions = channel_positions
        self.fir_bits = fir_bits or config.PRECISION_FIR
        self.whiten_bits = whiten_bits or config.PRECISION_WHITEN
        if precision_bits is not None:
            self.fir_bits = precision_bits
            self.whiten_bits = precision_bits
        self.skip_car = skip_car
        self.skip_fir = skip_fir
        self.fir_filter = None
        self.whitening_processor = None
        self.groups = None
        self._calibrated = False

    def calibrate(self, calibration_data):
        """Offline calibration phase.

        Signal quantization is applied during calibration so that whitening
        matrices are computed from DAC/ADC-quantized signals, matching
        the streaming signal distribution.
        """
        print(f"\n{'='*50}")
        print(f"[CALIB] precision: FIR={self.fir_bits}-bit weights, "
              f"Whiten={self.whiten_bits}-bit weights, "
              f"ADC={config.PRECISION_ADC}-bit, DAC={config.PRECISION_DAC}-bit")
        print(f"{'='*50}")

        if self.skip_car or self.skip_fir:
            print(f"[CALIB] Data pre-processed: "
                  f"skip_CAR={self.skip_car}, skip_FIR={self.skip_fir}")

        rms_raw = np.sqrt(np.mean(calibration_data ** 2))
        print(f"[CALIB] Raw calibration:  shape={calibration_data.shape}, "
              f"RMS={rms_raw:.4f}")

        # 1. CAR (digital, no quantization) — skip if data already CAR'd
        if self.skip_car:
            data_car = calibration_data
            print(f"[CALIB] CAR:              SKIPPED (already applied)")
        else:
            data_car = apply_car(calibration_data)
            rms_car = np.sqrt(np.mean(data_car ** 2))
            print(f"[CALIB] After CAR:        RMS={rms_car:.4f}")

        # 2. FIR filter setup — always create (needed for group delay tracking)
        coeffs = design_fir_highpass()
        self.fir_filter = FIRFilter(coeffs, precision_bits=self.fir_bits)
        self.fir_filter.reset(calibration_data.shape[1])

        # 3. Filter calibration data through FIR — skip if data already bandpassed
        if self.skip_fir:
            filtered = data_car
            print(f"[CALIB] FIR:              SKIPPED (already applied)")
        else:
            cs = config.CHUNK_SAMPLES
            chunks = []
            for s in range(0, data_car.shape[0] - self.fir_filter.num_taps, cs):
                c = data_car[s:s + cs]
                if c.shape[0] == cs:
                    # DAC quantization before FIR crossbar
                    if using_int_accum_quantization():
                        c_dac = c
                    else:
                        c_dac = quantize_signal(c, config.PRECISION_DAC)
                    # FIR process_chunk applies ADC on output
                    chunks.append(self.fir_filter.process_chunk(c_dac))
            filtered = np.vstack(chunks)
        rms_filt = np.sqrt(np.mean(filtered ** 2))
        print(f"[CALIB] After FIR stage:  shape={filtered.shape}, RMS={rms_filt:.4f}")

        if rms_filt ** 2 < config.WHITEN_EPSILON * 10:
            print(f"[CALIB] WARNING: signal variance ({rms_filt**2:.2e}) close to "
                  f"epsilon ({config.WHITEN_EPSILON:.1e})!")

        # 4. Channel groups
        groups = compute_channel_groups(self.channel_positions)
        self.groups = groups

        # 5. Whitening matrices (computed from ADC/DAC-quantized filtered data)
        wops = compute_whitening_matrices(filtered, groups,
                                          precision_bits=self.whiten_bits)
        self.whitening_processor = WhiteningProcessor(wops, self.whiten_bits)

        # Reset FIR for streaming
        self.fir_filter.reset(calibration_data.shape[1])
        self._calibrated = True
        print(f"[CALIB] Done. Output channels: "
              f"{self.whitening_processor.n_output_channels}")

    def calibrate_from_chunks(self, calibration_chunks, n_channels,
                              n_chunks=None, progress_label="CALIB",
                              progress_every=None):
        """Offline calibration from a chunk iterator.

        This preserves the same CAR -> FIR -> whitening calibration math as
        ``calibrate`` while avoiding one large in-memory calibration array.
        """
        print(f"\n{'='*50}")
        print(f"[CALIB] precision: FIR={self.fir_bits}-bit weights, "
              f"Whiten={self.whiten_bits}-bit weights, "
              f"ADC={config.PRECISION_ADC}-bit, DAC={config.PRECISION_DAC}-bit")
        print(f"{'='*50}")
        print("[CALIB] Streaming calibration chunks to avoid large RAM copies")

        if self.skip_car or self.skip_fir:
            print(f"[CALIB] Data pre-processed: "
                  f"skip_CAR={self.skip_car}, skip_FIR={self.skip_fir}")

        coeffs = design_fir_highpass()
        self.fir_filter = FIRFilter(coeffs, precision_bits=self.fir_bits)
        self.fir_filter.reset(n_channels)

        groups = compute_channel_groups(self.channel_positions)
        self.groups = groups
        cov_acc = WhiteningCovarianceAccumulator(groups)

        raw_sum_sq = 0.0
        car_sum_sq = 0.0
        filt_sum_sq = 0.0
        raw_count = 0
        car_count = 0
        filt_count = 0
        total_samples = 0
        n_seen = 0
        if progress_every is None:
            progress_every = max(1, (int(n_chunks) // 20) if n_chunks else 50)

        for n_seen, chunk in enumerate(calibration_chunks, start=1):
            if chunk.size == 0:
                continue
            chunk = np.asarray(chunk, dtype=np.float32)
            total_samples += int(chunk.shape[0])
            raw_sum_sq += float(np.einsum('ij,ij->', chunk, chunk,
                                          dtype=np.float64))
            raw_count += int(chunk.size)

            if self.skip_car:
                data_car = chunk
            else:
                data_car = apply_car(chunk)
            car_sum_sq += float(np.einsum('ij,ij->', data_car, data_car,
                                          dtype=np.float64))
            car_count += int(data_car.size)

            if self.skip_fir:
                filtered = data_car
            else:
                if using_int_accum_quantization():
                    data_dac = data_car
                else:
                    data_dac = quantize_signal(data_car, config.PRECISION_DAC)
                filtered = self.fir_filter.process_chunk(data_dac)

            filt_sum_sq += float(np.einsum('ij,ij->', filtered, filtered,
                                           dtype=np.float64))
            filt_count += int(filtered.size)
            cov_acc.update(filtered)

            if n_seen % progress_every == 0 or (
                    n_chunks and n_seen == int(n_chunks)):
                if n_chunks:
                    pct = 100.0 * n_seen / max(int(n_chunks), 1)
                    print(f"[{progress_label}] Calibration pass 1: "
                          f"{n_seen}/{int(n_chunks)} chunks ({pct:.1f}%)")
                else:
                    print(f"[{progress_label}] Calibration pass 1: "
                          f"{n_seen} chunks")

        if n_seen == 0:
            raise ValueError("No calibration chunks were provided")

        rms_raw = np.sqrt(raw_sum_sq / max(raw_count, 1))
        print(f"[CALIB] Raw calibration:  "
              f"shape=({total_samples}, {n_channels}), RMS={rms_raw:.4f}")

        if self.skip_car:
            print(f"[CALIB] CAR:              SKIPPED (already applied)")
        else:
            rms_car = np.sqrt(car_sum_sq / max(car_count, 1))
            print(f"[CALIB] After CAR:        RMS={rms_car:.4f}")

        if self.skip_fir:
            print(f"[CALIB] FIR:              SKIPPED (already applied)")

        rms_filt = np.sqrt(filt_sum_sq / max(filt_count, 1))
        print(f"[CALIB] After FIR stage:  "
              f"shape=({filt_count // max(n_channels, 1)}, {n_channels}), "
              f"RMS={rms_filt:.4f}")

        if rms_filt ** 2 < config.WHITEN_EPSILON * 10:
            print(f"[CALIB] WARNING: signal variance ({rms_filt**2:.2e}) close to "
                  f"epsilon ({config.WHITEN_EPSILON:.1e})!")

        wops = compute_whitening_matrices_from_covariances(
            cov_acc.covariances(), groups, precision_bits=self.whiten_bits)
        self.whitening_processor = WhiteningProcessor(wops, self.whiten_bits)

        self.fir_filter.reset(n_channels)
        self._calibrated = True
        print(f"[CALIB] Done. Output channels: "
              f"{self.whitening_processor.n_output_channels}")

    def process_chunk(self, raw_chunk):
        """Process one streaming chunk through full pipeline.

        CAR is digital (no quantization). DAC quantization is applied before
        each crossbar stage; ADC quantization is applied inside FIRFilter
        and WhiteningProcessor.
        """
        if not self._calibrated:
            raise RuntimeError("Not calibrated")
        # CAR — skip if data already CAR'd
        data = raw_chunk if self.skip_car else apply_car(raw_chunk)
        # FIR — skip if data already bandpassed
        if self.skip_fir:
            filtered = data
        else:
            if using_int_accum_quantization():
                data_dac = data
            else:
                data_dac = quantize_signal(data, config.PRECISION_DAC)
            filtered = self.fir_filter.process_chunk(data_dac)
        # Whitening crossbar (DAC on input, ADC on output is inside process_chunk)
        whitened = self.whitening_processor.process_chunk(filtered)
        return whitened
