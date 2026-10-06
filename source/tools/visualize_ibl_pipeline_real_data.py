"""Read-only, cache-backed real-data illustration of the four pipeline stages.

Replays 100 ms with existing software 8-bit calibration. Never fits a
sorter, calls ONE, or writes to the benchmark tree. Decoding reuses final spikes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "GUI-BCI-test-20260707"
EID = "9b5a1754-ac99-4d53-97d3-35c2f6638507"
BENCH = CODE / "benchmark_runs" / EID
RUN = BENCH / "runs" / "sw_8bit"
SESSION = CODE / "recordings/one_cache/mainenlab/Subjects/ZFM-01936/2021-01-22/002"
RAW_EXPORT = ROOT / "results_10ms/hardware_export/batches/batch_0000"
OUT = ROOT / "outputs/ibl_pipeline_real_data_20260916"


def lp(path):
    text = str(Path(path).absolute())
    return "\\\\?\\" + text if os.name == "nt" and not text.startswith("\\\\?\\") else text


def load(path):
    return np.load(lp(path), allow_pickle=False)


def read_json(path):
    with open(lp(path), encoding="utf-8") as file:
        return json.load(file)


def digest(path):
    hasher = hashlib.sha256()
    with open(lp(path), "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def prepare():
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    print("Loading saved configuration and pipeline modules", flush=True)
    sys.path.insert(0, str(CODE))
    import config
    from config_runtime import apply_config_snapshot

    snapshot = read_json(BENCH / "configs/sw_8bit.json")
    apply_config_snapshot(snapshot)
    config.QUANT_DIAGNOSTICS = False
    assert config.QUANTIZATION_BACKEND == "uint8_int_accum"
    assert config.CHUNK_SAMPLES == 3000
    assert not config.CLUST_ENABLE_CENTER_UPDATES

    import main as pipeline
    import preprocessing
    from threadpoolctl import threadpool_limits

    print("Reading calibration cache", flush=True)
    latest = read_json(RUN / "cali_cache/latest.json")
    bundle = RUN / "cali_cache" / (latest["fingerprint"] + ".pkl.gz")
    with gzip.open(lp(bundle), "rb") as file:
        payload = pickle.load(file)
    assert payload["fingerprint_payload"]["source_info"]["session_eid"] == EID
    fit = payload["final_fit"]
    preprocessing.set_quantization_fixed_ranges(payload["range_payload"]["ranges"])
    pipe, clust, sa = fit["pipe"], fit["clust"], fit["sa"]
    pipe.fir_filter.reset(384)
    for name in vars(clust):
        if name.startswith("all_") and isinstance(getattr(clust, name), list):
            setattr(clust, name, [])

    raw_manifest = read_json(RAW_EXPORT / "manifest.json")
    assert raw_manifest["source_info"]["session_eid"] == EID
    assert raw_manifest["source_info"]["batch_start_s"] == 240.0
    raw_path = RAW_EXPORT / "00_raw/raw_float.npy"
    raw = load(raw_path)
    assert raw.shape == (3000, 384)
    print("Loaded archived real AP excerpt: 240-240.1 s, 384 channels", flush=True)

    # Observe the real streaming processor, including its overlap handling.
    saved_compute = pipeline.compute_score_matrix
    observed = {}

    def capture_scores(*args, **kwargs):
        value = saved_compute(*args, **kwargs)
        observed["B"], observed["B_positions"] = value
        return value

    pipeline.compute_score_matrix = capture_scores
    processor = pipeline._SAStreamState()
    accum = {key: [] for key in ("times", "channels", "labels", "pca", "lda", "regions")}
    arrays = {}
    try:
        with threadpool_limits(limits=2):
            for i in range(1):
                chunk = raw[i * 3000:(i + 1) * 3000]
                car = preprocessing.apply_car(chunk)
                filtered = pipe.fir_filter.process_chunk(car)
                whitened = pipe.whitening_processor.process_chunk(filtered)
                events, labels = processor.process_chunk(
                    whitened, fit["wTEMP"], sa, clust, batch_time_offset=i * 3000)
                local = dict(events)
                local["times"] = events["times"] - i * 3000
                feature_path = pipeline._hw_compute_clustering_feature_path(clust, whitened, local)
                np.testing.assert_array_equal(feature_path["labels"], labels)
                accum["times"].append(events["times"])
                accum["channels"].append(events["channels"])
                accum["labels"].append(labels)
                accum["pca"].append(feature_path["pca_features"])
                accum["lda"].append(feature_path["features"])
                accum["regions"].append(clust.assigner.ch_to_region[events["channels"]])
                if i == 0:
                    candidates = sa.coarse_screen(observed["B"], verbose=False)
                    scores, _, _, _ = sa.sparse_aggregate(observed["B"], candidates)
                    arrays.update(raw=chunk, car=car, fir=filtered, whitened=whitened,
                                  B=observed["B"], B_positions=observed["B_positions"],
                                  candidates=candidates, candidate_scores=scores,
                                  batch_event_times=local["times"],
                                  batch_event_channels=local["channels"],
                                  batch_event_amplitudes=events["amplitudes"],
                                  batch_labels=labels)
                print("Replayed 1/1 chunk", flush=True)
    finally:
        pipeline.compute_score_matrix = saved_compute
    for key, values in accum.items():
        arrays["stream_" + key] = np.concatenate(values)
    arrays["templates"] = fit["wTEMP"]
    arrays["positions"] = clust.positions

    # Preserve the original benchmark's population, trial rules and CV method.
    import benchmark_recording_metrics as benchmark
    import decode_task
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    trials = benchmark._load_trials_from_session(Path(lp(SESSION)))
    spikes = decode_task.load_pipeline_spikes(lp(RUN))
    ids = benchmark._cluster_ids(spikes, "good")
    metrics = read_json(BENCH / "metrics/recording_benchmark_metrics.json")
    source = next(s for s in metrics["sources"] if s["name"] in ("sw_8bit", "sw_int8", "software_8bit"))
    event_times, alignment = decode_task.get_choice_event_times(trials)
    valid = trials["valid"] & (trials["stim_on"] >= 360.0)
    valid &= np.isfinite(event_times) & (event_times < metrics["test_end_s"])
    trial_idx = np.flatnonzero(valid)
    window = tuple(config.DECODE_CHOICE_WINDOW)
    x = decode_task.compute_firing_rates(spikes["times"], spikes["labels"], ids,
                                        event_times[trial_idx], window)
    active = x.std(axis=0) > 1e-6
    x, ids_used = x[:, active], np.asarray(ids)[active]
    y = (trials["choice"][trial_idx] == 1).astype(np.int32)
    prob, predicted, fold_index = np.zeros(len(y)), np.zeros(len(y), int), np.zeros(len(y), int)
    fold_scores = []
    fold_weights = []
    fold_biases = []
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y)):
        scaler = StandardScaler().fit(x[train])
        classifier = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                        max_iter=1000, random_state=42)
        classifier.fit(scaler.transform(x[train]), y[train])
        prob[test] = classifier.predict_proba(scaler.transform(x[test]))[:, 1]
        predicted[test] = classifier.predict(scaler.transform(x[test]))
        fold_index[test] = fold
        fold_scores.append(balanced_accuracy_score(y[test], predicted[test]))
        fold_weights.append(classifier.coef_[0] / scaler.scale_)
        fold_biases.append(classifier.intercept_[0] - np.dot(classifier.coef_[0], scaler.mean_ / scaler.scale_))
    accuracy = float(np.mean(fold_scores))
    old_accuracy = source["decoding"]["choice"]["balanced_accuracy"]
    assert len(y) == source["decoding"]["choice"]["n_trials"]
    np.testing.assert_allclose(accuracy, old_accuracy, atol=1e-10)
    print(f"Choice decoding reproduced: {accuracy:.8%}, {len(y)} trials, {x.shape[1]} features", flush=True)
    arrays.update(decode_rates=x, decode_y=y, decode_probability=prob, decode_prediction=predicted,
                  decode_fold=fold_index, decode_trial_index=trial_idx,
                  decode_cluster_ids=ids_used, decode_event_times=event_times[trial_idx],
                  decode_fold_scores=np.asarray(fold_scores), decode_weights=np.asarray(fold_weights),
                  decode_biases=np.asarray(fold_biases), decode_confusion=confusion_matrix(y, predicted))

    old_times = load(RUN / "clustering/spike_times.npy")
    old_channels = load(RUN / "clustering/spike_channels.npy")
    old_mask = old_times < 3000 - config.FIR_GROUP_DELAY
    new_times = arrays["stream_times"] - config.FIR_GROUP_DELAY
    old_pairs = old_times[old_mask].astype(np.int64) * 384 + old_channels[old_mask]
    new_pairs = new_times.astype(np.int64) * 384 + arrays["stream_channels"]
    intersection = np.intersect1d(old_pairs, new_pairs).size
    np.testing.assert_array_equal(np.sort(old_pairs), np.sort(new_pairs))
    provenance = {
        "eid": EID, "session": str(SESSION), "probe": "probe00",
        "source_run": str(RUN), "software": "8-bit uint8_int_accum sorting; FP logistic decoder",
        "replay_window_s": [240.0, 240.1], "display_chunk_s": [240.0, 240.1],
        "sample_rate_hz": 30000, "chunk_ms": 100, "calibration_s": 240,
        "fir_delay_samples": int(config.FIR_GROUP_DELAY),
        "cache_fingerprint": payload["fingerprint"], "cache_sha256": digest(bundle),
        "raw_file": str(raw_path), "raw_export_manifest": str(RAW_EXPORT / "manifest.json"),
        "raw_excerpt_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "replayed_events": len(new_pairs), "historical_events_in_window": len(old_pairs),
        "identical_time_channel_pairs": int(intersection),
        "decode": {"task": "choice", "alignment": alignment, "window_s": window,
                   "trials": len(y), "features": x.shape[1], "acg_pass_clusters": len(ids),
                   "mean_fold_balanced_accuracy": accuracy, "saved_benchmark_accuracy": old_accuracy,
                   "fold_scores": fold_scores, "pooled_confusion": arrays["decode_confusion"].tolist(),
                   "class_counts": np.bincount(y).tolist(), "folds": 5,
                   "neuron_selection": "post-merge ACG-pass; exact original benchmark population",
                   "classifier": "FP L1 logistic, C=1, liblinear; StandardScaler fitted in training fold",
                   "split": "StratifiedKFold(shuffle=True, random_state=42); offline cross-validation"},
        "code_sha256": {name: digest(CODE / name) for name in (
            "main.py", "preprocessing.py", "template_matching.py", "spatial_aggregation.py",
            "spike_clustering.py", "decode_task.py", "benchmark_recording_metrics.py")},
        "analysis_scope": [
            "Stages 1-3 replay the specified 100-ms AP excerpt through the software pipeline.",
            "Stage 3 labels are pre-merge calibration cluster assignments.",
            "Each regional LDA projection uses its own coordinate space.",
            "Stage 4 uses full-recording post-merge spikes for decoding.",
            "Stages 1-3 use the calibration cache identified in the input metadata.",
            "Software 8-bit denotes sorting; real_acc uses an FP logistic classifier.",
            "The decoding benchmark removes constant features before cross-validation.",
            "The 2-D feature view displays the PCA and LDA transformations.",
            "The first chunk starts with zero FIR history. The first 10 ms are marked/excluded in display and feature selection."
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(lp(OUT / "plot_data.npz"), **arrays)
    with open(lp(OUT / "provenance.json"), "w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2)
    print(json.dumps({key: provenance[key] for key in ("replayed_events", "historical_events_in_window", "identical_time_channel_pairs")}), flush=True)
    faulthandler.cancel_dump_traceback_later()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    d = load(OUT / "plot_data.npz")
    meta = read_json(OUT / "provenance.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#167d9a", "#d05b38", "#399365"]
    fig, axs = plt.subplots(3, 4, figsize=(18, 10.5))
    fig.subplots_adjust(left=.052, right=.966, top=.845, bottom=.13, wspace=.43, hspace=.7)
    fig.suptitle("From real extracellular signals to behavioral decoding", y=.975, fontsize=20)
    fig.text(.5, .932, "IBL ZFM-01936 | probe00 | software 8-bit sorting | 240 s calibration | 100 ms chunk",
             ha="center", fontsize=11, color="#555555")
    for col, title in enumerate(["Signal preprocessing", "Spike detection", "Neuron assignment", "Choice decoding"]):
        ax = axs[0, col]
        fig.text(ax.get_position().x0, .882, f"{chr(65 + col)}  {title}", fontsize=14, weight="bold")

    stable_events = d["batch_event_times"] >= 300
    ch_counts = np.bincount(d["batch_event_channels"][stable_events], minlength=384)
    ch = int(ch_counts.argmax())
    near = np.argsort(np.linalg.norm(d["positions"] - d["positions"][ch], axis=1))[:3]
    time_ms = np.arange(3000) / 30
    for row, key, label, units in [(0, "raw", "Raw + CAR", "uV"),
                                  (1, "fir", "FIR high-pass", "uV"),
                                  (2, "whitened", "ZCA output", "a.u.")]:
        ax = axs[row, 0]
        ydata = d[key][:, near]
        spacing = np.percentile(np.ptp(ydata, axis=0), 75) * 1.05
        for j, channel in enumerate(near):
            ax.plot(time_ms, ydata[:, j] + j * spacing, color=colors[j], lw=.65)
            if row == 0:
                ax.plot(time_ms, d["car"][:, channel] + j * spacing, color="#666666", lw=.5, alpha=.65)
        ax.set(xlim=(10, 100), xlabel="Time within chunk (ms)", ylabel=f"Amplitude ({units}; offset)", title=label)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    axs[0, 0].legend([Line2D([], [], color=colors[0]), Line2D([], [], color="#666666")],
                     ["Raw", "CAR"], loc="upper right", fontsize=8, frameon=False, ncol=2)

    ax = axs[0, 1]
    for index, template in enumerate(d["templates"]):
        ax.plot((np.arange(len(template)) - 30) / 30, template, lw=1, label=str(index))
    ax.set(title="Calibrated temporal templates", xlabel="Time (ms)", ylabel="Template weight (a.u.)")
    ax.text(.02, .96, "6 fixed templates", transform=ax.transAxes, va="top", fontsize=8)
    ax = axs[1, 1]
    xscore = d["B_positions"] / 30
    ax.plot(xscore, d["B"][:, ch, :].max(axis=1), color="#167d9a", lw=.9, label="Max template score")
    cand = d["candidates"]
    keep = cand[:, 1] == ch
    cx = d["B_positions"][cand[keep, 0]] / 30
    ax.scatter(cx, d["candidate_scores"][keep], c="#d05b38", s=22, zorder=3, label="SA candidate score")
    config = read_json(BENCH / "configs/sw_8bit.json")
    ax.axhline(config["SA_FINAL_THRESHOLD"], c="#555555", ls="--", lw=.8, label="SA threshold")
    ev = d["batch_event_channels"] == ch
    ax.scatter(d["batch_event_times"][ev] / 30, d["batch_event_amplitudes"][ev],
               marker="v", facecolors="none", edgecolors="#111111", s=50, zorder=4, label="NMS-kept event")
    ax.set(xlim=(10, 100), xlabel="Time within chunk (ms)", ylabel="Score (a.u.)")
    ax.set_title(f"Scores and detections: channel {ch}", pad=33)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1), ncol=2, columnspacing=.7)
    ax = axs[2, 1]
    ax.scatter(d["batch_event_times"] / 30, d["positions"][d["batch_event_channels"], 1],
               s=8, color="#167d9a", alpha=.75)
    ax.axvspan(0, 10, color="#888888", alpha=.18)
    ax.text(.04, .94, "Startup", transform=ax.transAxes, rotation=90, va="top", fontsize=7)
    ax.set(xlim=(0, 100), title=f"{int(stable_events.sum())} events after first 10 ms",
           xlabel="Time within chunk (ms)", ylabel="Probe depth coordinate (um)")

    stable = d["stream_times"] >= 300
    region = int(np.bincount(d["stream_regions"][stable]).argmax())
    mask = (d["stream_regions"] == region) & stable
    labels, counts = np.unique(d["stream_labels"][mask & (d["stream_labels"] >= 0)], return_counts=True)
    chosen = labels[np.argsort(-counts)[:3]]
    for row, key, xlabel, ylabel, title in [(0, "stream_pca", "Central-channel tPC1", "Central-channel tPC2", "Temporal PCA features"),
                                          (1, "stream_lda", "LD1", "LD2", "Regional LDA features")]:
        ax = axs[row, 2]
        other = mask & ~np.isin(d["stream_labels"], chosen)
        ax.scatter(d[key][other, 0], d[key][other, 1], s=6, color="#bbbbbb", alpha=.5, rasterized=False)
        for index, cluster in enumerate(chosen):
            selected = mask & (d["stream_labels"] == cluster)
            ax.scatter(d[key][selected, 0], d[key][selected, 1], s=14, color=colors[index], alpha=.7, label=f"Cluster {cluster}")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    axs[0, 2].legend(frameon=False, fontsize=7, loc="best")
    ax = axs[2, 2]
    for index, cluster in enumerate(chosen):
        selected = mask & (d["stream_labels"] == cluster)
        x = d["stream_times"][selected] / 30000 - meta["fir_delay_samples"] / 30000
        ax.eventplot(x, lineoffsets=index, linelengths=.55, colors=colors[index], linewidths=.8)
    ax.set(xlim=(.01 - meta["fir_delay_samples"] / 30000, .1), ylim=(-.6, 2.6), title=f"Same spikes, region {region} ({int(mask.sum())} events)",
           xlabel="Time from streaming start (s)", ylabel="Pre-merge cluster")
    ax.set_yticks(range(len(chosen)), [str(c) for c in chosen])

    y = d["decode_y"]
    ax = axs[0, 3]
    rate = d["decode_rates"]
    display_units = np.argsort(-rate.mean(axis=0))[:25]
    image = ax.imshow(np.log1p(rate[:60, display_units].T), aspect="auto", origin="lower",
                      cmap="viridis", interpolation="nearest", extent=(.5, 60.5, .5, 25.5))
    ax.set(title="Trial-wise neural input", xlabel="First 60 eligible trials", ylabel="25 highest-rate ACG-pass units")
    cb = fig.colorbar(image, ax=ax, pad=.02, fraction=.035)
    cb.set_label("log(1 + firing rate / Hz)", fontsize=7)
    ax = axs[1, 3]
    n = min(80, len(y))
    for label, color in [(0, colors[0]), (1, colors[1])]:
        sel = np.flatnonzero(y[:n] == label)
        ax.scatter(sel + 1, d["decode_probability"][sel], s=14, c=color, label=f"True choice {[-1, 1][label]:+d}")
    ax.axhline(.5, color="#555555", lw=.8, ls="--")
    ax.set(xlim=(0, n + 1), ylim=(-.04, 1.04), xlabel="Eligible trial index", ylabel="P(choice = +1)")
    ax.set_title("Held-out trial predictions", pad=24)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    ax = axs[2, 3]
    cm = d["decode_confusion"]
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.1%})", ha="center", va="center", color="white" if norm[i, j] > .65 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["-1", "+1"])
    ax.set_yticks([0, 1], ["-1", "+1"])
    ax.set(xlabel="Predicted choice", ylabel="True choice",
           title=f"5-fold mean balanced accuracy: {meta['decode']['mean_fold_balanced_accuracy']:.2%}")

    fig.text(.052, .068, "A-C: real 100 ms chunk at 240 s; first 10 ms excluded from traces/features; FIR delay = 4.27 ms. C: one regional LDA space; three clusters highlighted.", fontsize=9)
    fig.text(.052, .043, f"D: {len(y)} trials, {rate.shape[1]} active ACG-pass features; 100 ms before first movement; FP logistic regression on historical 8-bit-sorted spikes.", fontsize=9)
    fig.text(.052, .018, f"EID: {EID} | Software pipeline replay and offline cross-validation.", fontsize=8, color="#555555")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(lp(OUT / ("ibl_real_pipeline_overview." + ext)), dpi=220, facecolor="white")
    plt.close(fig)

    # Covariance is shown separately so the main overview remains readable.
    channels = np.sort(np.argsort(np.linalg.norm(d["positions"] - d["positions"][ch], axis=1))[:24])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    stats = {}
    for ax, key, title in zip(axs, ("fir", "whitened"), ("Before ZCA (FIR output)", "After local ZCA")):
        corr = np.corrcoef(d[key][300:, channels], rowvar=False)
        off = np.abs(corr[~np.eye(len(corr), dtype=bool)]).mean()
        stats[key] = float(off)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set(title=f"{title}\nMean |off-diagonal r| = {off:.3f}", xlabel="Local channel index", ylabel="Local channel index")
    fig.colorbar(im, ax=axs, label="Pearson correlation", shrink=.8)
    fig.suptitle("Real cross-channel correlation in the displayed streaming chunk", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(lp(OUT / ("ibl_whitening_detail." + ext)), dpi=220, facecolor="white")
    plt.close(fig)
    meta["plot_selection"] = {"example_channel": ch, "trace_channels": near.tolist(),
                              "lda_region": region, "highlighted_premerge_clusters": chosen.tolist(),
                              "regional_events": int(mask.sum()), "correlation_channels": channels.tolist(),
                              "startup_exclusion_ms": 10, "events_after_startup_exclusion": int(stable_events.sum()),
                              "mean_absolute_off_diagonal_correlation": stats,
                              "selection_rule": "most detected events in display chunk / replay region; not selected on separation or decoding accuracy"}
    with open(lp(OUT / "provenance.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print("Figures written to", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        prepare()
    if not args.prepare_only:
        plot()
