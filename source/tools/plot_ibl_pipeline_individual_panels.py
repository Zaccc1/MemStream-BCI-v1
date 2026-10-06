"""Readable, separate scientific panels from the verified real-data archive.

No sorting, calibration, decoder fitting or data download is performed.
"""
from __future__ import annotations

import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from visualize_ibl_pipeline_real_data import BENCH, OUT, load, lp, read_json

DEST = OUT / "individual_panels"
BLUE, ORANGE, GREEN = "#167d9a", "#d05b38", "#399365"
COLORS = [BLUE, ORANGE, GREEN]
FILES = []


def save(fig, name, explanation):
    fig.canvas.draw()
    for ext in ("png", "svg"):
        fig.savefig(lp(DEST / f"{name}.{ext}"), dpi=240, facecolor="white")
    plt.close(fig)
    FILES.append((name, explanation))


def axes_style(axes):
    for ax in np.ravel(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    d = load(OUT / "plot_data.npz")
    provenance = read_json(OUT / "provenance.json")
    selected = provenance["plot_selection"]
    ch = selected["example_channel"]
    times = np.arange(3000) / 30.0
    reference = np.median(d["raw"], axis=1)
    np.testing.assert_array_equal(d["raw"] - reference[:, None], d["car"])
    zoom = (times >= 10) & (times <= 30)

    # CAR: do not obscure the small subtraction by superimposing the traces.
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.8), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.83, hspace=.65)
    fig.suptitle("CAR: subtract the shared reference", y=.98, fontsize=15)
    fig.text(.5, .923, f"Channel {ch} | same 20 ms interval | median across all 384 channels",
             ha="center", fontsize=10, color="#555555")
    signals = [d["raw"][:, ch], reference, d["car"][:, ch]]
    titles = ["Input: raw voltage", "Subtract: median reference m(t)", "Output: raw voltage - m(t)"]
    limit = np.ceil(max(np.max(np.abs(s[zoom])) for s in (signals[0], signals[2])) / 20) * 20
    for ax, signal, title, color in zip(axes, signals, titles, (BLUE, ORANGE, GREEN)):
        ax.plot(times[zoom], signal[zoom], color=color, lw=1.05)
        ax.axhline(0, lw=.65, color="#cccccc", zorder=0)
        ax.set(ylabel=r"Voltage ($\mu$V)", xlim=(10, 30))
        ax.set_title(title, loc="left", pad=7)
    axes[0].set_ylim(-limit, limit)
    axes[2].set_ylim(-limit, limit)
    ref_limit = np.ceil(np.max(np.abs(reference[zoom])) / 10) * 10 + 10
    axes[1].set_ylim(-ref_limit, ref_limit)
    axes[1].text(.98, .88, "Reference has its own y-axis scale", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8, color="#555555")
    axes[2].set_xlabel("Time within the 100 ms chunk (ms)")
    axes_style(axes)
    rms_ref = float(np.sqrt(np.mean(reference[zoom] ** 2)))
    fig.text(.12, .037, f"Input and output use identical voltage limits. Removed reference RMS: {rms_ref:.1f} uV.", fontsize=9)
    save(fig, "01_car", "Raw, cross-channel median reference and CAR output are separated into three aligned traces. Input/output y-limits are identical; the reference has a visibly labeled independent scale. This shows the exact subtraction, not an assumed SNR gain.")

    # FIR: shift only the output time coordinate to compensate its known delay.
    delay = provenance["fir_delay_samples"]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.18, top=.83, hspace=.5)
    fig.suptitle("FIR high-pass filtering", y=.98, fontsize=15)
    fig.text(.5, .913, f"Channel {ch} | same voltage scale | output aligned by the known FIR delay", ha="center", fontsize=9)
    idx = np.flatnonzero(zoom)
    for ax, signal, title, color in [(axes[0], d["car"][idx, ch], "Before: CAR output", BLUE),
                                    (axes[1], d["fir"][idx + delay, ch], "After: 300 Hz high-pass", ORANGE)]:
        ax.plot(times[idx], signal, color=color, lw=1.05)
        ax.set(ylabel=r"Voltage ($\mu$V)", xlim=(10, 30))
        ax.set_title(title, loc="left")
    axes[1].set_xlabel("Time within chunk, delay corrected (ms)")
    fig.text(.13, .042, "Output shifted by 128 samples (4.27 ms); no amplitude normalization.", fontsize=9)
    axes_style(axes)
    save(fig, "02_fir", "CAR and FIR outputs are shown separately with the same voltage axis. The known 128-sample FIR group delay is compensated for visual correspondence.")

    # ZCA: remove the uninformative diagonal from the color comparison.
    channels = np.asarray(selected["correlation_channels"])
    corr = [np.corrcoef(d[key][300:, channels], rowvar=False) for key in ("fir", "whitened")]
    diag = np.eye(len(channels), dtype=bool)
    upper = np.triu_indices(len(channels), 1)
    off_mean = [float(np.mean(np.abs(c[upper]))) for c in corr]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d2d2d2")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.5))
    fig.subplots_adjust(left=.085, right=.87, bottom=.22, top=.73, wspace=.34)
    fig.suptitle("ZCA: reduce cross-channel correlation", fontsize=15, y=.97)
    fig.text(.5, .884, "Same 24 neighboring channels | same 90 ms of real data | same color scale", ha="center", fontsize=9)
    for ax, c, title, value in zip(axes, corr, ("Before ZCA", "After ZCA"), off_mean):
        im = ax.imshow(np.ma.array(c, mask=diag), vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
        ax.set_title(f"{title}\nMean off-diagonal |r| = {value:.3f}", fontsize=11, pad=10)
        ax.set(xlabel="Local channel index", ylabel="Local channel index")
        ax.set_xticks([0, 8, 16, 23])
        ax.set_yticks([0, 8, 16, 23])
    cax = fig.add_axes([.9, .252, .022, .442])
    colorbar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["-1", "0", "+1"])
    colorbar.set_label("Correlation r", fontsize=9)
    fig.text(.5, .095, "Color away from the diagonal = correlation between different channels.", ha="center", fontsize=10)
    fig.text(.5, .045, "White = near zero correlation. Gray diagonal = self-correlation (not compared).", ha="center", fontsize=9, color="#555555")
    save(fig, "03_zca", "Compare the off-diagonal cells. Saturated red/blue means correlated channels; white means near-zero correlation. The gray diagonal is masked self-correlation. Identical limits [-1,1] show the change in interchannel correlation after ZCA.")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.subplots_adjust(left=.15, right=.96, bottom=.21, top=.80)
    before, after = np.abs(corr[0][upper]), np.abs(corr[1][upper])
    ax.scatter(before, after, color=BLUE, s=18, alpha=.55, edgecolors="none")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Before ZCA: |correlation|", ylabel="After ZCA: |correlation|")
    fig.suptitle("ZCA: each dot is one channel pair", fontsize=14, y=.965)
    ax.text(.04, .91, "Below dashed line = lower correlation", transform=ax.transAxes, fontsize=9)
    fig.text(.5, .06, f"All {len(before)} pairs among the same 24 channels; no pair selection.", ha="center", fontsize=9)
    axes_style([ax])
    save(fig, "03b_zca_pairs", "All 276 unique channel pairs are compared before/after ZCA; points below the identity line have lower absolute correlation.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    fig.subplots_adjust(left=.12, right=.97, top=.79, bottom=.20, wspace=.42)
    fig.suptitle("Template matching", fontsize=15, y=.965)
    tscore = d["B_positions"] / 30
    score_zoom = (tscore >= 10) & (tscore <= 30)
    for index, template in enumerate(d["templates"]):
        color = plt.get_cmap("tab10")(index)
        axes[0].plot((np.arange(len(template)) - 30) / 30, template, color=color, lw=1.25)
        axes[1].plot(tscore[score_zoom], d["B"][score_zoom, ch, index], color=color, lw=1, label=f"T{index}")
    axes[0].set(title="Six calibrated templates", xlabel="Time (ms)", ylabel="Template weight (a.u.)")
    axes[1].set(title=f"Actual scores at channel {ch}", xlabel="Time within chunk (ms)", ylabel="Matching score (a.u.)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .005), ncol=6, frameon=False, fontsize=8)
    axes_style(axes)
    save(fig, "04_template_matching", "The six learned temporal templates and their actual software 8-bit template-matching scores at the same channel. A common color identifies a template across the two plots.")

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5), sharex=True)
    fig.subplots_adjust(left=.14, right=.97, bottom=.16, top=.80, hspace=.56)
    fig.suptitle("Spatial aggregation and event selection", fontsize=14, y=.975)
    cand = d["candidates"]
    candidate_time = d["B_positions"][cand[:, 0]] / 30
    candidate_mask = (cand[:, 1] == ch) & (candidate_time >= 10) & (candidate_time <= 30)
    axes[0].plot(tscore[score_zoom], d["B"][score_zoom, ch].max(axis=1), color=BLUE, lw=1.1)
    axes[0].set_title(f"1. Maximum template score, channel {ch}", loc="left")
    axes[0].set_ylabel("Template score")
    axes[1].scatter(candidate_time[candidate_mask], d["candidate_scores"][candidate_mask],
                    s=52, facecolor=ORANGE, label="SA candidate")
    threshold = read_json(BENCH / "configs/sw_8bit.json")["SA_FINAL_THRESHOLD"]
    axes[1].axhline(threshold, color="#666666", ls="--", lw=1, label=f"SA threshold = {threshold:g}")
    ev = (d["batch_event_channels"] == ch) & (d["batch_event_times"] >= 300) & (d["batch_event_times"] <= 900)
    axes[1].scatter(d["batch_event_times"][ev] / 30, d["batch_event_amplitudes"][ev], marker="v",
                    s=115, facecolor="none", edgecolor="#111111", lw=1.4, label="Kept after NMS")
    axes[1].set_title("2. Spatially aggregated candidate scores", loc="left")
    axes[1].set(xlim=(10, 30), xlabel="Time within chunk (ms)", ylabel="SA score", ylim=(0, max(16, float(d['candidate_scores'][candidate_mask].max()) * 1.15)))
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.53, .916), ncol=3, fontsize=8, frameon=False)
    fig.text(.14, .035, "Only evaluated candidates are plotted; no interpolated SA curve.", fontsize=9)
    axes_style(axes)
    save(fig, "05_spatial_aggregation", "Template matching is shown separately from spatial aggregation. Orange circles are real coarse-candidate SA evaluations, the dashed line is the final threshold, and open triangles are detections surviving NMS. This channel excerpt need not contain an example rejected by NMS.")

    region = selected["lda_region"]
    mask = (d["stream_regions"] == region) & (d["stream_times"] >= 300)
    clusters = selected["highlighted_premerge_clusters"]
    for name, key, title, xlabel, ylabel in [
        ("06_pca", "stream_pca", "Temporal PCA features", "Central-channel tPC1", "Central-channel tPC2"),
        ("07_lda", "stream_lda", "Regional LDA features", "LD1", "LD2")]:
        fig, ax = plt.subplots(figsize=(5.3, 4.4))
        fig.subplots_adjust(left=.17, right=.96, bottom=.17, top=.78)
        fig.suptitle(title, fontsize=15, y=.97)
        other = mask & ~np.isin(d["stream_labels"], clusters)
        ax.scatter(d[key][other, 0], d[key][other, 1], s=25, color="#c3c3c3", label="Other / rejected")
        for cluster, color in zip(clusters, COLORS):
            keep = mask & (d["stream_labels"] == cluster)
            ax.scatter(d[key][keep, 0], d[key][keep, 1], s=43, color=color, label=f"Cluster {cluster}")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54, .9), ncol=2, frameon=False, fontsize=8)
        fig.text(.5, .035, f"Same {int(mask.sum())} events, region {region}; colors are assigned labels.", ha="center", fontsize=9)
        axes_style([ax])
        save(fig, name, "The same 44 detected events in spatial region 1, with identical colors across PCA and LDA. The display uses the fitted projections and pre-merge cluster assignments. Gray includes other assigned clusters and rejects. Each regional LDA space is displayed separately.")

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    fig.subplots_adjust(left=.17, right=.96, bottom=.22, top=.76)
    fig.suptitle("Assigned neuron activity", fontsize=15, y=.965)
    for index, (cluster, color) in enumerate(zip(clusters, COLORS)):
        keep = mask & (d["stream_labels"] == cluster)
        x = (d["stream_times"][keep] - delay) / 30
        ax.eventplot(x, lineoffsets=index, linelengths=.5, colors=color, linewidths=1.7)
    ax.set(xlim=(10 - delay / 30, 100), ylim=(-.6, 2.6), xlabel="Time from streaming start, delay corrected (ms)", ylabel="Pre-merge cluster")
    ax.set_yticks(range(3), [str(c) for c in clusters])
    fig.text(.5, .845, "Each tick is one detected event assigned to that cluster", ha="center", fontsize=9)
    fig.text(.5, .045, "Same highlighted clusters and events as the PCA/LDA panels.", ha="center", fontsize=9)
    save(fig, "08_neuron_assignment", "The same three highlighted pre-merge clusters as the PCA and LDA panels. Tick positions use the original FIR delay correction. Final ACG-based filtering and merging are later steps, not shown by this pre-merge raster.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.75, wspace=.43)
    fig.suptitle("Choice decoding from sorted neural activity", fontsize=14, y=.97)
    ax = axes[0]
    for label, color in [(0, BLUE), (1, ORANGE)]:
        keep = np.flatnonzero(d["decode_y"][:60] == label)
        ax.scatter(keep + 1, d["decode_probability"][keep], s=21, c=color, label=f"True choice {[-1,1][label]:+d}")
    ax.axhline(.5, ls="--", color="#888888", lw=1)
    ax.set(xlim=(0, 61), ylim=(-.04, 1.04), xlabel="Eligible trial index", ylabel="P(choice = +1)", title="Held-out predictions")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.08, .885), ncol=2, frameon=False, fontsize=8)
    cm = d["decode_confusion"]
    norm = cm / cm.sum(axis=1, keepdims=True)
    axes[1].imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i,j]}\n{norm[i,j]:.1%}", ha="center", va="center", color="white" if norm[i,j]>.65 else "black", fontsize=10)
    axes[1].set_xticks([0,1], ["-1", "+1"])
    axes[1].set_yticks([0,1], ["-1", "+1"])
    axes[1].set(title="All 364 held-out trials", xlabel="Predicted choice", ylabel="True choice")
    fig.text(.5, .094, f"{d['decode_rates'].shape[1]} active ACG-pass features | 5-fold mean balanced accuracy: {provenance['decode']['mean_fold_balanced_accuracy']:.2%}", ha="center", fontsize=10)
    fig.text(.5, .042, "8-bit-sorted spikes; FP logistic decoder; 100 ms before first movement.", ha="center", fontsize=9)
    save(fig, "09_decoding", "The left panel shows the first 60 out-of-fold trial predictions; the right pools all 364 trials. Historical five-fold accuracy is reproduced. Software 8-bit applies to sorting, not the FP logistic classifier. Decoding uses the full eligible recording, not the 100-ms waveform excerpt.")

    audit = {"source": str(OUT / "plot_data.npz"), "eid": provenance["eid"],
             "car_method": "median across all 384 channels", "car_window_ms": [10,30],
             "car_subtraction_max_error": float(np.max(np.abs(d["raw"]-reference[:,None]-d["car"]))),
             "car_reference_rms_uv": rms_ref, "zca_time_window_ms": [10,100],
             "zca_channels": channels.tolist(), "zca_unique_pairs": len(before),
             "zca_mean_abs_correlation_before": off_mean[0], "zca_mean_abs_correlation_after": off_mean[1],
             "zca_pairs_decreasing": int((after < before).sum()), "figures": [f for f, _ in FILES]}
    with open(lp(DEST / "panel_audit.json"), "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
    lines = ["# Individual real-data pipeline panels", "", "All panels are exported as PNG and editable SVG. No sorting or model fitting was rerun.", "",
             "Source: IBL 9b5a1754-ac99-4d53-97d3-35c2f6638507 / probe00. Panels 01-08 use the verified 240.000-240.100 s software 8-bit replay; panel 09 uses historical full-recording sorted spikes and reproduced held-out predictions.", "",
             "CAR uses a median reference, despite the conventional CAR name. ZCA plots cross-channel Pearson correlation, not a claim that all removed components are noise. The short segment does not establish performance across recordings.", ""]
    for name, explanation in FILES:
        lines.extend([f"## {name}", "", explanation, "", f"[PNG]({name}.png) | [SVG]({name}.svg)", ""])
    with open(lp(DEST / "README.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    with zipfile.ZipFile(lp(OUT / "individual_panels_png_svg.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name, _ in FILES:
            for ext in ("png", "svg"):
                archive.write(lp(DEST / f"{name}.{ext}"), arcname=f"{name}.{ext}")
        for name in ("README.md", "panel_audit.json"):
            archive.write(lp(DEST / name), arcname=name)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
