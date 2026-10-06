"""
LDA Feature Space Sweep
========================
Runs multiple LDA configurations on the same KS4 cache and compares results.

Usage:
    python sweep_lda.py

Experiments:
    A. Per-region LDA (N=20)
    B. Post-PCA LDA N=20 + reject gate sweep (quantile 0.95/0.97/0.99, margin 1.05/1.10)
    C. Post-PCA LDA dimension sweep (N=20, 30, 40)
    + baseline (no LDA, reject on)

All runs use the same KS4 120s cache. Results parsed from logs.
"""

import sys, os, re, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# ================================================================
# Experiment definitions
# ================================================================
# Each experiment: (name, {config_attr: value, ...})
EXPERIMENTS = [
    # Baseline
    ("baseline_reject", {
        'CLUST_LDA_MODE': 'off',
        'CLUST_ASSIGN_REJECT_ENABLE': True,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),

    # A: Per-region LDA
    ("per_region_n20", {
        'CLUST_LDA_MODE': 'per_region',
        'CLUST_LDA_N_COMPONENTS': 20,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),

    # B: Post-PCA LDA N=20 + reject gate sweep
    ("postpca_n20_q99_m105", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 20,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),
    ("postpca_n20_q97_m105", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 20,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.97,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),
    ("postpca_n20_q95_m105", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 20,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.95,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),
    ("postpca_n20_q99_m110", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 20,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.10,
    }),

    # C: Post-PCA LDA dimension sweep
    ("postpca_n30_q99_m105", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 30,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),
    ("postpca_n40_q99_m105", {
        'CLUST_LDA_MODE': 'post_pca',
        'CLUST_LDA_N_COMPONENTS': 40,
        'CLUST_ASSIGN_REJECT_QUANTILE': 0.99,
        'CLUST_ASSIGN_MARGIN_MIN': 1.05,
    }),
]


# ================================================================
# Log parser
# ================================================================
def parse_log(log_path):
    """Extract key metrics from a pipeline log file."""
    metrics = {}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Fisher ratio from LDA fitting
    m = re.search(r'Fisher ratio in LDA space: median=([\d.]+), mean=([\d.]+), >1\.5: (\d+), >2\.0: (\d+)', text)
    if m:
        metrics['fisher_median'] = float(m.group(1))
        metrics['fisher_mean'] = float(m.group(2))
        metrics['fisher_gt15'] = int(m.group(3))
        metrics['fisher_gt20'] = int(m.group(4))

    # Version B summary (last occurrence)
    for m in re.finditer(r'Version B: spikes=(\d+), active=(\d+), good=(\d+), global_recall=([\d.]+), good_recall=([\d.]+)', text):
        metrics['pipe_spikes'] = int(m.group(1))
        metrics['active'] = int(m.group(2))
        metrics['good_units'] = int(m.group(3))
        metrics['global_recall'] = float(m.group(4))
        metrics['good_recall'] = float(m.group(5))

    # Detection precision (last occurrence for Version B)
    for m in re.finditer(r'Detection precision @100.m: \d+/\d+ = ([\d.]+)', text):
        metrics['precision_100um'] = float(m.group(1))

    # False positives
    for m in re.finditer(r'False positives: \d+ \(([\d.]+)%\)', text):
        metrics['fp_rate'] = float(m.group(1))

    # Purity and concentration (last occurrence = Version B)
    for m in re.finditer(r'Purity:\s+median=([\d.]+)%', text):
        metrics['purity_median'] = float(m.group(1))
    for m in re.finditer(r'Concentration:\s+median=([\d.]+)%', text):
        metrics['concentration_median'] = float(m.group(1))

    # Label preservation M1 (last occurrence)
    for m in re.finditer(r'Results for M1_recalibrate:\s*\n\s*Label preservation:\s*([\d.]+)%', text):
        metrics['label_pres_m1'] = float(m.group(1))

    # Accuracy from Version B "Our good matched" section (last occurrence)
    for m in re.finditer(r'Accuracy: mean=([\d.]+), median=([\d.]+), >0\.5: (\d+), >0\.8: (\d+)', text):
        metrics['acc_mean'] = float(m.group(1))
        metrics['acc_median'] = float(m.group(2))
        metrics['acc_gt05'] = int(m.group(3))
        metrics['acc_gt08'] = int(m.group(4))

    # Cosine similarity
    for m in re.finditer(r'Cosine similarity \(matched pairs\): mean=([\d.]+), median=([\d.]+)', text):
        metrics['cosine_median'] = float(m.group(2))

    return metrics


# ================================================================
# Run one experiment
# ================================================================
def run_experiment(name, overrides, output_base="results_ibl"):
    """Patch config, run pipeline, return metrics."""
    print(f"\n{'#' * 70}")
    print(f"  SWEEP: {name}")
    print(f"{'#' * 70}")

    # Save originals and patch
    originals = {}
    for key, val in overrides.items():
        originals[key] = getattr(config, key, None)
        setattr(config, key, val)
    print(f"  Config overrides: {overrides}")

    # Ensure common settings
    config.DIAG_M1_ONLY = True
    config.CLUST_ASSIGN_REJECT_ENABLE = overrides.get('CLUST_ASSIGN_REJECT_ENABLE', True)
    config.KS4_CACHE_ENABLE = True

    try:
        from main import run_ibl
        results = run_ibl(output_dir=output_base)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results = None
    finally:
        # Restore originals
        for key, val in originals.items():
            if val is not None:
                setattr(config, key, val)
        # Clean up any leaked TeeLogger from a crashed run_ibl
        from run_logger import TeeLogger
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()  # restores original stdout and closes log file

    # Find the latest log file
    log_files = sorted(glob.glob(os.path.join(output_base, "log_*.txt")),
                       key=os.path.getmtime)
    metrics = {}
    if log_files:
        metrics = parse_log(log_files[-1])
    metrics['name'] = name
    return metrics


# ================================================================
# Summary table
# ================================================================
def print_summary(all_metrics):
    """Print comparison table of all experiments."""
    print(f"\n\n{'=' * 100}")
    print("  LDA SWEEP — COMPARISON SUMMARY")
    print(f"{'=' * 100}")

    cols = [
        ('name', 'Experiment', '{}', 22),
        ('fisher_median', 'Fisher', '{:.3f}', 8),
        ('good_units', 'Good', '{:d}', 6),
        ('good_recall', 'G.Recall', '{:.3f}', 8),
        ('precision_100um', 'Prec@100', '{:.3f}', 8),
        ('fp_rate', 'FP%', '{:.1f}', 6),
        ('purity_median', 'Purity%', '{:.1f}', 8),
        ('concentration_median', 'Conc%', '{:.1f}', 7),
        ('acc_median', 'AccMed', '{:.3f}', 8),
        ('acc_gt08', '>0.8', '{:d}', 5),
        ('label_pres_m1', 'LblPres%', '{:.1f}', 8),
        ('cosine_median', 'CosMed', '{:.3f}', 8),
    ]

    header = ""
    for key, label, fmt, width in cols:
        header += f" {label:>{width}}"
    print(header)
    print("─" * len(header))

    for m in all_metrics:
        row = ""
        for key, label, fmt, width in cols:
            val = m.get(key, None)
            if val is None:
                row += f" {'—':>{width}}"
            else:
                row += f" {fmt.format(val):>{width}}"
        print(row)

    # Highlight best values
    print(f"\n{'─' * len(header)}")
    best_row = ""
    for key, label, fmt, width in cols:
        if key == 'name':
            best_row += f" {'BEST':>{width}}"
            continue
        if key == 'fp_rate':
            # Lower is better
            vals = [(m.get(key), m.get('name')) for m in all_metrics if m.get(key) is not None]
            if vals:
                best_val, best_name = min(vals, key=lambda x: x[0])
                best_row += f" {best_name[:width]:>{width}}"
            else:
                best_row += f" {'—':>{width}}"
        elif key in ('fisher_median', 'good_units', 'good_recall', 'precision_100um',
                     'purity_median', 'concentration_median', 'acc_median', 'acc_gt08',
                     'label_pres_m1', 'cosine_median'):
            # Higher is better
            vals = [(m.get(key), m.get('name')) for m in all_metrics if m.get(key) is not None]
            if vals:
                best_val, best_name = max(vals, key=lambda x: x[0])
                best_row += f" {best_name[:width]:>{width}}"
            else:
                best_row += f" {'—':>{width}}"
        else:
            best_row += f" {'':>{width}}"
    print(best_row)


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LDA sweep")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Run specific experiments by name (default: all)")
    parser.add_argument("--output", default="results_ibl",
                        help="Output directory")
    args = parser.parse_args()

    # Filter experiments if specified
    exps = EXPERIMENTS
    if args.experiments:
        exps = [(n, o) for n, o in EXPERIMENTS if n in args.experiments]
        if not exps:
            print(f"No matching experiments. Available: {[n for n, _ in EXPERIMENTS]}")
            sys.exit(1)

    print(f"Running {len(exps)} experiments:")
    for n, o in exps:
        print(f"  - {n}: {o}")

    all_metrics = []
    t_start = time.time()
    for name, overrides in exps:
        m = run_experiment(name, overrides, output_base=args.output)
        all_metrics.append(m)

    print_summary(all_metrics)
    print(f"\nTotal sweep time: {time.time() - t_start:.0f}s")
