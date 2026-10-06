# MemStream-BCI

Software for streaming Neuropixels spike sorting and neural decoding with
configurable full-precision and software 8-bit computation.

The processing pipeline comprises common-average referencing, FIR high-pass
filtering, local ZCA whitening, template matching, spatial aggregation,
temporal PCA, regional LDA, cluster assignment, merging and unit-quality checks.
IBL analysis supports choice, feedback, wheel velocity and wheel speed decoding.
A Qt GUI provides configuration, data preview, execution and result inspection.

## Contents

| Directory or file | Purpose |
| --- | --- |
| `source/` | General pipeline, GUI, tests and software export routines |
| `source/benchmarks/ibl/` | IBL sorting, reference comparison and decoding entry points |
| `source/configs/ibl/` | Per-recording FP/SW8 parameter examples |
| `demo/` | Demo runner, data downloader and integrity checks |
| `data/` | IBL sample, fitted calibration models, KS4 teacher and input metadata |
| `expected/` | Expected arrays and summaries for the bundled demo |
| `requirements-demo-lock.txt` | Exact demo dependency versions |
| `DEMO_ENVIRONMENT.json` | Demo Python and dependency information |
| `MANIFEST_SHA256.json` | Package file checksums |

## Requirements

The tested demo configuration is **Windows 11, 64-bit Python 3.12.3**, using the
dependency versions in `requirements-demo-lock.txt`. Use this environment with
the supplied serialized models.

The default demo runs on CPU. Recommended resources are 8 GB RAM and 2 GB free
disk for the package, environment and small outputs.
Optional KS4 execution requires PyTorch; a supported CUDA GPU can accelerate it.

Extract to a short path such as `C:\BCI` to avoid Windows path-length errors.

## Installation

Open a terminal in the package root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo-lock.txt
python demo/verify_package.py
```

If PowerShell activation is restricted, use `.venv\Scripts\python.exe` directly
instead of changing the execution policy. Dependencies are downloaded from PyPI.
Environment creation and installation took approximately 1.4 minutes on the
test workstation with possible pip-cache reuse. Network and hardware affect this
time.

Additional source entry points require the relevant extras:

```text
python -m pip install "./source[gui,ibl,ks4,analysis,export]"
```

This additional stack is separate from the minimal demo environment.

## Offline Demo

The package contains two seconds of real IBL AP data, from 240 to 242 s, with
384 channels at 30 kHz. FP and SW8 each use their own calibration model fitted
on 0-240 s. No raw-data download or new KS4 execution is needed for this demo.

```text
python demo/run_demo.py --mode fp --output outputs/demo_fp --check-expected
python demo/run_demo.py --mode sw8 --output outputs/demo_sw8 --check-expected
```

Use a new output directory for each run. The demo refuses to overwrite results.
Expected spike counts are **3,056 (FP)** and **2,952 (SW8)**; exact array checks
use the included reference outputs. Typical observed times on the test
workstation were approximately 8 s and 17 s respectively for the CPU software
demo, including model loading and post-processing.

Key outputs are:

- `DEMO_COMPLETE.txt`: successful completion marker.
- `demo_summary.json`: settings, counts, elapsed time and output checksums.
- `reference_check.json`: agreement with the included expected arrays.
- `clustering/spike_times.npy`: sample indices relative to the streaming interval,
  corrected for FIR delay.
- `clustering/spike_times_ap_seconds.npy`: AP-clock seconds, calculated as
  `sample_index / 30000 + 240`; use AP-to-behavior synchronization for event alignment.
- `clustering/spike_clusters.npy` and other sorting arrays.
- Cluster/merge summaries, a drift plot and a timestamped log.

The demo uses 100-ms chunks, fixed calibration-derived quantization ranges and
disabled center updates to process the supplied two-second interval.
Calibration model files use Python pickle and can execute code; load only
checksum-verified files from a trusted distribution.

## Ten-Minute Example

IBL EID: `9b5a1754-ac99-4d53-97d3-35c2f6638507`, `probe00`.

```text
python demo/download_10min.py --plan
python demo/download_10min.py
python demo/run_demo.py --mode fp --input downloaded/ibl_10min/recording.ap.cbin --output outputs/tenmin_fp
python demo/run_demo.py --mode sw8 --input downloaded/ibl_10min/recording.ap.cbin --output outputs/tenmin_sw8
```

The downloader retrieves approximately 5.12 GB of compressed public data with
bounded HTTP byte ranges, resumes compatible partial downloads and verifies the
complete prefix checksum. The input remains compressed; expansion to int16
including sync would require approximately 13.86 GB. Allow the download size plus
2 GB for cached-model inference and outputs.

The interval contains 0-240 s calibration data and 240-600 s streaming data.
The commands reuse the supplied models and process the latter 360 s. Estimated
runtime is 20-30 minutes, excluding download, installation and calibration
fitting. Actual elapsed time is recorded in the run summary.

To fit calibration again using the supplied 240-s KS4 teacher:

```text
python demo/run_demo.py --mode sw8 --input downloaded/ibl_10min/recording.ap.cbin --recalibrate --output outputs/refit_sw8
```

This option does not start a new KS4 job and fails if the teacher cannot be
reused. Allow at least 32 GB RAM and 50 GB temporary disk headroom, then monitor
actual use. Calibration fitting adds runtime to the streaming estimate above.

## Your Own Recording

Do not use the bundled fitted models on a different recording. Calibrate on
your input using the general or IBL entry point and its matching configuration.
The two entry points are separate implementations; do not mix their modules.

For IBL sorting and analysis:

```text
cd source/benchmarks/ibl
python main.py --help
python main.py --source ibl --eid YOUR_EID --probe probe00 --output YOUR_NEW_OUTPUT
python benchmark_recording_metrics.py --help
python 09_decode_wheel_movement.py --help
```

Use `--code-dir . --clean-root .` and actual input/output paths for standalone
wheel analysis. `benchmark_recording_metrics.py` calls the wheel functions
directly. Replace machine-specific paths in parameter examples before use.
Reference jobs can download substantial data and create large temporary files;
inspect `KS4_FULL_ENABLE` and CLI options before starting.

For the GUI, run `python source/run_gui.py` from the package root. Select data
and a fresh output directory in Config, inspect Preview if useful, start in Run,
and examine logs, arrays and plots in Results. Stop terminates the worker
subprocess. Pipeline errors are written to the output's `crash_report.txt`.

Local Open Ephys recordings need the AP binary and `structure.oebin` metadata
with sample rate, channel count and voltage conversion. From `source/`, run:

```text
python main.py --source local --dir YOUR_OPEN_EPHYS_DIRECTORY --output YOUR_NEW_OUTPUT
```

Local settings include `LOCAL_RECORDING_DIR`, `LOCAL_AP_FOLDER`,
`LOCAL_KS_SETTINGS_PATH`, optional `LOCAL_REFERENCE_PATH`, and `LOCAL_TTL_DIR`.
Supply the correct probe geometry. Check `LOCAL_PREPROCESSING_MODE` to avoid
repeating CAR/FIR when the input has already been processed.

## Quantization and Decoding

The software 8-bit path uses unsigned affine activation codes, signed weight
codes and int32 VMM accumulation. Calibration selects fixed ranges using a
99.99% central percentile by default. Per-recording configuration controls the
actual settings. `QUANT_RANGE_STAGE_OVERRIDES` or `QUANT_RANGE_OVERRIDE_PATH`
can set stage-specific percentiles or numeric ranges. `--calibration-only`
writes range diagnostics without starting the streaming phase.

Choice classification uses -100 to 0 ms around first wheel movement, with
fallback event fields when movement timing is unavailable. Feedback uses 0 to
200 ms after feedback onset. Check the selected entry point and configuration
for the exact alignment. Behavior analysis requires valid event labels and
AP-to-behavior clock synchronization. SW8 describes sorting-stage precision;
downstream classifiers and regressors can use floating-point arithmetic.

ACG-pass selects units using the pipeline's autocorrelogram-based quality criteria.

## License

Software is provided under the [MIT License](LICENSE). IBL example data use
[CC BY 4.0, with required attribution](DATA_LICENSE.md). Third-party dependencies
retain their own licenses.
