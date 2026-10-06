"""
BCI Neural Processing Pipeline Configuration
======================================
Single-value parameters. Results appended to results_log.xlsx after each run.
"""

# Data Source Selection
DATA_SOURCE = "ibl"  # "ibl" or "local"
DEBUG_VERBOSE = False

# IBL Data Source
IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"
IBL_DATASET_TYPE = "ephysData.raw.ap"
IBL_SESSION_EID = ""
IBL_PROBE_LABEL = "probe00"
# Keep downloaded ONE/IBL files inside this worktree for repeatable benchmarks.
# Can be overridden with environment variable BCI_ONE_CACHE_DIR.
IBL_CACHE_DIR = "recordings/one_cache"

# Local Data Source (Open Ephys)
LOCAL_RECORDING_DIR = ""
LOCAL_AP_FOLDER = ""              # auto-detected from oebin if empty
LOCAL_KS_SETTINGS_PATH = ""       # path to continuous_ksSettings.mat
LOCAL_REFERENCE_PATH = ""         # path to st0.mat for comparison (optional)
LOCAL_TTL_DIR = ""                # path to TTL events directory (optional)
LOCAL_SKIP_BRAIN_REGIONS = True
# Local preprocessing policy:
# "auto"      -> skip CAR/FIR when Open Ephys metadata says they were applied
# "force_raw" -> always run this pipeline's CAR and FIR, even if metadata says applied
# "skip_all"  -> always skip this pipeline's CAR and FIR
LOCAL_PREPROCESSING_MODE = "auto"

# Raw Data Properties
N_CHANNELS = 384
SAMPLE_RATE = 30_000
DTYPE_RAW = "int16"
UV_PER_BIT = 2.34375

# Calibration
CALIBRATION_DURATION_S = 240.0
CALI_CACHE_ENABLE = True
CALI_CACHE_SUBDIR = "cali_cache"
CALI_CACHE_VERSION = 1

# CAR
CAR_METHOD = "median"

# FIR High-Pass Filter
FIR_NUM_TAPS = 257
FIR_HIGHPASS_FREQ = 300.0
FIR_WINDOW = "hamming"
# Group delay of a linear-phase FIR: (N-1)/2 samples.
# Spike times are corrected by this offset in get_results().
FIR_GROUP_DELAY = (FIR_NUM_TAPS - 1) // 2  # 128 samples = 4.27ms

# Whitening (ZCA, local)
WHITEN_MODE = "overlap"
WHITEN_NRANGE = 32
WHITEN_OVERLAP = 8
WHITEN_EPSILON = 1e-6

# AD/DA Converter Precision (signal quantization at crossbar boundaries)
PRECISION_ADC = 8
PRECISION_DAC = 8

# Quantization backend
# "fake": existing float32 fake quantization (round -> dequantized float)
# "int_accum": signed integer input/weight codes with int32 VMM accumulation
# "uint8_int_accum": uint8 input codes + signed weights + int32 accumulation
QUANTIZATION_BACKEND = "uint8_int_accum"
QUANT_DIAGNOSTICS = True
QUANT_DIAG_MAX_PRINTS_PER_OP = 3
QUANT_CLIP_WARN_FRAC = 0.01

# Fixed quantization ranges for uint8_int_accum are collected during calibration.
# The percentile is central mass, so 99.0 means [0.5, 99.5] percentiles.
QUANT_RANGE_PERCENTILE = 99.99
QUANT_RANGE_SAMPLE_LIMIT = 200_000
QUANT_RANGE_SAMPLE_PER_UPDATE = 4096
QUANT_RANGE_SYMMETRIC = False
QUANT_CALIBRATION_SUBDIR = "quant_calibration"
# Stage-boundary ranges must be shared by the producer ADC and consumer DAC.
# Each pair is (producer_adc_stage, consumer_input_stage).
QUANT_RANGE_TIED_STAGE_PAIRS = [
    ("FIR.adc", "WHITEN.input"),
    ("WHITEN.adc", "MATCH.input"),
    ("MATCH.adc", "SA.sparse.input"),
    ("MATCH.adc", "SA.dense.input"),
]
# After fixed uint8 ranges are frozen, replay calibration once more and fit
# templates/LDA/centers/radii in the same fixed-range signal path used later.
QUANT_REFIT_AFTER_RANGE_FREEZE = True
# Optional per-stage overrides. Keys are stage names from quant_ranges.json
# such as "FIR.input", "FIR.adc", "WHITEN.input", "WHITEN.adc",
# "MATCH.input", "MATCH.adc", "SA.sparse.input", "ASSIGN.input".
# Supported forms:
#   {"FIR.input": {"percentile": 99.5}}
#   {"WHITEN.adc": {"min": -6.0, "max": 6.0}}
#   {"MATCH.*": {"range": [-12.0, 12.0]}}
QUANT_RANGE_STAGE_OVERRIDES = {}
# Optional JSON file containing either the override mapping directly or a
# quant_ranges.json-style {"ranges": {...}} object. Inline overrides win.
QUANT_RANGE_OVERRIDE_PATH = ""

# Per-stage precision (bits) for memristor simulation.
# Activations are uint8 at stage boundaries; weights are signed int8.
PRECISION_FIR = 8
PRECISION_WHITEN = 8
PRECISION_MATCH = 8
PRECISION_SA = 8
PRECISION_PCA = 8
PRECISION_ASSIGN = 8
PRECISION_LDA = 8

# Real-Time Chunk
CHUNK_DURATION_MS = 100.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000.0)  # 3000
MAX_TEST_BATCHES = 0

# Hardware golden-batch export. This is a main-pipeline mode, like
# calibration-only: run calibration, export one real post-cali batch, then stop.
HARDWARE_EXPORT_ENABLE = False
HARDWARE_EXPORT_SUBDIR = "hardware_export"
HARDWARE_EXPORT_BATCH_INDEX = 0
HARDWARE_EXPORT_STAGES = [
    "raw", "car", "fir", "whitening", "template_matching",
    "spatial_aggregation", "tpca", "lda", "assignment",
]
HARDWARE_EXPORT_MAX_VMM_ROWS = 20000
HARDWARE_EXPORT_ONNX = True
HARDWARE_EXPORT_PLOTS = True
HARDWARE_EXPORT_SPLIT_MODELS = True
HARDWARE_EXPORT_PLOT_MAX_POINTS = 50000

# Template Matching
MATCH_NT = 61
MATCH_N_TEMPLATES = 6
MATCH_STRIDE = 1
MATCH_THRESHOLD = 10.0
MATCH_NMS_SAMPLES = 20
MATCH_TH_SINGLE_CH = 8.0

# Template extraction from calibration data
TEMPLATE_FROM_DATA = True
TEMPLATE_TH_DETECT = 6.0
TEMPLATE_ISOLATION_T = 30
TEMPLATE_ISOLATION_CH = 6
TEMPLATE_MAX_CLIPS = 50000

# Calibration data usage
N_CALIB_CHUNKS = 2400

# Spatial Aggregation
SA_NEAREST_CHANNELS = 10
SA_N_SCALES = 5
SA_SIGMA_UM = [10.0, 20.0, 30.0, 40.0, 50.0]
SA_COARSE_THRESHOLD = 8.0
SA_FINAL_THRESHOLD = 10.0
SA_NMS_SAMPLES = 20
SA_NMS_CHANNELS = 5
SA_MODE = "gaussian"
SA_DENSE = False

# Clustering
CLUST_N_NEARBY = 10
CLUST_WAVEFORM_COLLECT_N_NEARBY = CLUST_N_NEARBY
CLUST_SNIPPET_HW = 30
CLUST_MIN_COUNT = 5

# Temporal PCA
CLUST_TEMPORAL_PCA_NC = 6
CLUST_KMEANS_CLEAN_K = 50
CLUST_KMEANS_CLEAN_MIN = 10
CLUST_WAVEFORM_MAX_CLIPS = 50000

# LDA Feature Projection
# Mode: 'off'        = pure PCA 60d, no LDA
#        'post_pca'  = PCA(6x61) -> 60d -> LDA(Nx60) -> Nd
#        'direct'    = raw waveform(n_ch x 61) -> LDA -> Nd
#        'per_region' = per-region PCA 60d -> region-specific LDA Nd
CLUST_LDA_MODE = 'per_region'
CLUST_LDA_N_COMPONENTS = 20
CLUST_LDA_MIN_SAMPLES_PER_CLASS = 10
CLUST_LDA_DIRECT_N_CHANNELS = 5

# Alignment diagnostic
DIAG_M1_ONLY = True

# Waveform cleaning KMeans
KMEANS_CLEAN_MAX_ITER = 20
KMEANS_CONVERGENCE_TOL = 1e-4

# Regional Clustering
CLUST_REGION_SIZE = 48
CLUST_REGION_OVERLAP = 12

# Streaming assignment gate / center update
CLUST_ASSIGN_REJECT_ENABLE = True
CLUST_ASSIGN_REJECT_MIN_SAMPLES = 20
CLUST_ASSIGN_REJECT_QUANTILE = 0.99
CLUST_ASSIGN_REJECT_SCALE = 1.10
CLUST_ASSIGN_MARGIN_MIN = 1.1

CLUST_ENABLE_CENTER_UPDATES = False
CLUST_UPDATE_HIGHCONF_ONLY = True
CLUST_UPDATE_QUANTILE = 0.90
CLUST_UPDATE_SCALE = 1.00
CLUST_UPDATE_MARGIN_MIN = 1.10

# Cluster Merging
MERGE_SIMILARITY_THRESHOLD = 0.5
MERGE_ACG_THRESHOLD = 0.20
MERGE_CCG_THRESHOLD = 0.25
MERGE_SPATIAL_SIGMA = 100.0
MERGE_MIN_SPIKES = 20

# CCG / ACG parameters
CCG_TBIN = 0.001
CCG_NBINS = 500
CCG_R12_ACG_THRESHOLD = 0.2
CCG_R12_CCG_THRESHOLD = 0.05

# Visualization
VIS_CHANNELS = [0, 50, 100, 191, 200, 300, 383]
VIS_DURATION_MS = 50.0

# Diagnostics
DIAG_DETECTION_AUDIT = True
DIAG_FN_ANALYSIS = True
DIAG_FEATURE_SEPARABILITY = True
DIAG_LABEL_ANALYSIS = True

# Reproducibility
RANDOM_SEED = 42

# ==============================================================
# Official KS4 calibration backend
# ==============================================================
KS4_IMPORT_MODE = 'package_or_source'
KS4_SOURCE_DIR = None
KS4_RESULTS_SUBDIR = 'official_ks4_cali'
KS4_KEEP_CALI_BINARY = True
KS4_SAVE_EXTRA_VARS = False
KS4_DEVICE = 'auto'
KS4_RUN_IN_SUBPROCESS = True
KS4_THREAD_LIMIT = 1
KS4_SUBPROCESS_TIMEOUT_S = 0.0  # 0 disables timeout
KS4_BATCH_SIZE = 60_000
KS4_NBLOCKS = 1
KS4_NT = 61
KS4_N_PCS = 6
KS4_TH_UNIVERSAL = 10.0
KS4_TH_LEARNED = 8.0
KS4_TMIN = 0.0
KS4_TMAX = None
KS4_ARTIFACT_THRESHOLD = float('inf')
KS4_DO_CAR = True
KS4_INVERT_SIGN = False

# Streaming center match
CALI_CENTER_MATCH_MAX_DT_SAMPLES = 3

# ==============================================================
# KS4 Result Caching
# ==============================================================
KS4_CACHE_ENABLE = True
KS4_CACHE_SUBDIR = 'ks4_cache'

# ==============================================================
# Decoding
# ==============================================================
DECODE_VAL_DURATION_S = 120.0
DECODE_ACC_THRESHOLD = 0.8
# IBL-aligned decoding windows (see ibl-task.png):
#   choice   = first movement aligned, -100 ms -> 0 ms
#   feedback = feedback onset aligned, 0 -> +200 ms
DECODE_CHOICE_WINDOW = (-0.1, 0.0)
DECODE_FEEDBACK_WINDOW = (0.0, 0.2)
DECODE_N_SPLITS = 5
DECODE_N_SHUFFLE = 200
DECODE_QUANT_BITS = [2, 4, 6, 8]

# ==============================================================
# KS4 Full-Recording Reference (evaluation only)
# ==============================================================
KS4_FULL_ENABLE = False
KS4_FULL_CACHE_SUBDIR = 'ks4_cache_full'
KS4_FULL_DEVICE = 'auto'
KS4_FULL_KEEP_BINARY = False
