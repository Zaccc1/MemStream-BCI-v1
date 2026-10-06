"""
Local Data Loader — Open Ephys Binary Format
===============================================
Loads Neuropixels recordings from local Open Ephys directories,
providing the same interface as data_loader.RawDataStreamer.
"""

import json
import os
import numpy as np
from pathlib import Path
import config


# ==============================================================
# OEBIN Parsing
# ==============================================================

def parse_oebin(recording_dir):
    """Parse structure.oebin to extract AP stream metadata.

    Parameters
    ----------
    recording_dir : str
        Path to the recording directory containing structure.oebin.

    Returns
    -------
    dict with keys: sample_rate, num_channels, bit_volts, folder_name,
                    source_processor_name, oebin_path, ap_data_dir
    """
    oebin_path = _find_oebin(recording_dir)
    if oebin_path is None:
        raise FileNotFoundError(
            f"No structure.oebin found in {recording_dir}")

    with open(oebin_path, 'r') as f:
        data = json.load(f)

    rec_root = os.path.dirname(oebin_path)

    # Find the AP continuous stream (Neuropixels AP). Do not select by
    # channel count alone first: NP1 LFP streams can also expose 384 channels.
    streams = data.get('continuous', [])
    ap_stream = None
    for stream in streams:
        folder = stream.get('folder_name', '')
        if 'AP' in folder.upper():
            ap_stream = stream
            break

    if ap_stream is None:
        candidates = []
        for stream in streams:
            sr = float(stream.get('sample_rate', 0) or 0)
            n_ch = int(stream.get('num_channels', 0) or 0)
            if sr >= 20_000 and n_ch >= 300:
                candidates.append(stream)
        if len(candidates) == 1:
            ap_stream = candidates[0]
            print("[LOCAL] WARNING: AP stream inferred from high sample rate "
                  f"({ap_stream.get('sample_rate')} Hz) and "
                  f"{ap_stream.get('num_channels')} channels.")
        elif len(candidates) > 1:
            available = [s.get('folder_name', '<unknown>') for s in candidates]
            raise ValueError(
                f"Multiple AP-like streams found in {oebin_path}: {available}. "
                "Set LOCAL_AP_FOLDER explicitly.")

    if ap_stream is None:
        raise ValueError(
            f"No AP stream found in {oebin_path}. "
            f"Available: {[s['folder_name'] for s in data.get('continuous', [])]}")

    folder_name = ap_stream['folder_name'].rstrip('/')
    bit_volts = ap_stream['channels'][0]['bit_volts']

    # Detect preprocessing chain from channel history
    history = ap_stream['channels'][0].get('history', '')
    has_car = 'common avg ref' in history.lower() or 'car' in history.lower()
    has_bandpass = 'bandpass' in history.lower() or 'highpass' in history.lower()

    return {
        'sample_rate': ap_stream['sample_rate'],
        'num_channels': ap_stream['num_channels'],
        'bit_volts': bit_volts,
        'folder_name': folder_name,
        'source_processor_name': ap_stream.get('source_processor_name', ''),
        'oebin_path': oebin_path,
        'ap_data_dir': os.path.join(rec_root, 'continuous', folder_name),
        'preprocessing_history': history,
        'has_car': has_car,
        'has_bandpass': has_bandpass,
    }


def _find_oebin(recording_dir):
    """Recursively search for structure.oebin in recording_dir."""
    for root, dirs, files in os.walk(recording_dir):
        if 'structure.oebin' in files:
            return os.path.join(root, 'structure.oebin')
    return None


# ==============================================================
# Channel Geometry
# ==============================================================

def load_local_channel_geometry(recording_dir, ks_settings_path=None):
    """Load channel positions from local Kilosort settings or default geometry.

    Parameters
    ----------
    recording_dir : str
        Path to the recording directory.
    ks_settings_path : str or None
        Path to continuous_ksSettings.mat. If None, searches in the AP
        data directory automatically.

    Returns
    -------
    positions : (n_ch, 2) float32 — x,y coordinates in micrometers
    raw_ind : (n_sorted_ch,) int32 or None — sorted→raw channel mapping
    """
    # Auto-discover ks_settings if not provided
    if not ks_settings_path:
        ks_settings_path = _find_ks_settings(recording_dir)

    if ks_settings_path and os.path.exists(ks_settings_path):
        return _load_geometry_from_ks_settings(ks_settings_path)

    print("[LOCAL] No ksSettings found, using default NP1 geometry")
    from data_loader import _default_neuropixels_geometry
    return _default_neuropixels_geometry(), None


def _find_ks_settings(recording_dir):
    """Search for continuous_ksSettings.mat in recording_dir."""
    for root, dirs, files in os.walk(recording_dir):
        for f in files:
            if f == 'continuous_ksSettings.mat':
                return os.path.join(root, f)
    return None


def _load_geometry_from_ks_settings(ks_settings_path):
    """Extract xcoords, ycoords, chanMap from a Kilosort settings .mat file.

    Handles both MATLAB v5 (scipy.io.loadmat) and v7.3 (h5py) formats.
    """
    try:
        return _load_geometry_h5py(ks_settings_path)
    except Exception:
        pass
    try:
        return _load_geometry_scipy(ks_settings_path)
    except Exception as e:
        print(f"[LOCAL] Failed to load geometry from {ks_settings_path}: {e}")
        from data_loader import _default_neuropixels_geometry
        return _default_neuropixels_geometry(), None


def _load_geometry_h5py(ks_settings_path):
    """Load from MATLAB v7.3 (HDF5) format."""
    import h5py
    with h5py.File(ks_settings_path, 'r') as f:
        cm = f['saveDat/ops/chanMap']
        xcoords = cm['xcoords'][:].flatten().astype(np.float32)
        ycoords = cm['ycoords'][:].flatten().astype(np.float32)
        chan_map = cm['chanMap'][:].flatten().astype(np.int32)

    n_sorted = len(xcoords)
    # chanMap is 1-indexed in MATLAB → convert to 0-indexed
    raw_ind = (chan_map - 1).astype(np.int32)

    n_raw = max(int(raw_ind.max()) + 1, config.N_CHANNELS)
    positions = np.zeros((n_raw, 2), dtype=np.float32)
    positions[raw_ind, 0] = xcoords
    positions[raw_ind, 1] = ycoords

    print(f"[LOCAL] Channel geometry from ksSettings (h5py): "
          f"{n_sorted} sorted channels → {n_raw} raw positions")
    print(f"[LOCAL] x range: [{xcoords.min():.0f}, {xcoords.max():.0f}] μm, "
          f"y range: [{ycoords.min():.0f}, {ycoords.max():.0f}] μm")
    return positions, raw_ind


def _load_geometry_scipy(ks_settings_path):
    """Load from MATLAB v5 format."""
    import scipy.io
    mat = scipy.io.loadmat(ks_settings_path, squeeze_me=True)

    # Top-level xcoords/ycoords (standard chanMap .mat files)
    if 'xcoords' in mat:
        xcoords = np.atleast_1d(mat['xcoords']).flatten().astype(np.float32)
        ycoords = np.atleast_1d(mat['ycoords']).flatten().astype(np.float32)
        chan_map = np.atleast_1d(mat['chanMap']).flatten().astype(np.int32)
    else:
        raise KeyError("No xcoords/ycoords found in mat file")

    n_sorted = len(xcoords)
    raw_ind = (chan_map - 1).astype(np.int32)

    n_raw = max(int(raw_ind.max()) + 1, config.N_CHANNELS)
    positions = np.zeros((n_raw, 2), dtype=np.float32)
    positions[raw_ind, 0] = xcoords
    positions[raw_ind, 1] = ycoords

    print(f"[LOCAL] Channel geometry from ksSettings (scipy): "
          f"{n_sorted} sorted channels → {n_raw} raw positions")
    return positions, raw_ind


# ==============================================================
# Local Data Streamer
# ==============================================================

class LocalDataStreamer:
    """Stream raw AP data from local Open Ephys continuous.dat file.

    Provides the same interface as data_loader.RawDataStreamer.
    """

    def __init__(self, recording_dir, ap_folder=None):
        self._recording_dir = recording_dir
        self._ap_folder = ap_folder
        self._mmap = None
        self._total_samples = 0
        self._n_channels = config.N_CHANNELS
        self._bit_volts = None
        self._sample_rate = None
        # Preprocessing state detected from oebin history
        self._has_car = False
        self._has_bandpass = False
        self._preprocessing_history = ''

    def open(self):
        """Parse oebin and memory-map the continuous.dat file."""
        meta = parse_oebin(self._recording_dir)
        self._n_channels = meta['num_channels']
        self._bit_volts = meta['bit_volts']
        self._sample_rate = meta['sample_rate']
        self._has_car = meta.get('has_car', False)
        self._has_bandpass = meta.get('has_bandpass', False)
        self._preprocessing_history = meta.get('preprocessing_history', '')

        ap_dir = meta['ap_data_dir']
        if self._ap_folder:
            # Override auto-detected AP folder
            rec_root = os.path.dirname(meta['oebin_path'])
            ap_dir = os.path.join(rec_root, 'continuous', self._ap_folder)

        dat_path = os.path.join(ap_dir, 'continuous.dat')
        if not os.path.exists(dat_path):
            raise FileNotFoundError(f"continuous.dat not found at {dat_path}")

        file_size = os.path.getsize(dat_path)
        expected_samples = file_size // (self._n_channels * 2)  # int16 = 2 bytes
        self._mmap = np.memmap(dat_path, dtype='int16', mode='r').reshape(
            -1, self._n_channels)
        self._total_samples = self._mmap.shape[0]

        dur = self._total_samples / self._sample_rate
        print(f"[LOCAL] Opened {dat_path}")
        print(f"[LOCAL] {self._total_samples} samples ({dur:.1f}s), "
              f"{self._n_channels} ch, bit_volts={self._bit_volts}")

    def _ensure_open(self):
        if self._mmap is None:
            raise RuntimeError("LocalDataStreamer: call open() before "
                               "reading data")

    def _to_uv(self, raw):
        """Convert raw int16 data to microvolts using oebin bit_volts."""
        data = raw[:, :self._n_channels].astype(np.float32)
        data *= self._bit_volts
        return data

    def read_calibration_data(self, duration_s=None, start_s=0.0):
        """Read calibration block. Returns (n_samples, n_ch) float32 in uV."""
        self._ensure_open()
        dur = duration_s or config.CALIBRATION_DURATION_S
        sr = self._sample_rate
        start = int(start_s * sr)
        n = min(int(dur * sr), self._total_samples - start)
        raw = self._mmap[start:start + n]
        data = self._to_uv(raw)
        print(f"[LOCAL] Calibration: {data.shape} ({n / sr:.1f}s)")
        return data

    def stream_chunks(self, start_s=None, duration_s=None, chunk_samples=None):
        """Yield (chunk_samples, n_ch) float32 chunks in uV."""
        self._ensure_open()
        cs = chunk_samples or config.CHUNK_SAMPLES
        sr = self._sample_rate
        start_time = config.CALIBRATION_DURATION_S if start_s is None else start_s
        start = int(start_time * sr)
        end = self._total_samples
        if duration_s is not None:
            end = min(start + int(duration_s * sr), end)
        pos = start
        while pos + cs <= end:
            raw = self._mmap[pos:pos + cs]
            yield self._to_uv(raw)
            pos += cs

    def _read(self, start, n):
        """Read raw int16 data from memory-mapped file."""
        self._ensure_open()
        return self._mmap[start:start + n]

    def close(self):
        if self._mmap is not None:
            try:
                mmap_obj = getattr(self._mmap, '_mmap', None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except Exception:
                pass
            self._mmap = None
        print("[LOCAL] Closed.")

    @property
    def total_duration_s(self):
        return self._total_samples / (self._sample_rate or config.SAMPLE_RATE)


# ==============================================================
# Local Reference (Kilosort results from st0.mat)
# ==============================================================

def load_local_reference(st0_path, sample_rate=None):
    """Load existing Kilosort spike sorting results from st0.mat.

    Parameters
    ----------
    st0_path : str
        Path to st0.mat file.
    sample_rate : float or None
        Sample rate for time conversion. Uses config.SAMPLE_RATE if None.

    Returns
    -------
    ref : dict compatible with comparison.py's reference format, or None.
    """
    sr = sample_rate or config.SAMPLE_RATE
    if not st0_path or not os.path.exists(st0_path):
        print(f"[LOCAL] Reference not found: {st0_path}")
        return None

    st0 = _load_mat_variable(st0_path, 'st0')
    if st0 is None:
        return None

    # st0 columns: [sample_index, cluster_id, amplitude, ...]
    # Kilosort versions vary but first two columns are always time and cluster
    spike_samples = st0[:, 0].astype(np.float64)
    spike_clusters = st0[:, 1].astype(np.int32)

    # Convert sample indices to seconds
    spike_times = spike_samples / sr

    # Build cluster info
    unique_clusters = np.unique(spike_clusters)
    n_clusters = len(unique_clusters)
    max_cid = int(unique_clusters.max()) + 1

    # Arrays indexed by cluster ID (not ordinal) for compatibility with
    # comparison.py which does ref['cluster_labels'][cluster_id]
    spike_channels = np.zeros_like(spike_clusters)
    cluster_channels = np.zeros(max_cid, dtype=np.int32)

    # All clusters treated as "good" (label=1) since no quality info in st0
    cluster_labels = np.zeros(max_cid, dtype=np.int32)
    cluster_labels[unique_clusters] = 1

    print(f"[LOCAL] Reference from st0.mat: {len(spike_times)} spikes, "
          f"{n_clusters} clusters, time range [{spike_times.min():.1f}, "
          f"{spike_times.max():.1f}]s")

    return {
        'spike_times': spike_times,
        'spike_clusters': spike_clusters,
        'spike_channels': spike_channels,
        'cluster_channels': cluster_channels,
        'cluster_labels': cluster_labels,
        'cluster_ids': unique_clusters,
        'n_clusters': n_clusters,
        'n_good': n_clusters,
        'sorter': 'kilosort_local',
    }


def _load_mat_variable(mat_path, var_name):
    """Load a variable from a .mat file (v5 or v7.3)."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        if var_name in mat:
            return np.asarray(mat[var_name])
    except NotImplementedError:
        pass

    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            if var_name in f:
                return f[var_name][:].T  # h5py stores MATLAB arrays transposed
    except Exception:
        pass

    print(f"[LOCAL] Could not load '{var_name}' from {mat_path}")
    return None


# ==============================================================
# TTL Events
# ==============================================================

def load_ttl_events(events_dir):
    """Load TTL event data from an Open Ephys events directory.

    Parameters
    ----------
    events_dir : str
        Path to the TTL events directory, e.g.
        .../events/NI-DAQmx-101.PXIe-6341/TTL/

    Returns
    -------
    dict with event data and derived trial structure, or None.
    """
    if not events_dir or not os.path.isdir(events_dir):
        print(f"[LOCAL] TTL events directory not found: {events_dir}")
        return None

    # Load raw TTL data
    try:
        sample_numbers = np.load(os.path.join(events_dir, 'sample_numbers.npy'))
        states = np.load(os.path.join(events_dir, 'states.npy'))
        timestamps = np.load(os.path.join(events_dir, 'timestamps.npy'))
    except FileNotFoundError as e:
        print(f"[LOCAL] Missing TTL file: {e}")
        return None

    full_words = None
    fw_path = os.path.join(events_dir, 'full_words.npy')
    if os.path.exists(fw_path):
        full_words = np.load(fw_path)

    print(f"[LOCAL] TTL events: {len(states)} events, "
          f"channels used: {sorted(set(np.abs(states)))}")
    print(f"[LOCAL] Time range: [{timestamps.min():.2f}, {timestamps.max():.2f}]s")

    # states: positive = rising edge on channel N, negative = falling edge
    # Derive trial structure from rising edges on channel 1
    # (This is a default; users may need to customize for their protocol)
    rising_ch1 = timestamps[states == 1]  # rising edges on channel 1

    if len(rising_ch1) < 2:
        print("[LOCAL] WARNING: fewer than 2 rising edges on TTL channel 1. "
              "TTL-to-trial mapping may need customization for your protocol.")
        return {
            'event_times': timestamps.astype(np.float64),
            'event_states': states,
            'event_words': full_words,
            'n_trials': 0,
            'stim_on_times': np.array([]),
            'choice_times': np.array([]),
            'choice': np.array([], dtype=np.int32),
            'feedbackType': np.array([], dtype=np.int32),
            'feedback_times': np.array([]),
            'valid': np.array([], dtype=bool),
        }

    # Use rising edges on ch1 as trial start times
    stim_on = rising_ch1.astype(np.float64)
    n_trials = len(stim_on)

    # Placeholder trial labels — user must customize based on protocol
    choice = np.ones(n_trials, dtype=np.int32)
    feedback_type = np.ones(n_trials, dtype=np.int32)

    # Estimate feedback times from falling edges or next trial
    falling_ch1 = timestamps[states == -1].astype(np.float64)
    feedback_times = np.full(n_trials, np.nan)
    for i, t in enumerate(stim_on):
        later = falling_ch1[falling_ch1 > t]
        if len(later) > 0:
            feedback_times[i] = later[0]

    valid = ~np.isnan(feedback_times)

    print(f"[LOCAL] Derived {n_trials} trials from TTL ch1 rising edges, "
          f"{valid.sum()} with feedback times")

    return {
        'event_times': timestamps.astype(np.float64),
        'event_states': states,
        'event_words': full_words,
        'stim_on_times': stim_on,
        'choice_times': np.full(n_trials, np.nan, dtype=np.float64),
        'choice': choice,
        'feedbackType': feedback_type,
        'feedback_times': feedback_times,
        'valid': valid,
        'n_trials': n_trials,
    }


def find_ttl_dir(recording_dir):
    """Auto-discover the NI-DAQmx TTL events directory."""
    for root, dirs, files in os.walk(recording_dir):
        if 'TTL' in dirs and 'NI-DAQmx' in root:
            ttl_path = os.path.join(root, 'TTL')
            if os.path.exists(os.path.join(ttl_path, 'states.npy')):
                return ttl_path
    return None
