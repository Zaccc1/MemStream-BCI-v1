"""
Data Loader — IBL ONE API
===========================
Setup: pip install ONE-api ibllib spikeglx mtscomp
"""

import os
import re
import shutil
import numpy as np
from pathlib import Path
import config

try:
    from one.api import ONE
    HAS_ONE = True
except ImportError:
    HAS_ONE = False


def connect_one(base_url=None, silent=True):
    """Connect to IBL public database."""
    if not HAS_ONE:
        raise RuntimeError("Run: pip install ONE-api ibllib spikeglx mtscomp")
    url = base_url or config.IBL_BASE_URL
    cache_dir = (os.environ.get("BCI_ONE_CACHE_DIR")
                 or getattr(config, "IBL_CACHE_DIR", "") or "")
    local_only = os.environ.get("BCI_ONE_LOCAL_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    kwargs = ({"mode": "local", "silent": silent} if local_only else {
        "base_url": url, "password": "international", "silent": silent
    })
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        if not cache_path.is_absolute():
            cache_path = Path(__file__).resolve().parent / cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_path)
    try:
        one = ONE(**kwargs)
    except TypeError:
        # Older ONE versions may not accept cache_dir in the constructor.
        kwargs.pop("cache_dir", None)
        one = ONE(**kwargs)
    if local_only:
        print("[IBL] ONE local-only mode")
    else:
        print(f"[IBL] Connected to {url}")
    if cache_dir:
        print(f"[IBL] ONE cache: {cache_path}")
    return one


def find_good_session(one, brain_region="VISp", n_show=5):
    """Search for Neuropixels sessions with raw AP data.
    Returns (eid, probe_label).
    """
    eids = None

    try:
        eids = one.search(dataset_type='ephysData.raw.ap',
                          atlas_acronym=brain_region)
    except (ValueError, TypeError):
        pass

    if not eids:
        try:
            sessions = one.alyx.rest('sessions', 'list',
                                     dataset_types='ephysData.raw.ap',
                                     atlas_acronym=brain_region,
                                     limit=n_show)
            eids = [s['id'] for s in sessions]
        except Exception:
            pass

    if not eids:
        try:
            eids = one.search(dataset_type='ephysData.raw.ap')
        except (ValueError, TypeError):
            try:
                sessions = one.alyx.rest('sessions', 'list',
                                         dataset_types='ephysData.raw.ap',
                                         limit=n_show)
                eids = [s['id'] for s in sessions]
            except Exception as e:
                raise ValueError(f"Cannot search IBL database: {e}")

    if not eids:
        raise ValueError("No raw AP sessions found")

    print(f"[IBL] {len(eids)} sessions found, first {min(n_show, len(eids))}:")
    for i, eid in enumerate(eids[:n_show]):
        try:
            info = one.get_details(eid)
            print(f"  [{i}] {eid}  ({info.get('subject','?')} / {info.get('date','?')})")
        except Exception:
            print(f"  [{i}] {eid}")

    eid = eids[0]
    collections = one.list_collections(eid)
    probes = sorted(set(c.split('/')[1] for c in collections
                        if c.startswith('raw_ephys_data/probe')))
    probe = probes[0] if probes else config.IBL_PROBE_LABEL
    print(f"[IBL] Selected {eid}, probe={probe} (available: {probes})")
    return eid, probe


_UUID_PART_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MISSING_FILE_RE = re.compile(
    r"No such file or directory: ['\"]([^'\"]+)['\"]")


def _repair_one_uuid_cache_miss(exc):
    """Create a UUID-suffixed ALF cache alias from the base local cache file."""
    match = _MISSING_FILE_RE.search(str(exc))
    if not match:
        return False

    raw_path = match.group(1)
    missing = Path(raw_path)
    if not missing.exists() and (chr(92) * 2) in raw_path:
        missing = Path(raw_path.replace(chr(92) * 2, chr(92)))
    if missing.exists():
        return False

    parts = missing.name.split(".")
    if len(parts) < 4 or not _UUID_PART_RE.match(parts[-2]):
        return False

    base_name = ".".join(parts[:-2] + parts[-1:])
    base_path = missing.with_name(base_name)
    if not base_path.exists():
        return False

    try:
        missing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, missing)
        print(f"[IBL] Repaired ONE cache alias: {missing.name} <- {base_path.name}")
        return True
    except OSError as copy_error:
        print(f"[IBL] Could not repair ONE cache alias {missing}: {copy_error}")
        return False

def load_channel_geometry(one, eid, probe_label):
    """Load channel x,y positions (um) and rawInd mapping.

    Returns
    -------
    positions : (n_ch, 2) float32
    raw_ind : (n_sorted_ch,) int or None
    """
    collections_to_try = [
        f'alf/{probe_label}',
        f'alf/{probe_label}/pykilosort',
        f'alf/{probe_label}/iblsorter',
        f'raw_ephys_data/{probe_label}',
    ]

    for coll in collections_to_try:
        channels = None
        for _ in range(21):
            try:
                channels = one.load_object(eid, 'channels', collection=coll)
                break
            except Exception as e:
                if _repair_one_uuid_cache_miss(e):
                    continue
                print(f"[IBL] Channel map not in '{coll}': {e}")
                break
        if channels is None:
            continue

        try:
            if hasattr(channels, 'localCoordinates') and channels.localCoordinates is not None:
                pos_sorted = channels.localCoordinates.astype(np.float32)
                raw_ind = None
                if hasattr(channels, 'rawInd') and channels.rawInd is not None:
                    raw_ind = channels.rawInd.astype(np.int32)
                    n_raw = max(raw_ind.max() + 1, config.N_CHANNELS)
                    pos_raw = np.zeros((n_raw, 2), dtype=np.float32)
                    pos_raw[raw_ind] = pos_sorted[:len(raw_ind)]
                    print(f"[IBL] Channel map from '{coll}': {pos_sorted.shape}, "
                          f"rawInd: {raw_ind.shape} -> "
                          f"positions remapped to raw index ({n_raw} ch)")
                    return pos_raw, raw_ind
                else:
                    print(f"[IBL] Channel map from '{coll}': {pos_sorted.shape} "
                          f"(no rawInd, assuming raw-indexed)")
                    return pos_sorted, None
        except Exception as e:
            print(f"[IBL] Channel map not in '{coll}': {e}")
            continue

    print("[IBL] Using default NP1 geometry (no rawInd)")
    return _default_neuropixels_geometry(), None


def load_channel_regions(one, eid, probe_label):
    """Load brain region acronym for each channel, remapped to raw index.

    Tries three approaches in order:
      1. SpikeSortingLoader
      2. brainLocationIds_ccf_2017 + iblatlas mapping
      3. Direct acronym attribute on channels object

    Returns
    -------
    regions : list of str, length n_channels, or None if unavailable.
    """
    from collections import Counter

    # Approach 1: SpikeSortingLoader
    try:
        from brainbox.io.one import SpikeSortingLoader
        ssl = SpikeSortingLoader(one=one, eid=eid, pname=probe_label)
        _, _, channels = ssl.load_spike_sorting()
        if 'acronym' in channels and channels['acronym'] is not None:
            acronyms = list(channels['acronym'])
            raw_ind = None
            if 'rawInd' in channels and channels['rawInd'] is not None:
                raw_ind = np.asarray(channels['rawInd'], dtype=np.int32)
            regions_raw = _remap_regions(acronyms, raw_ind)
            counts = Counter(regions_raw)
            top = counts.most_common(8)
            print(f"[IBL] Channel regions via SpikeSortingLoader: {len(acronyms)} channels")
            print(f"[IBL] Regions: {', '.join(f'{r}({n})' for r, n in top)}")
            return regions_raw
    except Exception as e:
        print(f"[IBL] SpikeSortingLoader regions failed: {e}")

    # Approach 2: atlas ID -> acronym mapping
    collections_to_try = [
        f'alf/{probe_label}/pykilosort',
        f'alf/{probe_label}',
        f'alf/{probe_label}/iblsorter',
    ]
    for coll in collections_to_try:
        try:
            channels = one.load_object(eid, 'channels', collection=coll)

            if hasattr(channels, 'brainLocationIds_ccf_2017') and \
               channels.brainLocationIds_ccf_2017 is not None:
                atlas_ids = np.asarray(channels.brainLocationIds_ccf_2017)
                try:
                    from iblatlas.regions import BrainRegions
                    br = BrainRegions()
                    acronyms = list(br.id2acronym(atlas_ids))
                    raw_ind = None
                    if hasattr(channels, 'rawInd') and channels.rawInd is not None:
                        raw_ind = channels.rawInd.astype(np.int32)
                    regions_raw = _remap_regions(acronyms, raw_ind)
                    counts = Counter(regions_raw)
                    top = counts.most_common(8)
                    print(f"[IBL] Channel regions via iblatlas from '{coll}': {len(acronyms)} channels")
                    print(f"[IBL] Regions: {', '.join(f'{r}({n})' for r, n in top)}")
                    return regions_raw
                except ImportError:
                    print("[IBL] iblatlas not installed, cannot map atlas IDs to acronyms")
                except Exception as e2:
                    print(f"[IBL] iblatlas mapping failed: {e2}")

            # Direct acronym attribute
            for attr in ('acronym', 'brainLocationAcronyms_ccf_2017'):
                if hasattr(channels, attr) and getattr(channels, attr) is not None:
                    acronyms = list(getattr(channels, attr))
                    raw_ind = None
                    if hasattr(channels, 'rawInd') and channels.rawInd is not None:
                        raw_ind = channels.rawInd.astype(np.int32)
                    regions_raw = _remap_regions(acronyms, raw_ind)
                    counts = Counter(regions_raw)
                    top = counts.most_common(8)
                    print(f"[IBL] Channel regions from '{coll}' ({attr}): {len(acronyms)} channels")
                    print(f"[IBL] Regions: {', '.join(f'{r}({n})' for r, n in top)}")
                    return regions_raw

        except Exception as e:
            print(f"[IBL] Channel regions not in '{coll}': {e}")
            continue

    print("[IBL] WARNING: Could not load channel brain regions")
    return None


def _remap_regions(acronyms, raw_ind):
    """Remap region list from sorted-channel order to raw-channel order."""
    if raw_ind is not None:
        n_raw = max(int(raw_ind.max()) + 1, config.N_CHANNELS)
        regions_raw = ['unknown'] * n_raw
        for i, ri in enumerate(raw_ind):
            if i < len(acronyms):
                regions_raw[int(ri)] = acronyms[i]
        return regions_raw
    else:
        return list(acronyms[:config.N_CHANNELS])


class RawDataStreamer:
    """Stream raw AP data from IBL in chunks. Handles .cbin and .bin."""

    def __init__(self, one, eid, probe_label):
        self.one = one
        self.eid = eid
        self.probe_label = probe_label
        self._reader = None
        self._mmap = None
        self._total_samples = 0
        self._n_channels = config.N_CHANNELS
        self._mode = None

    def open(self):
        collection = f"raw_ephys_data/{self.probe_label}"
        try:
            import spikeglx
            bin_file = self.one.load_dataset(
                self.eid, '_spikeglx_ephysData_g*_t0.imec*.ap.cbin',
                collection=collection, download_only=True)
            for ext in ('ch', 'meta'):
                try:
                    self.one.load_dataset(
                        self.eid,
                        f'_spikeglx_ephysData_g*_t0.imec*.ap.{ext}',
                        collection=collection, download_only=True)
                except Exception:
                    pass
            try:
                self._reader = spikeglx.Reader(bin_file)
            except (AssertionError, Exception):
                nc = self._n_channels + 1
                self._reader = spikeglx.Reader(
                    bin_file, nc=nc, fs=config.SAMPLE_RATE, dtype='int16')
            self._total_samples = self._reader.ns
            self._n_channels = self._reader.nc - 1
            self._mode = 'spikeglx'
        except Exception as e1:
            print(f"[IBL] cbin failed ({e1}), trying .bin...")
            try:
                bin_file = self.one.load_dataset(
                    self.eid, '_spikeglx_ephysData_g*_t0.imec*.ap.bin',
                    collection=collection, download_only=True)
                nc = self._n_channels + 1
                self._mmap = np.memmap(bin_file, dtype='int16', mode='r').reshape(-1, nc)
                self._total_samples = self._mmap.shape[0]
                self._mode = 'mmap'
            except Exception as e2:
                raise RuntimeError(f"Cannot open data:\n  {e1}\n  {e2}")

        dur = self._total_samples / config.SAMPLE_RATE
        print(f"[IBL] {self._total_samples} samples ({dur:.1f}s), "
              f"{self._n_channels} ch, mode={self._mode}")

    def _to_uv(self, raw):
        """Convert raw data to uV."""
        data = raw[:, :self._n_channels].astype(np.float32)
        if self._mode == 'spikeglx':
            data *= 1e6  # V -> uV
        else:
            data *= config.UV_PER_BIT  # int16 -> uV
        return data

    def read_calibration_data(self, duration_s=None, start_s=0.0):
        """Read calibration block. Returns (n_samples, n_ch) float32 in uV."""
        dur = duration_s or config.CALIBRATION_DURATION_S
        start = int(start_s * config.SAMPLE_RATE)
        n = min(int(dur * config.SAMPLE_RATE), self._total_samples - start)
        raw = self._read(start, n)
        data = self._to_uv(raw)
        print(f"[IBL] Calibration: {data.shape} ({n/config.SAMPLE_RATE:.1f}s)")
        return data

    def stream_chunks(self, start_s=None, duration_s=None, chunk_samples=None):
        """Yield (chunk_samples, n_ch) float32 chunks in uV."""
        cs = chunk_samples or config.CHUNK_SAMPLES
        start_time = config.CALIBRATION_DURATION_S if start_s is None else start_s
        start = int(start_time * config.SAMPLE_RATE)
        end = self._total_samples
        if duration_s is not None:
            end = min(start + int(duration_s * config.SAMPLE_RATE), end)
        pos = start
        while pos + cs <= end:
            raw = self._read(pos, cs)
            yield self._to_uv(raw)
            pos += cs

    def _read(self, start, n):
        if self._mode == 'spikeglx':
            return self._reader.read(nsel=slice(start, start + n), sync=False)
        return self._mmap[start:start + n]

    def close(self):
        if self._mode == 'spikeglx' and self._reader:
            self._reader.close()
        self._mmap = None
        print("[IBL] Closed.")

    @property
    def total_duration_s(self):
        return self._total_samples / config.SAMPLE_RATE


# ============================================================
# Default NP1 Geometry
# ============================================================

def _default_neuropixels_geometry(n_channels=None):
    n_ch = n_channels or config.N_CHANNELS
    positions = np.zeros((n_ch, 2), dtype=np.float32)
    for i in range(n_ch):
        row, col = i // 2, i % 2
        positions[i, 0] = (16.0 + col * 32.0) if row % 2 == 0 else (col * 32.0)
        positions[i, 1] = row * 20.0
    return positions
