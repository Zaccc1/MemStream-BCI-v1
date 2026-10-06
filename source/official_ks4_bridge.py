
"""Official KS4 calibration bridge.

This module runs the *official* Kilosort4 calibration pipeline (via the Python
package `kilosort`, or a flat official source tree if configured), then maps
its spike labels back into this project's streaming 60d feature space.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from scipy.spatial.distance import cdist

import config


_REQUIRED_FLAT_SOURCE = [
    'parameters.py', 'utils.py', 'CCG.py', 'swarmsplitter.py', 'hierarchical.py',
    'postprocessing.py', 'preprocessing.py', 'io.py', 'datashift.py',
    'spikedetect.py', 'template_matching.py', 'clustering_qr.py', 'run_kilosort.py',
]


def _load_module(fullname: str, path: Path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot create spec for {fullname} from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_flat_source_as_package(source_dir: Path):
    source_dir = Path(source_dir)
    missing = [f for f in _REQUIRED_FLAT_SOURCE if not (source_dir / f).exists()]
    if missing:
        raise RuntimeError(
            'KS4 flat source bootstrap requested, but files are missing: ' + ', '.join(missing)
        )

    pkg = types.ModuleType('kilosort')
    pkg.__file__ = str(source_dir / '__init__.py')
    pkg.__path__ = [str(source_dir)]
    sys.modules['kilosort'] = pkg

    order = [
        'parameters', 'utils', 'CCG', 'swarmsplitter', 'hierarchical',
        'postprocessing', 'preprocessing', 'io', 'datashift',
        'spikedetect', 'template_matching', 'clustering_qr', 'run_kilosort'
    ]
    for name in order:
        mod = _load_module(f'kilosort.{name}', source_dir / f'{name}.py')
        setattr(pkg, name, mod)

    # Common package-level exports used by official modules
    if hasattr(pkg.parameters, 'DEFAULT_SETTINGS'):
        pkg.DEFAULT_SETTINGS = pkg.parameters.DEFAULT_SETTINGS
    if hasattr(pkg.utils, 'PROBE_DIR'):
        pkg.PROBE_DIR = pkg.utils.PROBE_DIR
    if hasattr(pkg.utils, 'DOWNLOADS_DIR'):
        pkg.DOWNLOADS_DIR = pkg.utils.DOWNLOADS_DIR

    return pkg


def import_official_kilosort():
    """Import the official `kilosort` package.

    Import order:
      1. Installed Python package (`pip install kilosort`)
      2. Optional flat source directory (if KS4_IMPORT_MODE permits)
    """
    mode = getattr(config, 'KS4_IMPORT_MODE', 'package')
    source_dir = getattr(config, 'KS4_SOURCE_DIR', None)

    try:
        pkg = importlib.import_module('kilosort')
        rk = importlib.import_module('kilosort.run_kilosort')
        pp = importlib.import_module('kilosort.postprocessing')
        return pkg, rk, pp
    except Exception as e_pkg:
        if mode not in ('package_or_source', 'source'):
            raise RuntimeError(
                'Official KS4 backend requested, but `kilosort` could not be '
                'imported as a package. Install it with `pip install kilosort` '
                'and ensure its dependencies are available.'
            ) from e_pkg
        if source_dir is None:
            raise RuntimeError(
                'Official KS4 backend requested in source fallback mode, but '
                '`config.KS4_SOURCE_DIR` is not set.'
            ) from e_pkg
        try:
            pkg = _bootstrap_flat_source_as_package(Path(source_dir))
            rk = importlib.import_module('kilosort.run_kilosort')
            pp = importlib.import_module('kilosort.postprocessing')
            return pkg, rk, pp
        except Exception as e_src:
            raise RuntimeError(
                'Failed to import official KS4 from both installed package and '
                'flat source tree. If using a flat source tree, ensure official '
                'KS4 dependencies are installed (notably faiss, probeinterface, '
                'spikeinterface, torch, sklearn, numba).'
            ) from e_src


def make_probe_dict(channel_positions: np.ndarray) -> Dict[str, np.ndarray]:
    n_ch = channel_positions.shape[0]
    return {
        'chanMap': np.arange(n_ch, dtype=np.int32),
        'xc': channel_positions[:, 0].astype(np.float32),
        'yc': channel_positions[:, 1].astype(np.float32),
        'kcoords': np.zeros(n_ch, dtype=np.float32),
        'n_chan': n_ch,
    }


def write_int16_binary(raw_data_uv: np.ndarray, path: Path, uv_per_bit: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_i16 = np.clip(np.round(raw_data_uv / uv_per_bit), -32768, 32767).astype(np.int16)
    raw_i16.tofile(path)
    return path


def _pick_nearest_channels(spike_xy: np.ndarray, channel_positions: np.ndarray) -> np.ndarray:
    if len(spike_xy) == 0:
        return np.zeros(0, dtype=np.int32)
    d = cdist(spike_xy.astype(np.float32), channel_positions.astype(np.float32))
    return np.argmin(d, axis=1).astype(np.int32)


def _build_ks4_settings(raw_data: np.ndarray) -> Dict[str, Any]:
    duration_s = raw_data.shape[0] / float(config.SAMPLE_RATE)
    tmax = getattr(config, 'KS4_TMAX', None)
    if tmax is None:
        tmax = duration_s
    settings = {
        'n_chan_bin': int(raw_data.shape[1]),
        'fs': float(config.SAMPLE_RATE),
        'batch_size': int(getattr(config, 'KS4_BATCH_SIZE', 60000)),
        'nblocks': int(getattr(config, 'KS4_NBLOCKS', 1)),
        'nt': int(getattr(config, 'KS4_NT', 61)),
        'n_pcs': int(getattr(config, 'KS4_N_PCS', 6)),
        'Th_universal': float(getattr(config, 'KS4_TH_UNIVERSAL', 9.0)),
        'Th_learned': float(getattr(config, 'KS4_TH_LEARNED', 8.0)),
        'tmin': float(getattr(config, 'KS4_TMIN', 0.0)),
        'tmax': float(tmax),
        'artifact_threshold': float(getattr(config, 'KS4_ARTIFACT_THRESHOLD', np.inf)),
        'templates_from_data': True,
    }
    return settings


def resolve_ks4_device(device_preference: str | None):
    """Resolve 'auto' / explicit KS4 device preferences into a torch.device."""
    try:
        import torch
    except Exception:
        return None

    requested = str(device_preference or 'auto').strip().lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError, ValueError):
        print(f"[KS4] Invalid device '{device_preference}', falling back to CPU")
        device = torch.device('cpu')

    if device.type == 'cuda' and not torch.cuda.is_available():
        print('[KS4] CUDA not available, falling back to CPU')
        device = torch.device('cpu')

    return device


def _ks4_thread_limit() -> int | None:
    try:
        limit = int(getattr(config, 'KS4_THREAD_LIMIT', 1))
    except (TypeError, ValueError):
        limit = 1
    return limit if limit > 0 else None


def _ks4_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    limit = _ks4_thread_limit()
    if limit is not None:
        for name in (
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ):
            env[name] = str(limit)
    # On macOS, torch/faiss/sklearn wheels may bring separate OpenMP runtimes.
    # Keep the risky combination contained in the child and bias it to one worker.
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env.setdefault('KMP_INIT_AT_FORK', 'FALSE')
    env.setdefault('KMP_WARNINGS', '0')
    return env


def _run_official_ks4_on_binary(
    bin_path: Path,
    settings: Dict[str, Any],
    probe: Dict[str, np.ndarray],
    channel_positions: np.ndarray,
    ks4_dir: Path,
    device_preference: str | None,
    do_car: bool,
    invert_sign: bool,
    save_extra_vars: bool,
    keep_binary: bool,
    fir_group_delay: int,
    torch_thread_lim: int | None,
) -> Dict[str, Any]:
    """Run official KS4 in the current process on an already-written binary."""
    kilosort_pkg, rk_mod, pp_mod = import_official_kilosort()
    run_kilosort = rk_mod.run_kilosort
    compute_spike_positions = pp_mod.compute_spike_positions
    remove_duplicates = pp_mod.remove_duplicates

    device = resolve_ks4_device(device_preference)
    print(f'[KS4] Device: {device}')
    if torch_thread_lim is not None:
        print(f'[KS4] Torch/OpenMP thread limit: {torch_thread_lim}')

    kwargs = dict(
        settings=settings,
        probe=probe,
        filename=bin_path,
        results_dir=ks4_dir,
        data_dtype='int16',
        do_CAR=do_car,
        invert_sign=invert_sign,
        device=device,
        save_extra_vars=save_extra_vars,
    )
    if torch_thread_lim is not None:
        kwargs['torch_thread_lim'] = int(torch_thread_lim)

    ks4_out = run_kilosort(**kwargs)

    # Official Kilosort package APIs differ slightly by version.
    # Newer releases return 9 values, adding `kept_spikes`.
    if not isinstance(ks4_out, (tuple, list)):
        raise ValueError(f'Unexpected run_kilosort return type: {type(ks4_out)!r}')
    if len(ks4_out) == 9:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = ks4_out
    elif len(ks4_out) == 8:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = ks4_out
        kept_spikes = None
    else:
        raise ValueError(f'Unexpected run_kilosort return length: {len(ks4_out)}')

    st = np.asarray(st)
    clu = np.asarray(clu)

    def _slice_like(x, idx):
        if x is None:
            return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                if isinstance(idx, np.ndarray):
                    if idx.dtype == bool:
                        idx_t = torch.as_tensor(idx, dtype=torch.bool, device=x.device)
                    else:
                        idx_t = torch.as_tensor(idx, dtype=torch.long, device=x.device)
                else:
                    idx_t = idx
                return x[idx_t]
        except Exception:
            pass
        return x[idx]

    # Keep tF in its native type (official KS4 typically returns a torch.Tensor).
    # compute_spike_positions() expects torch-like tF, so do not coerce to numpy here.

    # Prefer official kept_spikes mask when available.
    if kept_spikes is not None:
        kept_spikes = np.asarray(kept_spikes, dtype=bool)
        if kept_spikes.ndim != 1 or kept_spikes.shape[0] != clu.shape[0]:
            raise ValueError(
                f'Invalid kept_spikes shape {kept_spikes.shape}; expected ({clu.shape[0]},)'
            )
        st_kept0 = st[kept_spikes]
        clu_kept0 = clu[kept_spikes]
        if tF is not None and len(tF) == len(clu):
            tF_kept0 = _slice_like(tF, kept_spikes)
        else:
            tF_kept0 = tF
    else:
        st_kept0 = st
        clu_kept0 = clu
        tF_kept0 = tF

    # Match official saved-output convention by removing duplicate same-cluster spikes.
    spike_times_raw0 = st_kept0[:, 0].astype(np.int64)
    spike_labels_raw0 = np.asarray(clu_kept0, dtype=np.int32)
    dup_dt = int(ops['settings'].get('duplicate_spike_bins', 15)) if isinstance(ops, dict) else 15
    spike_times_raw, spike_labels_raw, keep = remove_duplicates(spike_times_raw0, spike_labels_raw0, dt=dup_dt)
    st_kept = st_kept0[keep]
    if tF_kept0 is not None and len(tF_kept0) == len(st_kept0):
        tF_kept = _slice_like(tF_kept0, keep)
    else:
        tF_kept = tF_kept0

    # Some package versions may return tF as numpy; official compute_spike_positions
    # expects torch.Tensor, so convert only if needed.
    try:
        import torch
        if tF_kept is not None and not isinstance(tF_kept, torch.Tensor):
            tF_kept = torch.as_tensor(tF_kept)
    except Exception:
        pass

    xs, ys = compute_spike_positions(st_kept, tF_kept, ops)
    spike_xy = np.column_stack([xs, ys]).astype(np.float32)
    spike_channels = _pick_nearest_channels(spike_xy, channel_positions)

    # Convert to this pipeline's internal whitened-data time base.
    spike_times_internal = spike_times_raw + int(fir_group_delay)

    n_units = int(np.unique(spike_labels_raw).size) if len(spike_labels_raw) > 0 else 0
    n_good = int(np.asarray(is_ref).sum()) if is_ref is not None else 0
    print(f'[KS4] Final official output: {len(spike_times_internal)} spikes, {n_units} clusters, good={n_good}')

    if not keep_binary:
        try:
            os.remove(bin_path)
        except OSError:
            pass

    return {
        'backend': 'official_ks4',
        'all_times': spike_times_internal.astype(np.int64),
        'all_times_raw': spike_times_raw.astype(np.int64),
        'all_channels': spike_channels.astype(np.int32),
        'all_labels': spike_labels_raw.astype(np.int32),
        'spike_xy': spike_xy,
        'ks4_results_dir': str(ks4_dir),
        'ops': ops,
        'st': st_kept,
        'clu': spike_labels_raw.astype(np.int32),
        'tF': tF_kept,
        'Wall': Wall,
        'similar_templates': similar_templates,
        'is_ref': np.asarray(is_ref) if is_ref is not None else None,
        'est_contam_rate': np.asarray(est_contam_rate) if est_contam_rate is not None else None,
        'summary': {
            'n_spikes': int(len(spike_times_internal)),
            'n_clusters': int(n_units),
            'n_good_units': int(n_good),
        },
    }


def _save_subprocess_result(path: Path, result: Dict[str, Any]) -> None:
    save_dict = {
        'all_times': np.asarray(result['all_times'], dtype=np.int64),
        'all_times_raw': np.asarray(result['all_times_raw'], dtype=np.int64),
        'all_channels': np.asarray(result['all_channels'], dtype=np.int32),
        'all_labels': np.asarray(result['all_labels'], dtype=np.int32),
        '_backend': np.array([str(result.get('backend', 'official_ks4'))]),
        '_ks4_results_dir': np.array([str(result.get('ks4_results_dir', ''))]),
        '_summary_json': np.array([json.dumps(result.get('summary', {}))]),
    }
    if result.get('spike_xy') is not None:
        save_dict['spike_xy'] = np.asarray(result['spike_xy'], dtype=np.float32)
    if result.get('is_ref') is not None:
        save_dict['is_ref'] = np.asarray(result['is_ref'])
    if result.get('est_contam_rate') is not None:
        save_dict['est_contam_rate'] = np.asarray(result['est_contam_rate'])
    np.savez_compressed(path, **save_dict)


def _load_subprocess_result(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result = {
            'backend': str(data['_backend'][0]) if '_backend' in data else 'official_ks4_subprocess',
            'all_times': data['all_times'].astype(np.int64),
            'all_times_raw': data['all_times_raw'].astype(np.int64),
            'all_channels': data['all_channels'].astype(np.int32),
            'all_labels': data['all_labels'].astype(np.int32),
        }
        if 'spike_xy' in data:
            result['spike_xy'] = data['spike_xy'].astype(np.float32)
        if 'is_ref' in data:
            result['is_ref'] = data['is_ref']
        if 'est_contam_rate' in data:
            result['est_contam_rate'] = data['est_contam_rate']
        if '_ks4_results_dir' in data:
            result['ks4_results_dir'] = str(data['_ks4_results_dir'][0])
        if '_summary_json' in data:
            result['summary'] = json.loads(str(data['_summary_json'][0]))
        else:
            result['summary'] = {
                'n_spikes': int(len(result['all_times'])),
                'n_clusters': int(np.unique(result['all_labels']).size),
            }
    return result


def _run_official_ks4_subprocess(
    bin_path: Path,
    settings: Dict[str, Any],
    channel_positions: np.ndarray,
    ks4_dir: Path,
) -> Dict[str, Any]:
    request_path = ks4_dir / 'ks4_subprocess_request.json'
    result_path = ks4_dir / 'ks4_subprocess_result.npz'
    if result_path.exists():
        result_path.unlink()

    thread_limit = _ks4_thread_limit()
    request = {
        'bin_path': str(bin_path),
        'results_dir': str(ks4_dir),
        'result_path': str(result_path),
        'settings': settings,
        'channel_positions': channel_positions.astype(np.float32).tolist(),
        'device_preference': getattr(config, 'KS4_DEVICE', 'auto'),
        'do_car': bool(getattr(config, 'KS4_DO_CAR', True)),
        'invert_sign': bool(getattr(config, 'KS4_INVERT_SIGN', False)),
        'save_extra_vars': bool(getattr(config, 'KS4_SAVE_EXTRA_VARS', False)),
        'keep_binary': bool(getattr(config, 'KS4_KEEP_CALI_BINARY', True)),
        'fir_group_delay': int(getattr(config, 'FIR_GROUP_DELAY', 0)),
        'torch_thread_lim': thread_limit,
        'config_overrides': {
            'KS4_IMPORT_MODE': getattr(config, 'KS4_IMPORT_MODE', 'package'),
            'KS4_SOURCE_DIR': getattr(config, 'KS4_SOURCE_DIR', None),
        },
    }
    with open(request_path, 'w', encoding='utf-8') as f:
        json.dump(request, f)

    cmd = [sys.executable, str(Path(__file__).resolve()), '--ks4-worker', str(request_path)]
    print(f'[KS4] Subprocess isolation: enabled')
    print(f'[KS4] Worker command: {cmd[0]} {Path(cmd[1]).name} --ks4-worker ...')

    timeout_s = float(getattr(config, 'KS4_SUBPROCESS_TIMEOUT_S', 0.0) or 0.0)
    timeout = timeout_s if timeout_s > 0 else None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_ks4_subprocess_env(),
        cwd=str(Path(__file__).resolve().parent),
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        output, _ = proc.communicate()
        if output:
            print(output, end='' if output.endswith('\n') else '\n')
        raise RuntimeError(
            f'Official KS4 subprocess timed out after {timeout_s:.1f}s'
        ) from exc

    if output:
        print(output, end='' if output.endswith('\n') else '\n')

    if proc.returncode != 0:
        if proc.returncode < 0:
            reason = f'signal {-proc.returncode}'
        else:
            reason = f'exit code {proc.returncode}'
        raise RuntimeError(
            f'Official KS4 subprocess failed with {reason}. '
            f'Last KS4 log: {ks4_dir / "kilosort4.log"}'
        )
    if not result_path.exists():
        raise RuntimeError(
            f'Official KS4 subprocess finished but did not write {result_path}'
        )

    result = _load_subprocess_result(result_path)
    result['backend'] = 'official_ks4_subprocess'
    return result


def run_official_ks4_calibration(raw_data: np.ndarray,
                                 channel_positions: np.ndarray,
                                 output_dir: str | os.PathLike) -> Dict[str, Any]:
    """Run official KS4 on calibration raw data and return spike labels.

    Returned times are converted into this pipeline's *internal* time base by
    adding FIR group delay, because downstream 60d feature extraction is done
    on this project's whitened data (which is FIR-delayed until final output
    correction in `get_results()`).
    """
    out_root = Path(output_dir)
    ks4_dir = out_root / getattr(config, 'KS4_RESULTS_SUBDIR', 'official_ks4_cali')
    ks4_dir.mkdir(parents=True, exist_ok=True)
    bin_path = ks4_dir / 'calibration_ap.bin'

    write_int16_binary(raw_data, bin_path, uv_per_bit=float(config.UV_PER_BIT))
    probe = make_probe_dict(channel_positions)
    settings = _build_ks4_settings(raw_data)

    print('\n' + '=' * 60)
    print('  OFFICIAL KS4 CALIBRATION')
    print('=' * 60)
    print(f'[KS4] Binary: {bin_path}')
    print(f'[KS4] Duration: {raw_data.shape[0] / config.SAMPLE_RATE:.1f}s, channels={raw_data.shape[1]}')

    if bool(getattr(config, 'KS4_RUN_IN_SUBPROCESS', True)):
        return _run_official_ks4_subprocess(
            bin_path=bin_path,
            settings=settings,
            channel_positions=channel_positions,
            ks4_dir=ks4_dir,
        )

    return _run_official_ks4_on_binary(
        bin_path=bin_path,
        settings=settings,
        probe=probe,
        channel_positions=channel_positions,
        ks4_dir=ks4_dir,
        device_preference=getattr(config, 'KS4_DEVICE', 'auto'),
        do_car=bool(getattr(config, 'KS4_DO_CAR', True)),
        invert_sign=bool(getattr(config, 'KS4_INVERT_SIGN', False)),
        save_extra_vars=bool(getattr(config, 'KS4_SAVE_EXTRA_VARS', False)),
        keep_binary=bool(getattr(config, 'KS4_KEEP_CALI_BINARY', True)),
        fir_group_delay=int(getattr(config, 'FIR_GROUP_DELAY', 0)),
        torch_thread_lim=_ks4_thread_limit(),
    )


def _run_ks4_worker(request_path: str | os.PathLike) -> int:
    try:
        with open(request_path, 'r', encoding='utf-8') as f:
            request = json.load(f)

        for key, value in request.get('config_overrides', {}).items():
            setattr(config, key, value)

        channel_positions = np.asarray(
            request['channel_positions'], dtype=np.float32)
        probe = make_probe_dict(channel_positions)
        result = _run_official_ks4_on_binary(
            bin_path=Path(request['bin_path']),
            settings=request['settings'],
            probe=probe,
            channel_positions=channel_positions,
            ks4_dir=Path(request['results_dir']),
            device_preference=request.get('device_preference', 'auto'),
            do_car=bool(request.get('do_car', True)),
            invert_sign=bool(request.get('invert_sign', False)),
            save_extra_vars=bool(request.get('save_extra_vars', False)),
            keep_binary=bool(request.get('keep_binary', True)),
            fir_group_delay=int(request.get('fir_group_delay', 0)),
            torch_thread_lim=request.get('torch_thread_lim', None),
        )
        _save_subprocess_result(Path(request['result_path']), result)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--ks4-worker':
        raise SystemExit(_run_ks4_worker(sys.argv[2]))
    raise SystemExit('Usage: official_ks4_bridge.py --ks4-worker REQUEST_JSON')
