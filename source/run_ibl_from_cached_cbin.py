"""Run the normal IBL pipeline while opening a known cached AP cbin directly."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--raw-cbin", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--eid", required=True)
    parser.add_argument("--probe", default="probe00")
    parser.add_argument("--region", default="VISp")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for required in (args.raw_cbin, args.positions, args.config_json):
        if not required.exists():
            raise FileNotFoundError(f"required cached input is missing: {required}")

    os.chdir(args.worktree)
    sys.path.insert(0, str(args.worktree))

    import spikeglx
    import data_loader

    positions = np.load(args.positions, allow_pickle=False)

    def open_cached(streamer: data_loader.RawDataStreamer) -> None:
        streamer._reader = spikeglx.Reader(args.raw_cbin)
        streamer._total_samples = streamer._reader.ns
        streamer._n_channels = streamer._reader.nc - 1
        streamer._mode = "spikeglx"
        print(
            "[CACHED-IBL] Opened AP cbin directly: "
            f"{args.raw_cbin} ({streamer.total_duration_s:.1f}s)",
            flush=True,
        )

    data_loader.RawDataStreamer.open = open_cached

    import main as pipeline_main

    pipeline_main.load_channel_geometry = lambda one, eid, probe: (positions, None)
    pipeline_main.load_channel_regions = lambda one, eid, probe: None

    return pipeline_main._main_cli(
        [
            "--source",
            "ibl",
            "--eid",
            args.eid,
            "--probe",
            args.probe,
            "--region",
            args.region,
            "--config-json",
            str(args.config_json),
            "--output",
            str(args.output),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
