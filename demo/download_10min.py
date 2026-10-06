"""Fetch only the compressed prefix covering [0, 600) s from public IBL S3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from review_common import ROOT, long_path, read_json, sha256, verified_asset, write_json


def fetch_range(url, start, stop, destination, total_size, opener=urlopen):
    """Require byte-range service; never accept an accidental whole-file response."""
    request = Request(url, headers={"Range": f"bytes={start}-{stop - 1}",
                                    "Accept-Encoding": "identity"})
    with opener(request, timeout=90) as response:
        expected = f"bytes {start}-{stop - 1}/{total_size}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("Server did not honor the exact byte range; download stopped.")
        remaining = stop - start
        while remaining:
            block = response.read(min(1024 * 1024, remaining))
            if not block:
                raise IOError("Interrupted range response")
            destination.write(block)
            remaining -= len(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "downloaded/ibl_10min")
    parser.add_argument("--plan", action="store_true", help="Print size and source without downloading")
    args = parser.parse_args()
    descriptor = read_json(verified_asset("data/download_descriptor.json"))
    if args.plan:
        print(json.dumps(descriptor, indent=2))
        return
    output = long_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "recording.ap.cbin"
    part = output / "recording.ap.cbin.part"
    identity = output / "download_identity.json"
    n_bytes = descriptor["prefix_bytes"]
    expected_identity = {k: descriptor[k] for k in ("url", "prefix_bytes", "prefix_sha256")}
    if identity.exists() and read_json(identity) != expected_identity:
        raise RuntimeError("Incompatible existing download. Choose another output directory.")
    if final.exists():
        if final.stat().st_size != n_bytes or sha256(final) != descriptor["prefix_sha256"]:
            raise RuntimeError("Existing final file does not match the pinned prefix.")
    else:
        if part.exists() and not identity.exists():
            raise RuntimeError("Unidentified partial file. Choose another output directory.")
        write_json(identity, expected_identity)
        offset = part.stat().st_size if part.exists() else 0
        if offset > n_bytes:
            raise RuntimeError("Partial file is larger than the requested prefix.")
        if shutil.disk_usage(output).free < n_bytes - offset + 1024**3:
            raise RuntimeError("Not enough disk space for the compressed prefix and 1 GiB headroom.")
        with part.open("ab") as target:
            while offset < n_bytes:
                stop = min(offset + 32 * 1024**2, n_bytes)
                for attempt in range(4):
                    try:
                        fetch_range(descriptor["url"], offset, stop, target,
                                    descriptor["original_file_bytes"])
                        target.flush()
                        break
                    except Exception:
                        target.seek(offset)
                        target.truncate()
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                offset = stop
                print(f"Downloaded {offset / n_bytes:.1%}: {offset:,}/{n_bytes:,} bytes", flush=True)
        if sha256(part) != descriptor["prefix_sha256"]:
            raise RuntimeError("Prefix checksum failed; partial file retained for inspection.")
        part.rename(final)
    for suffix in ("ch", "meta"):
        shutil.copyfile(verified_asset(f"data/prefix.ap.{suffix}"), output / f"recording.ap.{suffix}")
    write_json(output / "provenance.json", descriptor)
    print(f"READY: {final}\nOnly the requested prefix was fetched, not the full recording.")


if __name__ == "__main__":
    main()
