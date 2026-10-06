"""Verify the review package against its per-file manifest."""
from review_common import ROOT, read_json, sha256


def main():
    manifest = read_json(ROOT / "MANIFEST_SHA256.json")
    failures = []
    for name, record in manifest.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            failures.append(name)
    if failures:
        raise SystemExit("Integrity failures:\n" + "\n".join(failures))
    print(f"PASS: {len(manifest)} files match the manifest.")


if __name__ == "__main__":
    main()
