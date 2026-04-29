#!/usr/bin/env python3
"""Restore the paper-data archive at the repository root.

Downloads (or accepts a local path to) the Zenodo deposit containing the
gitignored paper-data result trees, verifies its SHA256, and extracts it
at the 4th_gen_dh repository root.

Usage
-----
    python scripts/restore_paper_data.py
        download from Zenodo, verify SHA256, extract

    python scripts/restore_paper_data.py /path/to/local.tar.gz
        use a pre-downloaded archive

    python scripts/restore_paper_data.py --no-verify
        skip SHA256 check (not recommended)

After extraction, run::

    pytest -m regression

to confirm all 29 regression tests pass against the paper-aligned baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ZENODO_URL = (
    "https://zenodo.org/records/19894657/files/"
    "4th_gen_dh_paper_data_2026-04-29.tar.gz"
)
EXPECTED_SHA256 = (
    "18d00e3431460e08ba068ef47d9266abe6d8ec051915d519a18bdf2f0a78f01e"
)
ARCHIVE_NAME = "4th_gen_dh_paper_data_2026-04-29.tar.gz"
REPO_MARKERS = ("pyproject.toml", "config.py", "building_analysis")


def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if all((candidate / m).exists() for m in REPO_MARKERS):
            return candidate
    raise SystemExit(
        "Could not locate the 4th_gen_dh repository root.\n"
        f"Looked for markers {REPO_MARKERS} in {cwd} and its parents.\n"
        "Run this script from inside the repository tree."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    print("  (~5.7 GB; expect several minutes)")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1 << 20
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB"
                        f" ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        help=(
            "Local path to the paper-data tarball. "
            "If omitted, downloads from the Zenodo deposit."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the SHA256 integrity check (not recommended).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    print(f"Repository root: {repo_root}")

    if args.archive is not None:
        archive = args.archive.expanduser().resolve()
        if not archive.is_file():
            sys.exit(f"Archive not found: {archive}")
    else:
        archive = repo_root / ARCHIVE_NAME
        if archive.exists():
            print(f"Archive already present at {archive}; skipping download.")
        else:
            download(ZENODO_URL, archive)

    if args.no_verify:
        print("WARNING: skipping SHA256 verification (--no-verify).")
    else:
        print("Verifying SHA256...")
        actual = sha256_file(archive)
        if actual != EXPECTED_SHA256:
            sys.exit(
                "SHA256 mismatch:\n"
                f"  expected {EXPECTED_SHA256}\n"
                f"  actual   {actual}\n"
                "Refusing to extract a corrupted or altered archive."
            )
        print("SHA256 OK.")

    print(f"Extracting at {repo_root}...")
    with tarfile.open(archive, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(path=repo_root, filter="data")
        else:
            tar.extractall(path=repo_root)
    print("Extraction complete.")
    print()
    print("Next: pytest -m regression  # 29 tests should pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
