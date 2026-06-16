#!/usr/bin/env python3
"""Apply the Madeline patch to a local DeepSpeed installation.

Usage:
    python tools/apply_deepspeed_patch.py --deepspeed-root /path/to/deepspeed

Or, if DeepSpeed is installed as a package:
    python tools/apply_deepspeed_patch.py

The patch file is located at ``patches/madeline-deepspeed.patch`` relative to
this script's repository root.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_deepspeed_root() -> Path:
    """Try to locate DeepSpeed source root from the current Python env."""
    try:
        import deepspeed
        pkg_dir = Path(deepspeed.__file__).parent
        # Check if it looks like an editable install with source files
        candidate = pkg_dir / "runtime" / "zero" / "stage3.py"
        if candidate.exists():
            return pkg_dir
    except ImportError:
        pass
    raise RuntimeError(
        "Cannot find DeepSpeed installation. Please pass --deepspeed-root explicitly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Madeline patch to DeepSpeed")
    parser.add_argument(
        "--deepspeed-root",
        type=Path,
        default=None,
        help="Root directory of the DeepSpeed source tree (contains deepspeed/)",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=None,
        help="Path to the patch file (default: patches/madeline-deepspeed.patch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying",
    )
    args = parser.parse_args()

    deepspeed_root = args.deepspeed_root or find_deepspeed_root()
    if not deepspeed_root.exists():
        print(f"ERROR: DeepSpeed root does not exist: {deepspeed_root}", file=sys.stderr)
        return 1

    if args.patch is None:
        script_dir = Path(__file__).parent.resolve()
        repo_root = script_dir.parent
        patch_file = repo_root / "patches" / "madeline-deepspeed.patch"
    else:
        patch_file = args.patch

    if not patch_file.exists():
        print(f"ERROR: Patch file not found: {patch_file}", file=sys.stderr)
        return 1

    cmd = [
        "git",
        "-C",
        str(deepspeed_root),
        "apply",
        "--check" if args.dry_run else "",
        str(patch_file),
    ]
    cmd = [c for c in cmd if c]

    print(f"DeepSpeed root : {deepspeed_root}")
    print(f"Patch file     : {patch_file}")
    print(f"Command        : {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Patch application failed:\n{result.stderr}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry-run succeeded — patch can be applied cleanly.")
    else:
        print("Patch applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
