"""
Build the submission zip: source code, README, report and results, without the virtual
environment, the dataset cache or model checkpoints (the assignment asks to leave those out).

Works on any OS (uses Python's zipfile, no `zip` command needed).

Example:
    python make_submission.py                       # writes ../q3_submission.zip
    python make_submission.py --out group7_q3.zip
"""
import argparse
import os
import zipfile
from pathlib import Path

SKIP_DIRS = {".venv", "venv", "data_cache", "__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = {".pt", ".pyc", ".zip"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="../q3_submission.zip")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)   # do not descend into these
        for name in sorted(filenames):
            f = Path(dirpath) / name
            if f.suffix not in SKIP_SUFFIXES and f != out:
                files.append(f)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, Path(root.name) / f.relative_to(root))
    print(f"Wrote {out} ({len(files)} files, {out.stat().st_size / 2 ** 20:.1f} MB)")
    if not (root / "report").is_dir():
        print("Warning: no report/ folder found - add the final report before submitting.")


if __name__ == "__main__":
    main()
