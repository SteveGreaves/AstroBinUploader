#!/usr/bin/env python3
"""
Golden Test Harness - AstroBin Upload Utility

Replays every committed fixture in golden_tests/fixtures/ through the
pipeline via the --test CSV-injection path, and byte-compares the result
against golden_tests/references/. No filesystem scan, no machine-specific
data, no network access -- this is meant to run anywhere the repo is
checked out.

Fixtures are captured with:
    python3 AstroBinUpload.py <data_dir> --debug
    cp <data_dir>/AstroBinUploadInfo/debug_step_00_RawHeaders.csv \\
       golden_tests/fixtures/<name>_raw.csv

Output filenames (and the embedded acquisition-CSV filename inside the
summary text) are derived from the basename of the original data
directory, which need not match the short reference slug used here (e.g.
data directory "Sadr Region" -> basename "Sadr_Region", reference slug
"sadr"). Where that differs, record it alongside the fixture in
golden_tests/fixtures/<name>.basename (plain text, no trailing newline).
If that sidecar file is absent, the slug itself is used as the basename.

The --test path was verified (see REMEDIATION_PLAN.md P0) to reproduce a
full disk scan byte-for-byte, so replaying the captured CSV is a faithful
substitute for re-scanning the original files.

Usage:
    python3 golden_tests/run_golden.py            # run all fixtures
    python3 golden_tests/run_golden.py sadr        # run one fixture
    python3 golden_tests/run_golden.py --bless     # overwrite references
                                                    # with current output
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REFERENCES_DIR = Path(__file__).resolve().parent / "references"
RUN_OUTPUT_DIR = Path(__file__).resolve().parent / "run_output"

# Only the generation timestamp is expected to vary between runs.
_GENERATED_LINE = re.compile(r"^Generated .*$", re.MULTILINE)


def normalise(text: str) -> str:
    """Strip the one line that legitimately differs run to run."""
    return _GENERATED_LINE.sub("Generated <TIMESTAMP>", text)


def discover_fixtures() -> list[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*_raw.csv"))


def run_fixture(fixture_csv: Path, bless: bool) -> tuple[str, bool, str]:
    """
    Replays one fixture and compares it to its reference.

    Returns (name, passed, detail).
    """
    name = fixture_csv.name.removesuffix("_raw.csv")
    basename_sidecar = FIXTURES_DIR / f"{name}.basename"
    basename = basename_sidecar.read_text(encoding="utf-8").strip() if basename_sidecar.exists() else name

    work_dir = RUN_OUTPUT_DIR / name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # The exporter names output files after the basename of the first
    # directory argument, so the scratch directory itself must carry that
    # original basename for the output to match the reference exactly.
    scenario_dir = work_dir / basename
    scenario_dir.mkdir()
    local_csv = scenario_dir / "raw.csv"
    shutil.copy(fixture_csv, local_csv)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "AstroBinUpload.py"),
         str(scenario_dir), "--test", str(local_csv)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        return name, False, f"pipeline exited {proc.returncode}\n{proc.stderr[-2000:]}"

    out_dir = scenario_dir / "AstroBinUploadInfo"
    summary_path = out_dir / f"{basename}_session_summary.txt"
    csv_path = out_dir / f"{basename}_acquisition.csv"

    ref_summary = REFERENCES_DIR / f"{name}_summary.txt"
    ref_csv = REFERENCES_DIR / f"{name}_acquisition.csv"

    if bless:
        if summary_path.exists():
            shutil.copy(summary_path, ref_summary)
        if csv_path.exists():
            shutil.copy(csv_path, ref_csv)
        return name, True, "blessed"

    problems = []

    if not ref_summary.exists():
        problems.append(f"no reference summary at {ref_summary}")
    elif not summary_path.exists():
        problems.append("pipeline produced no summary file")
    else:
        got = normalise(summary_path.read_text(encoding="utf-8"))
        want = normalise(ref_summary.read_text(encoding="utf-8"))
        if got != want:
            problems.append(_first_diff("summary", want, got))

    if not ref_csv.exists():
        problems.append(f"no reference CSV at {ref_csv}")
    elif not csv_path.exists():
        problems.append("pipeline produced no acquisition CSV")
    else:
        got = csv_path.read_text(encoding="utf-8")
        want = ref_csv.read_text(encoding="utf-8")
        if got != want:
            problems.append(_first_diff("acquisition CSV", want, got))

    if problems:
        return name, False, "\n".join(problems)
    return name, True, "identical"


def _first_diff(label: str, want: str, got: str) -> str:
    want_lines, got_lines = want.splitlines(), got.splitlines()
    for i, (w, g) in enumerate(zip(want_lines, got_lines)):
        if w != g:
            return f"{label} differs at line {i + 1}:\n  reference: {w!r}\n  actual:    {g!r}"
    if len(want_lines) != len(got_lines):
        return f"{label} line count differs: reference={len(want_lines)} actual={len(got_lines)}"
    return f"{label} differs (byte-level only)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="fixture name(s) to run, e.g. 'sadr' (default: all)")
    parser.add_argument("--bless", action="store_true",
                         help="overwrite references with current output instead of comparing")
    args = parser.parse_args()

    fixtures = discover_fixtures()
    if args.names:
        wanted = set(args.names)
        fixtures = [f for f in fixtures if f.name.removesuffix("_raw.csv") in wanted]

    if not fixtures:
        print("No fixtures found in golden_tests/fixtures/.", file=sys.stderr)
        sys.exit(2)

    failures = []
    for fixture in fixtures:
        name, ok, detail = run_fixture(fixture, args.bless)
        status = "BLESSED" if args.bless else ("PASS" if ok else "FAIL")
        print(f"[{status}] {name}")
        if not ok:
            print(f"  {detail}")
            failures.append(name)
        elif args.bless:
            print(f"  {detail}")

    if RUN_OUTPUT_DIR.exists():
        shutil.rmtree(RUN_OUTPUT_DIR)

    if failures and not args.bless:
        print(f"\n{len(failures)}/{len(fixtures)} fixture(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print(f"\n{len(fixtures)}/{len(fixtures)} fixture(s) {'blessed' if args.bless else 'passed'}.")


if __name__ == "__main__":
    main()
