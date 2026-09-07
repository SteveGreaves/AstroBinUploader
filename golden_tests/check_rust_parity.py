#!/usr/bin/env python3
"""
Differential check for the Rust port's Phase 1 parsing.

Produces the same canonical dump from Python (configobj + pandas) that
`astrobin-upload --dump-parity` produces from Rust, and diffs them. This is
what stops the Rust config parser and CSV reader from drifting away from the
libraries they are emulating -- a drift the byte-level golden diff would not
catch until much later phases, if at all.

Usage:
    python3 golden_tests/check_rust_parity.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from configobj import ConfigObj

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CONFIG = REPO_ROOT / "golden_tests" / "golden_config.ini"
FIXTURES = sorted((REPO_ROOT / "golden_tests" / "fixtures").glob("*_raw.csv"))
RUST_BIN = REPO_ROOT / "rust" / "target" / "debug" / "astrobin-upload"

US = "\x1f"  # element separator inside a list value


def python_dump(config_path: Path, csv_path: Path | None) -> list[str]:
    lines = []

    def walk(prefix: str, section) -> None:
        for k, v in section.items():
            if hasattr(v, "items"):  # nested Section
                walk(f"{prefix}/{k}", v)
            elif isinstance(v, list):
                lines.append(f"{prefix}\t{k}\tlist\t{US.join(str(i) for i in v)}")
            else:
                lines.append(f"{prefix}\t{k}\tstr\t{v}")

    cfg = ConfigObj(str(config_path), encoding="utf-8")
    for name, sec in cfg.items():
        walk(name, sec)
    out = sorted(f"CONFIG\t{l}" for l in lines)

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        df.columns = [c.upper() for c in df.columns]
        out.append(f"CSV\trows\t{len(df)}")
        cols = [
            f"CSV\tcol\t{c}\t{df[c].dtype}\t{int(df[c].isna().sum())}"
            for c in df.columns
        ]
        out.extend(sorted(cols))
    return out


def rust_dump(config_path: Path, csv_path: Path | None) -> list[str]:
    cmd = [str(RUST_BIN), str(REPO_ROOT), "--config", str(config_path), "--dump-parity"]
    if csv_path is not None:
        cmd += ["--test", str(csv_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"rust binary exited {proc.returncode}")
    return proc.stdout.splitlines()


def compare(label: str, py: list[str], rs: list[str]) -> bool:
    if py == rs:
        print(f"[PASS] {label}  ({len(py)} lines)")
        return True
    print(f"[FAIL] {label}")
    only_py = [l for l in py if l not in rs]
    only_rs = [l for l in rs if l not in py]
    for l in only_py[:15]:
        print(f"  python only: {l!r}")
    for l in only_rs[:15]:
        print(f"  rust   only: {l!r}")
    if len(only_py) > 15 or len(only_rs) > 15:
        print(f"  ... {len(only_py)} python-only, {len(only_rs)} rust-only in total")
    return False


def main() -> int:
    if not RUST_BIN.exists():
        raise SystemExit(f"build the binary first: cargo build --manifest-path rust/Cargo.toml")

    ok = compare("config: golden_config.ini",
                 python_dump(GOLDEN_CONFIG, None),
                 rust_dump(GOLDEN_CONFIG, None))

    for fx in FIXTURES:
        name = fx.name.removesuffix("_raw.csv")
        ok &= compare(
            f"csv dtypes: {name}",
            python_dump(GOLDEN_CONFIG, fx),
            rust_dump(GOLDEN_CONFIG, fx),
        )

    print("\nall parity checks passed." if ok else "\nPARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
