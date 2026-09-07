## 🎯 Objective

Full remediation of AstroBinUpload.py v2.0.3 (a FITS/XISF metadata pipeline
for AstroBin bulk uploads) per an audit that produced two plan documents:
`REMEDIATION_PLAN.md` (bug fixes) and `RUST_PORT_PLAN.md` (feasibility study
for a future standalone Rust port, not started). Work happened on branch
`remediation/v2.1.0`. The branch is now pushed to `origin` and **PR #12**
(<https://github.com/SteveGreaves/AstroBinUploader/pull/12>) is open against
`main`, awaiting review/merge. The `v2.1.0` tag is still **local-only** — by
explicit user instruction it is not pushed until PR #12 merges. `main` is untouched.

The repo was also relocated this session, from `/mnt/raid0/AgentCode_old/AstroBinUpload`
(misleading historical name) to `/mnt/raid0/Agent_Code/Astronomy/AstroBinUploader`
(matches the GitHub repo name). **This is now the canonical location** — the
old directory still exists on disk but should be treated as stale.

## ✅ Completed Work

**Everything in the remediation plan is done and tagged `v2.1.0`** (annotated
git tag, local-only, points at commit `c25cd4c`). 34 commits on the branch.
Summary by phase — see `REMEDIATION_PLAN.md` for full detail on every item:

- **P0** — Built a golden regression harness (`golden_tests/run_golden.py`)
  and committed a fixture corpus (previously the whole `golden_tests/`
  directory was gitignored). `tests/test_imports.py` fixed and pytest
  installed into the project venv (`/mnt/raid0/Code/venvs/.astrovenv`) for
  the first time.
- **Bucket A (A1–A14, 14 items)** — output-changing defects, each its own
  commit with an attributed diff. Notably: A13 and A14 were **found live**
  during validation against real data, not in the original code-reading
  audit. About half of the originally-suspected bugs (A5, A7's HISTORY
  claim, A8) turned out, on rigorous re-verification against the real
  pipeline, to describe a real mechanism but not a currently-reachable bug —
  fixed defensively anyway, and the plan doc corrected rather than left
  overstated. A10/A11 required a user decision (master-preference semantics)
  which was obtained and implemented.
- **Bucket B (B1–B15, 15 items)** — hygiene/robustness, batched into 6
  commits, each verified with an **empty** golden diff (no behavior change).
  Includes: pyflakes-verified dead-code removal, deprecated
  `groupby(axis=1)` replaced with a dtype-safe alternative (the
  pandas-suggested fix `frame.T.groupby(...)` was tested and found unsafe —
  it silently corrupts dtypes on this codebase's mixed-type frames), worker
  logging fixed for spawn-based platforms (macOS/Windows), path validation
  added, named magic-number constants, requirements.txt trimmed to actual
  deps.
- **A14 (found after initial "complete" declaration)** — while validating
  against the user's real *unprocessed* calibration data (see below), found
  that `engine/reports.py`'s calibration report table was grouping
  darks/bias by filter, when darks/bias are physically filter-independent
  (mirrors `calibration.py`'s own matching, which never constrained on
  filter). Fixed, verified live.
- **Real-data validation** — this session used three real datasets beyond
  the two committed fixtures, at the user's request, specifically because
  the corpus originally had zero calibration-frame coverage:
  1. `/home/steve/Desktop/Pixinsight/SH2 101` — a single directory with
     pre-built master calibration frames (found A13 here).
  2. `/mnt/raid0/AstroImaging/Preselected/SH2 101` (lights) +
     `/mnt/raid0/AstroImaging/Preselected/Calibration data/9th August 2026`
     (calibration) — the user's real, **unprocessed** raw calibration subs.
     This is the dataset that exercises genuine calibration matching end to
     end for the first time, and is what caught A14. Committed to the
     corpus as `golden_tests/fixtures/sh2101_calib_raw.csv` (1693 frames,
     header metadata only, no pixel data) with blessed references.
- User also ran the tool independently against both real directories this
  session and confirmed it works. One user-side hiccup (not a code bug):
  running from `/home/steve` without `cd`-ing into the project directory
  caused `--config config.ini`'s relative path to resolve against the wrong
  CWD, auto-generating a stray `/home/steve/config.ini`. Explained; the file
  may still be sitting there — user was offered to have it deleted, no
  answer captured before this handoff.

**Golden harness status**: `golden_tests/run_golden.py` — 2/2 fixtures pass
(`sadr`, `sh2101_calib`). `pytest tests/` — 2/2 pass. `pyflakes` — clean
across the whole codebase (verified as part of this session, not previously
run). All confirmed passing as of the last commit.

## 🚧 Current Blockers & Technical Debt

None blocking. Everything planned is complete and verified. Residual items,
none urgent:

- The old repo location (`/mnt/raid0/AgentCode_old/AstroBinUpload`) still
  exists on disk, untouched, same branch/tag as the canonical copy. Never
  deleted per earlier explicit choice to let the user do it themselves.
- `RUST_PORT_PLAN.md` exists but **no Rust work has been started** — it's a
  feasibility/planning document only, written before Bucket A/B existed, so
  its "hazard" analysis should be re-checked against the actual final fixes
  (e.g. hazard 3's rounding contract, hazard 2's group-key ordering) before
  any Rust implementation begins.
- Two decisions were deliberately left to the user and never revisited:
  whether to correct the live `config.ini`'s dead `SWCREATOR` typo (would
  change real output — left alone on purpose), and whether the Rust port's
  `to_string()` fidelity should be byte-exact (`RUST_PORT_PLAN.md`, "Decision
  required" section) — recommend re-asking if Rust work starts.

## 🚀 Next Steps

Nothing is queued — the user has not yet said what they want next. Likely
candidates, in rough order of what was implied during the session:

1. **PR #12 is open** (branch pushed, tag held back per user). Next action
   is the user's review + merge. Once merged: push the `v2.1.0` tag to
   `origin` (`git push origin v2.1.0`).
2. **If continuing code work**: re-read `REMEDIATION_PLAN.md`'s status
   (all items marked done) to confirm nothing regressed, then decide
   whether to start `RUST_PORT_PLAN.md`'s Phase 0 (freeze the Python
   parity target — already effectively done via the `v2.1.0` tag) through
   Phase 1 (Cargo scaffold, config parser, `--test` CSV ingest path).
3. **If the user wants more real-data validation**: they may have further
   directories to test against, following the same pattern used this
   session (back up any existing `AstroBinUploadInfo` before running,
   diff against expectations, restore or leave in place per what's
   normal usage, offer to add a good fixture to the corpus).

_Done this session: stray `/home/steve/config.ini` deleted (confirmed by user)._

## 📂 Files to Load

- `REMEDIATION_PLAN.md` — the full bug-fix plan, now entirely marked
  complete with verification notes per item; read this first for what was
  actually done and why.
- `RUST_PORT_PLAN.md` — Rust port feasibility plan, not started; read if
  the next task is starting the port.
- `golden_tests/run_golden.py` — the regression harness; run this first
  in any new session before making changes, to confirm the starting state
  is still green.
- `AstroBinUpload.py` — entry point, small, good orientation read.
- `constants.py` — central column-name/keyword constants, referenced
  throughout; useful for orientation.

Venv for running anything: `/mnt/raid0/Code/venvs/.astrovenv/bin/python3`.
Always `cd` into `/mnt/raid0/Agent_Code/Astronomy/AstroBinUploader` before
running the tool directly (not via the harness), or `config.ini` resolves
against the wrong directory.
