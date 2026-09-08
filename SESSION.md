## 🎯 Objective

Python FITS/XISF metadata ETL producing AstroBin bulk-upload CSVs.

**This project is complete and released at v2.1.2.** Active development has moved to the
Rust port in a separate repository — see `RUST_PORT.md`. Work here now is maintenance only.

## ✅ Completed Work

- **v2.1.0** — full remediation of v2.0.3 per `REMEDIATION_PLAN.md`: Bucket A (A1–A14,
  output-affecting defects) and Bucket B (B1–B15, hygiene), each its own commit with an
  attributed diff. Merged via PR #12, tagged.
- **v2.1.1** — GitHub issue follow-ups: #3 (generated config seeded a stale `ROTATOR` key),
  #6 (`AOCSKYQU`/`AOCAMBT` → SQM/FOCTEMP now default `[override]` entries), #5 (new optional
  `[equipmentoverrides]` section forcing a display value into a column, `NormalizeHeadersStep`
  Stage 3b).
- **Golden harness made machine-independent** — `run_golden.py` ran without `--config`, so it
  used the *gitignored* `config.ini`; the committed references only reproduced on one machine
  (a clean clone failed 2/2). Now ships `golden_tests/golden_config.ini` and passes it via
  `--config`. A fresh clone with no `config.ini` passes 2/2.
- **`REMEDIATION_PLAN.md` gained a GitHub issue reconciliation table** mapping every open
  issue to its Bucket A finding and verified status. The original plan never cited an issue
  number, which is why 4 hours of "full remediation" left the tracker looking untouched.
- **Rust port split out** to its own repository; `RUST_PORT_PLAN.md` moved there as
  `PORT_PLAN.md`, `RUST_PORT.md` left as a pointer. That port is now functionally
  complete and re-verifies parity against this code on every push.
- **v2.1.2** — calibration sections were labelled `MASTERxxx` unconditionally, even for a
  session built entirely from raw `DARK`/`FLAT`/`BIAS` frames. Found by running the Rust
  port against real unstructured data. The code's comment claimed "v1.4.7 standards" for
  behaviour v1.4.7 never had — the real v1.4.7 labelled each section by its literal
  `IMAGETYP`. `format_image_type_table` now derives the label from what the table holds.
  Both golden references re-blessed (`sadr` timestamp only; `sh2101_calib`'s three
  calibration headers now read `FLATS:`/`BIAS:`/`DARKS:`).
- **Issues #9 and #10 re-tested and closed** — see the reconciliation table in
  `REMEDIATION_PLAN.md`. Both reproduced from the figures in the reports (the reporters'
  data was never available) and cross-checked against the Rust implementation.
- **Docs brought current for the v2.1.2 release**: `README.md` had accumulated real
  staleness — an installation file list naming modules that have not existed since v2.0,
  a live `[secrets]`/API-key section for network calls removed in v2.1.0, a `[sites]`
  section claiming automatic updates that no longer happen, a stale `ROTATOR` key in the
  config walkthrough, and no documentation at all for `[equipmentoverrides]`. All fixed.
  `future_work.md` reviewed item by item against the current code.

Current state: `main`, v2.1.2, golden 2/2, pytest 5/5.

## 🚧 Current Blockers & Technical Debt

- **GitHub tracker is completely clear** — 0 open PRs, **0 open issues** as of 2026-09-08.
  #3, #4, #5, #6, #11 closed 2026-09-07; #9 and #10 closed 2026-09-08 after the re-test they
  were held open for. Per-issue status and evidence: the reconciliation table at the end of
  `REMEDIATION_PLAN.md`.
  - Still unbuilt, tracked in `future_work.md` rather than on the tracker: the
    `[calibrationoverrides]` ini fallback floated in #10. It only matters for a master frame
    carrying no recoverable sub-exposure count, which no observed file does.
- `golden_tests/fixtures/binary/` was never built here (P0 item 3): this repository's own
  FITS/XISF parsing paths have no regression fixtures. **The Rust port built one** —
  `parity/fixtures/binary/`, 241 header-only real files plus synthetic HDU cases — and its
  differential harness exercises this code's readers through it on every push. Worth copying
  back here if this repository ever needs standalone binary coverage.
- Only two golden fixtures exist. `DARKFLAT` / flat-dark matching (A11) is covered by neither.
- **Do not regenerate `golden_tests/fixtures/sadr_raw.csv`** — it predates A2 and has no
  `SOURCE_PATH`, making it the only fixture exercising the degraded filename-only dedup branch.
- The user is content that their home address appears in the public golden references. Settled;
  do not raise again.

## 🚀 Next Steps

Nothing queued for this repository. If asked to work here:

1. Clear the GitHub tracker (needs the `gh` permission above).
2. If output behaviour ever changes and the references are re-blessed, **re-copy the corpus**
   into the Rust repo's `parity/` directory and update its `parity/CORPUS.md` — the port's
   parity contract names Python v2.1.1 specifically.

Active work is in the Rust port: `/mnt/raid0/Agent_Code/Astronomy/AstroBinUploaderRust`,
whose own `SESSION.md` has its next steps.

## 📂 Files to Load

- `REMEDIATION_PLAN.md` — what was fixed and why; issue reconciliation table at the end.
- `RUST_PORT.md` — pointer to the port and the corpus-resync obligation.
- `golden_tests/run_golden.py` — run first to confirm the starting state is green.
- `AstroBinUpload.py`, `constants.py` — orientation.

Venv: `/mnt/raid0/Code/venvs/.astrovenv/bin/python3`.
