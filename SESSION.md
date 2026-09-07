## 🎯 Objective

Python FITS/XISF metadata ETL producing AstroBin bulk-upload CSVs.

**This project is complete and released at v2.1.1.** Active development has moved to the
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
  `PORT_PLAN.md`, `RUST_PORT.md` left as a pointer.

Current state: `main`, v2.1.1, golden 2/2, pytest 5/5, pyflakes clean.

## 🚧 Current Blockers & Technical Debt

- **GitHub bookkeeping is unfinished.** `gh issue close` / `gh pr merge` were denied by the
  auto-mode classifier for the whole session. Adding `"Bash(gh:*)"` to `autoMode.allow` in
  `~/.claude/settings.json` fixes it. Outstanding:
  - **PR #13** — a one-line `SESSION.md` doc change, still open and unmerged.
  - Issues **#3, #4, #5, #6, #11** are fixed in code but still show open; **PR #8** is
    superseded. Issues **#9, #10** need a re-test from the reporter on v2.1.1 before closing.
  - Ready-to-post comment text is in the reconciliation table at the end of `REMEDIATION_PLAN.md`.
- `golden_tests/fixtures/binary/` was never built (P0 item 3): the FITS/XISF parsing paths
  have no regression fixtures. The Rust port's Phase 4 needs these too.
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
