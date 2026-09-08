# Rust port

The Rust port of this project lives in its own repository:

**<https://github.com/SteveGreaves/AstroBinUploaderRust>**

It was developed here initially (under `rust/`) and split out on 2026-09-07 so
the two projects stay independent. The full port plan — parity contract,
dependency mapping, the ranked hazard list, phasing and effort — moved with it
and is now `PORT_PLAN.md` in that repository.

## What this repository still owes the port

The Rust project keeps **copies** of the golden corpus (`golden_tests/fixtures/`,
`golden_tests/references/`, `golden_tests/golden_config.ini`) so it can build and
test standalone. Those copies go stale whenever this project re-blesses its
references.

**If you change output behaviour here and re-bless, re-copy the corpus into the
Rust repository's `parity/` directory and update its `parity/CORPUS.md`.**

The parity contract there names a specific Python version — currently
**v2.1.2** — so a release that changes output is also a decision to move that
target. `parity/check_steps.py` enforces it: the harness refuses to run
against a checkout at any other version, rather than silently comparing
against the wrong baseline.

**Status:** the port is functionally complete. All six phases of its
`PORT_PLAN.md` are done — byte-identical output to this code across four CSV
fixtures and three binary FITS/XISF scenarios, built and tested for five
platforms, with a differential harness re-verifying parity against this
repository on every push.
