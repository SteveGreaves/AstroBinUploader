# Rust Port Plan — `astrobin-upload`

## Verdict

**Yes, this is possible.** There is no capability in the Python codebase that
Rust cannot provide, and no dependency that forces a C runtime if the FITS
reader is hand-written. The result is a small (~3–5 MB) self-contained binary
per target — Windows, Linux and macOS — with no Python, no libcfitsio, and no
shared-library requirements (a fully static `musl` build on Linux).

Two honest caveats, stated up front rather than buried:

1. **The parity target must be the *repaired* Python, frozen at a tag** — not
   today's `v2.0.3`. Today's code is non-deterministic (`REMEDIATION_PLAN.md`
   A9: `as_completed` yields in completion order and `sort_values` is unstable),
   so it does not produce a single well-defined output to be exact *against*.
   Fix Bucket A first, tag `v2.1.0`, regenerate the golden corpus, and that
   becomes the contract.
2. **One output block is expensive to match byte-for-byte** — the
   `pandas.DataFrame.to_string()` table appended to the bottom of the session
   summary. It is emulable, but it is the single largest risk item in the
   port. Decided: reproduce it byte-faithfully (see "Decisions" below).

Everything else — FITS/XISF parsing, the six pipeline steps, the acquisition
CSV, the entire structured part of the text report — is mechanical.

---

## Parity contract

> The Rust binary reproduces, byte for byte, the `*_acquisition.csv` and
> `*_session_summary.txt` produced by Python `v2.1.0` for every fixture in
> `golden_tests/fixtures/`, with the sole exception of the
> `Generated <timestamp>` line.

Enforced by a differential harness (Phase 6) that runs both implementations over
the committed corpus in CI. This is why `REMEDIATION_PLAN.md` P0 is a hard
prerequisite: without a portable fixture corpus there is nothing to test parity
against, and the port becomes unfalsifiable.

---

## Dependency mapping

| Python | Rust | Notes |
|---|---|---|
| `argparse` | `clap` (derive) | trivial |
| `astropy.io.fits` | **hand-written** | see below |
| XISF via `struct` + `ElementTree` | `quick-xml` | pure Rust |
| `configobj` | **hand-written** | see below |
| `pandas` | plain structs + `BTreeMap` | see below |
| `numpy` | std / `itertools` | only used for `sqrt` and masks |
| `geopy.distance` | `geo` crate, or ~20 lines of haversine | needed by remediation A4 |
| `concurrent.futures.ProcessPoolExecutor` | `rayon` | threads, not processes — no GIL to escape |
| `csv` writing | `csv` crate | |
| `logging` | `tracing` + `tracing-subscriber` | |
| datetime | `chrono` | |
| — | `anyhow` + `thiserror` | replaces the silent `except: pass` blocks |

### FITS: hand-write it, don't bind cfitsio

`fitsio` wraps the cfitsio C library, which defeats "self-contained" and
complicates static linking. This program only ever reads **headers** — it never
touches pixel data. The FITS header format is trivial: 2880-byte blocks of
80-byte fixed-width card images, `KEYWORD = value / comment`, terminated by
`END`. A correct reader for this use is ~200 lines, has no dependencies, and is
faster than cfitsio because it stops at the first `END` and never mmaps the data
unit.

It must implement remediation A7: select the first HDU with a non-trivial
header (not unconditionally HDU 0), and preserve **repeated** `HISTORY` and
`COMMENT` cards, which the master sub-exposure count depends on.

### Config: hand-write the parser

configobj's nested-section syntax under `[sites]` —

```ini
[sites]
        [["My Full Site Address, Country, Postcode"]]
                latitude = 0.0000
```

— is **not** standard INI. `rust-ini`, `configparser` and `serde_ini` do not
support `[[...]]` depth nesting; `[[x]]` in TOML means an array-of-tables, which
is different semantics. Budget a small recursive-descent parser (~150 lines)
that handles depth-by-bracket-count, `#` comments, quoted section names, and
comma-separated values. It must reproduce configobj's whitespace and quote
stripping exactly, since `[override]` values feed column matching.

### Data layer: structs, not polars

Recommend `Vec<Frame>` with explicit `BTreeMap` grouping over pulling in polars.

- The datasets are thousands of rows, not millions — the reference sets are
  221–1608 frames. There is no performance argument for a columnar engine.
- polars adds ~40 MB to the binary and brings its own null/NaN and sort
  semantics, which would then have to be reconciled against *pandas'* semantics
  anyway. That is strictly more work than writing the semantics out explicitly.
- Explicit code makes each pandas behaviour being emulated auditable and
  testable, which is the whole game here.

Model missing values as `Option<T>` and be deliberate at every call site about
whether pandas would have skipped or propagated.

---

## Parity hazards, ranked

### 1. `acq_df.to_string()` appended to the summary — highest risk
`engine/exporter.py:122`

The bottom of every session summary is a pandas text render of the acquisition
table. The formatting rules are non-obvious. Confirmed empirically:

**Floats get a per-column common decimal count**, chosen as the minimum that
round-trips every value in that column:

```
      v
  4.600
  4.750
100.125     ← the 3-decimal value forces 4.6 to render as "4.600"
```

So a single value elsewhere in the column changes how *this* value prints. The
Rust implementation must compute the column-wide decimal count first, then
format — not format each value independently.

**Column width** is `max(len(header), max(len(formatted_value)))`, values
right-aligned including string columns.

**The separator is not uniform.** Measured on a mixed frame:

```
'  a  bbbbbb    s'
'  1    1.50    x'
'222    2.25 yyyy'
```

Two spaces between the numeric columns, one before the string column — because
pandas' numeric array formatter injects a leading space into each value that the
object formatter does not. Reproducing this requires matching pandas' per-dtype
formatter behaviour, not just a column-width calculation.

None of this is intractable. All of it is fiddly, and it is the part most likely
to produce a one-byte diff that costs a day to find.

### 2. Group-key ordering *is* row ordering

`df.groupby(agg_cols)` sorts by key tuple by default, and that order becomes the
row order of the acquisition CSV and of every report table. The Rust
implementation must sort by the same tuple with the same comparison semantics.

`BTreeMap` gives ordering for free — but only once the key types are clean.
Remediation **A5** is now fixed and landed clean: on investigation the
`fillna("None")`-promotes-`gain`-to-object-dtype scenario it describes was
never actually reachable in the pipeline as it exists (`gain`/`xbinning`/
`exposure` are unconditionally hardened to non-null numeric types upstream),
so there's no live mixed-type column to reproduce here. The fix was made
defensive rather than corrective. Good news for the port: `gain` stays a
clean numeric type at the group-key stage, so `BTreeMap`'s natural ordering
is a direct match without any pandas mixed-type sort fallback to replicate.

Also replicate: `groupby(dropna=True)` is the default, so **rows with a null in
any group key are silently dropped**.

### 3. Rounding — replicate the sequence, not just the final format

The code rounds *then* formats: `round(x, 2)` followed by `f"{x:.2f}"`. These
are not the same operation applied twice.

**There are three distinct rounding algorithms in play and they disagree**
(measured, see `REMEDIATION_PLAN.md` A12 — 5 of 10 boundary values differ, all
visible after `%.2f`):

| | Algorithm | Compatible with Python `round()`? |
|---|---|---|
| Python builtin `round(x, n)` | decimal-correct half-to-even | — (the current target) |
| pandas `Series.round(n)` | numpy multiply–`rint`–divide | **no** |
| Rust `format!("{:.2}")` | decimal-correct half-to-even | yes |
| Rust `f64::round()` | half-away-from-zero | **no** — never use it |

Today the Python code uses the builtin, so Rust's `{:.2}` matches. But
remediation **A12** vectorises `OpticalParameterStep`, and if that rewrite
reaches for pandas `.round()` the target algorithm silently changes underneath
this port.

**Action: pin it.** A12 specifies keeping builtin-`round` semantics via an
explicit helper. Confirm that landed before Phase 2, then implement
`python_round(x: f64, n: u32) -> f64` in Rust and route every rounding call
through it — never `f64::round`. If A12 is instead resolved in favour of pandas
semantics, this helper must emulate numpy's `rint` path instead; the two are
not interchangeable and the choice must be recorded in the parity contract.

`seconds_to_hms` needs equal care: `int(seconds // 3600)` is float floor-division
then truncation, and Python's `%` takes the sign of the divisor. For the
non-negative durations here the behaviours coincide, but pin it with tests.

### 4. Floating-point accumulation order

pandas delegates `mean`/`sum` to numpy, which uses pairwise summation; a naive
left fold in Rust can differ in the last ULP. After `{:.2}` this is invisible
except when a value sits exactly on a `.005` boundary. Mitigate by summing in
the same order and routing through `python_round`; accept the residual risk and
let the differential harness catch it.

### 5. pandas string-op null semantics

`str.contains(..., na=False)`, `.str.upper()` on a non-string yielding null,
`.str.lower()` on numerics — the calibration matcher and `reports.py` are full
of these. Each needs a deliberate Rust equivalent; `Option::map_or(false, ...)`
is usually the right shape for the `na=False` cases.

### 6. `agg('first')` skips nulls; `.iloc[0]` does not

Both appear in this codebase, sometimes within a few lines of each other
(`aggregate.py` uses `'first'`; `reports.py` uses `.iloc[0]`). They are different
functions. Port each faithfully to its own call site.

### 7. `pd.to_datetime(errors='coerce')`

Accepts a very wide range of inputs. Pin the Rust parser to the formats actually
present in FITS `DATE-OBS` — ISO 8601 with and without fractional seconds, `T`
or space separator — and coerce anything else to null rather than erroring, to
match. Enumerate the formats found across the fixture corpus first.

### 8. Traversal order

`os.walk` order feeds dedup tie-breaks and every `first()` pick. Remediation A9
makes the Python side sort explicitly; the Rust side must apply the **same**
sort (by absolute path, bytewise) rather than relying on `walkdir`'s
platform-dependent order.

### 9. Drop the version handshake

`verify_engine_integrity` and the 14 duplicated `__version__` strings exist to
detect mixed-file installations. That failure mode cannot occur in a single
static binary. Do not translate it; a `--version` flag replaces it.

---

## Phasing

Each phase ends with the differential harness green over whatever subset it
covers.

| Phase | Scope | Why here |
|---|---|---|
| **0** | Freeze the contract: tag repaired Python `v2.1.0`, regenerate goldens from fixtures, commit | Nothing testable without it |
| **1** | Cargo scaffold, `clap` CLI, config parser, `--test` CSV ingest **only** | Reaches end-to-end on committed fixtures without writing a single byte of FITS parsing |
| **2** | The six pipeline steps as pure functions over `Vec<Frame>` | The bulk of the logic; fully exercised by Phase 1's CSV path |
| **3** | Exporter + `reports.py` — the byte-parity grind | Hazard 1 lives here |
| **4** | FITS and XISF readers | The only part the CSV fixtures cannot exercise; validate against the synthetic binary fixtures from remediation P0 |
| **5** | `rayon` parallelism, musl static build, cross-compilation | Optimise only once correct |
| **6** | Differential harness in CI: both binaries over the full corpus, byte-compare modulo the `Generated` line | Ongoing guarantee |

Phase 1 is deliberately ordered before Phase 4. Because `--test` replay was
verified byte-identical to a disk scan, the entire transformation and reporting
pipeline can be built and proven correct before any binary file parsing exists.
That removes FITS/XISF parsing from the critical path.

---

## Effort

Roughly 3,500–5,000 lines of Rust. The distribution is lopsided and worth
knowing in advance:

| Area | ~LOC | ~Effort |
|---|---|---|
| Pipeline steps (Phase 2) | 1,500 | 25% |
| Report + exporter formatting (Phase 3) | 700 | **40%** |
| FITS + XISF readers (Phase 4) | 600 | 15% |
| Config parser, CLI, plumbing | 700 | 10% |
| Harness + tests | 800 | 10% |

Phase 3 is 15% of the lines and 40% of the effort. That asymmetry is the single
most important thing to plan around, and it is almost entirely hazard 1.

---

## Decisions (settled 2026-09-07)

1. **`to_string()` fidelity: (a) byte-faithful.** The requirement is "exactly
   as this code does", so Phase 3 reimplements the closed subset of pandas'
   formatter this program actually exercises (all-numeric plus one or two
   string columns, no nulls, no truncation) and property-tests it against real
   pandas output over generated frames. ~150 lines + test harness; the bulk of
   Phase 3.
2. **Target platforms: Windows, Linux, and macOS — all three.** Build matrix:
   `x86_64-unknown-linux-musl` (static), `x86_64-pc-windows-msvc` and
   `aarch64-pc-windows-msvc`, `x86_64-apple-darwin` and `aarch64-apple-darwin`.
   The hand-written FITS reader keeps every target free of a C runtime. No
   architectural impact — only the release/CI matrix grows.
3. **`--debug` CSV dumps: keep `debug_step_00_RawHeaders.csv` only.** That file
   is the fixture-capture mechanism the parity strategy depends on, so it must
   stay byte-comparable. The remaining per-step dumps become structured
   `tracing` output (not byte-compared).
4. **Rust is the eventual replacement, not a permanent twin.** The differential
   harness runs during development and for one release of overlap, then
   retires. Test infrastructure is not to be over-built beyond that horizon —
   the current two-fixture golden corpus is the working baseline for now.
