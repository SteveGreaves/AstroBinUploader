# Rust Port Plan — `astrobin-upload`

> **Revised 2026-09-07, after the Python work completed.** This plan was first
> written against `v2.0.3`, before any remediation existed. Since then the
> Python side shipped `v2.1.0` (Bucket A A1–A14, Bucket B B1–B15) and `v2.1.1`
> (GitHub #3/#5/#6). What that changed here:
>
> - The parity target is **`v2.1.1`**, not `v2.1.0` and certainly not `v2.0.3`.
> - `geopy` is **gone** from the Python side — `GeocodeStep` now hand-rolls a
>   specific haversine. The Rust port must reproduce *that* formula, not reach
>   for a crate (dependency table below, hazard 11).
> - The FITS reader has a requirement the first draft missed entirely:
>   **tile-compressed `.fits.fz` / `CompImageHDU`** files, where the real
>   metadata lives on an extension, not the primary HDU.
> - `--test` CSV ingest turns out to carry the sharpest Phase 1 parity trap —
>   `pd.read_csv` dtype inference (new hazard 14).
> - Config gained `[equipmentoverrides]` (v2.1.1) and the pipeline gained
>   `NormalizeHeadersStep` Stage 3b to apply it.
> - Hazards 3 and 8 became *resolved* questions; 10–14 are new.

## Verdict

**Yes, this is possible.** There is no capability in the Python codebase that
Rust cannot provide, and no dependency that forces a C runtime if the FITS
reader is hand-written. The result is a small (~3–5 MB) self-contained binary
per target — Windows, Linux and macOS — with no Python, no libcfitsio, and no
shared-library requirements (a fully static `musl` build on Linux).

Two honest caveats, stated up front rather than buried:

1. **The parity target must be the *repaired* Python, frozen at a tag.**
   `v2.0.3` was non-deterministic (`REMEDIATION_PLAN.md` A9: `as_completed`
   yields in completion order, `sort_values` was unstable), so it had no single
   well-defined output to be exact *against*. **Settled:** Bucket A/B and the
   issue-tracker follow-ups are all fixed and merged; the corpus is
   regenerated; **`v2.1.1` is tagged on `main`** and is the contract.
2. **One output block is expensive to match byte-for-byte** — the
   `pandas.DataFrame.to_string()` table appended to the bottom of the session
   summary. It is emulable, but it is the single largest risk item in the
   port. Decided: reproduce it byte-faithfully (see "Decisions" below).

Everything else — FITS/XISF parsing, the six pipeline steps, the acquisition
CSV, the entire structured part of the text report — is mechanical.

---

## Parity contract

> The Rust binary reproduces, byte for byte, the `*_acquisition.csv` and
> `*_session_summary.txt` produced by Python `v2.1.1` for every fixture in
> `golden_tests/fixtures/`, with the sole exception of the
> `Generated <timestamp>` line.
>
> Pin the target by reading the code **at the `v2.1.1` tag**, not by trusting a
> green differential diff — see hazard 13: config-driven behaviour the
> two-fixture corpus does not exercise.

Both implementations run under the **committed** `golden_tests/golden_config.ini`,
passed via `--config`. This is not incidental: until 2026-09-07 the harness used
the repo's gitignored `config.ini`, so the references only reproduced on one
machine (a clean clone failed 2/2 — wrong site, wrong `USEOBSDATE` grouping,
different FWHM/SQM). A differential harness needs a configuration both binaries
can actually load, so the Rust port must accept `--config` and parse that exact
file.

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
| `numpy` | std | `radians`/`sin`/`cos`/`arcsin`/`sqrt` for haversine, boolean masks, and `mean` (**pairwise** summation — hazard 4) |
| ~~`geopy.distance`~~ | **hand-written, matching Python exactly** | No longer a Python dependency. A3/A4 replaced it with a specific vectorized haversine (`EARTH_RADIUS_M = 6371000.0`, `2·arcsin(√a)`). Do **not** substitute the `geo` crate — it uses a different mean radius and formula family, which moves cluster boundaries. See hazard 11. |
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

Two requirements beyond the basic format:

**A7 — HDU selection.** Scan HDUs in order and take the **first whose header
contains `IMAGETYP`**; fall back to HDU 0 if none does (which preserves
`[defaults]` behaviour for files that legitimately lack it). Not
unconditionally HDU 0.

**Tile-compressed FITS.** This is the reason A7 exists, and the first draft of
this plan missed it. `.fits.fz` / `CompImageHDU` files store the real metadata
(`IMAGETYP`, `EXPOSURE`, `DATE-OBS`, …) on the **first image extension**,
leaving the primary HDU with only `SIMPLE`/`BITPIX`/`NAXIS`/`EXTEND`. astropy
hides this; a hand-written reader will not. In the file the compressed HDU is a
`BINTABLE` carrying `ZIMAGE = T`, and the user keywords sit in that binary-table
header as ordinary cards — so a reader that walks **every** HDU header
(not just image HDUs) and applies the `IMAGETYP` rule handles it correctly.
Do not filter HDUs by `XTENSION` type.

**Repeated cards.** `HISTORY` and `COMMENT` appear many times per header and the
master sub-exposure count (`ImageIntegration.numberOfImages`) is parsed out of
them, so the reader must expose them as a **sequence**, not a last-write-wins
scalar. (Note: A7's original "astropy collapses HISTORY" claim was investigated
and *disproven* — `dict(header)['HISTORY']` yields a list-like object. The
requirement stands for the Rust reader regardless; it just isn't a bug being
fixed.)

**Value cleanup.** After reading, every string value is stripped of surrounding
`'` and `"` (`extractor.py:177`), and `SOURCE_PATH` is set to the **absolute**
path — after the quote-strip, since it is not header text.

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

Sections to support, and what each does:

| Section | Semantics |
|---|---|
| `[defaults]` | Keys upper-cased with spaces stripped; injected only for columns still absent **after** case normalisation (A8). |
| `[override]` | Keyword remap, internal key → one or more raw header keys. A comma-separated value arrives from configobj as a **native list**, not a string (A6) — the parser must produce a list either way. Unknown targets warn and no-op; they must not error. |
| `[equipmentoverrides]` | **New in v2.1.1.** Value replacement, not keyword remap. Blank or the sentinel `None` ⇒ skip; anything else ⇒ force that literal into the column for every row, applied after default injection (`NormalizeHeadersStep` Stage 3b). |
| `[filters]` | Filter name → AstroBin numeric code. An unmapped name passes through **as the original string**. |
| `[sites]` | The nested `[[…]]` case above. Bare and quoted section names both occur in real configs. |
| `[secret]` | Present in older user configs; unused by the summary/CSV path. Parse and ignore. |

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

### Pipeline order — pin it

`AstroBinUpload.py:202–207`. The order is load-bearing and not the one you'd
guess, so it is recorded here rather than re-derived:

```
1  NormalizeHeadersStep     overrides → lowercase → defaults → 3b equipment
                            overrides → drop MASTERLIGHT → master preference
                            → IMAGETYP normalisation → type hardening
2  OpticalParameterStep     HFR / IMSCALE / FWHM      (before dedup!)
3  DeduplicateStep          WBPP survivor pick
4  CalibrationMatcherStep   gain handshake + cal counts
5  GeocodeStep              coordinate align → cluster → site lookup
6  AggregationStep          temporal split → groupby → filter codes
```

Two orderings worth noting because reversing them changes output: optical
metrics are computed **before** deduplication (so on pre-dedup rows), and
geocoding runs **after** calibration matching (so calibration frames are
matched before their coordinates are snapped).

---

## Parity hazards, ranked

> Re-validated against the landed `v2.1.1` code on 2026-09-07.
>
> - **3 and 8** are no longer open questions — the algorithm is pinned in
>   Python source, not merely proposed.
> - **2** gained concrete detail on mixed-type group keys and stable sorts.
> - **10–12** cover behaviour the A1/A2, A3/A4 and A13/A14 fixes introduced,
>   none of which existed when this plan was first drafted.
> - **13–14** came out of the post-completion revisit: config surface the
>   corpus cannot see, and `pd.read_csv` dtype inference.
>
> 1 remains the highest-risk item and 14 is the one most likely to be
> underestimated, because it looks like three lines of Python.

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
any group key are silently dropped** — but note `AggregationStep` now fills
every group key first (`fillna(0)` for numeric keys, `fillna("None")` for
object keys, literal `"None"` for a missing column), so in practice nothing is
dropped there; `reports.py`'s inner `groupby` calls are the ones still exposed
to the drop.

Confirmed against v2.1.0:
- `agg_cols` mixes types — `site_name`/`session_date` (a `datetime.date`)/
  `image_type`/`filter`/`target` are object, `gain`/`xbinning`/`duration` are
  numeric. `groupby` sorts by the key tuple with **per-column** ordering
  (lexicographic on strings, numeric on the rest, `date` chronological). A
  single `BTreeMap<CompositeKey, _>` works only if `CompositeKey`'s `Ord` is
  column-type-aware.
- `AggregationStep` does a `sort_values(DATE_OBS, kind='mergesort')` **before**
  grouping. `pd.to_datetime(errors='coerce')` puts unparseable dates at `NaT`,
  and a stable sort sends `NaT` to the **end**. The `agg('first')` columns
  (focal length, pixel size, camera, filenames, …) read from this order, so
  the Rust sort must be stable and NaT-last.
- `DeduplicateStep`'s survivor pick is `sort_values(['ext_rank', filename],
  key=len-on-filename-only, kind='mergesort').iloc[0]` — stable sort on
  `(ext_rank, char_len(filename))`, ties broken by dispatch order.

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

**Resolved (v2.1.0).** A12 landed as `engine/steps/optical.py::_python_round`
— `series.apply(lambda x: round(x, ndigits))`, i.e. CPython builtin
round-half-to-even on the true decimal value. So the target algorithm is
fixed: implement `python_round(x: f64, n: u32) -> f64` in Rust (a correctly
rounded decimal round, not `f64::round`, which is half-away-from-zero) and
route HFR / IMSCALE / MEAN_FWHM through it, then `{:.2}` on top. Rust's
`{:.N}` formatter is itself half-to-even and matches the trailing format
step. **Do not use `f64::round()` anywhere.**

One asymmetry to reproduce deliberately: `geocode.py::_find_site_in_db`
compares `db_lat.round(precision)` (pandas/numpy rint) against
`round(lat, precision)` (builtin) — the two sides of that equality use
*different* rounding. It only affects whether an averaged centroid matches a
site-DB entry, but a faithful port needs numpy-rint on the DB column and
builtin-style on the query scalar, not one helper for both.

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
makes the Python side sort explicitly — `file_paths.sort()` over the
**`os.path.join(root, file)` strings exactly as constructed from the CLI
arguments** (relative if the argument was relative — *not* canonicalised to
absolute), with all input directories merged into one list before the sort.
Python's `str` sort is by Unicode scalar value, which for UTF-8 coincides with
a bytewise sort. The Rust side must reproduce that exact string and that exact
order rather than relying on `walkdir`'s platform-dependent traversal.
Extension filter: case-insensitive `.fits/.fit/.fts/.xisf`.

### 9. Drop the version handshake

`verify_engine_integrity` and the 14 duplicated `__version__` strings exist to
detect mixed-file installations. That failure mode cannot occur in a single
static binary. Do not translate it; a `--version` flag replaces it. (B8 already
collapsed the 14 strings to one source, so there is only one value to expose.)

### 10. Deduplication key and WBPP regex (A1, A2 — new since the audit)

`DeduplicateStep` changed materially in v2.1.0:
- The base-name split is now `constants.RegexPatterns.WBPP_FILENAME`, an
  **anchored** pattern applied case-insensitively via `.str.extract(...)[0]`
  (group 1 = base name; a non-match yields null and that row is skipped):
  `(.+?)(_(?:c|cc)(?:_(?:cc|rn|r|d|b|s|lps))*)?(\.xisf|\.fits|\.fit|\.fts)$`.
  The lazy `(.+?)` plus end-anchor means the Rust regex must be equally lazy
  and anchored — a greedy or unanchored translation reintroduces A1.
- The dedup group key is now `(dirname(source_path), base_filename)`, not
  `base_filename` alone. `dirname` is `os.path.dirname` semantics on the
  `source_path` string. When the `source_path` column is absent (a `--test`
  CSV captured before A2) the code degrades to a `''` directory for every row
  and logs a warning — the port must reproduce both the fallback and the
  warning so old fixtures still replay identically.

### 11. GPS greedy single-linkage clustering (A3, A4 — new since the audit)

`GeocodeStep` is no longer coordinate rounding. It is now:
1. `_align_coordinates`: for each non-Light row, if lat/lon are both present,
   snap to the **haversine-nearest** Light frame (`dist.idxmin()` — first index
   wins ties); if either is missing, snap to `lights.iloc[0]`.
2. Build `unique_coords` = `coords_df.drop_duplicates()` (first-occurrence
   order, after `pd.to_numeric(errors='coerce').fillna(0.0)` on both columns).
3. Greedy single-linkage over `unique_coords` **in row order**: each still-
   unclaimed point opens a new cluster and claims every still-unclaimed point
   with haversine distance **strictly** `< 110.0 m`. Order-dependent by
   design — the Rust iteration order must match exactly.
4. Cluster coordinate = arithmetic mean of member lat/lon (numpy `mean`,
   pairwise summation).
5. Site-DB lookup per cluster with the mixed-rounding equality from hazard 3.

`_haversine_distance_m`: `EARTH_RADIUS_M = 6371000.0`, standard haversine
(`2 * arcsin(sqrt(a))`), radians via `np.radians`. Pin the constant and the
formula shape; do not substitute a different great-circle form.

### 12. Calibration report-table grouping (A13, A14 — new since the audit)

- **A13**: master-preference filtering runs in `base.py` **before** IMAGETYP
  normalization (so `FLAT`/`MASTERFLAT` substring matching still works), and
  the normalization pass must not re-run over its own output (that was the
  bug — it re-mapped `MASTER DARK` → `MASTERDARK` → matched `MASTERDARK` as a
  substring and clobbered the label). Port the ordering and the single-pass
  guarantee.
- **A14**: in `reports.py::format_image_type_table`, only
  `filter_matters_for_type = 'FLAT' in imagetype.upper()` (i.e. FLAT and
  DARKFLAT) groups/labels by filter. DARK and BIAS are filter-independent —
  their table rows carry a blank filter regardless of any stray `FILTER` tag
  on the frame. This mirrors `calibration.py`, which never constrains
  dark/bias matching on filter.

### 13. Config-driven behaviour the two-fixture corpus does not exercise

The differential harness only proves parity for the config the fixtures run
under. Checked against the repo's live `config.ini` (2026-09-07):

| Path | Covered by corpus? | Notes for the port |
|---|---|---|
| `USEOBSDATE = False` — overnight date-shift (`aggregate.py` `calculate_ref_date`, noon boundary) | **Yes** — live config sets `False` | The `True` branch (plain `dt.date`) is **not** exercised; unit-test it directly. |
| `[sites]` DB lookup, incl. nested `[[...]]` and a bare `[[Name]]` section | **Yes** — live config has 4 site entries | configobj depth-by-bracket-count parsing + the mixed-rounding equality in `_find_site_in_db` (hazard 3). |
| `[override]` keyword remap, incl. a **dead** entry (`SWCREATOR` — not a real internal column, warns and no-ops, A6) | **Yes** — live config carries both live and dead entries | Port must emit the same warning and no-op, not error. |
| `[equipmentoverrides]` (v2.1.1) | **No** — generated template is all sentinels | Sentinel/blank ⇒ skip; any other value ⇒ force into the column for every row, after default injection. |
| `[filters]` name→code map | Partial — only the filters present in the two fixtures | Unmapped filter ⇒ the original name string passes through (`aggregate.py::map_filter`). |
| `[secret]` / light-pollution API | Not used by the summary/CSV path | Out of parity scope; omitted from `golden_config.ini`. |

Fixture coverage of the *data* paths, checked the same way:

| Path | Covered? | Notes |
|---|---|---|
| Directory-aware dedup key (A2) | **Yes** — `sh2101_calib_raw.csv` carries `SOURCE_PATH` | Normal `(dirname, base)` keying. |
| Degraded filename-only dedup + warning | **Yes** — `sadr_raw.csv` predates A2 and has no `SOURCE_PATH` | The corpus exercises both branches of hazard 10 by accident. Preserve that: do not regenerate `sadr_raw.csv`. |
| `DARKFLAT` / flat-dark matching (A11) | **No** — neither fixture contains one | Verified in Python with a targeted synthetic case; do the same in Rust. |

Whatever is not covered here gets a targeted Rust unit test rather than
leaning on the byte diff.

### 14. `pd.read_csv` dtype inference — the Phase 1 trap

`extract_from_csv` is three lines (`extractor.py`): `pd.read_csv(path)`, then
upper-case the column names. Those three lines carry more parity risk than the
config parser, because **the inferred dtype of every column propagates all the
way to the acquisition CSV**, and int-vs-float is directly visible there.

Measured on `golden_tests/fixtures/sadr_raw.csv` (221 rows, 64 columns):

| Column | Inferred dtype | Renders as |
|---|---|---|
| `GAIN`, `XBINNING`, `NUMBER` | `int64` | `100`, `1`, `1` |
| `EGAIN`, `EXPOSURE`, `CCD-TEMP`, `FOCALLEN`, `SITELAT` | `float64` | `0.246657639741898`, `600.0`, `-10.0` |
| `IMAGETYP`, `FILTER` | `object` | as written |

63 of the 64 columns contain no nulls. That matters: pandas types an all-integer
column `int64` **only while it has no nulls** — a single empty field promotes it
to `float64`, and `100` becomes `100.0` in the output AstroBin consumes. This is
the same failure surface as A5, reached through the reader rather than through
`fillna`.

So the Rust CSV reader cannot be "parse into `String` and coerce later". It must
run pandas' inference rules per column up front — all-integer-and-no-blanks ⇒
integer; otherwise numeric-parseable ⇒ float; otherwise string; empty ⇒ null —
and carry that decision as the column's type for the rest of the run. Float
formatting must round-trip (`repr`-shortest), not fixed precision.

---

## Phasing

Each phase ends with the differential harness green over whatever subset it
covers.

| Phase | Scope | Why here |
|---|---|---|
| **0** | Freeze the contract — **done**: `v2.1.1` tagged and on `main`, goldens regenerated. | Nothing testable without it |
| **1** | Cargo scaffold, `clap` CLI, config parser (incl. `[equipmentoverrides]`, v2.1.1), `--test` CSV ingest **only** — with pandas-equivalent dtype inference (hazard 14) | Reaches end-to-end on committed fixtures without writing a single byte of FITS parsing |
| **2** | The six pipeline steps as pure functions over `Vec<Frame>`, incl. `NormalizeHeadersStep` Stage 3b (equipment value overrides, v2.1.1) | The bulk of the logic; fully exercised by Phase 1's CSV path |
| **3** | Exporter + `reports.py` — the byte-parity grind | Hazard 1 lives here |
| **4** | FITS and XISF readers | The only part the CSV fixtures cannot exercise. **Prerequisite:** `REMEDIATION_PLAN.md` P0 item 3 — hand-built FITS/XISF fixtures under `golden_tests/fixtures/binary/` — was never done and that directory does not exist. Build it as the first task of this phase, including a tile-compressed `.fits.fz` case. |
| **5** | `rayon` parallelism; release matrix for Windows / Linux (`musl` static) / macOS, x86-64 and arm64 | Optimise only once correct |
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
| Pipeline steps (Phase 2) | 1,600 | 25% |
| Report + exporter formatting (Phase 3) | 700 | **35%** |
| FITS + XISF readers (Phase 4) | 750 | 15% |
| Config parser, CLI, CSV ingest, plumbing | 900 | 15% |
| Harness + tests | 900 | 10% |

Revised upward from the first draft by ~400 LOC. The additions are the
tile-compressed FITS path (hazard/FITS section), the `[equipmentoverrides]`
section and Stage 3b, and a real dtype-inferring CSV reader (hazard 14) in place
of the "just read strings" the first draft implicitly assumed. Phase 3 drops
from 40% to 35% of effort only because the denominator grew — its absolute cost
is unchanged and it is still the single largest block.

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
