# Remediation Plan — AstroBinUpload v2.0.3 → v2.1.0

Audit performed 2026-09-07 against the working tree at commit `394e7d0`.
Supersedes `future_work.md`, which remains accurate but incomplete: it lists
hygiene items and misses every one of the output-corrupting defects in Bucket A
below.

Findings marked **[proven]** were reproduced empirically in this session, not
inferred from reading.

---

## P0 — Build the verification substrate first

Nothing else in this plan is safe to start until there is a machine-independent
way to prove a change did or did not alter output. Right now there isn't one.

### The problem

`.gitignore:168–174` excludes `config.ini`, `tests/`, `golden_tests/` **and**
`golden_tests/references/`. The six Golden Tests in `GEMINI.md` reference
absolute paths on one workstation. The entire acceptance criterion for this
project is therefore untracked and unreproducible.

Current state of that data on this machine:

| Test | Data | Status |
|---|---|---|
| 1 (Michael, CSV) | `~/Downloads/Jason Astrobin Data` | dir present, **0 frames**, CSV missing |
| 2 (31st May) | `~/Desktop/Pixinsight/LBN 548` | **missing** |
| 3 (13th June, standard sanity check) | `~/Desktop/Pixinsight/LBN 548` | **missing** (calibration dir present, 812 frames) |
| 4 (Mosaic) | NGC 6997 Mosaic | not checked, depends on missing LBN 548 calib pairing |
| 5 (Alpha, CSV) | `~/Downloads/Jason Astrobin Data` | dir present, **0 frames**, CSV missing |
| 6 (Sadr) | `Preselected/Sadr Region` | **present, 221 frames** |

Five of the six Golden Tests cannot currently be run at all.

### What was verified **[proven]**

TEST 6 was executed against `v2.0.3`. Both artefacts are **byte-identical** to
`golden_tests/references/sadr_*` after normalising the `Generated <timestamp>`
line. So the committed references are current for the one scenario that is
still runnable — the baseline is trustworthy, just nearly inaccessible.

More importantly, the `--test` replay path was validated end to end:

```
scan from disk           → Sadr_Region_session_summary.txt
--debug                  → debug_step_00_RawHeaders.csv   (140 KB, 17 KB gzipped)
--test debug_step_00.csv → Sadr_Region_session_summary.txt
```

The replayed output is byte-identical to the disk scan. **`--test` is a faithful
substitute for the filesystem.** That makes a committed fixture corpus possible.

### Actions

1. **Un-ignore the reference outputs.** Remove `golden_tests/` and
   `golden_tests/references/` from `.gitignore`. Keep `tests/`, `config.ini`,
   `GEMINI.md` and `MEMORY.md` ignored, per the standing rule in `GEMINI.md`.
2. **Capture a fixture corpus.** For each scenario, run with `--debug` and commit
   the resulting `debug_step_00_RawHeaders.csv` as
   `golden_tests/fixtures/<name>_raw.csv`. At ~17 KB gzipped each these are
   comfortably committable and contain only header metadata, no pixel data.

   **Be clear about what this can deliver today: Sadr only.** Fixtures cannot be
   captured for TESTs 1–5 because their source data is absent from this machine
   (table above) — the capture requires the original frames. Until you restore
   that data the corpus is a single scenario, which exercises no calibration
   matching at all (Sadr has zero darks/flats/bias) and therefore leaves A1, A2,
   A10 and A11 without regression cover. Treat restoring TEST 2/3 data as the
   highest-value unblocking action in this entire plan.
3. **Add synthetic binary fixtures** for the paths CSV replay cannot exercise:
   the XISF XML parser, the FITS HDU selection, and the filename GAIN/FILTER
   fallbacks. Hand-built FITS/XISF files with valid headers and a 1×1 pixel
   array are a few KB each. Commit under `golden_tests/fixtures/binary/`.
4. **Install pytest into the venv** — `/mnt/raid0/Code/venvs/.astrovenv` does
   not have it, so `tests/` has never actually run.
5. **Fix the existing test.** `tests/test_imports.py:47` asserts `'2.0.2'`
   against modules declaring `'2.0.3'`; the suite fails on contact. Replace the
   hardcoded literal with an import from the single version source (item B8).
6. **Write `golden_tests/run_golden.py`** — replays every fixture through
   `--test`, diffs against `references/`, normalising only the `Generated` line.
   One command, no external data, machine-independent.

Only once step 6 goes green on the current code does the rest of this plan begin.

---

## Bucket A — defects that change output

**Status: complete.** Fourteen items total (A1–A14 — A13 and A14 both found
live during validation, not in the original plan) are fixed, each in its own
commit with its own verification. Roughly half turned out, on rigorous
re-checking against the real pipeline rather than the code-reading-only
original audit, to describe a real mechanism but not a currently-reachable
bug (A5, A7's HISTORY half, A8) — those are noted and fixed defensively
rather than corrected as regressions, since there was no live behaviour to
compare against. The other half were proven live, several only visible once
tested against real target directories beyond the single committed Sadr
fixture — SH2 101 (a second full dataset, masters only) and, for A14, a third
scenario: the user's actual light-frame directory alongside its real
*unprocessed* calibration directory (raw bias/dark/flat subs, not masters) —
the first time this session anything exercised genuine calibration matching
end to end rather than a synthetic reproduction: A1, A3/A4, A6, A10/A11, A13,
A14. See each item below for what was actually verified and how.

### A14. Darks/bias report table incorrectly grouped by filter **[proven live, fixed]**
`engine/reports.py`

Found only once real unprocessed calibration data was available: the capture
software stamps a `FITSKeyword FILTER='Ha'` into every bias and dark frame's
header too, not just lights' — presumably recording whatever filter happened
to be mounted, which is physically meaningless for a bias/dark exposure.
Confirmed directly in the raw XISF header, not just the filename convention
that also carries it.

`format_image_type_table`'s calibration branch always grouped by
`FILTER_NAME` regardless of type. In the tested dataset every dark/bias frame
happened to carry the same filter tag, so the only visible symptom was a
misleading label (`MASTERDARKS` showing `Ha` where it should be blank) — but
a dataset with dark/bias frames captured across more than one actual filter
position would have fragmented one logical calibration set into multiple
report rows, each understating its own Frames count.

`calibration.py`'s actual matching (`dark_candidates`, `bias_candidates`) was
never affected — neither has ever constrained on filter — so the acquisition
CSV's `darks`/`bias` columns were correct throughout; this was a
session-summary display bug only. **Fixed** by mirroring `calibration.py`'s
own semantics: the report's group key now drops `FILTER_NAME` for Dark and
Bias, keeping it for Flat and FlatDark (which do depend on it there too).
Verified live: `MASTERBIAS`/`MASTERDARKS` rows now correctly show a blank
filter column; counts unchanged from before the fix.

These are the real bugs. Each **must** get its own commit with its own attributed
golden diff: every changed byte traced to a named fix, anything else is a
regression. Do not batch these.

### A1. Deduplication regex silently destroys exposure data **[proven]**
`engine/steps/deduplicate.py:56`

```python
df['base_filename'] = df[FILENAME].str.extract(
    r'(.+?)(?:_c.*)?(\.xisf|\.fits|\.fit|\.fts)', flags=re.IGNORECASE)[0]
```

`.+?` is non-greedy and `str.extract` uses `re.search` (unanchored), so the
engine finds the *shortest* prefix that lets the rest match. Any `_c` anywhere
in the filename truncates the key:

| Filename | Extracted base |
|---|---|
| `M31_Light_001.fits` | `M31_Light_001` ✅ |
| `M31_Light_001_c.xisf` | `M31_Light_001` ✅ |
| `NGC7000_Light_005.fits` | `NGC7000_Light_005` ✅ |
| `NGC7000_calibrated_Light_005.fits` | **`NGC7000`** ❌ |

Every frame of a target whose filename contains `_c` — `_calibrated`,
`_cropped`, a filter named `_clear`, a target like `IC_1396` written
`ic_cocoon_...` — collapses into a **single** row. The frames are discarded
silently, total integration time is under-reported, and nothing is logged
because `DeduplicateStep` only logs the aggregate count removed.

**Fix.** Anchor the pattern and require the postfix to sit immediately before
the extension, with an explicit alternation of known WBPP postfixes
(`_c`, `_cc`, `_r`, `_rn`, `_d`, `_b`, `_s` and their combinations) rather than
`_c.*`. Move the pattern to a `RegexPatterns` class in `constants.py` (see B-items).
Add a DEBUG log line naming each file dropped and the survivor it lost to.

### A2. Deduplication key ignores the directory
`engine/steps/deduplicate.py:71`, caused by `engine/extractor.py:139,166`

The extractor stores only `os.path.basename(filepath)`; the full path is thrown
away at read time. Dedup then groups on that basename alone. Two sessions that
each contain `Light_0001.fits` — the default naming of several capture packages,
and the norm when you pass two directories on the command line — collapse to one
frame.

This is the single most likely cause of an under-count in ordinary use, and it
compounds A1.

**Fix.** Add a `source_path` column carrying the absolute path. Key dedup on
`(dirname, base_filename)`.

**Backwards compatibility — do not skip this.** Adding a required column breaks
replay of any `emergency_raw_dump.csv` or `debug_step_00_RawHeaders.csv` written
by an earlier version. `PROGRAM_OVERVIEW.md` sells that exact flow as the crash
recovery mechanism ("preserving scanned metadata for immediate recovery using
the `--test` flag"), so a hard requirement here is a user-facing regression, not
merely a fixture chore. `extract_from_csv` must detect an absent `source_path`,
fall back to filename-only keying, and log a warning naming the limitation. The
P0 fixture corpus must also be regenerated after this commit — sequence it early.

### A3. GPS clustering reassigns points away from their cluster **[proven]**
`engine/steps/geocode.py:68–85`

```python
unique_coords.loc[dist < dist_threshold, 'site_cluster'] = cluster_id
```

The mask is unconditional — it overwrites points **already assigned** to an
earlier cluster. Three colinear readings 0.0008° apart (well inside the 0.001°
threshold, so morally one site):

```
   sitelat  sitelong  site_cluster
0  52.0000       0.0             0
1  52.0008       0.0             1     ← stolen from cluster 0
2  52.0016       0.0             1
```

Cluster 0 is stripped of its only other member, so its "centroid" is a single
un-averaged reading — defeating the entire stated purpose of the step. One
imaging site becomes two, and the report emits two site blocks.

Compounding it: the closing log line reports `cluster_id`, which counts *seeds*,
not surviving clusters, so the count shown to the user can exceed reality.

**Fix.** Restrict the assignment to `site_cluster == -1`, which yields a
deterministic greedy clustering; or replace the whole loop with proper
single-linkage agglomeration, which is what the docstring actually describes.
Report `nunique()`, not the seed counter.

### A4. Euclidean distance on degrees
`engine/steps/geocode.py:65,79`

`np.sqrt(dlat² + dlon²)` treats a degree of longitude as equal to a degree of
latitude. At the site in the reference data (52.25°N) 0.001° of longitude is
~68 m, not the ~110 m the comment claims; at 60°N it is ~56 m. The cluster
radius silently narrows as you move north, and the same code is used for the
calibration-frame coordinate alignment in `_align_coordinates`.

**Fix.** Use `geopy.distance.distance` — already a dependency — and express the
threshold in metres as a named constant.

### A5. `fillna("None")` corrupts numeric group keys **[proven, fixed]**
`engine/steps/aggregate.py`, Stage 3 (`agg_cols` grouping-key fill)

```python
for col in agg_cols:
    ...
    df[col] = df[col].fillna("None")
```

`agg_cols` includes `gain`, `exposure` and `xbinning`. Filling a numeric column
with a *string* promotes it to `object` dtype:

```
dtype after fillna: object
    gain filter  n
0    0.0     Ha  4      ← was int64 100 → now float in an object column
1  100.0     Ha  3
2   None     Ha  3
```

So a single NaN anywhere in `gain` would make the acquisition CSV write
`100.0` where it should write the integer `100`.

**Correction after re-checking against the real pipeline, not just the
standalone snippet above**: this does not currently happen. `gain`,
`xbinning` and `exposure` are all unconditionally filled and cast to a
non-null numeric dtype by `NormalizeHeadersStep`'s Stage 7 hardening
(`base.py`) before `AggregationStep` ever runs, and nothing between the two
steps can null them out again — so the promotion shown above is not reachable
today. **Fixed anyway**: the column is now checked with
`pd.api.types.is_numeric_dtype` and filled with `0` rather than `"None"` when
numeric, so a future change to that hardening guarantee fails safe instead of
silently corrupting the acquisition CSV. Verified against both fixtures with
no output change, as expected for something that wasn't live.

The two `agg_cols` entries that genuinely can be null when this step runs —
`filter` and `target` — are string columns, where `"None"` is the correct,
harmless fallback; confirmed live in the SH2 101 data (calibration masters
with no FILTER/OBJECT header). The `filter` case's downstream display symptom
was already fixed as part of A13.

Sort-order note for the Rust port still stands, but the premise changes: the
group-key sort order is type-dependent *by design* here (numeric columns stay
numeric), not as a side effect of null corruption. See `RUST_PORT_PLAN.md`
hazard 2.

### A6. Dead `[override]` entries — typo'd target and list-value corruption **[proven, fixed]**
`config.ini.example`, `engine/loader.py`

`[override] SWCREATOR = CREATOR` writes a column named `swcreator`. Every
consumer reads `InternalColumns.SWCREATE == 'swcreate'`. The capture-software
override has never worked; users editing it see no effect and no warning.
Confirmed present in the user's own real `config.ini`, not just the example.

**Second bug found in the same function while fixing the first**:
`_normalize_overrides` only checked `isinstance(v, str)`. ConfigObj parses a
comma-separated override value (`SQM = AOCSKYQ, AOCSKYQU`, also present in
the user's real config) into a native Python **list** before this method ever
runs, so it fell into the scalar branch and got `str()`'d whole — a
one-element list containing the literal text `"['AOCSKYQ', 'AOCSKYQU']"`,
which can never match a real column. That override — try `AOCSKYQ`, fall
back to `AOCSKYQU` — has been entirely dead too.

**Fixed**: corrected the shipped template's key (the user's own `config.ini`
is left untouched — that's a behavioural choice for them, not a code-fix);
added an `isinstance(v, list)` branch using the list's own items; added
load-time validation of every `[override]` target against the known internal
column names, logging a warning naming the likely typo. Confirmed the warning
fires correctly against the user's real config, naming `SWCREATOR` and
suggesting `swcreate`.

Both bugs were live but produced no observed output change on either
fixture — neither's raw headers contain a `CREATOR` or `AOCSKYQ`/`AOCSKYQU`
keyword — so this closes two dead configuration paths without an available
before/after example. The fix is `git log` `91b77f6`.

### A7. Only HDU 0 is read — real, but the "HISTORY collapse" half wasn't **[proven, fixed]**
`engine/extractor.py`, `_read_fits`

Originally written up as two problems. Both were re-verified empirically
before fixing anything, since A5 had already shown a read-the-code-only audit
can overstate a bug:

- **"`dict()` collapses repeated `HISTORY` cards" — does not hold up.**
  astropy's `Header` object already aggregates every `HISTORY` card under one
  key as a single `_HeaderCommentaryCards` sequence; `dict()` carries that
  through unchanged, and iterating it (as `_get_fit_number` already does)
  correctly finds the right line regardless of where it sits among other
  `HISTORY` entries. Tested with the target line first, last, and buried in
  the middle: all three parsed correctly with the *existing*, unmodified
  code. No fix needed, none made.
- **"Files whose primary HDU is empty ... yield a blank header" — real.**
  Confirmed with a genuine compressed-FITS structure (`CompImageHDU`, the
  `.fits.fz` convention): `IMAGETYP`/`EXPOSURE`/etc. land on the image
  extension (HDU 1), and the primary HDU (HDU 0) carries only structural
  boilerplate (`SIMPLE`/`BITPIX`/`NAXIS`/`EXTEND`). `dict(hdul[0].header)`
  unconditionally reads the boilerplate-only header, silently missing every
  real keyword.

**Fix.** `_read_fits` now scans the HDU list for the first one containing
`IMAGETYP` and reads that one, falling back to HDU 0 (today's behaviour) if
none carry it. Verified against a synthetic `CompImageHDU` file (correctly
now reads `IMAGETYP`/`EXPOSURE` from HDU 1) and a normal uncompressed FITS
file (unaffected). No fixture in the committed corpus exercises FITS file
reading at all — both replay via `--test` CSV injection, which bypasses it —
so this fix has no regression coverage from either fixture; P0 already calls
out synthetic binary fixtures as future work for exactly this gap.

### A8. Defaults were injected before case normalisation **[fixed defensively]**
`engine/steps/base.py`

Originally written up as: Stage 2 injects `[defaults]` under uppercase keys;
Stage 3 lowercases all columns; duplicates are then coalesced with
`groupby(level=0, axis=1).first()`, which picks by column position, so
whether a real header value or an injected constant survives depends on
column ordering.

**Re-verified before fixing, since A5/A7 had already shown a
read-the-code-only audit can overstate a bug.** Reconstructed the exact
Stage 2 → Stage 3 sequence in isolation and could not reproduce the claimed
overwrite. Two things account for it: default injection only ever *appends*
a new column (it's skipped outright when the exact-case key already exists),
so an injected default always lands to the right of any pre-existing genuine
column; and `groupby(axis=1).first()` genuinely coalesces row-by-row — first
*non-null* wins, verified separately with complementary-NaN columns — not
"leftmost column regardless of content." Given real FITS/XISF data's
consistent uppercase-keyword convention, no realistic case produces the
described overwrite.

**Fixed anyway**, as a defensive simplification rather than a behavioural
correction: column case is now normalised *before* default injection, which
removes the implicit dependency on append-order and file-scan-order the old
sequence rested on, regardless of whether it was ever shown to misbehave. No
output change on either fixture, as expected.

The deprecated `axis=1` groupby call itself is **not** removed by this —
duplicate columns can still arise from genuinely differently-cased raw
headers even post-reorder, so the call remains, and its pandas 2.2
deprecation (removed in pandas 3) is left for the Bucket B batch (B2) as a
separate, mechanical fix.

### A9. Row order is non-deterministic — `first()` aggregations are unstable
`engine/extractor.py:71–77`, `engine/steps/aggregate.py:50`

`as_completed(futures)` yields futures in **completion** order, so `raw_df` row
order varies between runs on identical input. `df.sort_values(DATE_OBS)` then
uses pandas' default `kind='quicksort'`, which is **not stable**, so frames
sharing a timestamp are ordered arbitrarily.

Everything downstream that resolves a tie by position inherits this: the dedup
`.iloc[0]` survivor pick, `agg('first')` for `instrume`/`telescop`/`focallen`/
`filename`, the `iloc[0]` reads throughout `reports.py`, and the A8 column
coalesce. TEST 6 reproduces only because those columns happen to be constant
within every group.

This is a prerequisite for the Rust port: byte-parity against a target that
isn't deterministic is not a well-defined goal.

**Fix.** Sort `file_paths` before dispatch and sort the assembled DataFrame by
`source_path` after collection; use `kind='mergesort'` (stable) for every
`sort_values`.

### A10. Master preference discards legitimate masters **[resolved, fixed]**
`engine/steps/base.py`, `engine/steps/calibration.py`

Both sites did `candidates[is_master_mask].iloc[[0]]` — keep exactly one
master per hardware group, arbitrarily by scan order. If two master darks at
the same gain/binning/duration genuinely existed from different dates, one
was dropped along with its `NUMBER` sub-exposure count.

**User decision**: masters should always be the latest available; there
normally shouldn't be two in a directory for a given gain/offset in the
first place, but if there are, prefer the most recent by `DATE-OBS` rather
than an arbitrary pick.

**Fixed** in both locations that make this choice: `base.py`'s
`_execute_master_preference` (Stage 5, general ingest filter) and
`calibration.py`'s `resolve_count()` (per-light candidate matching — this
was unreachable before A13 fixed IMAGETYP normalization; this is the first
fix to actually exercise it). Both fall back to "first found under the
deterministic scan order" (A9) if no candidate has a usable `DATE-OBS`.

Verified live against the SH2 101 fixture (not in the committed corpus): it
has a genuine duplicate `masterBias` pair (a literal `_(1)` copy file), and
the debug log now reads the group and the `DATE-OBS` kept for both affected
groups. Output there is otherwise unchanged — the duplicate pair has
matching `NUMBER`/exposure/gain, so which one backs the aggregate doesn't
move the reported numbers.

### A11. Flat-dark matching skipped binning *and* master preference **[proven live, fixed]**
`engine/steps/calibration.py`

Darks, bias and flats are all resolved through `resolve_count()` — which
applies master preference — and all constrain on `BINNING`. Flat-darks did
neither, summing master + raw candidates unconditionally with no binning
check.

**User decision**: make it consistent with the other three types.

**Verification found this is a real, live bug, not just a consistency
gap** — but only once isolated from two pre-existing, overlapping defenses
that hid it in simpler test cases: `base.py`'s Stage 5 master preference and
`calibration.py`'s own orphan-pruning both already key on binning for every
calibration type, including `DARKFLAT`, via an incidental substring match on
`"DARK"`. The live-reachable case needed two lights at *different* binnings
(so neither binning's calibration anchors get orphaned) plus two legitimate
flat-dark masters, one per binning. Under the old code, **both** lights were
assigned `flatDarks = 1039` (`40 + 999`, both masters summed) regardless of
their own binning; the fix correctly assigns `40` to the bin-1 light and
`999` to the bin-2 light.

**Fixed**: routed flat-darks through `resolve_count()` and added the
`BINNING` constraint, making all four calibration classes identical in
structure. Neither fixture contains a `DARKFLAT` frame, so this was verified
with a targeted synthetic reproduction rather than either fixture.

### A12. Vectorising `OpticalParameterStep` changes rounding **[proven]**

Filed here rather than in Bucket B, where it superficially belongs, because the
rewrite is **not** value-preserving and cannot be held to an empty golden diff.

`optical.py:80` currently uses Python's builtin `round(x, 2)`, which is
decimal-correct half-to-even. The natural vectorised replacement is pandas
`.round(2)`, which is numpy's multiply–`rint`–divide on the binary double. They
disagree, and the disagreement survives `%.2f` formatting into the report:

```
   value   py round()  pd .round()
   2.675         2.67         2.68    ← report prints 2.67 vs 2.68
   2.665         2.67         2.66
   1.115         1.11         1.12
   0.005         0.01         0.00
   3.345         3.35         3.34
```

5 of 10 tested boundary values differ; all 5 are visible in the rendered report.
`hfr`, `imscale` and `fwhm` all flow through this path, and `meanFwhm` is a
column of the acquisition CSV.

Two further traps in the same rewrite: HFR extraction moving from per-row
`re.search` (with its `> 0` guard and default fallback) to `.str.extract` +
`fillna` changes the null path; and `float(...)` inside `try/except` becoming
`pd.to_numeric(errors='coerce')` means the `flen > 0` guard must now also handle
NaN, which `NaN > 0` silently answers `False`.

**Fix.** Vectorise, but route rounding through an explicit helper that
reproduces builtin `round` semantics rather than calling `.round()`. Then
diff-check `hfr`/`imscale`/`fwhm` across the Sadr fixture before blessing.
Whatever is decided here becomes the rounding contract the Rust port must match
— see `RUST_PORT_PLAN.md` hazard 3.

### A13. IMAGETYP normalization erases its own MASTER labels **[proven, fixed]**
`engine/steps/base.py`, Stage 6 (IMAGETYP Normalization)

Found live, testing against a real user directory (SH2 101) rather than
either committed fixture — the Sadr fixture's darks happen to already be
labeled plain `DARK` in the raw data, so it never exercised this path.

The keyword-replacement loop matched each keyword against `df[itype_col]`
*after* prior iterations had already mutated that same column. Traced step
by step:

```
keyword='MASTER DARK' -> 'MASTERDARK'   (before='MASTER DARK')  -- correct
keyword='MASTERDARK'  -> 'MASTERDARK'   (before='MASTERDARK')   -- harmless no-op
keyword='DARK'        -> 'DARK'         (before='MASTERDARK')   -- clobbered back down
```

`'DARK'` (4 chars, sorted last since the loop processes longest keywords
first) matches `'MASTERDARK'` as a substring of *its own normalized
output*, so every master calibration frame silently lost its master
designation — unconditionally, regardless of naming format, not something
specific to PixInsight's current WBPP output as first suspected.

**Fixed**: match against a frozen snapshot of the original values, and mark
each row assigned once a keyword claims it, so shorter/later keywords never
re-examine an already-normalized value (`git log` `a927727`).

**Consequence found while fixing it**: `format_image_type_table` in
`reports.py` re-filtered its input for an exact match against the
category's *base* type only. That was silently dropping every row whenever
a group contained solely its MASTER variant — invisible before only
because the base.py bug independently collapsed those rows down to the
base type anyway; two bugs canceling out. Fixed alongside A13 since leaving
it would have made the base.py fix a visible regression (entire calibration
sections vanishing for master-only datasets, which is the common case).

**Second consequence, unrelated to either bug above**: the Filter column
was leaking the literal text `"None"` for calibration frames with no
FILTER header (XISF masters commonly have none) in a dataset where other
frames do carry a FILTER column. Two different "no filter" sentinels exist
— `'No Filter'` (the configured default, injected only when a file has no
FILTER *column* at all) and `'None'` (`AggregationStep`'s null-safety fill,
applied per-cell when the column exists but one row's value is missing).
Only the former was recognized by the blank-out check. Fixed in the same
commit.

**What this does *not* change**, contrary to the initial hypothesis: a
clean `git stash`/pop A-B comparison against the SH2 101 fixture showed
exposure times and frame counts were already correct before this fix, for
this dataset — `NormalizeHeadersStep`'s Stage 5 master-preference filtering
already drops raw subs in favour of a same-signature master *before* Stage
6 runs, independent of this bug. The originally reported symptom ("masters
report zero exposure") did not reproduce against this repository on either
`main` or this branch; whatever produced it on the user's machine was a
different code path, not this one. The one proven, isolated effect was the
Filter-column leak above.

**Structural risk fixed regardless**: `CalibrationMatcherStep.resolve_count()`
checks for a `'MASTER'` substring to prefer a master over raw subs when
both are present as *candidates* for a given light frame. That check was
permanently unreachable before this fix — no row could ever carry a
`'MASTER'`-containing label by the time it ran. No live double-counting was
found in the tested dataset (Stage 5 already prevents the raw/master
coexistence that would trigger it), but the check is now live rather than
dead code, and is exactly the mechanism A10/A11 depend on to matter at all.

---

## Bucket B — corrections with no output change

**Status: complete.** All fifteen items (B1–B15, less B13, reclassified as A12)
are fixed, batched into six commits as originally planned, each verified with
an empty golden diff. Two items surfaced real, worth-recording surprises while
being implemented — B2's "obvious" pandas-suggested fix (`frame.T.groupby(...)`)
turned out to be dtype-unsafe on this codebase's mixed-type frames, and fixing
B3 in `base.py` uncovered a genuine `NameError`-in-waiting (a closure
referencing `logger` before it was assigned in the enclosing scope, never
triggered because nothing had used it) — see each item below.

| # | Item | Location | Status |
|---|---|---|---|
| B1 | Fake progress loop — `for i in range(1, total+1): print(...)` iterates doing nothing but I/O | `steps/aggregate.py:107–111` | done |
| B2 | `groupby(level=0, axis=1)` — FutureWarning on pandas 2.2, **removed** in pandas 3 | `steps/base.py:96` | done — the stdlib-suggested `.T.groupby(...).T` replacement was dtype-unsafe here (upcasts every column to object on a mixed-type frame, reproducing A5's bug at a new site); replaced with a per-duplicate-group coalesce instead, verified with `.equals()` |
| B3 | Silent `except: pass` / `except Exception: pass` — add specific types and a `logger.debug` | `AstroBinUpload.py:278`; `extractor.py:212,223,239`; `geocode.py:175,205`; `reports.py:44` | done — plus 5 more found in `base.py`/`calibration.py` not in this original location list; fixing `base.py`'s uncovered a real latent `NameError` (a closure used `logger` before its enclosing-scope assignment) |
| B4 | `inspect.stack()` runs on **every log record** — walks and reads source for every frame; also `%(lineno)d` reports the logging call site, not the resolved frame. Replace with `%(funcName)s` | `AstroBinUpload.py:100–131` | done — the `%(lineno)d` half of this claim didn't hold up on inspection (it's captured before any Filter runs, so was never actually affected); corrected while fixing |
| B5 | Worker logging is fork-only: `getLogger("AstroBinV2")` in a spawned process has no handlers, so all per-file parse errors vanish on macOS/Windows | `extractor.py:110` | done — added a `ProcessPoolExecutor` initializer; verified live against SH2 101 (2587 DEBUG lines correctly attributed) |
| B6 | Dead code — unused `Nominatim` import; `[secret]` plumbed through `AppConfig` but never read (the light-pollution API is documented, never implemented); `pipeline.py` is an empty version-handshake stub; unused `Path`, `numpy`, `Tuple`, `ConfigSections` imports | `geocode.py:20`; `loader.py:75`; `pipeline.py`; various | done — ran pyflakes across the whole codebase to find every instance definitively rather than continuing to spot them incidentally |
| B7 | `requirements.txt` is a raw `pip freeze` — matplotlib, bs4, jupyter, ipython, requests, PyYAML, debugpy are not used. Replace with real deps (`astropy`, `pandas`, `numpy`, `configobj`, `geopy`) plus a separate lock file | `requirements.txt` | done — trimmed to `astropy`, `pandas`, `numpy`, `configobj`; `geopy` dropped too, since A3/A4's haversine fix made it a non-dependency |
| B8 | `__version__` duplicated across 14 files; `verify_engine_integrity` trips on any partial edit. Collapse to one `_version.py` imported everywhere | all modules | done in P0 phase |
| B9 | Test suite asserts `'2.0.2'` vs actual `'2.0.3'` → fails; pytest not installed in the venv | `tests/test_imports.py:47` | done in P0 phase |
| B10 | No validation that input paths exist or are readable before the pipeline runs | `AstroBinUpload.py:188` | done — verified a bad path now fails immediately with exit 1 and a clear message |
| B11 | Magic numbers → named constants with provenance comments: 5 h session gap, 0.001° cluster radius, 0.0001 EGAIN tolerance, 206.265 arcsec conversion, `FWHM = HFR × 2` | `aggregate.py:57`, `geocode.py:65`, `calibration.py:39`, `optical.py:70,77` | done — cluster radius was already named as part of A3/A4 |
| B12 | Exporter hardcodes `'imagetyp'` instead of the constant, and `acq_source[list(mapping.keys())]` raises a bare `KeyError` if any column is absent | `exporter.py:60,96` | done |
| ~~B13~~ | *Moved to **A12** — vectorising this step changes rounding and therefore changes output. It cannot be held to an empty golden diff.* | `steps/optical.py:85–89` | done as A12 |
| B14 | Version drift in docs — `PROGRAM_OVERVIEW.md` and every module docstring still say v2.0.2 | repo-wide | done — `CHANGELOG.md`/`ReleaseNotes.md` deliberately left untouched (historical record) |
| B15 | Remove the `[secret]` section from the shipped example rather than leaving an unused credential slot as an attractive nuisance. (No real key is committed: `config.ini` is correctly gitignored and holds only a placeholder.) | `config.ini.example:38` | done — also removed from the auto-generated template in `loader.py`; the user's own real `config.ini` left untouched |

---

## Sequencing

```
P0   verification substrate            ← blocks everything          [done, a2a0741]
 │
B8   single version source             ← blocks partial commits      [done, 63aa3a7]
     (integrity check dropped per user decision, not just centralized)
 │
A9   determinism                       ← blocks meaningful diffs     [done, 4fc8382]
A2   source_path in extractor          ← changes fixture schema      [done, 53ffb42]
A13  IMAGETYP master-label collapse    ← found live on SH2 101,      [done, a927727]
     (+ its reports.py knock-on)         not in either fixture
A1   dedup regex over-truncation       ← anchored, vocab from real   [done, ff3d19b]
                                          WBPP output (SH2 101)
A3 · A4  cluster stealing / geodesic   ← haversine, not raw geopy    [done, 2091c68]
                                          (perf; see commit message)
A5   numeric group-key fillna          ← not actually reachable      [done, 950ffca]
                                          today; fixed defensively
A6   dead [override] entries           ← 2nd bug (SQM list) found    [done, 91b77f6]
                                          fixing the 1st
A7   HDU selection                     ← "HISTORY collapse" half     [done, 63f5886]
                                          disproven; other half real
A8   default-injection ordering        ← not reachable on real data; [done, 784f831]
                                          fixed defensively anyway
A10 · A11  calibration semantics       ← resolved per user decision   [done, 00cb110]
                                          A11 proven live once isolated
A12  vectorize OpticalParameterStep    ← highest-rigor verification:  [done, 6b32663]
                                          772/772 real frames match
 │
B1–B15 (less B13)                      ← batched, 6 commits            [done, 760f0a5..94360e6]
                                          golden diff empty throughout
 │
tag v2.1.0                             ← parity contract for the Rust port    [remaining]
```

**All of P0, Bucket A, and Bucket B are complete.** The only remaining step is
tagging `v2.1.0`, which becomes the parity contract `RUST_PORT_PLAN.md` is
written against.

Rationale for the two blockers: B8 first because the original
`verify_engine_integrity` aborted the program on any version mismatch, which
made incremental commits unrunnable — resolved by dropping that check
entirely rather than just centralizing the version strings, per user
decision. A9 next because a golden diff against non-deterministic output
cannot distinguish a fix from noise.

A13 was not in the original plan — found by testing against a second, real
target directory (SH2 101) rather than relying on the Sadr fixture alone.
That directory is not part of the committed corpus (kept local-only, per
user decision) but the finding and fix are real and apply regardless.

## Estimate

P0 is the bulk of the calendar time and most of it is data recovery only you can
do. The code work is roughly: Bucket A ~2 days, Bucket B ~1 day, test suite
~2 days.

## Open questions for you

1. **A10** — which master-preference semantics do you want?
2. **TEST 1–5 source data** — can it be restored, or should the corpus be
   rebuilt from whatever datasets you still hold?
3. Bucket A changes output **by design**. The reference files must be
   regenerated and re-blessed after each fix. Do you want to review each
   attributed diff before it is blessed, or only the cumulative one at v2.1.0?

---

## GitHub issue reconciliation

The audit above was written as an internal findings list and never cited an
issue number, even though most of Bucket A was derived from the open tracker.
This section closes that loop. Verified against the `v2.1.0` code
(2026-09-07).

| Issue | Reporter's ask | Maps to | Status | Close action |
|---|---|---|---|---|
| **#11** | A `_c`/`_C` token anywhere in a filename collapses every frame into one row | **A1** + **A2** | **Resolved.** Anchored `WBPP_FILENAME` regex; dedup key is now `(dirname(source_path), base_filename)`. Verified against the exact `CandidateDolphin` filenames in the report. | Close, referencing `ff3d19b` / #12 |
| **#4** + PR **#8** | `--config` flag to select an `.ini` per scope | pre-existing (v2.0.1) | **Resolved.** `--config` / `-c`, `AstroBinUpload.py:120`. | Close #4; close PR #8 as superseded |
| **#10** | Read `ImageIntegration.numberOfImages` from a master's HISTORY instead of reporting `1` | extractor deep-inspection | **Core resolved.** Parsed from XISF `ProcessingHistory` table rows, XISF FITSKeyword COMMENT/HISTORY, and FITS `HISTORY` (`_get_fit_number`). The optional `[calibrationoverrides]` ini fallback is **not** built. | Comment: ask for a re-test on `v2.1.0`; split the ini fallback into a separate `enhancement` |
| **#9** | Both a 180 s and a 600 s master dark detected; one dark per light | **A11** (binning), **A10** (master pref), **A13/A14** (labels + report) | **Mechanisms present.** Dark matching keys on gain + binning + duration; report no longer groups darks by filter. Reporter's output was pre-fix (v1.4.5). | Comment: ask for a re-test on `v2.1.0` before closing |
| **#3** | `ROTATOR` carries the mechanical angle, not the name | keyword constants | **Structurally resolved.** Name keys on `ROTNAME`, angle on `ROTANTANG`; `ROTATOR` is no longer read as the name. Vestige: the auto-generated default config still seeds `ROTATOR` rather than `ROTNAME`. | Fix the default-dict key, then close |
| **#6** | Built-in mapping of `AOCSKYQU`→SQM, `AOCAMBT`→FOCTEMP | **A6** (`[override]` plumbing) | **Achievable by config today** (`SQM = AOCSKYQU`, `FOCTEMP = AOCAMBT` in `[override]`). No zero-config default. | Decide: ship as default aliases, or document the `[override]` recipe and close |
| **#5** | Override *values* (e.g. `EAF` → full focuser name) | — | **Not built.** `[override]` remaps keywords, not values. Genuine feature request. | Keep open as `enhancement` |

**Blocked bookkeeping**: the `gh issue close` / `gh pr merge` calls above are
denied by this session's auto-mode classifier. They need `Bash(gh:*)` added to
`autoMode.allow`, or the maintainer runs them.

**Remaining code work implied by the table**: the #3 default-dict key
(one line); the maintainer's decision on #6 (default aliases vs documented
recipe); #5 and the #10 ini fallback if wanted (both `enhancement`, out of the
v2.1.0 scope).
