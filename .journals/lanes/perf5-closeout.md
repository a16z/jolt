# PERF-5 closeout

Date: 2026-09-04. Base: `70250c497`. Mac mini M4;
`CARGO_TARGET_DIR=/Volumes/Dev/target/perf5-closeout`.

## Fixes

- `kzg_verify_batch`: `ecAdd = (k − 1) + 4 + 2 = k + 5`.
  ecMul, field operations, pairing pairs, transcript, and proof encoding unchanged.
- Bound-column extraction returns `Result`; its default returns
  `MissingEvaluationSource { kind: "bound columns" }`. Every extraction member
  supplies an explicit implementation, including column-less member 1 (hash wiring reuses hash-row columns) and
  the scalar link (which reuses T2/carry columns). A forgotten implementation now fails
  even against an empty plan. This avoids adding unused no-op implementations
  throughout the non-wrapper sumcheck kernels.
- Gate pins and current verifier-cost rows updated. `pr-body.md` untouched.

## Verifier cost

The old estimator bundled one multiplication and one addition into 7,700 gas
per MSM term. Separating the 150-gas addition component gives
`7,550 × ecMul + 150 × ecAdd`; the remaining calibrated overhead stays attached
to ecMul. This preserves the old estimate when both counts are equal and
reduces either packing by exactly 300 gas for the two removed additions.
This is modeled gas, not a new EVM measurement.

| packing | ecMul | ecAdd old → new | gas old → new |
|---|---:|---:|---:|
| k=16 default | 233 | 233 → 231 | 4,944,149 → 4,943,849 |
| k=32 comparison | 216 | 216 → 214 | 4,800,225 → 4,799,925 |

Other MSM accounting conventions are unchanged; this lane corrects the
specified `kzg_verify_batch` count only. Historical lane tables keep their
original measurements/accounting.

## Test audit

| retained coverage | independent oracle / distinct failure |
|---|---|
| `even_terminal_rechecks_claim` (new) | honest opening accepts; changing only the verifier claim rejects with `FoldingConsistencyFailed` at 2/4/6 variables |
| `hybrid_signed_chains_match_naive` (new) | established scalar-mul-plus-add oracle; signed bucket chains, mixed widths, u32 windows, identities, duplicate/negated points at 64/65/129/1025 points |
| `inconsistent_fold_with_valid_kzg_openings_rejects` (extended) | production standalone KZG verifier accepts; protocol rejects degree-127 middle/terminal folds at odd/even dimensions; existing coefficient mutation retained |
| `bound_columns_require_complete_consistent_coverage` (new) | full production column evaluation; missing or conflicting bound columns reject |
| `typed_rlc_skips_only_padding_across_row_blocks` (new) | integer weighted-sum ground truth across all four packed scalar types, short/full row blocks, with/without padding skipping |
| `unimplemented_bound_values_reject_empty_output` (new) | default extraction must return an error even for an empty result vector |

Five new tests, one existing test extended. The review's separate unbounded
fold fixture was merged into the existing valid-batch negative. Predicted-point
and residue-swap variants were dropped: they add coefficient-mutation cases
to the same fold-chain rejection already tested. Removed both environment-gated
runtime probes and their naive-prefix helper; no review-only code remains.
No old implementation was copied as an oracle.

`scheme.rs` grows 1,000 → 1,053 lines, entirely inside its existing test module.
Kept local to reuse its setup and private KZG helpers without moving unrelated
code or adding a second fixture helper.

## Verification

- `cargo fmt -q --message-format=short`: pass.
- All-target `cargo check` and `cargo clippy -q -- -D warnings` for
  `jolt-wrapper`, `jolt-crypto`, and `jolt-hyperkzg`, with
  `--features prover-fixtures`: pass.
- Six-worker unit suite: **244/244** in 130.479 s; crypto **146**, HyperKZG
  **31**, wrapper **67**. The first run caught the missing explicit hash-wiring
  implementation; the complete rerun above includes that fix.
- Prebuilt real-fixture gates: **1/1** at each packing; every tamper rejected.

| real gate | k=16 default | k=32 |
|---|---:|---:|
| payload / bincode / statement (B) | 7,392 / 7,533 / 352 | 7,104 / 7,232 / 352 |
| ecMul / ecAdd / pairing pairs | 233 / 231 / 8 | 216 / 214 / 8 |
| Fr multiplications / inversions / Keccak | 123,144 / 8 / 852 | 123,121 / 8 / 839 |
| modeled gas | 4,943,849 | 4,799,925 |
| honest online wall | 16.902 s | 20.029 s |
| mutex window (ET) | 00:24:29–00:24:56 | 00:26:02–00:26:37 |
| one-minute load on entry | 2.758 | 2.610 |
| competing compiler/test processes observed | 0 | 0 |

The runner acquires `/tmp/wrapper-gate.lock` with `mkdir`, retries after 60 s
unless one-minute load is below 4 and no competing compiler/test is active,
and releases with `rmdir` on every exit. Ten Rayon threads. Five-minute loads
were still high after compilation; these are correctness runs, not a new
performance comparison. Existing timing rows in `pr-tables.md` are unchanged.

Prebuild: `cargo nextest run -p jolt-wrapper --features prover-fixtures
--cargo-quiet real_wrapper --no-run`. Binary metadata uses
`cargo nextest list --list-type binaries-only --message-format json` with the
same package/features; runtime reuses it via `--binaries-metadata` and
`--cargo-metadata`, selecting `real_wrapper --no-capture`. Metadata execution
omits `--cargo-quiet`, which nextest rejects in that mode. An initial metadata
preflight failed before test execution (test-list JSON instead of binary-only
JSON); corrected before either accepted gate.

Logs and runner: `/tmp/perf5-closeout/{unit,check,clippy,default,k32}.log`,
`gate.py`, `binaries.json`, and `cargo.json`.
