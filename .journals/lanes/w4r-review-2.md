# W4-R review #2

Scope: committed tree `1f7ce468a`, reviewed read-only in a detached worktree against review #1,
the W4-R journal, the native Jolt stage implementations, and `dory-pcs-0.4.2`.

## Findings

1. **MINOR (score 50) — `crates/jolt-wrapper/src/relation/mod.rs:194`: the native parity
   diagnostic runs unconditionally in production witness generation.** `generate_witness` first
   replays the native verifier, then `native::check` rebuilds every stage's sumcheck owners and
   re-evaluates their derived terms. This check was specified as a debug-mode drift assertion in
   `.journals/plan-relation.md:92,125`; the change also exposes the test-only `NativeParity` counter
   through the public `Witness` API (`mod.rs:42,151-152`). Gate the second reconstruction to debug
   or test builds and keep diagnostic coverage out of the production witness type.

2. **MINOR (score 50) — `.journals/lanes/w4r-relation.md:56,59-60,100`: the headline counts were
   not fully updated after adding `Chi(sigma)`.** Stage 8 at 2^18 is 279 rows, not 278; the 2^20
   total/stage-8 counts are 5,455/292, not 5,454/291; and `DoryLinks` now has 175 named scalars,
   not 174. The response at lines 171-173 and the fixture constants carry the corrected row counts.

## Review-1 closure

- **Delta setup index fixed:** `relation/dory.rs:182-185` names the round-`j` scalars
  `Delta1R(sigma - j)` and `Delta2R(sigma - j)`. Native `DoryVerifierState::process_round` reads
  `setup.delta_1r[self.num_rounds]` and `setup.delta_2r[self.num_rounds]` at
  `dory-pcs-0.4.2/src/reduce_and_fold.rs:707,714`; `num_rounds` starts at `sigma`.
- **Real-proof negative control present:** `tests/relation_dory_native.rs:223-233` accepts the native
  `k -> k` base pairing and rejects the former `k -> k - 1` pairing; lines 235-248 also reject a
  Delta1R/Delta2R swap. The old name-to-base index cannot fail the relation's `check_witness`
  because GT setup bases are intentionally outside this R1CS; the deferred pairing test is the
  check that pins that external mapping.
- **Native derived parity present:** `relation/native.rs:185-219` compares every symbolic input and
  output `Source::Derived` and `Source::Challenge` through `derive_input_term`,
  `derive_output_term`, and `resolve_challenge`; lines 596-617 reject any registered id without an
  owner. The fixture pins 214 derived ids and 19 semantic challenge ids. Coverage includes Spartan
  Az/Bz weights/constants and TauKernel, uni-skip Lagrange terms, all stage 2-7 eq/LT/kernel/public
  terms, all 54 table terms, bytecode stage terms, and all 39 virtualization terms. Arithmetic
  intermediates are R1CS-constrained rather than separately registered as native ids.
- **Both review-1 minors fixed:** the journal's opaque inventory is 137 elements = 68 GT + 35 G1 +
  34 G2, and the public `table_gadget_values` helper plus duplicate `relation_tables` integration
  test are gone.

## Counts, tamper coverage, and external links

- Real 2^18 fixture: 5,254 rows, 6,761 variables, 45 public columns; the single added row is the
  materialized `Chi(sigma) = 1`. The code pins 2^20 at 5,455 rows. No unexplained relation growth.
- Tampers still fail at their intended owners: stage-1 round coefficient at row 38
  (`stage1/remainder`), `ValIo` at row 626 (`stage2/expected`), Dory gamma at row 5,082
  (`stage8/dory`).
- External interfaces at 2^18: 376 unconstrained squeeze variables require T1 hash-chain binding;
  1,199 absorbed-Fr occurrences before the final Dory squeeze require T1 byte binding; 175 named
  scalar entries feed T2; 45 public columns comprise 7 verifier-computed evaluations and 38 copied
  challenge outputs. These sets overlap. The relation schedule has 1,222 Fr occurrences total; its
  final 23 are the 22-coordinate opening point plus evaluation absorbed after the last squeeze, so
  no later Fiat-Shamir value depends on them.
- Non-variable external data: 137 opaque Dory elements and the 32-byte `state_in`. T1 must hash the
  same encodings; T2 must perform subgroup/encoding checks and the GT/four-pair equation. No
  accidental unowned relation source was found.

## Checks

- `cargo clippy -p jolt-wrapper --all-targets -q -- -D warnings`: passed.
- `cargo nextest run -p jolt-wrapper --cargo-quiet -E 'test(relation)'`: 7 passed.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_fixture --no-capture`:
  1 passed, 1 ignored; cached 2^18 proof used.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_dory_native --no-capture`:
  1 passed.

VERDICT: 0 blockers, 0 majors, 2 minors
