# W4-T1 review #2

Scope: `c4a218b14` plus `72190b623`; diff from `1c6155bb5` limited to
`crates/jolt-wrapper/src/hash_table/**`, hash-table tests, and `wrap.rs`.

## Findings

1. **BLOCKER — `crates/jolt-wrapper/src/hash_table/terms.rs:423-462`: both field-word forms accept
   non-canonical transcript bytes.** The link proves only a field equality:

   ```text
   canonical bytes(x)  -> fr_word = x
   bytes(x + q)        -> fr_word = x + q = x in Fr
   ```

   `q = Fr::MODULUS`, and `x + q < 2^255` for every canonical `x`, so every absorbed field value
   has at least one distinct 32-byte alias. Blake3 hashes the different bytes while the CopyLink
   sees the same `Fr`. The real-fixture oracle at `tests/hash_table_fixture.rs:372-377` calls
   `from_bytes_le_reduced`, so it accepts the same alias and cannot detect this. The aligned and
   two-byte-shifted forms have the same defect. This gives the prover per-message Fiat-Shamir
   grinding choices and does not model `AppendToTranscript`'s canonical serialization. **Fix:**
   constrain each 256-bit wire encoding to be `< q` (or link its exact bits to one constrained
   canonical decomposition), then link that decomposition to the relation scalar. Add `x + q`
   rejection tests for aligned and shifted wires.

2. **BLOCKER — `crates/jolt-wrapper/src/hash_table/schedule.rs:314-527`,
   `crates/jolt-wrapper/src/wrap.rs:140-160`: `SymbolicSchedule` is still built from the current
   proof's recorded run, not from the profile/verifier key.** The constructor consumes recorded
   append lengths, labels, squeeze positions, constants, and public bytes; lines 499-516 also put
   the proof's preamble-tail values inside `SymbolicSchedule`. The determinism test at
   `tests/hash_table_relation.rs:356-365` clears `tail` before comparing schedules, masking that
   dependency. `WrapPreparation::new` then derives the four alleged VK columns from that same
   per-proof schedule. There is no independently built schedule/key to compare against and no
   verifier path deriving `state_in`, first length/flags, or the tail from the external statement.
   **Fix:** build the byte-source schedule and four fixed-column commitments from
   `WrapperProfile` during key generation; keep only public offsets in it. Derive public values
   from preprocessing/public IO, replay proof bytes only as witness data, and reject any structural
   mismatch against the fixed schedule.

3. **BLOCKER — `crates/jolt-wrapper/src/wrap.rs:68-85`,
   `crates/jolt-wrapper/src/hash_table/prover.rs:50-79`,
   `crates/jolt-wrapper/src/hash_table/wiring_prover.rs:72-90`: the constraint-compression points
   and coefficients are caller inputs with no post-commitment Fiat-Shamir derivation.** Both T1
   provers need `Relation`/`WiringStatement` coefficients and `tau` before `wrap` commits the
   columns. The target stream draws only the member batching coefficients after commitments
   (`stream.rs:644-676`); it never draws the row/wiring `tau` or their constraint-batching
   coefficients. If those values are known at commitment time, a prover can cancel nonzero row or
   constraint errors in the one weighted sum. The negative tests do not exercise verification:
   `tests/hash_table_relation.rs:482-499` compares `WiringProver::input_claim()` with the expected
   value, while the stride case at lines 554-581 only calls the final formula with a different
   `tau`. **Fix:** add a post-phase-commit callback that draws all T1 row/wiring randomizers from
   the wrapper transcript, constructs both members, and gives the verifier the same derivation.
   Add full proof-verification negatives for label, preamble, next length/flags, `din`, `m_in`,
   round-0 `m`, and kernel stride.

4. **MAJOR — `crates/jolt-wrapper/src/hash_table/terms.rs:23-28,112-159,191-206`: the export is a
   lane-local API, not the shared term-stage contract.** Its `ColumnId` is a flat `usize`, it owns
   duplicate `AffineForm`/`Term` types, and it has no `TermExporter` implementation or observed
   arithmetic. The target assembly still serializes `factor_columns` claims
   (`stream/protocol.rs:79-110,206-256`), so this commit does not establish the claimed zero
   per-column-claim verifier path or its measured field-operation count. Flat IDs also collide
   across T1/T2/R without the shared `{group, slot}` mapping. **Fix:** consume
   `stream::{ColumnId, AffineForm, Term, TermContext, TermExporter}`, map T1 columns through the
   packing layout, implement observed term construction, and route the real verifier through the
   term stage.

5. **MINOR — `crates/jolt-wrapper/src/hash_table/terms.rs:37-109,161-186,341-379` and
   `crates/jolt-wrapper/src/hash_table/wiring_prover.rs:254-263`: public integration/diagnostic
   surface has no production caller.** `column_specs`, `members`, `kernel_counts`, and
   `WiringProver::final_parts` evade dead-code checks only because they are public; `kernel_counts`
   is test-only. **Fix:** have the shared exporter/assembly consume the required pieces and move
   diagnostics under tests; delete the rest until its first caller.

6. **MINOR — `crates/jolt-wrapper/tests/hash_table_fixture.rs:376`: an added fully-qualified trait
   path violates the nominal-import rule.** **Fix:** import `CanonicalEncoding` and use
   `<Fr as CanonicalEncoding>::from_bytes_le_reduced`.

## Enforced properties

- `wiring::source(position, slot)` is fixed code over 128-row cells. Its `Cell`, `Previous`, and
  `Next` reindexing is algebraically consistent: same-cell uses `eq`, previous reads use
  `eq+1(r, tau)`, and next reads use `eq+1(tau, r)`, with no wrap at either domain edge.
- The wiring member includes every entry of all 64 wired bit columns and 15 wired word columns.
  Constants, padding, and each following cell's length/flags are half-word-pinned by the four VK
  columns; the first state/length/flags and 22-byte tail enter the formula as public values. These
  facts become verifier enforcement only after findings 2 and 3 are fixed.
- The half-G permutation, rotations 16/12/8/7, seven rounds, chaining rows, and two challenge rows
  match `hash_table/blake3.rs`; that model matches the `blake3` crate for boundary lengths.
- Local term algebra is correct: the real fixture reports 230 terms, degree at most 2, and
  `sum(coeff * product(L(v)))` equals the batched row-plus-wiring final claim. Coefficients depend
  only on the supplied verifier context. Finding 4 blocks this from being the production verifier
  path.
- Challenge forms probe the production decoders. `challenge125` ignores the top three bits through
  `Mask(29)`; `challenge_scalar128` applies the required word-byte swaps. `LinkMap` covers all 376
  real challenges and 1,199 real field wires; the synthetic test covers the shifted wire class.
- Kernel construction visits 724 fixed entries and 30 value forms, independent of the number of
  cells. The estimated ~4.5k `Fr` work is plausible for `terms()`, but no production observer
  measures it in this target.

## Checks

- `cargo nextest run -p jolt-wrapper --features prover-fixtures hash_table --cargo-quiet`: 2 passed.
- Synthetic `hash_table_relation`: 4 passed; 724 entries, 537 structural kernels, 30 value forms.
- Real `hash_table_fixture`: passed; 1,819 active cells, 376 challenges, 1,199 wires, 230 terms,
  degree 2; stage A 0.741 s.
- `uv run scripts/check_style_invariants.py --base 1c6155bb5`: passed.

**VERDICT: 3 blockers / 1 major / 2 minors.**
