# W4-T1 review #1

Scope: committed tree at `985422c28`; `crates/jolt-wrapper/src/hash_table/**`,
`crates/jolt-wrapper/tests/hash_table_*.rs`, the W4-T1/M3 journals, the production
`Blake3Transcript`, and the R1CS BLAKE3 gadget.

## Findings

1. **BLOCKER — `crates/jolt-wrapper/src/hash_table/schedule.rs:39`,
   `crates/jolt-wrapper/src/hash_table/table.rs:76`: public and constant message links have no
   verifier-independent external identity.** `ItemClass::Wire`, `Element`, and `Squeeze` carry an
   indexed source, but `Public` and `Constant` are bare tags. Their `MessageLink` identifies only a
   `Recorded` log item and offset; neither `HashTable` nor `JoltSchedule` retains the public-field
   source or the expected constant byte. The only remaining byte value is in
   `chain.blocks[*].compression.block`, which is witness data. A stream/R1CS linker therefore must
   trust prover-derived bytes or reimplement the transcript schedule to pin the 22 public tail
   bytes, 23,584 constant bytes, block lengths, and flags. Without those pins, a prover can satisfy
   the local G relation for a different domain/preamble/message schedule and derive a different
   Fiat–Shamir chain. **Fix:** define one profile-derived symbolic schedule whose items carry
   `PublicSource` or exact constant bytes plus wire/element/challenge IDs; build verifier links and
   `Feed::Const` values from it, and use the recorded log only to fill and validate witness bytes.
   Add link-relation negatives for a public-tail byte, a label byte, a block length, and ROOT/CHUNK
   flags.

2. **MAJOR — `crates/jolt-wrapper/src/hash_table/table.rs:102`: the exported wiring form makes the
   sound verifier evaluate Ω(rows) metadata, with no structural evaluator for the EVM path.** At
   the measured shape this is 262,144 `RowFeeds` entries (1,310,720 `Feed` slots), including
   1,098,920 active feed slots, plus 174,624 message-copy rows, 116,416 byte links, and 376
   challenge links before 32-bit word-copy expansion. `Relation::final_check` itself is O(256), but
   it accepts claimed wired evaluations; binding those evaluations to committed columns and public
   inputs requires traversing these tables. This is the public-matrix O(rows) verifier path that the
   architecture marks as unusable on EVM. **Fix:** make fixed `(block, position)` wiring kernels and
   compact schedule/link ranges the verifier-owned representation; evaluate their MLEs in size
   independent of the 219,784 active rows. Keep explicit feeds only for witness materialization and
   cross-check them against the structural form.

3. **MINOR — `crates/jolt-wrapper/src/hash_table/schedule.rs:69`,
   `crates/jolt-wrapper/src/hash_table/table.rs:421`: state-row selection mixes global and
   table-local block indices.** `JoltSchedule::rlc_block` indexes `chain.blocks`, while
   `HashTable::chaining_rows` indexes `block_rows` within `schedule.blocks`; both are plain `usize`.
   The fixture has to subtract `schedule.blocks.start` manually at
   `crates/jolt-wrapper/tests/hash_table_fixture.rs:207`. Passing `rlc_block` directly selects the
   wrong Dory block for this fixture rather than failing. **Fix:** use distinct index types or have
   `HashTable::build(&JoltSchedule)` store typed `state_rlc_rows` and `state_out_rows` accessors.

4. **MINOR — `crates/jolt-wrapper/Cargo.toml:21`: the fixture feature's `tracer` dependency is
   absent from the committed lockfile entry for `jolt-wrapper`.** The requested fresh
   `prover-fixtures` run modified `Cargo.lock` by adding `"tracer"`; a clean locked build rejects
   the committed tree. **Fix:** regenerate and commit the lockfile after enabling the feature.

## Verified

- The half-G relation contains all 163 booleanity constraints, 64 XOR constraints, both half-step
  add equations with three boolean carries, and rotation wiring for 16/12/8/7. Chaining rows bind
  `out[0..8]`; squeeze rows bind `out[8..12]` and both challenge decoders match the production field
  functions.
- `1,819 = 267 + 1,017 + 535` compressions, 376 squeezes, and
  `120 * 1,819 + 4 * 376 = 219,784` active rows. Columns are 163 committed plus 64 wired bits and
  three wired words. The row polynomial is quadratic, hence degree 3 with `eq`.
- Witness generation records an accepted verifier run and reimplements only the hash trace. The
  real fixture checks every recorded state/challenge through the Dory `d` squeeze and reports
  1,199 Fr wires, 41 commitment GTs, 68 Dory GTs, 35 G1s, and 34 G2s.
- The single-bit tests alter relation inputs, not lengths. The missing negatives are verifier-link
  failures covered by finding 1.
- Scoped source files are 27–433 lines; no `#[allow]`, dead scoped helper, or second witness-side
  verifier implementation found.

## Checks

- `cargo clippy -p jolt-wrapper --all-targets -q -- -D warnings`: passed.
- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 9 passed, 1 fixture skipped; nextest marked the
  row-relation process leaky after success.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E
  'binary(hash_table_fixture)' --no-capture`: passed; recorded verify 0.120 s, build 0.304 s, row
  sumcheck 0.520 s; the run exposed finding 4's lockfile delta.

VERDICT: 1 blocker, 1 major, 2 minors
