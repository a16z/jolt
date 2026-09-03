# W4-T1 review #4

Scope: `ded155c15` on `wrap/spartan-hyperkzg`, limited to `hash_table/`, the pinning and
statement paths it calls, review #3, and Fix #3. Review work ran in detached `w4t1-review4`;
the added checks are saved in `.journals/lanes/w4t1-review-4-tests.patch`.

**VERDICT: 0 blockers / 1 major / 2 minors.**

## Findings

1. **MAJOR — `crates/jolt-wrapper/src/hash_table/eq.rs:1-5` and
   `crates/jolt-wrapper/src/wrap.rs:166-176,335-344`: the returned `VerifierCost` still omits
   T1 statement construction.** `verify_wrapped_with_key` derives `T1Challenges` and the two
   input claims through the unobserved `from_challenges` / `input_claims` methods before calling
   `verify_assembly_with_cost`; its returned counter therefore contains the 4,206 exporter
   multiplications but not the real fixture's 705 statement multiplications. The journal's manual
   sum 4,206 + 705 = 4,911 is correct, but the new module-level claim that T1's reported
   `VerifierCost` is execution-derived is false. Scratch repro: synthetic verification returns
   `fr_mul = 9,963`; the same verifier work including its 703 statement multiplications is 10,666.
   **Fix:** pass the verifier observer through `WrapVerifierKey::statement` using
   `from_challenges_with` and `input_claims_with`, then return the accumulated cost with the stream
   verifier's cost. Pin the full production-path count.

2. **MINOR — `crates/jolt-wrapper/src/hash_table/adapter.rs:71-76,102-105`: public correlated key
   fields can silently unpin a VK group.** `schedule`, `vk`, `packing`, and `commitments` can be
   changed independently; `pinned_commitments` zips the expected two-group range with the supplied
   vector and returns success when it is short. A truncated trusted key therefore hands the omitted
   VK group back to the proof. This is not a proof-only attack—the verifier key is trusted—but the
   public type does not preserve the invariant `commitments.len() == vk_group_range(...).len()`.
   **Fix:** make the fields private, expose read-only accessors, and make pin extraction reject any
   length mismatch.

3. **MINOR — `crates/jolt-wrapper/src/hash_table/layout.rs:96-99`: `FrLo2` has contradictory byte
   documentation.** The first two lines call it the high half-word of `m` two rows later; the next
   line correctly calls it the low half of that word. It is block bytes 8–9, hence shifted-field
   bytes 6–7. **Fix:** replace the three lines with the latter description.

## Adversarial answers

- **B1 byte derivation:** let `s[0..32]` be the field's canonical little-endian bytes. Transcript
  absorption reverses them, so `t[i] = s[31-i]` and the numeric top 64 bits are `t[0..8]`
  (`s[24..32]` in reverse order). At a shifted start, row `p` is
  `[old0, old1, t0, t1]`, row `p+1` is `[t2,t3,t4,t5]`, and row `p+2` begins `[t6,t7]`.
  Thus `2^48·bswap16(hi m(p)) + 2^16·bswap(m(p+1)) + bswap16(lo m(p+2))` is exactly
  `u64::from_be_bytes(t[0..8])`. The aligned form is exactly
  `2^32·bswap(m(p)) + bswap(m(p+1))`.
- **B1 coverage/attack:** the real fixture has 400 aligned wires at position 0 and 799 at position
  8; zero shifted wires. No real wire straddles a block. Synthetic shifted positions 5 and 13 cover
  the alternate class, including a position-13 cross-block value. The 35 Dory 32-byte values are
  `ElementKind::DoryG1`, not `Wire`; squeeze boundaries start new chain cells. `carry_case(1..=8)`
  carries into each top-window byte 0..7, and the `x[8..10] = 0x47b0` case is review #3's case.
  Every representable alias is `x+k·r`, `k=1..=5`; each has top word at least `MODULUS_HI`, while
  equality is also rejected. No non-canonical alias passes.
- **B2 pins/transcript:** with `k=16`, local ids 0..306 occupy 20 prover groups; ids 307..312 map to
  the two key groups. The proof must carry 20 commitments; a proof carrying 22 fails `StageCount`.
  `full_commitments` inserts key groups 20 and 21 before phase challenge derivation, term export,
  column reduction, and the final HyperKZG opening. The VK evaluations at `r_A` therefore come from
  the pinned polynomials. The full list, pins included, is absorbed before the phase's 38
  challenges.
- **B2 trust:** `SymbolicSchedule::from_reference` fixes the label, cell bytes/classes,
  block lengths/flags, tail geometry, wire and squeeze rows/counts, RLC cell, and last squeeze cell.
  Replay proves only internal transcript consistency. The production trust boundary is
  `WrapHashKey::from_reference`, which first runs native `jolt_verifier::verify`; trusted inputs are
  the verifier code, preprocessing/profile, public IO, and one natively accepted reference proof.
  Calling `HashTableKey::new` on an arbitrary schedule makes that schedule verifier policy.
- **Fr/terms:** raw field `*` operations remain only in unobserved prover/test oracles
  (`plain`, `Relation::evaluate/final_check`, `WiringStatement::final_check`); the exporter itself
  routes its multiplications through `mul`. `eq_helpers_match_jolt_poly` passes. The term merge still
  satisfies each member separately and their `rho` batch under high-to-low binding: `T = 232`,
  `d = 2`, 313 logical ids over 352 packed columns (20 prover + 2 key groups).

## Checks

- `cargo nextest run -p jolt-wrapper --test hash_table_relation --cargo-quiet`: 11 passed.
- Real fixture: passed; 1,199 wires, `T = 232`, `d = 2`, 4,206 + 705 Fr multiplications; scratch
  position census `{(0, aligned): 400, (8, aligned): 799}`.
- Scratch `canonicality_windows_match_every_representable_alias_class`: passed for both alignments
  and `k=1..=5`; `verifier_cost_includes_statement_derivation`: failed as expected, 9,963 vs 10,666.
- Clippy passed for `--lib --test hash_table_relation --test hash_table_fixture --test
  wrap_real_t1_r`, with and without `prover-fixtures`. The requested superset including
  `perf1_profile` is blocked by five post-`ded155c15` T2 API mismatches (`Wiring`, `Slot::y_sign`,
  and changed function arities), outside this commit's hash-table delta.
