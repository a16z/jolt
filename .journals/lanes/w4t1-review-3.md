# W4-T1 review #3

Scope: `eb318731d` on `wrap/spartan-hyperkzg`, limited to `hash_table/`, its
`wrap.rs` and test callers, review #2, and the Fix #2 journal entry. Review work ran in detached
`w4t1-review3`; the adversarial additions are saved in
`.journals/lanes/w4t1-review-3-tests.patch`.

**VERDICT: 4 blockers / 1 major / 2 minors.**

## Findings

1. **BLOCKER — `crates/jolt-wrapper/src/hash_table/wiring.rs:367-375,764-795`:
   shifted canonicality reads bytes 8–9 in place of bytes 6–7.** A shifted wire starts at byte 2
   of row `p`; `FrNext(1)` reads bytes 2–5 and the low half of row `p + 2` is bytes 6–7.
   `FrHi2` instead uses `BswapHi16`, which reads bytes 8–9. The constrained value is therefore
   `bytes[0..6] || bytes[8..10]`, while `fr_word_shifted()` uses all bytes in order. Repro: choose
   canonical `x` with `x[8..10] = 0x47b0`; `x + r` has prefix
   `30644e72e131a02a0000`. The wrong `w_hi = 0x30644e72e1310000` is below `r_hi`, the value form
   still aliases `x`, and the wrapper verifier accepts the proof. The existing `1 + r` case
   rejects accidentally because its bytes 8–9 are `b850 > a029`. **Fix:** source `FrHi2` with
   `BswapLo16`, rename it to match the low-half meaning, and keep the carry-wrap repro.

2. **BLOCKER — `crates/jolt-wrapper/src/hash_table/adapter.rs:43-68`:
   the six alleged VK columns are prover commitments.** `StreamColumns::new` puts
   `lo_is_const`, `lo_const`, `hi_is_const`, `hi_const`, `wire_aligned`, and `wire_shifted` in the
   proof's `Column` list. `AssemblyStatement` carries no expected commitments for them; the
   verifier exporter receives only their physical ids. A malicious prover can set both wire
   selectors to zero and disable canonicality. Scratch proofs with `1 + r` then verify for both
   alignments. The same defect lets the prover disable label/block pins; a table with one extra
   commitment also verifies under the original schedule key. `Column::Bits` checks honest prover
   input only; no verifier relation makes these selectors boolean or equal to the key's rows.
   **Fix:** store these column commitments and the id map in the wrapper key, make the verifier use
   those exact commitments in the opening/transcript, and omit their 64 wire bytes from each
   proof. Commitment equality is the full-column pin.

3. **BLOCKER — `crates/jolt-wrapper/src/wrap.rs:134-157`:
   production preparation still derives `hash_key` from the proof being wrapped.**
   `SymbolicSchedule::from_reference(&records, ...)` is called after recording that proof, then
   `JoltSchedule::witness` compares the proof to the key just made from it. `verify_wrapped` has no
   wrapper-key argument and no code that constructs the T1 statement/exporter from a stored
   schedule. `from_reference` trusts the reference's labels, append lengths/classes, constants,
   squeeze positions/count, and Dory segment shape; replay proves only that the supplied record is
   internally consistent. This is sound only as trusted key generation after the native verifier
   accepts the reference. **Fix:** add a `WrapKey` generated once from a trusted reference/profile;
   pass it into proof preparation; derive `JoltSchedule::witness`, VK columns, member layout,
   phase layout, ids, and the exporter only from that key. Bind its schedule/commitments in
   `key_digest`.

4. **BLOCKER — `crates/jolt-wrapper/src/wrap.rs:71-81` and
   `crates/jolt-wrapper/src/hash_table/adapter.rs:80-105`:
   the production wrapper cannot construct T1 members after commitment.** `wrap` receives
   prebuilt `StageMember`s, then calls `commit_packed`; `Members::new` already requires the T1
   challenges. Only the test harness manually commits, derives 38 challenges, constructs members,
   and calls `prove_assembly`. There is no production caller of `Members` or
   `StreamTermExporter`. **Fix:** split commit/prove in the public wrapper or add a per-phase
   post-commit constructor. The key must own `CommitmentPhase { group_count: 22,
   challenge_count: 38 }`, challenge offsets, member slots, and column ids. The existing own-τ
   test is only a prover-side `Err(StageLink)`; the scratch patch also builds a complete proof with
   prover-chosen randomizers and confirms that the honest low-level verifier rejects it. Thus the
   low-level transcript rule is sound; the production integration is missing.

5. **MAJOR — `crates/jolt-wrapper/src/hash_table/adapter.rs:170-175` and
   `crates/jolt-wrapper/src/hash_table/terms.rs:59-60,181-223,248,253-276,326-332`:
   the reported 3,278 verifier Fr multiplications omit many operations.** `terms_observed` passes
   an observer closure into selected products, but `powers`, `eq_rounds`, both seven-variable eq
   tables, both eleven-variable `EqPlusOne::evaluate` calls, `eq_points`, `r_first_cell`, one tail
   product, and every `mul_pow_2` remain direct field operations. The non-`mul_pow_2` omissions
   alone are about 1.3k, so the real count is already about 4.6k before hundreds of observed-aware
   power-of-two multiplies and before `WiringStatement::input_claim`. The `<= 5_000` test checks an
   incomplete counter. **Fix:** give the canonical eq/shift/power helpers observer-aware forms,
   call `mul_pow_2_observed`, and count statement construction. Pin the exact count, not only an
   upper bound.

6. **MINOR — `crates/jolt-wrapper/src/hash_table/terms.rs:185-197`:
   64 quadratic terms can be removed.** Each XOR operand column currently emits
   `e·γ_sq·v_j²` and `e·γ_cross·v_j·w_j` separately. Emit
   `e·v_j·(γ_sq·v_j + γ_cross·w_j)` instead. This changes `T = 296` to `T = 232`, keeps `d = 2`,
   drops the term stage from nine rounds to eight, and saves 64 proof bytes. No other exact merge
   is apparent without raising degree or adding cross terms.

7. **MINOR — `crates/jolt-wrapper/src/hash_table/table.rs:31`,
   `crates/jolt-wrapper/src/hash_table/layout.rs:192,267-281`, and
   `crates/jolt-wrapper/src/hash_table/schedule.rs:4`:
   protocol documentation disagrees with the code.** The table has 227 committed bit columns, not
   163; the row relation has 293 batching coefficients, not 229; high-to-low round `i` binds
   `tau[i]`, not `tau[n - 1 - i]`; and `JoltSchedule::new` no longer exists. **Fix:** update these
   statements with the same commit as the protocol changes.

## Adversarial answers

- **B1:** `MODULUS_HI = 0x30644e72e131a029` is the modulus's top 64-bit big-endian word. All 64
  `CANON` columns fall inside `COMMITTED = 227`, so `Relation::new` gives them booleanity terms.
  The aligned `w_hi` is exact. The shifted `w_hi` is not; finding 1 gives an accepted
  non-canonical word. `fr_word_shifted` does not use `FrHi2`, so the two forms read different
  source bits. The real fixture has 1,199 aligned and zero shifted wires. Every current raw
  32-byte pre-Dory field append becomes a `Wire`; current Dory raw 32-byte values are 35 G1
  encodings, not fields. The selector and all other VK rows are prover-controlled per finding 2.
- **B2:** `JoltSchedule::witness(log, key)` rejects structural differences when the honest caller
  invokes it, and `PublicInputs::from_preamble` correctly derives/pins the 22-byte tail plus first
  state/length/flags. The wrapper verifier does not consume `SymbolicSchedule`; the tests pass only
  `log_rows` and proof-owned VK column ids into the stream verifier. A foreign schedule proof is
  accepted. Reference trust is described in finding 3.
- **B3:** `assembly_transcript` absorbs each phase's commitments before drawing its challenges;
  `T1Challenges` maps 18 row τ values, 18 wiring τ values, and two coefficient bases in fixed
  order. Stream member batching coefficients are drawn later by stage A. A complete own-randomizer
  proof is rejected by the low-level verifier. The missing post-commit production constructor is
  finding 4.
- **Exporter:** both T1 members bind high-to-low; `eq_rounds` and the stream row point use the same
  order. Local ids map correctly to 352 physical columns across 22 groups. The real fixture and
  `StageLink` establish `sum(coeff * product(L(v))) == batched final claim` for `T = 296`, `d = 2`
  under stream rho coefficients and the degree-5 stage envelope. The 3,278 operation count is not
  complete.

## Byte levers

- Current groups: 19 bit groups (`227 + 64`, padded by 13 zeros), one 16-word u32 group, one
  four-selector VK-bit group padded by 12 zeros, one two-value VK-u16 group padded by 14 zeros:
  **22 groups**.
- Vector-affine witness columns: `Din[0..32]`, `Bin[0..32]`; and all 16 words `AIn`, `CIn`,
  `RotD`, `MIn`, `XIn`, `YIn`, `ZIn`, `FrNext(1..=7)`, `FrTail`, `FrHi2`. They are fixed linear
  copies/permutations of committed source bits plus public constants, but not same-row affine forms
  of the existing column evaluations. Removing only the words needs a shift-aware final-check
  rewrite and saves one group; removing all 80 saves five groups. They cannot simply be deleted
  under the current `AffineForm` contract.
- The six VK columns are fixed rather than witness-affine. Moving them into the key removes two
  prover-sent group commitments and fixes finding 2. Combined with virtualizing all 80 wired
  columns: 15 prover-sent groups, down from 22.
- The 39 zero padding columns are affine constants. Packing the 313 logical columns contiguously
  gives 20 groups, saving two commitments, but creates two mixed full-width Fr MSMs in
  `commit_packed`; poor prover-time trade at the current 176-byte margin. The term merge in finding
  6 is the cheaper 64-byte gain.

## Checks

- `cargo clippy -p jolt-wrapper -q --message-format=short --lib --test hash_table_relation --test
  hash_table_fixture -- -D warnings -A dead-code`: passed, with and without
  `--features prover-fixtures`. The allowance is only for the named pre-existing
  `relation_table/mod.rs` dead functions.
- `cargo nextest run -p jolt-wrapper --test hash_table_relation --cargo-quiet`: 12 passed with the
  scratch patch (one nextest leaky-process mark); all four added adversarial checks behaved as
  described.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --test hash_table_fixture
  --cargo-quiet --no-capture`: passed; stage A 0.884 s, build 0.312 s, 1,819 cells, 1,199 wires,
  `T = 296`, `d = 2`, 352 columns / 22 groups.
- `cargo fmt -q --message-format=short -- --check`: passed in the scratch tree.

