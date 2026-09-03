# W4-T2 limb-table review #1

Target: `ddd1fa814` (`crates/jolt-wrapper/src/limb_table/`)

## Verdict

**5 blockers / 1 major / 3 minors**

### Blockers

1. **`relation.rs:323` — the range-table tail is prover-owned.** `small` gates the inverse equation, but neither
   `MULT` nor `INV` is forced to zero on rows `2^16..2^18`. A prover can place an out-of-range chunk `w` there,
   commit `MULT[w]`, and choose `INV[w] = 1/(alpha-w)`. The grouped helper and both LogUp sums then pass. The
   scratch test `out_of_range_chunk_is_accepted_with_multiplicity_above_the_range_table` changes a free-row chunk
   to `2^16`; the full row sumcheck and exported final relation accept. **Fix:** add
   `(1-small) * MULT = 0` to the row relation and term export; optionally pin `INV=0` outside the table.

2. **`export.rs:19,49-57` — phase 2 sees the tuple-binding challenges before committing the tuples.** `X/Y`,
   fingerprints, and `H/G` are all committed after `fp_root`, `fp_combine`, `beta`, `copy_root`, and `gamma` under
   the documented contract. A prover can choose a nonzero change in the joint kernel of the known fingerprint
   vector and the row's `X` vector, changing three selected `Y` slots while preserving both the lookup fingerprint
   and limb product. `selected_operand_collision_is_accepted_when_fingerprint_root_is_known` constructs that
   collision; the full member accepts. Known `copy_root` and `gamma` permit analogous adaptive cancellations.
   **Fix:** with the current columns, use four prover phases: (1) chunks/digits/multiplicities; derive `xi,alpha`;
   (2) `X/Y` and range helpers; derive `fp_root`; (3) fingerprints; derive `beta,fp_combine,copy_root`; (4) `H/G`;
   only then derive `tau,gamma,lambda,lambda_lookup`. A different tuple-binding design must preserve the same
   commit-before-challenge order.

3. **`columns.rs:22`, `export.rs:163-170` — byte-linked Fq inputs are not canonical.** Sixteen chunks admit every
   integer below `2^256`; input rows are marked free and have no `< q` check. Replacing a serialized coordinate
   `x` by `x + kq < 2^256` changes the Blake transcript while representing the same Fq value in every T2 equation.
   This gives several Fiat-Shamir aliases per byte-linked coordinate and accepts encodings rejected by arkworks.
   Internal arithmetic rows may use any representative, but transcript-linked input rows may not. **Fix:** enforce
   `< q` on each byte-linked Fq coordinate, preferably sharing the borrow/range rows with the pending sign check.

4. **`tower.rs:126-129`, `schedule.rs:492-502` — untrusted GT inputs lack target-subgroup membership.** Negative
   centered digits use Fq12 conjugation as inversion, which holds for cyclotomic/target-group elements, not an
   arbitrary Fq12 input. Dory deserialization checks `x^r = 1`; this table only copies twelve coordinates. The
   scratch test `gt_conjugation_is_not_inverse_without_a_target_subgroup_check` exhibits the mismatch. Scalar
   algebra over Fr also needs r-torsion. **Fix:** prove `x^r = 1` for every proof-owned `GtBase::Input` (110 at the
   fibonacci profile), or bind the table to a separately verified target-group decoder.

5. **`ops.rs:364-479,515-569` — affine G1/G2 add and double gadgets do not constrain denominators nonzero.** They
   enforce only `lambda * den = num`. For `P = Q`, both are zero in the addition gadget, so any `lambda` passes and
   produces a point other than `2P`. The fixed offsets avoid this for honest random inputs, but proof points are
   adversarial; a prover can choose a base colliding with an accumulator/table point. The scratch test
   `affine_addition_accepts_an_arbitrary_slope_when_points_are_equal` uses the real `g1_add` template and a valid
   subgroup point. **Fix:** add denominator-inverse constraints for every affine add/double, rejecting exceptional
   inputs, or use complete projective formulas.

### Major

1. **`tests/limb_table_program.rs:432-446` — the reported 9,511 Fr count is not end-to-end observed.** The observer
   covers `public_evals` and `omega_eval`; `RowRelation::new`, `RowRelation::terms`, affine-form scaling, and final
   term evaluation perform direct multiplications. The test calls `terms()` after reading the counter and still
   asserts the 10,000 budget. **Fix:** thread `VerifierObserver` through challenge-power construction and term
   export/evaluation, then gate the returned verifier-path total.

### Minors

1. **`schedule.rs:1` — 2,682 lines, 2.7x the repository's 1,000-line soft limit.** It owns GT, G1, G2, Miller,
   final-exponentiation, and placement policy in one file. Split by operation family while keeping shared geometry
   in one owner.

2. **`dory.rs:645-660`, `tests/limb_table_program.rs:77-106` — `NativeCheck` is not an independent oracle.** It
   consumes the same `FlattenedCheck` base/wire list as the table, so a shared omission survives the component
   comparisons. The production Dory accept check triangulates the happy path but no verifier-path mutation test
   covers each input family. Compare against state extracted from the production verifier or add one mutation per
   message/setup family.

3. **`relation.rs:798-800`, `digit_link.rs:17-35` — production prover types expose test-only `cheat` switches.**
   They are temporary adversarial-test machinery in the shipped module. Move the forced-round-polynomial helpers
   under `cfg(test)` or into the scratch test harness before handoff.

## Checked and accepted

- The CRT core has four carry coefficients, covers all five 96-bit product positions, and uses both the random
  limb point and native Fr identity implied at `B=2^96`; the fixed program's measured `sum |kappa| = 102` fits the
  stated integer bound. All 61 chunk columns and five digit bits enter the grouped range products. The carry
  algebra passed; blocker 1 is the missing table-domain restriction.
- Operand key namespaces are disjoint (`row < 2^18`, negative GT offset `2^18`), and `TableRead.conjugated` has one
  sign owner. Zero and negative centered digits map to the intended identity/inverse rows.
- Kernel edge tests cover every program slot and fingerprint read at the tested profile; closed forms equal both
  the kernels and enumerated edge MLEs. The `place()` mask now restricts every empty-domain family to placed cells.
  No remaining a6e09751d padding/free-cell leak found.
- Digit-link indexing follows `wire_order`, divides repeated bases by their multiplicity, and matches R's
  `DoryScalarTermExporter` at the shared point. The local exports are 70 phase-1, 72 phase-2, five VK columns;
  `terms()` returns `T=131`, `d=4`, and matches the native row summand.
- The flattened transparent check matches dory-pcs 0.4.2's state updates and four-pair final equation. Fresh proof
  G1/G2 inputs get on-curve rows. G1 is cofactor-one.

## Known open items, excluded from the verdict

- No G2 subgroup check exists at `ddd1fa814`. Checking only the two proof-owned G2 values fed to Miller is sound
  if all preceding full-curve additions/scalar multiplications implement the group law; blocker 5 currently breaks
  that premise. Once blocker 5 is fixed, cofactor components may cancel before the pairing inputs, which supports
  the orchestrator's reduced scope.
- Compressed-point sign flags are absent and are already assigned to the lane.

## Scratch tests and verification

Patch: `.journals/lanes/w4t2-review-1-tests.patch`

- `cargo clippy -p jolt-wrapper --lib ... -D warnings` — pass.
- Clippy on `limb_table_e2e`, `limb_table_miller`, `limb_table_program` — pass.
- The three limb-table test binaries with the patch — 13/13 pass, including both accepted full-table forgeries and
  the two focused gadget counterexamples.
- `tests/perf1_profile.rs` intentionally excluded per the stated unrelated compile break.
