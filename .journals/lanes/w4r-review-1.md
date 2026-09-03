# W4-R review #1

Scope: committed tree `587c53525` (`261bff0ef` + `587c53525`), reviewed read-only in a
detached worktree against the native stages 1–8, `jolt-sumcheck`, the Dory verifier, and the W4-R
design notes.

## Findings

1. **BLOCKER (score 100) — `crates/jolt-wrapper/src/relation/dory.rs:178`: both deferred
   Delta scalars are named with the wrong setup index.** Native round `j` reads
   `setup.delta_{1r,2r}[sigma - j]` (`DoryVerifierState::process_round`, while `num_rounds`
   decreases from `sigma` to `1`), and the closed form in `dory-offload-study.md` uses the same
   `k = sigma - j`. The relation instead emits `Delta1R(sigma - 1 - j)` and
   `Delta2R(sigma - 1 - j)`. A T2 consumer following `DoryScalar` therefore applies every value to
   the preceding setup base: `Delta[*][sigma]` is omitted and `Delta[*][0]` gets the last-round
   value. This changes the deferred Dory equation; valid proofs fail and the checked equation is
   not the native verifier's equation. **Fix:** emit both Delta ids at `sigma - j`; keep `Chi` at
   `sigma - 1 - j`; correct the enum documentation and add a scalar-to-base parity test against
   the deferred checker for every round.

2. **MAJOR (score 75) — `crates/jolt-wrapper/src/relation/mod.rs:168`: assign mode does not perform
   the planned native-derived differential checks.** `replay()` records only transcript appends
   and squeezes; `walk()` then computes every derived value through the circuit gadgets. No
   native `derive_input_term` / `derive_output_term` value is recorded or compared, and the only
   production `debug_assert` checks the stage-2 address identity. The tests compare generic gadget
   families and table MLEs, but do not cover each derived id, Dory scalar name, or sha2-chain as
   required by `plan-relation.md` section 4. A transcription can therefore survive until a sampled
   proof happens to expose it; finding 1 is not detected by the present fixture. **Fix:** in assign
   mode evaluate each derived id through its native `ConcreteSumcheck` owner and
   `debug_assert_eq!` it with the gadget output; add per-id randomized parity and Dory
   scalar-to-base tests.

3. **MINOR (score 50) — `.journals/lanes/w4r-relation.md:83`: the opaque Dory element inventory is
   wrong.** For `sigma = 11`, the schedule contains 68 GT, 35 G1, and 34 G2 encodings: 137 opaque
   elements totaling the stated 29,408 bytes. The journal says 89 elements and 24 of each curve
   group. **Fix:** replace the element count and type breakdown; keep 29,408 bytes.

4. **MINOR (score 50) — `crates/jolt-wrapper/src/relation/mod.rs:224`: a public diagnostic API and
   integration test duplicate the internal table parity test.** `table_gadget_values` has no
   production caller, while `tests/relation_tables.rs` repeats the same 200 points, seed, native
   oracle, and row check already present in `relation/tables.rs`. **Fix:** retain the internal test;
   remove the public helper and duplicate integration test.

## Native checks confirmed

1. **Sumcheck rounds:** both uni-skip rounds constrain the centered-domain power sum to the input
   claim, Horner-evaluate at the raw 125-bit challenge, and bind the computed output claim. Every
   Boolean-hypercube round has a fixed compressed degree/count; reconstructing `c1 = claim - 2c0 -
   sum(c2..cd)` enforces `s(0) + s(1) = claim`, and Horner carries the next claim.
2. **Batch heads and tails:** member input claims use the symbolic `JoltExpr`; batching coefficients
   are 128-bit scalar challenges; the head applies `2^(max_rounds - member_rounds)`; instance
   offsets match the native head/suffix placement; each final claim equals the coefficient fold of
   every member's expected output.
3. **Stages 1–7:** all base clear-mode members are present in native order. Opening claims use their
   canonical field order. Stage-2 and stage-3 aliases share the source variable; the stage-6b
   bytecode/booleanity runtime alias shares the source variable. Opening points use the native
   reversals, phase slices, address chunks, and cycle suffixes. Checked derived families include
   Eq, LT, EqPlusOne, centered Lagrange/kernel, Spartan row weights, UnmapAddress, IoMask, entry,
   bytecode stage folds, hamming weights, and all 54 table MLEs including shift/rotate/PEXT.
4. **Public IO:** `val_io`, `init_eval`, and five `stage_values` are verifier-recomputed from public
   data at public challenge outputs. RAM read-write and output-check addresses are identical linear
   combinations of the same stage-2 point; RAM value-check reuses that address by construction.
   Register value-evaluation prepends the stage-4 read-write address, so the single seven-coordinate
   public address is justified.
5. **Stage 8 and Dory Fr:** final-opening order, increment zero-embedding, 41-term RLC, joint claim,
   24 inverse equations, `u_j`/`v_j`, alpha products, chi coefficients, `s1`/`s2`, HT,
   `beta_0 + d^2`, and pairing scalars match the native algebra except finding 1. Each inverse hint
   is pinned by `x * x_inv = 1`, matching native rejection when a challenge is zero.

## External obligations

- The relation leaves all 376 challenge wires free. `ScheduleEntry::Squeeze` labels 310
  `challenge()` wires for the 125-bit shifted decoder and 66 `challenge_scalar()` wires for the
  128-bit big-endian decoder; T1 must bind these wires to the Blake3 chain beginning at `state_in`.
- Opaque Dory encodings, subgroup checks, commitment combination, GT equation, and four-pair
  equation are outside this R1CS; T1/T2 must use the same proof elements. Public MLE evaluation is
  likewise outside the R1CS, with the points and returned values bound through the 45 public
  columns.

## Checks

- `cargo clippy -p jolt-wrapper --all-targets -q --message-format=short -- -D warnings`: passed.
- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 17 passed, 1 skipped.
- Cached fibonacci 2^18 fixture with `prover-fixtures`: passed; 5,253 rows, 6,760 variables, 45
  public columns; all three tampers failed in their pinned sections.
- Cached fibonacci 2^20 ignored fixture: passed; 5,454 rows, 7,031 variables, 45 public columns;
  per-stage snapshot matched.
- Touched source/test files are all below 1,000 lines; no added `#[allow]` or nominal-import defect
  found.

VERDICT: 1 blocker, 1 major, 2 minors
