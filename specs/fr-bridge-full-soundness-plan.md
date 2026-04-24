# Task #65 completion — Full FR Operand-Binding Plan

| Field   | Value                                                              |
|---------|--------------------------------------------------------------------|
| Status  | DESIGN. Not yet implemented.                                       |
| Blocker | Architecture — R1CS row alone is insufficient; needs cross-check.  |
| Effort  | ~600-900 LOC across 6 files, 1-2 focused sessions                  |

## What this fixes

Plan P (rows 29/30, already shipped) closes the compensating-tamper attack
against the limb-sum bridge. But the 10-agent cross-verification (2026-04-23)
identified a **second attack class**: operand-forgery via omitted FMov-I2F.

Concrete attack:
1. Honest program: `fmov_i2f(x20 → field_regs[0])` loads A into slot 0, then
   `FMUL frd, frs1=0, frs2=1` → field_regs[frd] = A·B.
2. Malicious prover: omits the FMov-I2F event entirely. FR Twist state at
   slot 0 stays at 0.
3. Prover sets `FieldOpPayload.a = A'` (arbitrary), and x10..x13 to A's
   limbs — satisfies row 29 (`V_LIMB_SUM_A == V_FIELD_OP_A = A'`).
4. FMUL rows (19-26) compute `V_FIELD_OP_RESULT = A' · B`.
5. FieldRegEvent fires at `slot=frd` with `new = A'·B`.
6. **Nothing checks that `V_FIELD_OP_A` equals `field_regs[frs1]` at the
   FMUL cycle** — the FR Twist state at slot 0 is still 0, but the R1CS uses
   A'. FR Twist doesn't complain; no row ties them.

## Root cause

`V_FIELD_OP_A/B/RESULT` are populated from `FieldOpPayload.{a, b, result}`
at `jolt-host/src/r1cs_witness.rs:324-331`. The FieldOpPayload is
**prover-declared** in the event record. No R1CS row or sumcheck binds
these columns to the FR Twist's committed `Val_fr(slot, cycle)` polynomial.

Same gap exists for FMov (task #64 was incomplete in the same way):
`V_FIELD_REG_READ_LIMB` / `V_FIELD_REG_WRITE_LIMB` populated from
`FMovPayload.limb` with no FR-Twist binding.

## Why the simple "add R1CS rows" approach doesn't work

A naive implementation would:
1. Add R1CS witness columns `V_FIELD_REG_READ_VALUE_A/B`,
   `V_FIELD_REG_WRITE_VALUE`.
2. Populate from event stream (same data as committed `FieldRegReadValue`).
3. Add rows 31/32/33: `IsFieldOpAny · (V_FIELD_OP_A − V_FR_READ_A) = 0` etc.

This is **vacuous**: both sides of the equality are populated from the same
prover-supplied event data. Row 31 says "A = A" tautologically. No soundness.

For the row to catch forgery, the right-hand side (`V_FR_READ_A`) must be
cryptographically bound to the FR Twist's committed state — not to the event
record directly. That binding is the real work.

## Required mechanism

Three things must be true at the opening point `stage1_cycle`:

1. `V_FIELD_OP_A(stage1_cycle)` = opening claim from Stage 1 Spartan
   (already exists).

2. `FieldRegReadValueA(stage1_cycle) == Val_fr(frs1(stage1_cycle), stage1_cycle)`
   — proven by an extension to the FR Twist read-check sumcheck. This binds
   `FieldRegReadValueA` to the committed `FieldRegVal` at the slot dictated
   by bytecode's `rs1` field for that cycle.

3. `V_FIELD_OP_A(stage1_cycle) == FieldRegReadValueA(stage1_cycle)` —
   enforced by R1CS row 31 (trivially, since both are scalars opened at
   the same point and the constraint is pointwise on cycles).

The third is trivial once (1) and (2) are in place. The hard work is (2).

## Concrete design

### New committed polynomials (2)

- `FieldRegReadValueA`: T-length, `value(c) = Val_fr(frs1(c), c)` on
  FieldOp cycles, zero elsewhere. Committed via Dory.
- `FieldRegReadValueB`: symmetric for frs2 (gated by IsFieldOpNoInv).

`FieldRegWriteValue` already exists (committed) — captures post-write at
slot `e.slot = frd` on FieldOp cycles. Reusable for result-side.

### New virtual polynomials (2)

- `FieldRegRs1Ra`: K×T one-hot at `(frs1(c), c)` on FieldOp cycles, zero
  elsewhere. Slot address comes from bytecode's `RS1_INDEX` committed field
  (already bound, used by integer Registers Twist for rs1_ra).
- `FieldRegRs2Ra`: symmetric for frs2. Slot from bytecode's `RS2_INDEX`.

These can be **virtual** (not separately committed) if we're willing to
materialize them from the bytecode preprocessed data at proving time and
the FR Twist sumcheck treats them as derived inputs. Precedent: integer
`reg_rs1_ra` in `crates/jolt-witness/src/derived.rs:783-819`.

### Extended FR Twist read-check sumcheck

Current FR Twist (Stage 2 instance [5]) proves:
```
FieldRegReadValue(r_cycle) · Ra(r_slot, r_cycle) =
    Σ_s Val_fr(s, r_cycle) · Ra(s, r_cycle)     (one-slot read per cycle)
```

Extend to γ-batched form:
```
input_claim = FieldRegReadValueA(r_cycle) · IsFieldOpAny(r_cycle)
            + γ_read · FieldRegReadValueB(r_cycle) · IsFieldOpNoInv(r_cycle)
            + γ_write · FieldRegWriteValue(r_cycle)         [existing]

output_check = [Σ_s Val_fr(s, r_cycle) · Rs1Ra(s, r_cycle)] · IsFieldOpAny
             + γ_read · [Σ_s Val_fr(s, r_cycle) · Rs2Ra(s, r_cycle)] · IsFieldOpNoInv
             + γ_write · [existing]
```

Mirrors the integer Registers Twist γ-batching of rs1/rs2/rd reads
(`jolt_core_module_with_fieldreg.rs:3821-3847`).

### Three new R1CS rows

```
Row 31: IsFieldOpAny   · (V_FIELD_OP_A      − V_FR_READ_VALUE_A)  = 0
Row 32: IsFieldOpNoInv · (V_FIELD_OP_B      − V_FR_READ_VALUE_B)  = 0
Row 33: IsFieldOpAny   · (V_FIELD_OP_RESULT − V_FR_WRITE_VALUE)   = 0
```

New R1CS witness columns at slots 49, 50, 51 (pushing V_BRANCH to 52,
V_NEXT_IS_NOOP to 53). NUM_R1CS_INPUTS: 48 → 51. NUM_EQ_CONSTRAINTS:
31 → 34. Pad stays at 64 (already sized for 34+3=37 matrix rows).

### R1CS-column ↔ committed-poly cross-check

R1CS column slots 49/50/51 must equal committed `FieldRegReadValueA/B` and
`FieldRegWriteValue` at `stage1_cycle`. Two implementation options:

**Option X (preferred)**: use the SAME PolynomialId for both roles.
Give `FieldRegReadValueA` an r1cs_variable_index=49, and have the R1CS
source materialize from the same data buffer that feeds the Dory
commitment. No duplication, no cross-check needed — one opening serves both
Stage 1 Spartan and Stage 2 FR Twist.

Blocker: the current `PolynomialDescriptor` match in
`crates/jolt-compiler/src/polynomial_id.rs:355-362` routes committed polys
via `PolySource::Witness` (not `R1cs`). Need a new descriptor variant like
`PolySource::R1csCommitted` — small compiler-side change.

**Option Y (fallback)**: keep R1CS column and committed poly separate, both
populated from the same source. Add a new `VerifierOp::AssertEqualEvals {
a: PolynomialId, b: PolynomialId, at_stage: VerifierStageIndex }` that
asserts `evaluations[a] == evaluations[b]` at the same opening point.
~50 LOC for the VerifierOp + matching prover-side `Op::AssertEqualEvals`.

### Slot-address binding to bytecode

Bytecode already commits `RS1_INDEX`, `RS2_INDEX`, `RD_INDEX` as virtual
polys evaluated at the BytecodeRaF opening point. For FieldOp cycles, these
5-bit fields encode frs1/frs2/frd (the low 4 bits used as slot indices
0..15).

To bind `FieldRegRs1Ra(s, c)` to `RS1_INDEX(c)`:
```
FieldRegRs1Ra(s, c) = 1 iff s == RS1_INDEX(c) AND IsFieldOpAny(c) = 1
```

Materialization: at proving time, for each FieldOp cycle c, read
`RS1_INDEX(c)` from bytecode-preprocessed data, set `Rs1Ra[RS1_INDEX(c), c] = 1`.

Verification: no new sumcheck needed — the FR Twist read-check sumcheck's
output claim includes a Lagrange evaluation against the bytecode address,
analogous to integer Registers Twist.

## Implementation checklist

Ordered so each step compiles and tests can run:

1. [ ] Add `PolynomialId::FieldRegReadValueA`, `FieldRegReadValueB`
       variants in `crates/jolt-compiler/src/polynomial_id.rs`.
2. [ ] Add R1CS slot constants V_FIELD_REG_READ_VALUE_A=49,
       V_FIELD_REG_READ_VALUE_B=50, V_FIELD_REG_WRITE_VALUE=51 in
       `rv64.rs`. Bump V_BRANCH, V_NEXT_IS_NOOP, NUM_R1CS_INPUTS.
3. [ ] Decide Option X vs Y for cross-check. If X: add
       `PolySource::R1csCommitted` variant in polynomial_id.rs and handle in
       `R1csSource::compute` + `DerivedSource`. If Y: add
       `Op::AssertEqualEvals` and matching `VerifierOp`.
4. [ ] Extend `FieldRegConfig::compute_read_value_a/b` in
       `crates/jolt-witness/src/derived.rs` — compute the per-cycle values.
5. [ ] Populate R1CS witness slots 49/50/51 in
       `crates/jolt-host/src/r1cs_witness.rs`.
6. [ ] Add `FieldRegRs1Ra`, `FieldRegRs2Ra` virtual polys mirroring
       `reg_rs1_ra`/`reg_rs2_ra` pattern in derived.rs.
7. [ ] Extend FR Twist Stage 2 instance [5] input_claim and output_check
       with γ-batched read terms in
       `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs`.
8. [ ] Add 3 R1CS rows (31/32/33) in rv64.rs. Bump NUM_EQ_CONSTRAINTS 31 → 34.
9. [ ] Update group0_indices / group1_indices in the fieldreg module to
       include new rows. 34 eq split = 17/17. Uniskip domain = 18 (next_pow2
       = 32, 2× outer uniskip cost — acceptable or further consolidate).
10. [ ] Update r1cs_input_polys list (46 → 49 entries including LimbSumA/B
        from Plan P plus the 3 new columns).
11. [ ] Verifier mirrors: extend output_check ClaimFormula for Stage 1
        Spartan to include the new rows' contributions.
12. [ ] Write PoC attack test: honest FMov-I2F + FMUL vs. adversarial
        (omitted FMov-I2F + forged operand). Adversarial should reject at
        Stage 1 or Stage 2 FR Twist.
13. [ ] Extend same pattern to FMov rows 27/28 for task #64 completion —
        `V_FIELD_REG_READ_LIMB` / `V_FIELD_REG_WRITE_LIMB` bound to FR Twist
        limb values (less urgent, separate task).

## Perf impact estimate

- 3 new R1CS eq rows → NUM_EQ=34. Uniskip domain 16 → 18, `next_power_of_two
  = 32` — outer Spartan Lagrange basis doubles. Est. +3-5% prove_ms.
- 2 new Dory commitments (FieldRegReadValueA/B). Commit phase adds ~2
  tier-1 batches. Est. +1-2% prove_ms.
- FR Twist sumcheck extended with 2 γ-batched terms. Per-round cost ~3×
  current. Stage 2 is small, net impact sub-1%.

Total estimate: 4-8% prove_ms over current Plan-P state. Proof size:
+2 Dory openings ≈ +200 bytes.

## Cross-verify

This plan WILL break cross-verify with jolt-core on the refactor-crates'
baseline module (the rv64.rs matrix goes to 37 rows, jolt-core stays at
matrix with its own layout). Acceptable per user guidance 2026-04-23.

## Acceptance criteria

1. All existing honest tests pass (muldiv, fieldreg self-verify, FMUL/FADD
   multi-limb).
2. `audit_poc_compensating_tamper_solved` still rejects (Plan P regression).
3. New test `fr_omit_fmov_i2f_forged_operand_rejects` rejects.
4. Extended test with tampered `FieldRegEvent.new` rejects (result-side).
5. Clippy clean in both `host` and `host,zk` modes.
6. `jolt-bench muldiv modular` prove_ms within +10% of current Plan-P state.
