# FR Twist Mirror Plan (v2 FR Coprocessor)

| Field | Value |
|---|---|
| Status | Implementation-ready. Hand to compiler without further iteration. |
| Supersedes | v1 FR Twist in `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs` |
| Reference | Integer Registers Twist in `crates/jolt-compiler/examples/jolt_core_module.rs` |
| Base spec | `specs/bn254-fr-coprocessor.md` |

**Principle (mzhu):** FR Twist commits ONLY `FieldRegInc` + `FieldRegRa` one-hot. `FieldRegReadValue` / `FieldRegWriteValue` are DELETED as polynomial IDs. `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue` are virtual R1CS input columns; their Stage-1 openings at `r_cycle` are proven by the FR Twist γ-batched sumcheck. This is a literal structural mirror of the integer Registers Twist (`Rs1Value`, `Rs2Value`, `RdWriteValue`).

---

## Registers Twist skeleton (what to mirror)

The integer Registers Twist is the chain of THREE sumchecks, not one:

1. **Stage 3 `RegistersClaimReduction`** — reduces Stage-1 openings of `Rs1Value`, `Rs2Value`, `RdWriteValue` (each committed-looking but actually virtual at `r_cycle`) into a single γ-batched input_claim that is then proven equal to a Twist-bound sumcheck.
   - `jolt_core_module.rs:2746-2776` — input_claim = `rd_write_value + γ·rs1_val + γ²·rs2_val`.
   - Kernel (`jolt_core_module.rs:2981-3061`): degree-2 dense, reduces the three virtual openings to the Stage-4 RegistersRW entry claim.
2. **Stage 4 `RegistersReadWriteChecking`** — the actual Twist. 2-phase: log_T cycle rounds (segmented, degree 3) + log_K_REG address rounds (dense, degree 2).
   - `jolt_core_module.rs:3137-3604`.
   - Input claim (`:3230-3253`): `rd_wv + γ·rs1_rv + γ²·rs2_rv` (same γ-batch, rebinding).
   - Phase 1 kernel (`:3336-3390`): `eq·rd_wa·val + eq·rd_wa·inc + γ·eq·rs1_ra·val + γ²·eq·rs2_ra·val`. Inputs: `[BatchEq(9), reg_ra_rs1, reg_ra_rs2, reg_wa, reg_val, rd_inc]`. Segmented `inner_only = [true, false, false, false, false, true]`.
   - Phase 2 kernel (`:3398-3455`): `ch_eq·rd_wa·val + ch_eq·ch_inc·rd_wa + γ·ch_eq·rs1_ra·val + γ²·ch_eq·rs2_ra·val`. ScalarCaptures: `BatchEq(9) → ch_reg_eq_bound`, `rd_inc → ch_reg_inc_bound`. Inputs: `[reg_wa, reg_val, reg_ra_rs1, reg_ra_rs2]`.
   - Stage-4 eval flush (`:3581-3599`): `[reg_val, reg_ra_rs1, reg_ra_rs2, reg_wa, rd_inc]` evaluated+absorbed at `FinalBind`.
3. **Stage 5 `RegistersValEvaluation`** — single-term degree-3 sumcheck over log_T rounds: reduces `RegistersVal(r_address, r_cycle)` (the virtual val evaluation emitted by Stage 4) through `RdInc · wa · lt`.
   - Kernel (`jolt_core_module.rs:3693-3725`): single product `Input(0)·Input(1)·Input(2)` = `rd_inc · eq_gather(r_address, rd[j]) · LT(r_cycle, j)`.
   - Input claim (`:4034-4040`): single term `Eval(reg_val)`.
   - Eval flush (`:4158-4160`): `[rd_inc, rd_wa_gather]` (where `rd_wa_gather = BatchEq(12)`).

Poly registration (`jolt_core_module.rs:327-329, 372-375`):
- `rs1_val = pt.add(Rs1Value, …, Virtual, log_t)` — R1CS variable index 9 (`polynomial_id.rs:493`).
- `rs2_val = pt.add(Rs2Value, …, Virtual, log_t)` — R1CS variable index 10.
- `rd_write_value = pt.add(RdWriteValue, …, Virtual, log_t)` — R1CS variable index 11.
- `reg_ra_rs1`, `reg_ra_rs2`, `reg_wa` — Virtual at `LOG_K_REG + log_t`. Their coefficients are derived from bytecode's `rs1_index`, `rs2_index`, `rd_index` fields (one-hots at those 7-bit slots).
- `reg_val` — Virtual at `LOG_K_REG + log_t` (running register-file state).

Ra generation: `reg_ra_rs1[k,t]`, `reg_ra_rs2[k,t]`, `reg_wa[k,t]` are derived polynomials — one-hot at `bytecode(PC(t)).rs1_index`, `rs2_index`, `rd_index` respectively. The bytecode commitment covers these 5-bit fields for free (see `PolynomialId::BytecodeField` and the bytecode-field index constants in `polynomial_id.rs:310-332`).

---

## Mirror mapping table

| Integer Registers Twist element | FR Twist counterpart | File:line (integer) | Notes |
|---|---|---|---|
| `rs1_val` (R1CS #9, Virtual log_t) | `field_rs1_value` — new `PolynomialId::FieldRs1Value`, R1CS virtual | `jolt_core_module.rs:327` | Add R1CS variable index (next free slot after LimbSum removal; 42-44 freed in Phase 1). |
| `rs2_val` (R1CS #10) | `field_rs2_value` — new `PolynomialId::FieldRs2Value` | `:328` | |
| `rd_write_value` (R1CS #11) | `field_rd_value` — new `PolynomialId::FieldRdValue` | `:329` | |
| `reg_ra_rs1` (LOG_K_REG + log_t) | `field_reg_ra_rs1` — Virtual, `log_k_fr + log_t` | `:373` | Derived from bytecode `frs1_index = rs1_index & 0xF` (low 4 bits). Same bytecode field — free. |
| `reg_ra_rs2` | `field_reg_ra_rs2` | `:374` | Same, from `frs2_index`. |
| `reg_wa` (write-address) | `field_reg_wa` | `:372` | Same, from `frd_index`. |
| `reg_val` (running state) | `field_reg_val` — rename from v1 `FieldRegVal`, unchanged | `:375` | Already exists in `polynomial_id.rs:39`. |
| `rd_inc` (Committed, log_t → padded log_k_reg+log_t) | `field_reg_inc` — **Committed**, log_t → padded `log_k_fr + log_t` | `:246` | Promote v1 `FieldRegInc` from Derived to Committed. |
| `Rs1Ra`/`Rs2Ra`/`RdWa` one-hot chunks (from bytecode, NOT directly committed — they're virtual; underlying `BytecodeRa(d)` IS committed) | `field_reg_ra` chunks — Committed one-hot, same structure as `RamRa(d)` | `:264-273` | Promote v1 `FieldRegRa` from Derived to Committed one-hot. Chunks identical to integer pattern. |
| `ch_registers_gamma` (Stage 3 mixing) | `ch_gamma_fr_cr` | `:2648` | Fiat-Shamir after Stage 2. |
| `ch_gamma_reg_rw` (Stage 4 Twist γ) | `ch_gamma_fr_rw` | `:3161` | Fiat-Shamir after Stage 3. |
| Stage-3 RegistersCR input_claim: `rd_wv + γ·rs1_val + γ²·rs2_val` | FR Stage-3 CR input_claim: `FieldRdValue + γ·FieldRs1Value + γ²·FieldRs2Value` | `:2746-2768` | Exact same shape. |
| Stage-3 RegistersCR kernel (degree 2) | FR Stage-3 CR kernel (degree 2) | `:2981-3061` | Structural copy: replace every `reg_*` / `rs*_val` / `rd_write_value` with `field_*` counterparts. |
| Stage-4 RegistersRW input_claim: `rd_wv + γ·rs1_rv + γ²·rs2_rv` (rebind) | FR Stage-4 RW input_claim: `FieldRdValue + γ·FieldRs1Value + γ²·FieldRs2Value` | `:3230-3253` | Same. |
| Stage-4 Phase-1 kernel: `eq·rd_wa·val + eq·rd_wa·inc + γ·eq·rs1_ra·val + γ²·eq·rs2_ra·val` | FR Stage-4 Phase-1 kernel: identical formula with `field_reg_*` polys | `:3336-3390` | Identical 4-term formula. |
| Stage-4 Phase-2 kernel (address, log_k_reg rounds) | FR Stage-4 Phase-2 kernel (address, **log_k_fr = 4** rounds) | `:3398-3455` | **Rounds change**: `log_k_reg = 7` → `log_k_fr = 4`. Formula structurally identical. |
| Stage-4 ScalarCaptures: `BatchEq(9) → ch_reg_eq_bound`, `rd_inc → ch_reg_inc_bound` | FR Stage-4 ScalarCaptures: `BatchEq(fr_eq_slot) → ch_fr_eq_bound`, `field_reg_inc → ch_fr_inc_bound` | `:3528-3537` | Pick fresh `BatchEq(…)` slot. |
| Stage-4 eval flush: `[reg_val, reg_ra_rs1, reg_ra_rs2, reg_wa, rd_inc]` | `[field_reg_val, field_reg_ra_rs1, field_reg_ra_rs2, field_reg_wa, field_reg_inc]` | `:3581-3589` | |
| Stage-5 RegValEval kernel: `rd_inc · eq_gather(r_addr, rd[j]) · LT(r_cycle, j)` | FR Stage-5 FieldRegValEval kernel: `field_reg_inc · eq_gather(r_addr_fr, frd[j]) · LT(r_cycle_fr, j)` | `:3693-3725` | Needs a new gather-index poly `FrdGatherIndex` analogous to `RdGatherIndex`. |
| Stage-5 input_claim: `Eval(reg_val)` | `Eval(field_reg_val)` | `:4034-4040` | Single term. |
| Stage-5 eval flush: `[rd_inc, rd_wa_gather]` | `[field_reg_inc, field_reg_wa_gather]` | `:4158-4160` | `field_reg_wa_gather` = the BatchEq slot the `EqGather` produces. |
| Stage-8 opening entries: `rd_inc` and `BytecodeRa(d)` Dory openings | `field_reg_inc` + `field_reg_ra` one-hot chunk Dory openings | | Add commits in the commit phase (see Phase 1 deletions below). |

### Ra generation (from bytecode)

Integer register Ra one-hots come from bytecode's 5-bit `rs1_index`/`rs2_index`/`rd_index` fields. Those are already committed as `PolynomialId::BytecodeField(RS1_INDEX=24)` / `(RS2_INDEX=25)` / `(RD_INDEX=23)` (`polynomial_id.rs:313-332`). Register Ra derivation reuses those same fields.

**FR Ra derivation: reuse the exact same bytecode fields, masked to the low 4 bits.** The bytecode binding is free — `frs1 = rs1 & 0xF`, `frs2 = rs2 & 0xF`, `frd = rd & 0xF`. The only change is a new `PolynomialId::FrdGatherIndex` populated by `frd_index[t] = rd_index(PC(t)) & 0xF` (sentinel 255 → 0). No new bytecode fields needed.

---

## v2 FR Twist module pseudocode

Drop this in place of v1's FR Twist sections in `jolt_core_module_with_fieldreg.rs`. Numbering mirrors the integer module's section headers; substitutions are noted inline.

```text
// --- PolynomialId additions (polynomial_id.rs) ---
// DELETE: FieldRegReadValue, FieldRegWriteValue, FieldRegReadLimb,
//         FieldRegWriteLimb, FieldOpOperandA/B/ResultValue,
//         BridgeValWeightA/B, BridgeAnchorA/B, BridgeValWeight,
//         BridgeAnchorWeight, IsFieldOpAny, FieldOpBGated,
//         IsFieldOpNoInv, LimbSumA, LimbSumB, RdIncAtBridge*,
//         WeightAOfRd, WeightBOfRd.
// ADD: FieldRs1Value (R1CS var idx 42), FieldRs2Value (43),
//      FieldRdValue (44), FrdGatherIndex.
// PROMOTE: FieldRegInc, FieldRegRa → Committed (was Derived).
// KEEP: FieldRegVal, FieldRegEqCycle as Virtual.

// --- Poly registration (in register_polys) ---
let field_rs1_value     = pt.add(FieldRs1Value, "FieldRs1Value", Virtual, p.log_t);
let field_rs2_value     = pt.add(FieldRs2Value, "FieldRs2Value", Virtual, p.log_t);
let field_rd_value      = pt.add(FieldRdValue,  "FieldRdValue",  Virtual, p.log_t);
let field_reg_wa        = pt.add(FieldRegWa,    "FieldRegWa",    Virtual, LOG_K_FR + p.log_t);
let field_reg_ra_rs1    = pt.add(FieldRegRaRs1, "FieldRegRaRs1", Virtual, LOG_K_FR + p.log_t);
let field_reg_ra_rs2    = pt.add(FieldRegRaRs2, "FieldRegRaRs2", Virtual, LOG_K_FR + p.log_t);
let field_reg_val       = pt.add(FieldRegVal,   "FieldRegVal",   Virtual, LOG_K_FR + p.log_t);
let field_reg_inc       = pt.add_committed(
    FieldRegInc, "FieldRegInc", Committed,
    p.log_t, LOG_K_FR + p.log_t,           // pad-to dim mirrors RdInc
);
let field_reg_ra: Vec<_> = (0..p.fr_d).map(|d|
    pt.add(FieldRegRa(d), &format!("FieldRegRa_{d}"), Committed,
           p.log_k_chunk + p.log_t)        // mirrors RamRa/BytecodeRa
).collect();

// --- r1cs_input_polys (Stage 1 Spartan outer) ---
// Add FieldRs1Value/Rs2Value/RdValue into the R1CS input column list at
// freed slots 42/43/44. R1CS rows per spec §Phase 3 (FieldAdd/Sub/
// AssertEq/Mov/SLL*/Mul/Inv).

// --- Commit phase ---
commit_pairs.push((p.field_reg_inc, DomainSeparator::Commitment));
for &ra in &p.field_reg_ra {
    commit_pairs.push((ra, DomainSeparator::Commitment));
}
// DELETE the v1 commit_pairs.push((p.field_reg_read_value, …)) and
// (p.field_reg_write_value, …) entries (lines 1735-1736 in v1).

// --- Stage 3: add FR Claim Reduction as instance [3] ---
let ch_gamma_fr_cr = ch.add("fr_cr_gamma",
    ChallengeSource::FiatShamir { after_stage: 2 });
ops.push(Op::Squeeze { challenge: ch_gamma_fr_cr });

let fr_cr_input_claim = ClaimFormula { terms: vec![
    ClaimTerm { coeff: 1, factors: vec![Eval(p.field_rd_value)] },
    ClaimTerm { coeff: 1, factors: vec![Challenge(ch_gamma_fr_cr),
                                        Eval(p.field_rs1_value)] },
    ClaimTerm { coeff: 1, factors: vec![Challenge(ch_gamma_fr_cr),
                                        Challenge(ch_gamma_fr_cr),
                                        Eval(p.field_rs2_value)] },
]};
// Batched instance: degree 2, log_T rounds, first_active = 0 within stage3.
// Kernel: identical shape to registers_kernel (jolt_core_module.rs:2981-3061)
// with inputs mapped to field_* counterparts.

// --- Stage 4: add FR Read-Write Checking (Twist proper) ---
let ch_gamma_fr_rw = ch.add("fr_rw_gamma",
    ChallengeSource::FiatShamir { after_stage: 3 });
ops.push(Op::Squeeze { challenge: ch_gamma_fr_rw });

let ch_fr_eq_bound  = ch.add("fr_eq_cycle_bound",  ChallengeSource::External);
let ch_fr_inc_bound = ch.add("fr_inc_bound",       ChallengeSource::External);

let fr_rw_rounds = LOG_K_FR + params.log_t;   // 4 + log_t
// Add as instance [N] of the stage-4 batch (alongside RegistersRW and
// RamValCheck — first_active chosen per stage4_max_rounds bookkeeping).

let fr_rw_input_claim = ClaimFormula { terms: /* same γ-batch as CR */ };

// Phase-1 kernel (cycle binding, log_t rounds, degree 3, segmented):
//   eq·field_reg_wa·field_reg_val
// + eq·field_reg_wa·field_reg_inc
// + γ·eq·field_reg_ra_rs1·field_reg_val
// + γ²·eq·field_reg_ra_rs2·field_reg_val
// Inputs: [BatchEq(fr_eq_slot), field_reg_ra_rs1, field_reg_ra_rs2,
//          field_reg_wa, field_reg_val, field_reg_inc]
// SegmentedConfig: inner_only = [true, false, false, false, false, true]
//                  inner_num_vars = log_t, outer_num_vars = LOG_K_FR

// Phase-2 kernel (address binding, LOG_K_FR=4 rounds, degree 2):
//   ch_fr_eq_bound·field_reg_wa·field_reg_val
// + ch_fr_eq_bound·ch_fr_inc_bound·field_reg_wa
// + γ·ch_fr_eq_bound·field_reg_ra_rs1·field_reg_val
// + γ²·ch_fr_eq_bound·field_reg_ra_rs2·field_reg_val
// ScalarCaptures at phase boundary:
//   BatchEq(fr_eq_slot) → ch_fr_eq_bound
//   field_reg_inc       → ch_fr_inc_bound

// Stage-4 eval flush order (append after integer RegistersRW's flush):
//   field_reg_val, field_reg_ra_rs1, field_reg_ra_rs2,
//   field_reg_wa, field_reg_inc.

// --- Stage 5: add FR Val Evaluation as instance [3] ---
// r_address_fr = s4.round_challenges[log_t..log_t+LOG_K_FR] reversed (BE)
// r_cycle_fr   = s4.round_challenges[0..log_t] reversed (BE)
// Kernel (degree 3, log_T rounds):
//   field_reg_inc · eq_gather(r_address_fr, frd[j]) · LT(r_cycle_fr, j)
// Inputs:
//   Provided { field_reg_inc }
//   EqGather { BatchEq(fr_val_eq_slot), r_address_fr, FrdGatherIndex }
//   LtTable  { BatchEq(fr_val_lt_slot), r_cycle_fr }
// Input claim: Eval(field_reg_val)
// Eval flush: field_reg_inc, field_reg_wa_gather (= BatchEq(fr_val_eq_slot))

// --- Stage 8: opening claims ---
// field_reg_inc: Dory opening at Stage-4 final point
// field_reg_ra[d]: Dory opening at Stage-4 final point (chunked one-hot,
//                  mirrors BytecodeRa/RamRa handling)
// DELETE v1's opening entries for field_reg_read_value / field_reg_write_value.
```

---

## Divergences from integer pattern

None of the integer-register-only quirks apply to FR:

1. **No carry/truncation.** Integer `Rs1Value`/`Rs2Value` hold u64 values embedded in Fr. FR values are already native field elements. Arithmetic constraints are trivial field equalities (no overflow bit, no mod-2⁶⁴ truncation).
2. **No lookup operand split.** Integer registers feed into `LeftLookupOperand`/`RightLookupOperand`/`LeftInstructionInput`/`RightInstructionInput` — the instruction lookup argument. FR ops don't use the lookup argument; `FieldRs1Value`/`FieldRs2Value`/`FieldRdValue` are consumed only by FieldAdd/Sub/Mul/Inv/Mov/SLL*/AssertEq R1CS rows (direct arithmetic) plus FieldMul/Inv routing through `V_PRODUCT`.
3. **No RdNotZero gating.** Integer `RdWriteValue` is zeroed on `rd == x0` via `IsRdNotZero`. `frd == f0` has no write-suppression semantics in v2 (f0 is a valid register). Skip the `IsRdNotZero`-style gating entirely in FR kernels.
4. **No `BranchFlag`/`JumpFlag` coupling.** Integer Registers Twist's input_claim is consumed unconditionally by Spartan rows. FR Twist's input_claim is consumed only by FR-opcode R1CS rows; these rows are already gated by per-instruction circuit flags (`IsFieldMul`, etc.) in Phase 3's R1CS additions. No extra gating needed inside the Twist.
5. **`FrdGatherIndex` sentinel.** Integer uses `RdGatherIndex` with sentinel 255 for "no write" cycles. FR analog: sentinel outside `0..=15` → `eq_gather` returns 0. Populate as `frd_index[t] = (bytecode[PC(t)].rd_index & 0xF)` when `IsFieldOp*` is set at cycle `t`, otherwise 255. This correctly zeros `FieldRegWa` on non-FR cycles so `Inc(t)=0` is consistent.

---

## Sumcheck dimensional differences

Only one constant changes between the integer and FR Twists: the register-file address width.

| Symbol | Integer | FR | Defined at |
|---|---|---|---|
| `LOG_K` (register-file address bits) | `LOG_K_REG = 7` (128 slots) | `LOG_K_FR = 4` (16 slots) | `jolt_core_module.rs:3203` |
| Stage-4 RW rounds | `7 + log_t` | `4 + log_t` | `:3204` |
| Stage-4 Phase-2 rounds | `7` | `4` | `:3453` |
| Stage-5 val-eval r_address slice | `s4.rc[log_t..log_t+7]` | `s4.rc[log_t..log_t+4]` | `:3661` |
| Segmented outer_num_vars | `log_k_reg = 7` | `log_k_fr = 4` | `:3516` |
| One-hot chunk count `fr_d` | integer uses `BytecodeRa(d)` with `d = bytecode_d` and `log_k_chunk` per chunk; integer registers' Ra is *derived from bytecode*, not committed standalone | FR Ra must be committed as its own one-hot set. Choose `fr_d = ceil(LOG_K_FR / log_k_chunk)`. With `log_k_chunk = 4`, `fr_d = 1` (single chunk) — one-hot lives in a single committed vector of length `2^4 · T = 16T`. | `config.rs` |
| Stage-4 `max_rounds` ceiling | `max(log_k_reg + log_t, log_t) = log_k_reg + log_t` | must compare `LOG_K_FR + log_t` against existing `reg_rw_rounds = 7 + log_t` → `reg_rw_rounds` still dominates. FR RW shorter than integer RW; `first_active_round = stage4_max_rounds - fr_rw_rounds = 3` | `:3206, 3220` |

`log_k_fr = 4` (not 5) per spec §ISA: "the same 5-bit bytecode fields as standard R-type `rs1`/`rs2`/`rd`, **low 4 bits used** as slot index 0..15." The MSB of each 5-bit field is ignored for FR addressing; this happens at bytecode derivation time (Ra-generation mask `& 0xF`).

**One-hot chunking choice.** With `LOG_K_FR = 4` and the default `log_k_chunk = 4`, FR Ra fits in exactly ONE chunk — no chunking required. This is simpler than integer Ra which chunks over `log_k_bytecode` (~16). `fr_d = 1` throughout. Still model it as `FieldRegRa(d)` with `d ∈ 0..1` to preserve the chunked-Ra opening machinery in Stage 8.

**Committed-poly size.** `FieldRegInc` padded to `LOG_K_FR + log_t = 4 + log_t`. `FieldRegRa(0)` committed at `log_k_chunk + log_t = 4 + log_t`. Commit cost is O(16T), dwarfed by integer RdInc (same shape) and RamRa (~log_t + 16).

---

## Hand-off summary

Everything above maps 1-to-1 onto existing integer-twist code. The implementer should:

1. Edit `polynomial_id.rs` per §v2 pseudocode "PolynomialId additions" (delete v1 zombie variants; promote Inc/Ra; add three virtual R1CS input variants).
2. Copy-paste the three integer-twist sections (`jolt_core_module.rs:2746-2776, 3137-3604, 3693-3725, 4034-4047, 4158-4160`) into the FR module, with the identifier substitutions in the mapping table.
3. Substitute the single constant `LOG_K_REG → LOG_K_FR = 4` everywhere the integer code references log_k_reg.
4. Append commit_pairs entries for `field_reg_inc` and `field_reg_ra[0]`; delete v1's `field_reg_read_value`/`field_reg_write_value` entries.
5. Wire the three new virtual R1CS columns into Spartan outer at the freed slots (42/43/44).

No new abstractions, no new Op variants, no new challenge sources beyond the two γ's and two ScalarCapture slots (`ch_gamma_fr_cr`, `ch_gamma_fr_rw`, `ch_fr_eq_bound`, `ch_fr_inc_bound`). The machinery already exists for the integer twin; FR is a pure substitution job.
