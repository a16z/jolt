# Stage Regrouping Plan

Goal: Match old jolt-core's batching exactly so all claims converge to 1 Dory proof.

## Reference: Old jolt-core Stage Batching

Each row = one batched sumcheck call. All instances in a row share the challenge vector.

| Old Stage | Instances | Rounds | Challenge Vector |
|-----------|-----------|--------|-----------------|
| **S1** | Spartan outer (UniSkip + remaining) | `3·log_T - 1` | `r_s1` |
| **S2** | ProductVirtualRemainder, RamRWChecking, InstructionLookupsClaimReduction, RamRafEvaluation, OutputSumcheck | `log_T` | `r_s2` |
| **S3** | Shift, InstructionInput, RegistersClaimReduction | `log_T` | `r_s3` |
| **S4** | RegistersRWChecking, RamValCheck | `log_T` | `r_s4` |
| **S5** | InstructionReadRaf, RamRaClaimReduction, RegistersValEval | `log_K + log_T` | `r_s5` |
| **S6** | BytecodeReadRaf, HammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncReduction, AdviceReduction(cycle) | `log_K + log_T` | `r_s6` |
| **S7** | HammingWeightReduction, AdviceReduction(address) | `log_K` | `r_s7` |
| **S8** | Point normalize → RLC → single Dory prove() | — | unified point |

## Mapping: Existing New Stages → Old Groups

### Group 1 (= Old S1): `s1_spartan`
Already a single stage. No changes needed.

### Group 2 (= Old S2): log_T rounds
| Old Instance | New Stage |
|---|---|
| ProductVirtualRemainder | `s2_product_virtual` |
| RamReadWriteChecking | `s4_ram_rw` |
| InstructionLookupsClaimReduction | `s3_claim_reductions` (instruction subset) |
| RamRafEvaluation | `s5_ram_checking` (RAF subset) |
| OutputSumcheck | `s5_ram_checking` (output subset) |

### Group 3 (= Old S3): log_T rounds
| Old Instance | New Stage |
|---|---|
| Shift | `s3_shift` |
| InstructionInput | `s3_instruction_input` |
| RegistersClaimReduction | `s3_claim_reductions` (registers subset) |

### Group 4 (= Old S4): log_T rounds
| Old Instance | New Stage |
|---|---|
| RegistersRWChecking | `s4_rw_checking` (registers portion) |
| RamValCheck | `s4_rw_checking` (ram val portion) |

Already grouped correctly — both in `s4_rw_checking`.

### Group 5 (= Old S5): log_K + log_T rounds
| Old Instance | New Stage |
|---|---|
| InstructionReadRaf | `s_instruction_read_raf` |
| RamRaClaimReduction | `s3_claim_reductions` (ram RA subset) |
| RegistersValEval | `s5_registers_val_eval` |

**Note**: RamRaClaimReduction is part of s3_claim_reductions but belongs in Group 5
(only log_T vars — participates in first log_T rounds only).

### Group 6 (= Old S6): log_K + log_T rounds
| Old Instance | New Stage | Vars |
|---|---|---|
| BytecodeReadRaf | `s_bytecode_read_raf` | log_K + log_T |
| HammingBooleanity | `s6_booleanity` | log_T |
| Booleanity (RA) | `s6_ra_booleanity` | log_K + log_T |
| RamRaVirtual | **TODO** (or subset of s2_ra_virtual) | log_K + log_T |
| LookupsRaVirtual | `s2_ra_virtual` | log_K + log_T |
| IncClaimReduction | `s3_claim_reductions` (inc subset) | log_T |
| AdviceReduction(cycle) | **TODO** | varies |

### Group 7 (= Old S7): log_K rounds
| Old Instance | New Stage |
|---|---|
| HammingWeightReduction | `s7_hamming_reduction` |
| AdviceReduction(address) | **TODO** |

### Group 8 (= Old S8): s8_opening
Point normalization + RLC + single Dory proof.

## The Claim Chain (how 3 points collapse to 1 Dory proof)

```
Group 6 (r_s6 = challenges[0..log_K+log_T])
  ├─ IncClaimReduction (log_T vars, LowToHigh) → RamInc, RdInc at r_cycle = r_s6[0..log_T]
  ├─ Booleanity (log_K+log_T vars, HighToLow) → RA claims at (r_addr_bool, r_cycle)
  └─ RA Virtual (log_K+log_T vars) → RA claims at (r_addr_virt, r_cycle)

Group 7 (r_s7 = challenges[0..log_K])
  ├─ HammingWeightReduction retrieves r_cycle from Group 6's Booleanity opening
  └─ Produces RA claims at UNIFIED POINT = (r_s7, r_cycle) length log_K + log_T

Group 8 (opening)
  ├─ RA polys: at unified point → scale = 1
  ├─ RamInc/RdInc: at r_cycle (log_T) → scale = ∏(1 - r_addr_i)
  ├─ Advice polys: at shorter point → scale = Lagrange factor
  └─ RLC all → single Dory prove()
```

## Key Mechanism: Point Sharing via Batched Sumcheck

When instances with different num_vars are batched together:
- All share the same challenge vector (length = max num_vars)
- Shorter instances only participate in their first `num_vars` rounds
- With LowToHigh binding: short instance uses challenges[0..num_vars]
- The longer instance's first num_vars rounds use the SAME challenges
- **This is how r_cycle is shared** between log_T-var and (log_K+log_T)-var instances

## Implementation Plan

### Step 1: StageGroup abstraction

Create a `StageGroup` that composes multiple `ProverStage`s:

```rust
pub struct StageGroup<F, T> {
    stages: Vec<Box<dyn ProverStage<F, T>>>,
}
```

`build()`: calls each sub-stage's `build()`, concatenates claims and witnesses into one StageBatch.
`extract_claims()`: dispatches to each sub-stage with the shared challenge vector.
Each sub-stage only "sees" challenges for its own num_vars rounds.

### Step 2: Split s3_claim_reductions

Currently one generic stage handling all reductions. Need to split into:
- `ClaimReductionStage::instruction_lookups()` → Group 2
- `ClaimReductionStage::registers()` → Group 3
- `ClaimReductionStage::ram_ra()` → Group 5
- `ClaimReductionStage::inc()` → Group 6

The existing stage likely already supports this via configuration — just instantiate
it multiple times with different config for each group.

### Step 3: Split s2_ra_virtual

Currently one stage for all RA virtual sumchecks. Need:
- Instruction RA virtual → Group 6
- RAM RA virtual → Group 6

Both go in Group 6 anyway, so they can stay together.

### Step 4: Split s5_ram_checking

Currently handles both RAF eval and output check. Need:
- RamRafEvaluation → Group 2
- OutputSumcheck → Group 2

Both go in Group 2, so they can stay together.

### Step 5: Split s4_ram_rw and s4_rw_checking

- `s4_ram_rw` (RamRWChecking) → Group 2
- `s4_rw_checking` (RegistersRWChecking + RamValCheck) → Group 4

These are already separate files — just need to assign to correct groups.

### Step 6: Wire the pipeline

```rust
// Pipeline configuration
let groups = vec![
    StageGroup::new(vec![s1_spartan]),                    // Group 1
    StageGroup::new(vec![                                  // Group 2
        s2_product_virtual, s4_ram_rw,
        s3_claim_reductions_instruction, s5_ram_checking,
    ]),
    StageGroup::new(vec![                                  // Group 3
        s3_shift, s3_instruction_input, s3_claim_reductions_registers,
    ]),
    StageGroup::new(vec![s4_rw_checking]),                 // Group 4
    StageGroup::new(vec![                                  // Group 5
        s_instruction_read_raf, s3_claim_reductions_ram_ra,
        s5_registers_val_eval,
    ]),
    StageGroup::new(vec![                                  // Group 6
        s_bytecode_read_raf, s6_booleanity, s6_ra_booleanity,
        s2_ra_virtual, s3_claim_reductions_inc,
        // advice_reduction_cycle (when implemented),
    ]),
    StageGroup::new(vec![                                  // Group 7
        s7_hamming_reduction,
        // advice_reduction_address (when implemented),
    ]),
];
```

### Step 7: Implement claim chaining in pipeline

```rust
let mut prior_claims: Vec<ProverClaim<F>> = vec![];
for group in &mut groups {
    let batch = group.build(&prior_claims, &mut transcript);
    let (proof, challenges, final_eval) = batched_sumcheck_prove(&batch, &mut transcript);
    prior_claims = group.extract_claims(&challenges, final_eval);
    proofs.push(proof);
}
```

### Step 8: Implement point normalization (Stage 8)

Point normalization for the 3 claim categories:

```rust
impl PointNormalizationReduction {
    /// Scale eval by Lagrange zero-selector: ∏(1 - r_extra_i)
    fn normalize_dense(eval: F, r_extra: &[F]) -> F {
        let factor: F = r_extra.iter().map(|r| F::one() - *r).product();
        eval * factor
    }

    /// Scale eval by Lagrange selector for advice embedding
    fn normalize_advice(eval: F, unified_point: &[F], advice_point: &[F]) -> F {
        let factor: F = unified_point.iter()
            .map(|r| if advice_point.contains(r) { F::one() } else { F::one() - *r })
            .product();
        eval * factor
    }
}
```

### Step 9: Implement RLC batching (Stage 8)

After normalization, all claims are at the unified point:
1. Sample gamma powers from transcript
2. Compute joint_claim = Σ gamma^i * scaled_claim_i
3. Build RLC polynomial = Σ gamma^i * poly_i (streaming from trace)
4. Single Dory::prove(joint_poly, unified_point)

### Step 10: E2E test

Run `cargo nextest run -p jolt-core muldiv` in both standard and ZK modes.

## Missing Stages (to implement later)

- **AdviceClaimReduction** (two-phase: cycle in Group 6, address in Group 7)
- **RamRaVirtual** (if not already part of s2_ra_virtual)

These can be deferred — they only matter when advice polynomials are present.

## Dependencies

- `PointNormalizationReduction` → jolt-openings (new `OpeningReduction` impl)
- `ZeroPaddedSource` → jolt-poly (new `EvaluationSource` wrapper)
- Fix `combine_hints` in jolt-dory (auto-pad shorter hints with identity)
- Wire `RlcSource` into reduction pipeline

## Risk: Binding Order Compatibility

Different instances in the same batch can use different binding orders (LowToHigh vs HighToLow).
The sumcheck protocol doesn't care — each instance binds independently. But extract_claims must
know each sub-stage's convention to interpret challenges correctly.

The `SumcheckCompute` trait already handles this via `binding_order()`. The `StageGroup` just
needs to track which challenges map to which sub-stage's variables.
