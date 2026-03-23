# T05: Stage Output Types

**Status**: `[x]` Done
**Depends on**: T01 (Claim Types), T02 (PolynomialTables)
**Blocks**: T06–T15 (everything downstream)
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~300 lines, type definitions only)

## Objective

Define the fully typed output struct for every stage in the DAG. These types
are the backbone of the stateless claim propagation — each stage function
returns one, and downstream stages accept references to upstream ones.

## Deliverables

File: `jolt-zkvm/src/stage_types.rs` (new file)

### Common Types

```rust
/// Sumcheck proof for one stage (may contain uni-skip data).
pub struct SumcheckStageProof<F: Field> {
    pub round_polys: Vec<UnivariatePoly<F>>,
    pub uni_skip: Option<UniSkipProof<F>>,
}
```

### SpartanOutput (S1)

```rust
pub struct SpartanOutput<F: Field> {
    pub proof: SpartanStageProof<F>,  // from jolt-spartan
    pub r_x: Vec<F>,
    pub r_y: Vec<F>,
    pub evals: SpartanVirtualEvals<F>,
}

pub struct SpartanVirtualEvals<F> {
    // Memory
    pub ram_read_value: VirtualEval<F>,
    pub ram_write_value: VirtualEval<F>,
    pub ram_address: VirtualEval<F>,
    // Lookups
    pub lookup_output: VirtualEval<F>,
    pub left_operand: VirtualEval<F>,
    pub right_operand: VirtualEval<F>,
    pub left_instruction_input: VirtualEval<F>,
    pub right_instruction_input: VirtualEval<F>,
    // Registers
    pub rd_write_value: VirtualEval<F>,
    pub rs1_value: VirtualEval<F>,
    pub rs2_value: VirtualEval<F>,
}
```

### Stage2Output (S2: RamRW + PV + InstrLookupsCR + RamRaf + OutputCheck)

```rust
pub struct Stage2Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    // From RamReadWriteChecking
    pub ram_rw_point: Vec<F>,         // (r_address || r_cycle) at stage 2
    pub ram_val_s2: VirtualEval<F>,
    pub ram_inc_at_s2: CommittedEval<F>,

    // From ProductVirtualRemainder
    pub pv_point: Vec<F>,             // r_cycle at stage 2
    pub next_is_noop_s2: VirtualEval<F>,
    pub left_instr_input_s2: VirtualEval<F>,
    pub right_instr_input_s2: VirtualEval<F>,

    // From InstructionLookupsClaimReduction
    pub lookup_output_s2: VirtualEval<F>,
    pub left_operand_s2: VirtualEval<F>,
    pub right_operand_s2: VirtualEval<F>,
    pub left_instr_input_cr_s2: VirtualEval<F>,
    pub right_instr_input_cr_s2: VirtualEval<F>,

    // From RamRafEvaluation
    pub ram_raf_point: Vec<F>,
    pub ram_raf_eval: VirtualEval<F>,

    // From OutputCheck
    pub output_check_point: Vec<F>,
    pub ram_val_final_s2: VirtualEval<F>,
}
```

### Stage3Output (S3: Shift + InstrInput + RegistersCR)

```rust
pub struct Stage3Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    // From Shift
    pub shift_point: Vec<F>,

    // From InstructionInput
    pub instr_input_point: Vec<F>,
    pub rs1_value_s3: VirtualEval<F>,
    pub rs2_value_s3: VirtualEval<F>,

    // From RegistersClaimReduction
    pub registers_cr_point: Vec<F>,
    pub rd_write_value_s3: VirtualEval<F>,
    pub rs1_value_cr_s3: VirtualEval<F>,
    pub rs2_value_cr_s3: VirtualEval<F>,
}
```

### Stage4Output (S4: RegistersRW + RamValCheck)

```rust
pub struct Stage4Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    // Committed evals for IncCR in S6
    pub ram_inc_at_s4: CommittedEval<F>,
    pub rd_inc_at_s4: CommittedEval<F>,

    // From RegistersRW
    pub registers_rw_point: Vec<F>,

    // From RamValCheck
    pub ram_val_check_point: Vec<F>,
}
```

### Stage5Output (S5: InstrReadRaf + RamRaCR + RegistersValEval)

```rust
pub struct Stage5Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    // Committed evals for IncCR in S6
    pub rd_inc_at_s5: CommittedEval<F>,

    // From InstructionReadRaf — RA claims for HammingWeightCR
    pub instruction_ra_at_s5: Vec<CommittedEval<F>>,

    // From RamRaCR
    pub ram_ra_cr_point: Vec<F>,

    // From RegistersValEval
    pub registers_val_eval_point: Vec<F>,
}
```

### Stage6Output (S6: BytecodeReadRaf + Booleanity + HammingBool + RamRaVirtual + InstrRaVirtual + IncCR)

```rust
pub struct Stage6Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    // From IncClaimReduction — the cycle point that feeds S7
    pub r_cycle_s6: Vec<F>,
    pub ram_inc_reduced: CommittedEval<F>,
    pub rd_inc_reduced: CommittedEval<F>,

    // From Booleanity — RA claims for HammingWeightCR
    pub instruction_ra_at_s6: Vec<CommittedEval<F>>,
    pub bytecode_ra_at_s6: Vec<CommittedEval<F>>,
    pub ram_ra_at_s6: Vec<CommittedEval<F>>,

    // From RA Virtual — additional RA claims for HammingWeightCR
    pub instruction_ra_virtual_s6: Vec<CommittedEval<F>>,
    pub ram_ra_virtual_s6: Vec<CommittedEval<F>>,

    // From HammingBooleanity
    pub hamming_evals_s6: Vec<VirtualEval<F>>,

    // From BytecodeReadRaf
    pub bytecode_ra_raf_s6: Vec<CommittedEval<F>>,
}
```

### Stage7Output (S7: HammingWeightCR)

```rust
pub struct Stage7Output<F: Field> {
    pub proof: SumcheckStageProof<F>,
    pub challenges: Vec<F>,

    /// THE unified opening point: (r_addr_s7 || r_cycle_s6)
    pub unified_point: Vec<F>,

    /// All RA polynomial evals at unified point
    pub instruction_ra: Vec<CommittedEval<F>>,
    pub bytecode_ra: Vec<CommittedEval<F>>,
    pub ram_ra: Vec<CommittedEval<F>>,
}
```

## Notes

- These are INITIAL types. Fields may be refined during stage implementation
  (T08–T13) as we discover exactly which evals each downstream stage needs.
- The types should be generic over `F: Field` only — no backend parameter.
  Backend genericity is in the stage FUNCTIONS, not the output types.
- Some fields may need adjustment after the IR audit (T03) clarifies exact
  data flow.
- Virtual evals that are read by exactly one downstream stage could arguably
  be omitted (computed inline). But for clarity and debuggability, include
  them all initially. We can trim later.

## Acceptance Criteria

- [ ] All 8 output types defined and compile
- [ ] All types are generic over `F: Field`
- [ ] All fields documented with which sub-instance produces them
- [ ] `cargo clippy -p jolt-zkvm` passes
