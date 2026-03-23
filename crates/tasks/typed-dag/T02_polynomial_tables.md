# T02: PolynomialTables

**Status**: `[x]` Done
**Depends on**: Nothing
**Blocks**: T05 (Stage Output Types), T08–T13 (all stage functions)
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~200 lines)

## Objective

Define `PolynomialTables<F>` — a fully typed struct with named fields for
every polynomial evaluation table the prover needs. Replaces the current
`CommittedTables<F>` (test-only, incomplete) and supplements `WitnessStore`
(tag-based, runtime-keyed).

## Deliverables

### 1. `PolynomialTables<F>` struct

File: `jolt-zkvm/src/tables.rs` (new file)

Three categories of fields:

**Committed** (opened via PCS):
- `ram_inc: Vec<F>` — dense, length `2^log_T`
- `rd_inc: Vec<F>` — dense, length `2^log_T`
- `instruction_ra: Vec<Vec<F>>` — `[instruction_d]`, each `2^(log_T + log_k_chunk)`
- `bytecode_ra: Vec<Vec<F>>` — `[bytecode_d]`
- `ram_ra: Vec<Vec<F>>` — `[ram_d]`

**Virtual** (from R1CS witness columns):
- `rd_write_value: Vec<F>`
- `rs1_value: Vec<F>`
- `rs2_value: Vec<F>`
- `hamming_weight: Vec<F>`
- `ram_address: Vec<F>`
- `ram_read_value: Vec<F>`
- `ram_write_value: Vec<F>`
- `lookup_output: Vec<F>`

**Trace-derived**:
- PV factors: `left_instruction_input`, `right_instruction_input`,
  `is_rd_not_zero`, `write_lookup_to_rd_flag`, `jump_flag`,
  `branch_flag`, `next_is_noop`
- Instruction input: `left_is_rs1`, `left_is_pc`, `right_is_rs2`,
  `right_is_imm`, `unexpanded_pc`, `imm`
- Register addresses: `rs1_ra`, `rs2_ra`, `rd_wa`
- Shift: `next_unexpanded_pc`, `next_pc`, `next_is_virtual`,
  `next_is_first_in_sequence`

### 2. Helper methods

```rust
impl<F: Field> PolynomialTables<F> {
    pub fn num_cycles(&self) -> usize;
    pub fn log_num_cycles(&self) -> usize;
    pub fn all_ra_polys(&self) -> Vec<&[F]>;
    pub fn total_d(&self) -> usize; // instruction_d + bytecode_d + ram_d
}
```

### 3. Constructor `from_witness`

```rust
pub fn from_witness(
    store: &WitnessStore<F>,
    r1cs_witness: &[Vec<F>],  // per-cycle R1CS witness vectors
    trace: &[Cycle],
    config: &ProverConfig,
) -> Self
```

Extracts all named fields from:
- `WitnessStore` (committed polys via tags)
- R1CS witness matrix (virtual polys via column indices)
- Trace (trace-derived polys via Cycle fields)

### 4. Delete or deprecate `CommittedTables`

The existing `CommittedTables<F>` in `prover.rs` should be replaced by
`PolynomialTables<F>`. If tests depend on `CommittedTables`, update them.

## Reference

- Current `CommittedTables`: `jolt-zkvm/src/prover.rs:60-105`
- Current extraction: `jolt-zkvm/src/prover.rs:156-254`
- WitnessStore: `jolt-zkvm/src/witness/store.rs`
- Polynomial tags: `jolt-ir/src/zkvm/tags.rs`
- R1CS column indices: `jolt-zkvm/src/r1cs.rs` (V_RD_WRITE_VALUE, etc.)

## Acceptance Criteria

- [x] `PolynomialTables<F>` struct compiles
- [x] `from_witness()` populates all fields correctly
- [ ] All existing tests that used `CommittedTables` updated (deferred to T18 cleanup — `CommittedTables` kept alongside per task spec)
- [ ] `cargo clippy -p jolt-zkvm` passes (blocked by pre-existing jolt-verifier `use_lt` error)
