//! Typed polynomial evaluation tables for the proving pipeline.
//!
//! [`PolynomialTables`] is a fully typed struct with named fields for every
//! polynomial the prover needs. It replaces tag-based lookups with compile-time
//! field access: stages read `tables.ram_inc` instead of `store.get(RAM_INC)`.
//!
//! Three categories:
//! - **Committed** — polynomials with PCS commitments (opened via Dory)
//! - **Virtual** — derived from R1CS witness columns (used in sumchecks)
//! - **Trace-derived** — extracted directly from the execution trace

use jolt_field::Field;
use jolt_instructions::flags::InstructionFlags;
use jolt_ir::zkvm::tags::poly;
use jolt_verifier::ProverConfig;
use tracer::instruction::Cycle;

use crate::witness::store::WitnessStore;

/// All polynomial evaluation tables the prover needs, grouped by origin.
///
/// Created once at the boundary between witness generation and proving via
/// [`from_witness`](Self::from_witness). Stages borrow individual fields
/// through shared `&PolynomialTables<F>` references.
pub struct PolynomialTables<F: Field> {
    // ── Committed (opened via PCS) ──────────────────────────────
    /// Dense increment polynomial for RAM timestamps, length `2^log_T`.
    pub ram_inc: Vec<F>,
    /// Dense increment polynomial for register timestamps, length `2^log_T`.
    pub rd_inc: Vec<F>,
    /// One-hot RA polynomials for instruction lookups.
    /// `instruction_ra[i]` has length `2^(log_T + log_k_chunk)`.
    pub instruction_ra: Vec<Vec<F>>,
    /// One-hot RA polynomials for bytecode lookups.
    pub bytecode_ra: Vec<Vec<F>>,
    /// One-hot RA polynomials for RAM address lookups.
    pub ram_ra: Vec<Vec<F>>,

    // ── Virtual (from R1CS witness columns) ─────────────────────
    /// Register write values from R1CS witness.
    pub rd_write_value: Vec<F>,
    /// RS1 register read values.
    pub rs1_value: Vec<F>,
    /// RS2 register read values.
    pub rs2_value: Vec<F>,
    /// Hamming weight indicator: `flag_load + flag_store` per cycle.
    pub hamming_weight: Vec<F>,
    /// RAM address for load/store instructions.
    pub ram_address: Vec<F>,
    /// RAM read values.
    pub ram_read_value: Vec<F>,
    /// RAM write values.
    pub ram_write_value: Vec<F>,
    /// Instruction lookup output.
    pub lookup_output: Vec<F>,

    // ── Trace-derived: product virtualization factors ────────────
    /// Left operand to the instruction lookup.
    pub left_instruction_input: Vec<F>,
    /// Right operand to the instruction lookup.
    pub right_instruction_input: Vec<F>,
    /// 1 if the destination register is not x0.
    pub is_rd_not_zero: Vec<F>,
    /// 1 if the lookup output should be written to rd.
    pub write_lookup_to_rd_flag: Vec<F>,
    /// 1 if the instruction is a jump (JAL/JALR).
    pub jump_flag: Vec<F>,
    /// 1 if the instruction is a branch (BEQ/BNE/...).
    pub branch_flag: Vec<F>,
    /// 1 if the next cycle is a no-op (padding or sequence end).
    pub next_is_noop: Vec<F>,

    // ── Trace-derived: instruction input decomposition ──────────
    /// 1 if left operand source is RS1 value.
    pub left_is_rs1: Vec<F>,
    /// 1 if left operand source is PC.
    pub left_is_pc: Vec<F>,
    /// 1 if right operand source is RS2 value.
    pub right_is_rs2: Vec<F>,
    /// 1 if right operand source is immediate.
    pub right_is_imm: Vec<F>,
    /// Unexpanded (physical) program counter.
    pub unexpanded_pc: Vec<F>,
    /// Immediate value.
    pub imm: Vec<F>,

    // ── Trace-derived: register addresses ───────────────────────
    /// RS1 register address (0 for NoOp).
    pub rs1_ra: Vec<F>,
    /// RS2 register address (0 for NoOp).
    pub rs2_ra: Vec<F>,
    /// RD write address (0 for NoOp).
    pub rd_wa: Vec<F>,

    // ── Trace-derived: shift (next-cycle) ───────────────────────
    /// Next cycle's unexpanded PC.
    pub next_unexpanded_pc: Vec<F>,
    /// Next cycle's expanded PC.
    pub next_pc: Vec<F>,
    /// 1 if the next instruction is virtual (part of an inline sequence).
    pub next_is_virtual: Vec<F>,
    /// 1 if the next instruction is the first in a virtual sequence.
    pub next_is_first_in_sequence: Vec<F>,
}

impl<F: Field> PolynomialTables<F> {
    /// Number of cycles (padded to power of 2).
    #[inline]
    pub fn num_cycles(&self) -> usize {
        self.ram_inc.len()
    }

    /// Log₂ of the number of cycles.
    #[inline]
    pub fn log_num_cycles(&self) -> usize {
        self.num_cycles().trailing_zeros() as usize
    }

    /// All RA polynomials as a flat list of slices, ordered:
    /// instruction, bytecode, RAM.
    pub fn all_ra_polys(&self) -> Vec<&[F]> {
        let total = self.instruction_ra.len() + self.bytecode_ra.len() + self.ram_ra.len();
        let mut ra = Vec::with_capacity(total);
        for p in &self.instruction_ra {
            ra.push(p.as_slice());
        }
        for p in &self.bytecode_ra {
            ra.push(p.as_slice());
        }
        for p in &self.ram_ra {
            ra.push(p.as_slice());
        }
        ra
    }

    /// Total number of RA chunks: `instruction_d + bytecode_d + ram_d`.
    #[inline]
    pub fn total_d(&self) -> usize {
        self.instruction_ra.len() + self.bytecode_ra.len() + self.ram_ra.len()
    }

    /// Extracts all polynomial tables from witness generation outputs.
    ///
    /// This is the single conversion point between witness generation and
    /// the typed proving pipeline. After this call, stages access named
    /// fields instead of tag-based lookups.
    pub fn from_witness(
        store: &WitnessStore<F>,
        r1cs_witness: &[Vec<F>],
        trace: &[Cycle],
        config: &ProverConfig,
    ) -> Self {
        let one_hot_params = config.one_hot_params_from_config();
        let padded_len = r1cs_witness.len();

        // ── Committed polynomials from WitnessStore ─────────────
        let ram_inc = store.get(poly::RAM_INC).to_vec();
        let rd_inc = store.get(poly::RD_INC).to_vec();

        let instruction_ra: Vec<Vec<F>> = (0..one_hot_params.instruction_d)
            .map(|i| store.get(poly::instruction_ra(i)).to_vec())
            .collect();
        let bytecode_ra: Vec<Vec<F>> = (0..one_hot_params.bytecode_d)
            .map(|i| store.get(poly::bytecode_ra(i)).to_vec())
            .collect();
        let ram_ra: Vec<Vec<F>> = (0..one_hot_params.ram_d)
            .map(|i| store.get(poly::ram_ra_committed(i)).to_vec())
            .collect();

        // ── Virtual polynomials from R1CS witness columns ───────
        let rd_write_value = extract_column(r1cs_witness, crate::r1cs::V_RD_WRITE_VALUE);
        let rs1_value = extract_column(r1cs_witness, crate::r1cs::V_RS1_VALUE);
        let rs2_value = extract_column(r1cs_witness, crate::r1cs::V_RS2_VALUE);
        let hamming_weight: Vec<F> = r1cs_witness
            .iter()
            .map(|w| w[crate::r1cs::V_FLAG_LOAD] + w[crate::r1cs::V_FLAG_STORE])
            .collect();
        let ram_address = extract_column(r1cs_witness, crate::r1cs::V_RAM_ADDRESS);
        let ram_read_value = extract_column(r1cs_witness, crate::r1cs::V_RAM_READ_VALUE);
        let ram_write_value = extract_column(r1cs_witness, crate::r1cs::V_RAM_WRITE_VALUE);
        let lookup_output = extract_column(r1cs_witness, crate::r1cs::V_LOOKUP_OUTPUT);

        // ── PV factor polynomials (mix of R1CS columns and trace flags) ─
        let left_instruction_input =
            extract_column(r1cs_witness, crate::r1cs::V_LEFT_INSTRUCTION_INPUT);
        let right_instruction_input =
            extract_column(r1cs_witness, crate::r1cs::V_RIGHT_INSTRUCTION_INPUT);
        let is_rd_not_zero =
            extract_flag_poly(trace, padded_len, InstructionFlags::IsRdNotZero as usize);
        let write_lookup_to_rd_flag =
            extract_column(r1cs_witness, crate::r1cs::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD);
        let jump_flag = extract_column(r1cs_witness, crate::r1cs::V_FLAG_JUMP);
        let branch_flag = extract_column(r1cs_witness, crate::r1cs::V_BRANCH);
        let next_is_noop = extract_column(r1cs_witness, crate::r1cs::V_NEXT_IS_NOOP);

        // ── Instruction input decomposition ─────────────────────
        let left_is_rs1 = extract_flag_poly(
            trace,
            padded_len,
            InstructionFlags::LeftOperandIsRs1Value as usize,
        );
        let left_is_pc = extract_flag_poly(
            trace,
            padded_len,
            InstructionFlags::LeftOperandIsPC as usize,
        );
        let right_is_rs2 = extract_flag_poly(
            trace,
            padded_len,
            InstructionFlags::RightOperandIsRs2Value as usize,
        );
        let right_is_imm = extract_flag_poly(
            trace,
            padded_len,
            InstructionFlags::RightOperandIsImm as usize,
        );
        let unexpanded_pc = extract_column(r1cs_witness, crate::r1cs::V_UNEXPANDED_PC);
        let imm = extract_column(r1cs_witness, crate::r1cs::V_IMM);

        // ── Register addresses (from trace) ─────────────────────
        let (rs1_ra, rs2_ra, rd_wa) = extract_register_addresses(trace, padded_len);

        // ── Shift (next-cycle) polynomials ──────────────────────
        let next_unexpanded_pc =
            extract_column(r1cs_witness, crate::r1cs::V_NEXT_UNEXPANDED_PC);
        let next_pc = extract_column(r1cs_witness, crate::r1cs::V_NEXT_PC);
        let next_is_virtual = extract_column(r1cs_witness, crate::r1cs::V_NEXT_IS_VIRTUAL);
        let next_is_first_in_sequence =
            extract_column(r1cs_witness, crate::r1cs::V_NEXT_IS_FIRST_IN_SEQUENCE);

        Self {
            ram_inc,
            rd_inc,
            instruction_ra,
            bytecode_ra,
            ram_ra,
            rd_write_value,
            rs1_value,
            rs2_value,
            hamming_weight,
            ram_address,
            ram_read_value,
            ram_write_value,
            lookup_output,
            left_instruction_input,
            right_instruction_input,
            is_rd_not_zero,
            write_lookup_to_rd_flag,
            jump_flag,
            branch_flag,
            next_is_noop,
            left_is_rs1,
            left_is_pc,
            right_is_rs2,
            right_is_imm,
            unexpanded_pc,
            imm,
            rs1_ra,
            rs2_ra,
            rd_wa,
            next_unexpanded_pc,
            next_pc,
            next_is_virtual,
            next_is_first_in_sequence,
        }
    }
}

/// Extracts a single variable column from per-cycle R1CS witness vectors.
fn extract_column<F: Field>(r1cs_witness: &[Vec<F>], var_index: usize) -> Vec<F> {
    r1cs_witness.iter().map(|w| w[var_index]).collect()
}

/// Extracts an instruction flag column from the trace, padded to `padded_len`.
///
/// NoOp cycles use the flag value from `Instruction::NoOp` so padding
/// is consistent with the R1CS witness generation.
fn extract_flag_poly<F: Field>(trace: &[Cycle], padded_len: usize, flag_idx: usize) -> Vec<F> {
    let noop_flags =
        crate::witness::flags::instruction_flags(&tracer::instruction::Instruction::NoOp);
    let pad_val = if noop_flags[flag_idx] {
        F::one()
    } else {
        F::zero()
    };

    let mut poly = Vec::with_capacity(padded_len);
    for cycle in trace {
        let iflags = crate::witness::flags::instruction_flags(&cycle.instruction());
        poly.push(if iflags[flag_idx] {
            F::one()
        } else {
            F::zero()
        });
    }
    poly.resize(padded_len, pad_val);
    poly
}

/// Extracts register address polynomials (rs1_ra, rs2_ra, rd_wa) from the trace.
fn extract_register_addresses<F: Field>(
    trace: &[Cycle],
    padded_len: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut rs1_ra = Vec::with_capacity(padded_len);
    let mut rs2_ra = Vec::with_capacity(padded_len);
    let mut rd_wa = Vec::with_capacity(padded_len);

    for cycle in trace {
        rs1_ra.push(
            cycle
                .rs1_read()
                .map_or(F::zero(), |(addr, _)| F::from_u64(addr as u64)),
        );
        rs2_ra.push(
            cycle
                .rs2_read()
                .map_or(F::zero(), |(addr, _)| F::from_u64(addr as u64)),
        );
        rd_wa.push(
            cycle
                .rd_write()
                .map_or(F::zero(), |(addr, _, _)| F::from_u64(addr as u64)),
        );
    }

    rs1_ra.resize(padded_len, F::zero());
    rs2_ra.resize(padded_len, F::zero());
    rd_wa.resize(padded_len, F::zero());

    (rs1_ra, rs2_ra, rd_wa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use tracer::instruction::{
        add::ADD,
        format::format_r::{FormatR, RegisterStateFormatR},
        RISCVCycle,
    };

    fn make_add_cycle(addr: u64, rs1: u64, rs2: u64) -> Cycle {
        Cycle::from(RISCVCycle {
            instruction: ADD {
                address: addr,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                ..ADD::default()
            },
            register_state: RegisterStateFormatR {
                rd: (0, rs1.wrapping_add(rs2)),
                rs1,
                rs2,
            },
            ram_access: (),
        })
    }

    #[test]
    fn from_witness_populates_all_fields() {
        use crate::witness::generate::generate_witnesses;

        let trace = vec![make_add_cycle(0x1000, 3, 4), make_add_cycle(0x1004, 10, 20)];
        let output = generate_witnesses::<Fr>(&trace);

        let tables = PolynomialTables::from_witness(
            &output.witness_store,
            &output.cycle_witnesses,
            &trace,
            &output.config,
        );

        // Padded to power of 2
        assert_eq!(tables.num_cycles(), 2);
        assert_eq!(tables.log_num_cycles(), 1);

        // Committed polys present and correctly sized
        assert_eq!(tables.rd_inc.len(), 2);
        assert_eq!(tables.ram_inc.len(), 2);

        // RA polys have the expected D counts
        let params = output.config.one_hot_params_from_config();
        assert_eq!(tables.instruction_ra.len(), params.instruction_d);
        assert_eq!(tables.bytecode_ra.len(), params.bytecode_d);
        assert_eq!(tables.ram_ra.len(), params.ram_d);
        assert_eq!(tables.total_d(), params.instruction_d + params.bytecode_d + params.ram_d);

        // Virtual polys sized to num_cycles
        assert_eq!(tables.rd_write_value.len(), 2);
        assert_eq!(tables.rs1_value.len(), 2);
        assert_eq!(tables.rs2_value.len(), 2);
        assert_eq!(tables.hamming_weight.len(), 2);
        assert_eq!(tables.ram_address.len(), 2);
        assert_eq!(tables.ram_read_value.len(), 2);
        assert_eq!(tables.ram_write_value.len(), 2);
        assert_eq!(tables.lookup_output.len(), 2);

        // Trace-derived polys sized to num_cycles
        assert_eq!(tables.left_instruction_input.len(), 2);
        assert_eq!(tables.is_rd_not_zero.len(), 2);
        assert_eq!(tables.rs1_ra.len(), 2);
        assert_eq!(tables.next_pc.len(), 2);

        // all_ra_polys returns the right count
        assert_eq!(tables.all_ra_polys().len(), tables.total_d());
    }

    #[test]
    fn register_addresses_extracted_correctly() {
        use crate::witness::generate::generate_witnesses;

        let trace = vec![make_add_cycle(0x1000, 3, 4)];
        let output = generate_witnesses::<Fr>(&trace);

        let tables = PolynomialTables::from_witness(
            &output.witness_store,
            &output.cycle_witnesses,
            &trace,
            &output.config,
        );

        // ADD uses rs1=2, rs2=3, rd=1
        assert_eq!(tables.rs1_ra[0], Fr::from_u64(2));
        assert_eq!(tables.rs2_ra[0], Fr::from_u64(3));
        assert_eq!(tables.rd_wa[0], Fr::from_u64(1));
    }
}
