//! The fused per-cycle record: every branch-resolved value the trace-walking
//! kernels read, derived with ONE instruction decode, ONE lookup query, and
//! ONE bytecode-PC mapping per row.
//!
//! The atomic extractors in [`crate::witnesses`] deliberately recompute from
//! row accessors — collecting a wide bundle through them repeats the decode
//! per field (a 35-field row performs ~19 decodes and 6 lookup-query
//! dispatches). This row IS those extractors' values — each field transcribes
//! the corresponding `Extract` impl against the shared decode — so consumers
//! switching from atomic bundles to record lanes see bit-identical scalars.
//! The `record_matches_atomic_extractors` test pins the transcription
//! field by field.
//!
//! Lookahead policy: the only stored `next`-dependent values are the two
//! cheap instruction-kind checks ([`NextIsNoop`]'s missing-successor-is-noop
//! convention and [`ShouldJump`]'s strict variant). The rest of the `Next*`
//! family (`NextPc`, `NextUnexpandedPc`, `NextIsVirtual`,
//! `NextIsFirstInSequence`) is a shifted read of the successor row's own
//! lanes — zero-filled at `T - 1`, exactly the extractors' `map_or`/
//! `is_some_and` semantics — so the record never decodes `next`.

use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::signed::{S128, S64};
use jolt_lookup_tables::{InstructionLookupTable, LookupQuery};
use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags as _, InstructionFlagSet, InstructionFlags,
    JoltInstruction, JoltTraceRow as TraceRow,
};

use crate::witnesses::{
    decode_instruction, lookup_query, ram_access_address, row_is_noop, WitnessEnv,
};
use crate::{WitnessBundle, WitnessError, RV64_XLEN};

/// One cycle's branch-resolved witness values. Field semantics match the
/// like-named atomic extractors ([`crate::witnesses`]) exactly; see the
/// module docs for the `Next*` shifted-read contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceRecordRow {
    /// [`crate::witnesses::Pc`] — bytecode-expanded PC (mapped at trace time).
    pub pc: u64,
    /// [`crate::witnesses::UnexpandedPc`] — the instruction's address.
    pub unexpanded_pc: u64,
    /// [`crate::witnesses::Imm`].
    pub imm: i128,
    /// `(register, read value)` of rs1, as traced.
    pub rs1: Option<(u8, u64)>,
    /// `(register, read value)` of rs2, as traced.
    pub rs2: Option<(u8, u64)>,
    /// `(register, pre-write value, post-write value)` of rd, as traced.
    pub rd: Option<(u8, u64, u64)>,
    /// [`crate::witnesses::RamAddress`] — raw address, 0 when no access.
    pub ram_address: u64,
    /// [`crate::witnesses::RemappedRamAddress`] — word index; `None` for
    /// no-ops and unremappable addresses.
    pub remapped_ram_address: Option<u64>,
    /// [`crate::witnesses::RamReadValue`] — pre-access word value.
    pub ram_read_value: u64,
    /// [`crate::witnesses::RamWriteValue`] — post-access word value.
    pub ram_write_value: u64,
    /// [`crate::witnesses::LeftLookupOperand`].
    pub left_lookup_operand: u64,
    /// [`crate::witnesses::RightLookupOperand`].
    pub right_lookup_operand: u128,
    /// [`crate::witnesses::LeftInstructionInput`].
    pub left_instruction_input: u64,
    /// [`crate::witnesses::RightInstructionInput`].
    pub right_instruction_input: i128,
    /// [`crate::witnesses::Product`] — truncated signed product of the
    /// instruction inputs.
    pub product: S128,
    /// [`crate::witnesses::LookupOutput`].
    pub lookup_output: u64,
    /// [`crate::witnesses::LookupIndex`] — the 128-bit lookup index.
    pub lookup_index: u128,
    /// [`crate::witnesses::TableIndex`] — which lookup table the
    /// instruction's lookup targets, if any.
    pub table_index: Option<usize>,
    /// The instruction's full circuit-flag set
    /// ([`crate::witnesses::OpFlag`] per flag).
    pub circuit_flags: CircuitFlagSet,
    /// The instruction's full instruction-flag set
    /// ([`crate::witnesses::InstructionFlag`] per flag).
    pub instruction_flags: InstructionFlagSet,
    /// [`crate::witnesses::ShouldBranch`].
    pub should_branch: bool,
    /// [`crate::witnesses::ShouldJump`] — jump taken; a missing successor
    /// does NOT count as a no-op here.
    pub should_jump: bool,
    /// [`crate::witnesses::NextIsNoop`] — a missing successor DOES count as
    /// a no-op.
    pub next_is_noop: bool,
}

impl WitnessBundle for TraceRecordRow {
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let instruction = decode_instruction(row)?;
        let circuit_flags = instruction.circuit_flags();
        let instruction_flags = instruction.instruction_flags();

        let query = lookup_query(row);
        let (left_lookup_operand, right_lookup_operand) =
            LookupQuery::<RV64_XLEN>::to_lookup_operands(&query);
        let (left_instruction_input, right_instruction_input) =
            LookupQuery::<RV64_XLEN>::to_instruction_inputs(&query);
        let lookup_output = LookupQuery::<RV64_XLEN>::to_lookup_output(&query);
        let lookup_index = LookupQuery::<RV64_XLEN>::to_lookup_index(&query);
        let table_index =
            <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&instruction)
                .map(|kind| kind.index());
        let product = S64::from_u64(left_instruction_input)
            .mul_trunc::<2, 2>(&S128::from_i128(right_instruction_input));

        let next_is_real_noop = next.is_some_and(row_is_noop);
        Ok(Self {
            pc: row.pc(),
            unexpanded_pc: row.unexpanded_pc(),
            imm: row.imm(),
            rs1: row.rs1_index().map(|register| (register, row.rs1_value())),
            rs2: row.rs2_index().map(|register| (register, row.rs2_value())),
            rd: row
                .rd_index()
                .map(|register| (register, row.rd_pre_value(), row.rd_write_value())),
            ram_address: row.ram_address(),
            remapped_ram_address: ram_access_address(row)
                .and_then(|address| {
                    env.preprocessing
                        .memory_layout
                        .remap_word_address(address)
                        .ok()
                })
                .flatten(),
            ram_read_value: row.ram_read_value(),
            ram_write_value: row.ram_write_value(),
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input,
            right_instruction_input,
            product,
            lookup_output,
            lookup_index,
            table_index,
            circuit_flags,
            instruction_flags,
            should_branch: instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
            should_jump: circuit_flags[CircuitFlags::Jump] && !next_is_real_noop,
            next_is_noop: next.is_none_or(row_is_noop),
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::testing::with_sample_backend;
    use crate::witnesses::{
        Extract, ExtractIndexed, Imm, LeftInstructionInput, LeftLookupOperand, LookupOutput,
        NextIsNoop, OpFlag, Pc, Product, RamAddress, RamReadValue, RamWriteValue,
        RemappedRamAddress, RightInstructionInput, RightLookupOperand, ShouldBranch, ShouldJump,
        UnexpandedPc,
    };
    use crate::BundleSource;
    use jolt_riscv::CIRCUIT_FLAGS;

    const INSTRUCTION_FLAG_LIST: [InstructionFlags; jolt_riscv::NUM_INSTRUCTION_FLAGS] = [
        InstructionFlags::LeftOperandIsPC,
        InstructionFlags::RightOperandIsImm,
        InstructionFlags::LeftOperandIsRs1Value,
        InstructionFlags::RightOperandIsRs2Value,
        InstructionFlags::Branch,
        InstructionFlags::IsNoop,
    ];

    /// Twin of the record built through the atomic extractors, one decode per
    /// field — the transcription oracle.
    #[derive(Clone, Copy, Debug)]
    struct AtomicTwin;

    impl WitnessBundle for AtomicTwin {
        fn from_row(
            row: &TraceRow,
            next: Option<&TraceRow>,
            env: &WitnessEnv<'_>,
        ) -> Result<Self, WitnessError> {
            let record = TraceRecordRow::from_row(row, next, env)?;
            assert_eq!(record.pc, Pc::extract(row, next, env)?.0);
            assert_eq!(
                record.unexpanded_pc,
                UnexpandedPc::extract(row, next, env)?.0
            );
            assert_eq!(record.imm, Imm::extract(row, next, env)?.0);
            assert_eq!(
                record.rs1.map_or(0, |(_, value)| value),
                crate::witnesses::Rs1Value::extract(row, next, env)?.0
            );
            assert_eq!(
                record.rs2.map_or(0, |(_, value)| value),
                crate::witnesses::Rs2Value::extract(row, next, env)?.0
            );
            assert_eq!(
                record.rd.map_or(0, |(_, _, post)| post),
                crate::witnesses::RdWriteValue::extract(row, next, env)?.0
            );
            assert_eq!(record.ram_address, RamAddress::extract(row, next, env)?.0);
            assert_eq!(
                record.remapped_ram_address,
                RemappedRamAddress::extract(row, next, env)?.0
            );
            assert_eq!(
                record.ram_read_value,
                RamReadValue::extract(row, next, env)?.0
            );
            assert_eq!(
                record.ram_write_value,
                RamWriteValue::extract(row, next, env)?.0
            );
            assert_eq!(
                record.left_lookup_operand,
                LeftLookupOperand::extract(row, next, env)?.0
            );
            assert_eq!(
                record.right_lookup_operand,
                RightLookupOperand::extract(row, next, env)?.0
            );
            assert_eq!(
                record.left_instruction_input,
                LeftInstructionInput::extract(row, next, env)?.0
            );
            assert_eq!(
                record.right_instruction_input,
                RightInstructionInput::extract(row, next, env)?.0
            );
            let product = Product::extract(row, next, env)?.0;
            assert_eq!(record.product.magnitude_limbs(), product.magnitude_limbs());
            assert_eq!(record.product.is_positive, product.is_positive);
            assert_eq!(
                record.lookup_output,
                LookupOutput::extract(row, next, env)?.0
            );
            assert_eq!(
                record.lookup_index,
                crate::witnesses::LookupIndex::extract(row, next, env)?.0
            );
            assert_eq!(
                record.table_index,
                crate::witnesses::TableIndex::extract(row, next, env)?.0
            );
            for flag in CIRCUIT_FLAGS {
                assert_eq!(
                    record.circuit_flags[flag],
                    OpFlag::extract_indexed(flag, row, next, env)?.0,
                    "circuit flag {flag:?}"
                );
            }
            for flag in INSTRUCTION_FLAG_LIST {
                assert_eq!(
                    record.instruction_flags[flag],
                    crate::witnesses::InstructionFlag::extract_indexed(flag, row, next, env)?.0,
                    "instruction flag {flag:?}"
                );
            }
            assert_eq!(
                record.should_branch,
                ShouldBranch::extract(row, next, env)?.0
            );
            assert_eq!(record.should_jump, ShouldJump::extract(row, next, env)?.0);
            assert_eq!(record.next_is_noop, NextIsNoop::extract(row, next, env)?.0);
            Ok(Self)
        }

        fn annotated_ids() -> Vec<JoltPolynomialId> {
            Vec::new()
        }
    }

    #[test]
    fn record_matches_atomic_extractors() {
        with_sample_backend(|backend| {
            // Assertions run inside `from_row`, per cycle (real + padding,
            // including the `next = None` tail).
            let rows: Vec<AtomicTwin> = backend.bundles().unwrap();
            assert_eq!(rows.len(), 4);
        });
    }
}
