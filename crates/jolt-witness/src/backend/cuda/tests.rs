#![expect(
    clippy::expect_used,
    reason = "test module: device and fixture errors fail loudly"
)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use jolt_program::execution::{OwnedTrace, RegisterState, TraceRow};
use jolt_riscv::{CircuitFlags, InstructionFlags, JoltInstructionKind};

use super::*;
use crate::backend::trace::TraceBackend;
use crate::backend::ProgramSource;
use crate::testing::{all_kinds_backend, supported_jolt_kinds};
use crate::witnesses::{
    BytecodePc, InstructionFlag, InstructionRafFlag, LeftInstructionInput, LeftLookupOperand,
    LookupIndex, LookupOutput, MappedPc, NextIsNoop, OpFlag, Product, RamHammingWeight, RamInc,
    RdAddress, RdInc, RemappedRamAddress, RightInstructionInput, RightLookupOperand, ShouldBranch,
    ShouldJump, TableIndex,
};
use crate::{collect_bundles, RowSource, WitnessBundle};

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct AtomProbe {
    #[opening(OpFlags(CircuitFlags::AddOperands))]
    add_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::SubtractOperands))]
    subtract_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::MultiplyOperands))]
    multiply_operands: OpFlag,
    #[opening(OpFlags(CircuitFlags::Load))]
    load: OpFlag,
    #[opening(OpFlags(CircuitFlags::Store))]
    store: OpFlag,
    #[opening(OpFlags(CircuitFlags::Jump))]
    jump: OpFlag,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    write_lookup_output_to_rd: OpFlag,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    virtual_instruction: OpFlag,
    #[opening(OpFlags(CircuitFlags::Assert))]
    assert_flag: OpFlag,
    #[opening(OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC))]
    do_not_update_unexpanded_pc: OpFlag,
    #[opening(OpFlags(CircuitFlags::Advice))]
    advice: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsCompressed))]
    is_compressed: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    is_first_in_sequence: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsLastInSequence))]
    is_last_in_sequence: OpFlag,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsPC))]
    left_is_pc: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsImm))]
    right_is_imm: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsRs1Value))]
    left_is_rs1: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsRs2Value))]
    right_is_rs2: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    branch: InstructionFlag,
    #[opening(InstructionFlags(InstructionFlags::IsNoop))]
    is_noop: InstructionFlag,
    #[opening(InstructionRafFlag)]
    raf_flag: InstructionRafFlag,
    next_is_noop: NextIsNoop,
    hamming: RamHammingWeight,
    table_index: TableIndex,
    bytecode_pc: BytecodePc,
    rd_address: RdAddress,
    rd_inc: RdInc,
    ram_inc: RamInc,
    left_instruction_input: LeftInstructionInput,
    right_instruction_input: RightInstructionInput,
    left_lookup_operand: LeftLookupOperand,
    right_lookup_operand: RightLookupOperand,
    lookup_output: LookupOutput,
    product: Product,
    should_branch: ShouldBranch,
    should_jump: ShouldJump,
}

impl AtomProbe {
    fn circuit_flag(&self, flag: CircuitFlags) -> bool {
        match flag {
            CircuitFlags::AddOperands => self.add_operands.0,
            CircuitFlags::SubtractOperands => self.subtract_operands.0,
            CircuitFlags::MultiplyOperands => self.multiply_operands.0,
            CircuitFlags::Load => self.load.0,
            CircuitFlags::Store => self.store.0,
            CircuitFlags::Jump => self.jump.0,
            CircuitFlags::WriteLookupOutputToRD => self.write_lookup_output_to_rd.0,
            CircuitFlags::VirtualInstruction => self.virtual_instruction.0,
            CircuitFlags::Assert => self.assert_flag.0,
            CircuitFlags::DoNotUpdateUnexpandedPC => self.do_not_update_unexpanded_pc.0,
            CircuitFlags::Advice => self.advice.0,
            CircuitFlags::IsCompressed => self.is_compressed.0,
            CircuitFlags::IsFirstInSequence => self.is_first_in_sequence.0,
            CircuitFlags::IsLastInSequence => self.is_last_in_sequence.0,
        }
    }

    fn instruction_flag(&self, flag: InstructionFlags) -> bool {
        match flag {
            InstructionFlags::LeftOperandIsPC => self.left_is_pc.0,
            InstructionFlags::RightOperandIsImm => self.right_is_imm.0,
            InstructionFlags::LeftOperandIsRs1Value => self.left_is_rs1.0,
            InstructionFlags::RightOperandIsRs2Value => self.right_is_rs2.0,
            InstructionFlags::Branch => self.branch.0,
            InstructionFlags::IsNoop => self.is_noop.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct ColumnProbe {
    lookup_index: LookupIndex,
    mapped_pc: MappedPc,
    remapped_ram: RemappedRamAddress,
}

const RAM_K: usize = 1 << 6;

struct Fixture {
    stream: Arc<CudaStream>,
    backend: TraceBackend<OwnedTrace>,
    trace: DeviceTrace,
    cycles: usize,
}

fn fixture(seed: u64) -> Option<Fixture> {
    let stream = CudaContext::new(0).ok()?.default_stream();
    let (backend, cycles, kinds) = all_kinds_backend(seed);
    let physical = backend.rows().expect("the fixture is slice-backed");
    let present: Vec<_> = physical
        .iter()
        .map(|row| row.instruction.instruction_kind)
        .collect();
    let missing: Vec<_> = supported_jolt_kinds()
        .into_iter()
        .filter(|kind| !present.contains(kind))
        .collect();
    assert!(
        missing.is_empty(),
        "{} supported kinds never appear in the fixture trace, so the device dispatch would be \
         untested on them: {:?}",
        missing.len(),
        missing.iter().take(8).collect::<Vec<_>>(),
    );
    assert!(
        kinds > 60,
        "only {kinds} kinds — too narrow to be a coverage claim",
    );

    let trace = DeviceTrace::upload(
        Arc::clone(&stream),
        physical,
        cycles,
        backend.program_preprocessing(),
    )
    .expect("device residency");
    Some(Fixture {
        stream,
        backend,
        trace,
        cycles,
    })
}

impl Fixture {
    fn u32s(&self, device: &CudaSlice<u32>) -> Vec<u32> {
        self.stream.clone_dtoh(device).expect("download")
    }

    fn u64s(&self, device: &CudaSlice<u64>) -> Vec<u64> {
        self.stream.clone_dtoh(device).expect("download")
    }

    fn probes<B: WitnessBundle + Copy + Send + Sync>(&self) -> Vec<B> {
        collect_bundles(&self.backend, self.cycles).expect("reference atoms")
    }

    fn row(&self, index: usize) -> TraceRow {
        self.backend
            .rows()
            .expect("slice-backed")
            .get(index)
            .copied()
            .unwrap_or_default()
    }

    fn register(&self, index: usize, slot: impl Fn(&RegisterState) -> Option<u8>) -> u32 {
        slot(&self.row(index).registers).map_or(REGISTER_ADDRESS_ABSENT, u32::from)
    }

    fn wide(limbs: &[u64], index: usize) -> i128 {
        Self::unsigned_wide(limbs, index) as i128
    }

    fn unsigned_wide(limbs: &[u64], index: usize) -> u128 {
        u128::from(limbs[2 * index]) | (u128::from(limbs[2 * index + 1]) << 64)
    }
}

#[test]
fn device_atom_columns_match_the_reference_extractors() {
    let Some(fixture) = fixture(7) else {
        return;
    };
    let columns = fixture.trace.atom_columns().expect("atom columns");
    let flags = fixture.u32s(&columns.flags);
    let table_index = fixture.u32s(&columns.table_index);
    let bytecode_pc = fixture.u64s(&columns.bytecode_pc);
    let rd_pre = fixture.u64s(&columns.rd_pre_value);
    let rs1_address = fixture.u32s(&columns.rs1_address);
    let rs2_address = fixture.u32s(&columns.rs2_address);
    let rd_address = fixture.u32s(&columns.rd_address);
    let rd_inc = fixture.u64s(&columns.rd_inc);
    let ram_inc = fixture.u64s(&columns.ram_inc);
    let left_input = fixture.u64s(&columns.left_instruction_input);
    let right_input = fixture.u64s(&columns.right_instruction_input);
    let left_operand = fixture.u64s(&columns.left_lookup_operand);
    let right_operand = fixture.u64s(&columns.right_lookup_operand);
    let lookup_output = fixture.u64s(&columns.lookup_output);
    let product = fixture.u64s(&columns.product_magnitude);
    let rows: Vec<AtomProbe> = fixture.probes();

    assert!(
        flags.iter().any(|&mask| mask != flags[0]),
        "every cycle has the same flag mask, so a kernel ignoring the kind would pass",
    );
    assert!(
        table_index.contains(&TABLE_INDEX_ABSENT),
        "no cycle lacks a lookup table, so the absent sentinel is untested",
    );
    assert!(
        rd_address.contains(&REGISTER_ADDRESS_ABSENT),
        "every cycle writes a register, so the absent sentinel is untested",
    );
    assert!(
        (0..fixture.cycles).any(|index| Fixture::wide(&ram_inc, index) != 0),
        "every RAM increment is zero, so the increment path is untested",
    );
    assert!(
        rows.iter().any(|row| row.right_instruction_input.0 < 0),
        "no cycle has a negative right instruction input, so the two's-complement limb pair is \
         untested",
    );
    assert!(
        rows.iter()
            .any(|row| row.product.0.magnitude_as_u128() != 0),
        "every product is zero, so the product path is untested",
    );
    assert!(
        rows.iter()
            .any(|row| !row.product.0.is_positive && row.product.0.magnitude_as_u128() != 0),
        "no cycle has a negative product, so the product sign bit is untested",
    );
    assert!(
        rows.iter()
            .any(|row| row.right_lookup_operand.0 > u128::from(u64::MAX)),
        "every right lookup operand fits in 64 bits, so the wide operand limb is untested",
    );
    assert!(
        rows.iter().any(|row| row.lookup_output.0 != 0),
        "every lookup output is zero, so output_of is untested",
    );
    assert!(
        rows.iter().any(|row| row.should_branch.0),
        "no cycle branches, so the should-branch bit is untested",
    );
    assert!(
        rows.iter().any(|row| row.should_jump.0),
        "no cycle jumps, so the should-jump bit is untested",
    );
    assert!(
        rows.last().is_some_and(|row| row.jump.0),
        "the fixture's last cycle does not jump, so ShouldJump's `is_some_and` lookahead is \
         indistinguishable from NextIsNoop's `is_none_or` at the one index where they differ",
    );

    for (index, row) in rows.iter().enumerate() {
        let mask = flags[index];
        let bit = |position: u32| mask >> position & 1 == 1;
        for (slot, flag) in PACK_CIRCUIT_ORDER.into_iter().enumerate() {
            assert_eq!(
                bit(FLAG_BIT_CIRCUIT_BASE + slot as u32),
                row.circuit_flag(flag),
                "{flag:?} at cycle {index}",
            );
        }
        for (slot, flag) in PACK_INSTRUCTION_ORDER.into_iter().enumerate() {
            assert_eq!(
                bit(FLAG_BIT_INSTRUCTION_BASE + slot as u32),
                row.instruction_flag(flag),
                "{flag:?} at cycle {index}",
            );
        }
        assert_eq!(bit(FLAG_BIT_RAF), row.raf_flag.0, "raf flag at {index}");
        assert_eq!(
            bit(FLAG_BIT_NOOP_ROW),
            fixture.row(index).instruction.instruction_kind == JoltInstructionKind::NoOp,
            "noop row at {index}",
        );
        assert_eq!(
            bit(FLAG_BIT_NEXT_IS_NOOP),
            row.next_is_noop.0,
            "next is noop at {index}",
        );
        assert_eq!(
            bit(FLAG_BIT_RAM_HAMMING),
            row.hamming.0,
            "ram hamming at {index}",
        );
        assert_eq!(
            table_index[index],
            row.table_index
                .0
                .map_or(TABLE_INDEX_ABSENT, |slot| slot as u32),
            "table index at {index}",
        );
        assert_eq!(
            bytecode_pc[index], row.bytecode_pc.0 as u64,
            "bytecode pc at {index}",
        );
        assert_eq!(
            rd_pre[index],
            fixture
                .row(index)
                .registers
                .rd
                .map_or(0, |write| write.pre_value),
            "rd pre at {index}",
        );
        assert_eq!(
            rs1_address[index],
            fixture.register(index, |registers| registers.rs1.map(|read| read.register)),
            "rs1 address at {index}",
        );
        assert_eq!(
            rs2_address[index],
            fixture.register(index, |registers| registers.rs2.map(|read| read.register)),
            "rs2 address at {index}",
        );
        assert_eq!(
            rd_address[index],
            fixture.register(index, |registers| registers.rd.map(|write| write.register)),
            "rd address at {index}",
        );
        assert_eq!(
            Fixture::wide(&rd_inc, index),
            row.rd_inc.0,
            "rd increment at {index}",
        );
        assert_eq!(
            Fixture::wide(&ram_inc, index),
            row.ram_inc.0,
            "ram increment at {index}",
        );
        assert_eq!(
            left_input[index], row.left_instruction_input.0,
            "left instruction input at {index}",
        );
        assert_eq!(
            Fixture::wide(&right_input, index),
            row.right_instruction_input.0,
            "right instruction input at {index}",
        );
        assert_eq!(
            left_operand[index], row.left_lookup_operand.0,
            "left lookup operand at {index}",
        );
        assert_eq!(
            Fixture::unsigned_wide(&right_operand, index),
            row.right_lookup_operand.0,
            "right lookup operand at {index}",
        );
        assert_eq!(
            lookup_output[index], row.lookup_output.0,
            "lookup output at {index}",
        );
        assert_eq!(
            Fixture::unsigned_wide(&product, index),
            row.product.0.magnitude_as_u128(),
            "product magnitude at {index}",
        );
        assert_eq!(
            bit(FLAG_BIT_PRODUCT_NEGATIVE),
            !row.product.0.is_positive && row.product.0.magnitude_as_u128() != 0,
            "product sign at {index}",
        );
        assert_eq!(
            bit(FLAG_BIT_SHOULD_BRANCH),
            row.should_branch.0,
            "should branch at {index}",
        );
        assert_eq!(
            bit(FLAG_BIT_SHOULD_JUMP),
            row.should_jump.0,
            "should jump at {index}",
        );
    }
}

#[test]
fn device_columns_match_the_reference_extractors() {
    let Some(fixture) = fixture(11) else {
        return;
    };
    let limbs = fixture.u64s(
        &fixture
            .trace
            .lookup_index_limbs()
            .expect("lookup index limbs"),
    );
    let pc = fixture.u32s(&fixture.trace.mapped_pc_words().expect("mapped pc words"));
    let (ram, _) = fixture
        .trace
        .remapped_ram_words(RAM_K)
        .expect("remapped ram words");
    let ram = fixture.u32s(&ram);
    let rows: Vec<ColumnProbe> = fixture.probes();

    assert!(
        limbs.iter().any(|&limb| limb != 0),
        "every lookup-index limb is zero, so the interleave path is untested",
    );
    assert!(
        ram.contains(&COLD),
        "no cycle is RAM-cold, so the cold sentinel is untested",
    );
    assert!(
        pc.iter().any(|&word| word != pc[0]),
        "every mapped PC is the same, so a kernel ignoring the address would pass",
    );

    for (index, row) in rows.iter().enumerate() {
        let got = u128::from(limbs[2 * index]) | (u128::from(limbs[2 * index + 1]) << 64);
        assert_eq!(got, row.lookup_index.0, "lookup index at {index}");
        assert_eq!(
            pc[index],
            row.mapped_pc.0.map_or(COLD, |slot| slot as u32),
            "mapped pc at {index}",
        );
        assert_eq!(
            ram[index],
            row.remapped_ram.0.map_or(COLD, |address| address as u32),
            "remapped ram at {index}",
        );
    }
}

#[test]
fn the_kernel_source_agrees_on_the_flag_bits() {
    let source = include_str!("kernels/atoms.cu");
    let jump = circuit_flag_bit(CircuitFlags::Jump).expect("Jump has a canonical bit");
    let branch =
        instruction_flag_bit(InstructionFlags::Branch).expect("Branch has a canonical bit");
    for (name, bit) in [
        ("FLAG_BIT_JUMP", jump),
        ("FLAG_BIT_BRANCH", branch),
        ("FLAG_BIT_NOOP_ROW", FLAG_BIT_NOOP_ROW),
        ("FLAG_BIT_NEXT_IS_NOOP", FLAG_BIT_NEXT_IS_NOOP),
        ("FLAG_BIT_RAM_HAMMING", FLAG_BIT_RAM_HAMMING),
        ("FLAG_BIT_SHOULD_BRANCH", FLAG_BIT_SHOULD_BRANCH),
        ("FLAG_BIT_SHOULD_JUMP", FLAG_BIT_SHOULD_JUMP),
        ("FLAG_BIT_PRODUCT_NEGATIVE", FLAG_BIT_PRODUCT_NEGATIVE),
    ] {
        let expected = format!("#define {name} {bit}");
        assert!(
            source.contains(&expected),
            "the CUDA source must declare `{expected}`",
        );
    }
}

#[test]
fn the_canonical_flag_layout_has_no_overlap() {
    let circuit = FLAG_BIT_CIRCUIT_BASE + PACK_CIRCUIT_ORDER.len() as u32;
    assert!(
        circuit <= FLAG_BIT_INSTRUCTION_BASE,
        "the circuit flags run into the instruction flags",
    );
    let instruction = FLAG_BIT_INSTRUCTION_BASE + PACK_INSTRUCTION_ORDER.len() as u32;
    assert!(
        instruction <= FLAG_BIT_RAF,
        "the instruction flags run into the row-derived bits",
    );
}

#[test]
fn the_flag_bit_accessors_agree_with_the_canonical_order() {
    for (slot, flag) in PACK_CIRCUIT_ORDER.into_iter().enumerate() {
        assert_eq!(
            circuit_flag_bit(flag),
            Some(FLAG_BIT_CIRCUIT_BASE + slot as u32),
            "{flag:?} resolves to the wrong canonical bit",
        );
    }
    for (slot, flag) in PACK_INSTRUCTION_ORDER.into_iter().enumerate() {
        assert_eq!(
            instruction_flag_bit(flag),
            Some(FLAG_BIT_INSTRUCTION_BASE + slot as u32),
            "{flag:?} resolves to the wrong canonical bit",
        );
    }
    assert_eq!(
        PACK_CIRCUIT_ORDER.len(),
        jolt_riscv::NUM_CIRCUIT_FLAGS,
        "a circuit flag was added to the enum without a canonical bit, so it silently has none",
    );
    assert_eq!(
        PACK_INSTRUCTION_ORDER.len(),
        jolt_riscv::NUM_INSTRUCTION_FLAGS,
        "an instruction flag was added to the enum without a canonical bit",
    );
}
