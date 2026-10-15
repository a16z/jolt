use super::CUDA_PROFILE;
use jolt_riscv::{CircuitFlags, JoltTraceRow as TraceRow, RamAccess};
use std::ops::Range;

use crate::{RowSource, WitnessError, JOLT_VM_LABEL};

pub const NO_SEQUENCE: u32 = u32::MAX;

pub const RAM_NO_ACCESS: u64 = u64::MAX;

pub const EXTRA_WORDS: usize = 11;
pub const EXTRA_BYTECODE_PC: usize = 10;

pub const EXTRA_RS1: usize = 0;
pub const EXTRA_RS2: usize = 1;
pub const EXTRA_RD_POST: usize = 2;
pub const EXTRA_RAM_READ: usize = 3;
pub const EXTRA_RAM_WRITE: usize = 4;
pub const EXTRA_IMM_LO: usize = 5;
pub const EXTRA_IMM_HI: usize = 6;
pub const EXTRA_KIND_BITS: usize = 7;
pub const EXTRA_REGISTERS: usize = 8;
pub const EXTRA_RD_PRE: usize = 9;

pub const REGISTER_ABSENT: u64 = 0xFF;

pub const KIND_UNMAPPED: u16 = u16::MAX;

pub const BIT_IS_COMPRESSED: u32 = 16;
pub const BIT_IS_FIRST_IN_SEQUENCE: u32 = 17;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedTrace {
    pub cycles: usize,
    pub is_noop: Vec<u8>,
    pub address: Vec<u64>,
    pub virtual_sequence: Vec<u32>,
    pub ram_address: Vec<u64>,
    pub extras: Vec<u64>,
}

#[cfg(feature = "parallel")]
const PACK_CHUNK: usize = 1 << 14;

impl PackedTrace {
    pub fn from_source(source: &dyn RowSource, range: Range<usize>) -> Result<Self, WitnessError> {
        let len =
            range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    label: JOLT_VM_LABEL,
                    reason: "reversed CUDA trace window".to_owned(),
                })?;
        let mut packed = Self::with_capacity(len);
        let mut offset = 0;
        source.visit_chunks(range, 1 << 14, &mut |rows, _, _| {
            let end = offset + rows.len();
            if end > len {
                return Err(WitnessError::InvalidDimensions {
                    label: JOLT_VM_LABEL,
                    reason: "row source exceeded the CUDA trace window".to_owned(),
                });
            }
            fill(
                0,
                rows,
                &TraceRow::default(),
                &mut packed.is_noop[offset..end],
                &mut packed.address[offset..end],
                &mut packed.virtual_sequence[offset..end],
                &mut packed.ram_address[offset..end],
                &mut packed.extras[offset * EXTRA_WORDS..end * EXTRA_WORDS],
            );
            offset = end;
            Ok(())
        })?;
        if offset != len {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!("CUDA trace window expected {len} rows, received {offset}"),
            });
        }
        packed.cycles = len;
        Ok(packed)
    }

    pub fn with_capacity(cycles: usize) -> Self {
        Self {
            cycles: 0,
            is_noop: vec![0u8; cycles],
            address: vec![0u64; cycles],
            virtual_sequence: vec![0u32; cycles],
            ram_address: vec![0u64; cycles],
            extras: vec![0u64; cycles * EXTRA_WORDS],
        }
    }

    pub fn fill_range(&mut self, rows: &[TraceRow], cycles: usize, base: usize, len: usize) {
        self.cycles = len;
        let padding = TraceRow::default();
        let physical = if base >= cycles { &[][..] } else { rows };
        let offset = base;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.is_noop[..len]
                .par_chunks_mut(PACK_CHUNK)
                .zip(self.address[..len].par_chunks_mut(PACK_CHUNK))
                .zip(self.virtual_sequence[..len].par_chunks_mut(PACK_CHUNK))
                .zip(self.ram_address[..len].par_chunks_mut(PACK_CHUNK))
                .zip(self.extras[..len * EXTRA_WORDS].par_chunks_mut(PACK_CHUNK * EXTRA_WORDS))
                .enumerate()
                .for_each(
                    |(chunk, ((((is_noop, address), virtual_sequence), ram_address), extras))| {
                        fill(
                            offset + chunk * PACK_CHUNK,
                            physical,
                            &padding,
                            is_noop,
                            address,
                            virtual_sequence,
                            ram_address,
                            extras,
                        );
                    },
                );
        }
        #[cfg(not(feature = "parallel"))]
        fill(
            offset,
            physical,
            &padding,
            &mut self.is_noop[..len],
            &mut self.address[..len],
            &mut self.virtual_sequence[..len],
            &mut self.ram_address[..len],
            &mut self.extras[..len * EXTRA_WORDS],
        );
    }

    pub fn pack(rows: &[TraceRow], cycles: usize) -> Result<Self, WitnessError> {
        Self::require_domain(rows, cycles)?;
        let mut packed = Self::with_capacity(cycles);
        packed.fill_range(rows, cycles, 0, cycles);
        Ok(packed)
    }

    pub fn require_domain(rows: &[TraceRow], cycles: usize) -> Result<(), WitnessError> {
        if rows.len() > cycles {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "physical trace has {} rows but the cycle domain has {cycles}",
                    rows.len()
                ),
            });
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one destination slice per packed column; bundling them would only move the arity"
)]
fn fill(
    base: usize,
    rows: &[TraceRow],
    padding: &TraceRow,
    is_noop: &mut [u8],
    address: &mut [u64],
    virtual_sequence: &mut [u32],
    ram_address: &mut [u64],
    extras: &mut [u64],
) {
    for offset in 0..is_noop.len() {
        let row = rows.get(base + offset).unwrap_or(padding);
        is_noop[offset] = row_is_noop_byte(row);
        address[offset] = row_address(row);
        virtual_sequence[offset] = row_virtual_sequence(row);
        ram_address[offset] = row_ram_address(row);

        let words = &mut extras[offset * EXTRA_WORDS..(offset + 1) * EXTRA_WORDS];
        let ram_access = row.ram_access();
        words[EXTRA_RS1] = row.rs1_value();
        words[EXTRA_RS2] = row.rs2_value();
        words[EXTRA_RD_POST] = row.rd_write_value();
        words[EXTRA_RAM_READ] = match ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.pre_value,
            RamAccess::NoOp => 0,
        };
        words[EXTRA_RAM_WRITE] = match ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.post_value,
            RamAccess::NoOp => 0,
        };
        let imm = row.imm();
        words[EXTRA_IMM_LO] = imm as u64;
        words[EXTRA_IMM_HI] = (imm >> 64) as u64;
        let register = |slot: Option<u8>| slot.map_or(REGISTER_ABSENT, u64::from);
        words[EXTRA_REGISTERS] = register(row.rs1_index())
            | register(row.rs2_index()) << 8
            | register(row.rd_index()) << 16;
        words[EXTRA_RD_PRE] = row.rd_pre_value();
        words[EXTRA_BYTECODE_PC] = row.pc();
        let circuit_flags = row.circuit_flags();
        words[EXTRA_KIND_BITS] = u64::from(row_kind_index(row))
            | (u64::from(circuit_flags.get(CircuitFlags::IsCompressed)) << BIT_IS_COMPRESSED)
            | (u64::from(circuit_flags.get(CircuitFlags::IsFirstInSequence))
                << BIT_IS_FIRST_IN_SEQUENCE);
    }
}

pub(crate) fn row_kind_index(row: &TraceRow) -> u16 {
    row.instruction_kind()
        .and_then(|kind| CUDA_PROFILE.jolt_dense_index(kind))
        .map_or(KIND_UNMAPPED, |index| index.0)
}

pub(crate) fn row_is_noop_byte(row: &TraceRow) -> u8 {
    u8::from(row.is_noop())
}

pub(crate) fn row_address(row: &TraceRow) -> u64 {
    row.unexpanded_pc()
}

pub(crate) fn row_virtual_sequence(row: &TraceRow) -> u32 {
    row.virtual_sequence_remaining()
        .map_or(NO_SEQUENCE, u32::from)
}

pub(crate) fn row_ram_address(row: &TraceRow) -> u64 {
    match row.ram_access() {
        RamAccess::Read(read) => read.address,
        RamAccess::Write(write) => write.address,
        RamAccess::NoOp => RAM_NO_ACCESS,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    #[test]
    fn the_kernel_source_agrees_on_the_row_stride() {
        let source = include_str!("kernels/atoms.cu");
        let expected = format!("#define EXTRA_WORDS {}", EXTRA_WORDS);
        assert!(
            source.contains(&expected),
            "the CUDA sources must declare `{expected}`; a stride mismatch silently makes every \
             kernel read the wrong row"
        );
    }

    use super::EXTRA_WORDS;
    use super::*;
    use jolt_riscv::{
        JoltInstructionKind, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite,
    };
    use jolt_riscv::{JoltInstructionRow, NormalizedOperands};
    use proptest::prelude::*;

    /// `TraceRow::new` rejects `u16::MAX`, which it reserves as the "no
    /// virtual sequence" sentinel.
    const VSR_MAX: u16 = u16::MAX;

    fn arb_kind() -> impl Strategy<Value = JoltInstructionKind> {
        prop_oneof![
            Just(JoltInstructionKind::ADD),
            Just(JoltInstructionKind::XOR),
            Just(JoltInstructionKind::MUL),
        ]
    }

    fn arb_registers() -> impl Strategy<Value = RegisterState> {
        (
            proptest::option::of(any::<u64>()),
            proptest::option::of(any::<u64>()),
            proptest::option::of((any::<u64>(), any::<u64>())),
        )
            .prop_map(|(rs1, rs2, rd)| RegisterState {
                rs1: rs1.map(|value| RegisterRead { register: 2, value }),
                rs2: rs2.map(|value| RegisterRead { register: 3, value }),
                rd: rd.map(|(pre_value, post_value)| RegisterWrite {
                    register: 1,
                    pre_value,
                    post_value,
                }),
            })
    }

    /// `TraceRow::new` ties the RAM access to the instruction's load/store
    /// flags, so the generator builds each shape with the registers and RAM
    /// access the constructor accepts.
    fn arb_instruction(
        kind: JoltInstructionKind,
    ) -> impl Strategy<Value = (JoltInstructionRow, i128)> {
        (
            any::<u32>(),
            proptest::option::of(0..VSR_MAX),
            any::<bool>(),
            any::<bool>(),
            any::<i64>(),
        )
            .prop_map(
                move |(address, virtual_sequence_remaining, is_first, is_compressed, imm)| {
                    let imm = i128::from(imm);
                    (
                        JoltInstructionRow {
                            instruction_kind: kind,
                            address: address as usize,
                            operands: NormalizedOperands {
                                imm,
                                ..NormalizedOperands::default()
                            },
                            virtual_sequence_remaining,
                            is_first_in_sequence: is_first,
                            is_compressed,
                        },
                        imm,
                    )
                },
            )
    }

    fn arb_alu_row() -> impl Strategy<Value = TraceRow> {
        (arb_kind(), arb_registers())
            .prop_flat_map(|(kind, registers)| {
                arb_instruction(kind).prop_map(move |(instruction, _)| (instruction, registers))
            })
            .prop_map(|(instruction, registers)| {
                let mut instruction = instruction;
                instruction.operands.rs1 = registers.rs1.map(|read| read.register);
                instruction.operands.rs2 = registers.rs2.map(|read| read.register);
                instruction.operands.rd = registers.rd.map(|write| write.register);
                TraceRow::new(instruction, registers, RamAccess::NoOp, 1).unwrap()
            })
    }

    fn arb_load_row() -> impl Strategy<Value = TraceRow> {
        (any::<u64>(), any::<u64>(), any::<u64>())
            .prop_flat_map(|(rs1_value, address, value)| {
                arb_instruction(JoltInstructionKind::LD)
                    .prop_map(move |(instruction, _)| (instruction, rs1_value, address, value))
            })
            .prop_map(|(mut instruction, rs1_value, address, value)| {
                instruction.operands.rs1 = Some(2);
                instruction.operands.rd = Some(1);
                TraceRow::new(
                    instruction,
                    RegisterState {
                        rs1: Some(RegisterRead {
                            register: 2,
                            value: rs1_value,
                        }),
                        rd: Some(RegisterWrite {
                            register: 1,
                            pre_value: 0,
                            post_value: value,
                        }),
                        ..Default::default()
                    },
                    RamAccess::Read(RamRead { address, value }),
                    1,
                )
                .unwrap()
            })
    }

    fn arb_store_row() -> impl Strategy<Value = TraceRow> {
        (any::<u64>(), any::<u64>(), any::<u64>())
            .prop_flat_map(|(rs1_value, address, post_value)| {
                arb_instruction(JoltInstructionKind::SD)
                    .prop_map(move |(instruction, _)| (instruction, rs1_value, address, post_value))
            })
            .prop_map(|(mut instruction, rs1_value, address, post_value)| {
                instruction.operands.rs1 = Some(2);
                instruction.operands.rs2 = Some(3);
                TraceRow::new(
                    instruction,
                    RegisterState {
                        rs1: Some(RegisterRead {
                            register: 2,
                            value: rs1_value,
                        }),
                        rs2: Some(RegisterRead {
                            register: 3,
                            value: post_value,
                        }),
                        ..Default::default()
                    },
                    RamAccess::Write(RamWrite {
                        address,
                        pre_value: 0,
                        post_value,
                    }),
                    1,
                )
                .unwrap()
            })
    }

    fn arb_row() -> impl Strategy<Value = TraceRow> {
        prop_oneof![arb_alu_row(), arb_load_row(), arb_store_row()]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn packed_trace_matches_trace_rows(
            rows in prop::collection::vec(arb_row(), 0..24usize),
            extra in 0..8usize,
        ) {
            let cycles = rows.len() + extra;
            let padding = TraceRow::default();
            let got = PackedTrace::pack(&rows, cycles).unwrap();

            prop_assert_eq!(got.cycles, cycles);
            for index in 0..cycles {
                let expected = rows.get(index).unwrap_or(&padding);
                prop_assert_eq!(got.is_noop[index], row_is_noop_byte(expected), "is_noop at {}", index);
                prop_assert_eq!(got.address[index], row_address(expected), "address at {}", index);
                prop_assert_eq!(
                    got.virtual_sequence[index],
                    row_virtual_sequence(expected),
                    "virtual sequence at {}",
                    index
                );
                prop_assert_eq!(
                    got.ram_address[index],
                    row_ram_address(expected),
                    "ram address at {}",
                    index
                );

                let words = &got.extras[index * EXTRA_WORDS..(index + 1) * EXTRA_WORDS];
                prop_assert_eq!(
                    words[EXTRA_RS1],
                    expected.rs1_read().map_or(0, |read| read.value),
                    "rs1 at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_RS2],
                    expected.rs2_read().map_or(0, |read| read.value),
                    "rs2 at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_RD_POST],
                    expected.rd_write().map_or(0, |write| write.post_value),
                    "rd post value at {}", index
                );
                let (read_value, write_value) = match expected.ram_access() {
                    RamAccess::Read(read) => (read.value, read.value),
                    RamAccess::Write(write) => (write.pre_value, write.post_value),
                    RamAccess::NoOp => (0, 0),
                };
                prop_assert_eq!(words[EXTRA_RAM_READ], read_value, "ram read at {}", index);
                prop_assert_eq!(words[EXTRA_RAM_WRITE], write_value, "ram write at {}", index);
                let imm = expected.imm();
                prop_assert_eq!(words[EXTRA_IMM_LO], imm as u64, "imm low at {}", index);
                prop_assert_eq!(words[EXTRA_IMM_HI], (imm >> 64) as u64, "imm high at {}", index);
                prop_assert_eq!(
                    (words[EXTRA_KIND_BITS] & 0xFFFF) as u16,
                    row_kind_index(expected),
                    "kind index at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_KIND_BITS] >> BIT_IS_COMPRESSED & 1 == 1,
                    expected.circuit_flags().get(CircuitFlags::IsCompressed),
                    "is_compressed at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_KIND_BITS] >> BIT_IS_FIRST_IN_SEQUENCE & 1 == 1,
                    expected.circuit_flags().get(CircuitFlags::IsFirstInSequence),
                    "is_first_in_sequence at {}", index
                );
                let register = |slot: Option<u8>| slot.map_or(REGISTER_ABSENT, u64::from);
                prop_assert_eq!(
                    words[EXTRA_REGISTERS] & 0xFF,
                    register(expected.rs1_read().map(|read| read.register)),
                    "rs1 register at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_REGISTERS] >> 8 & 0xFF,
                    register(expected.rs2_read().map(|read| read.register)),
                    "rs2 register at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_REGISTERS] >> 16 & 0xFF,
                    register(expected.rd_write().map(|write| write.register)),
                    "rd register at {}", index
                );
                prop_assert_eq!(
                    words[EXTRA_RD_PRE],
                    expected.rd_write().map_or(0, |write| write.pre_value),
                    "rd pre value at {}", index
                );
            }
        }
    }

    #[test]
    fn pack_rejects_a_trace_longer_than_the_domain() {
        let rows = vec![TraceRow::default(); 4];
        assert!(PackedTrace::pack(&rows, 2).is_err());
    }
}
