use common::constants::RAM_START_ADDRESS;
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite,
    RegisterRead, RegisterState, RegisterWrite, SourceTraceError, SourceTraceRow, TraceError,
    TraceInputs, TraceOutput,
};
use jolt_program::image::DecodeMode;
use jolt_riscv::{NormalizedOperands, SourceInstructionKind as Kind, RV64I};

use crate::{emulator::Emulator, instruction::Instruction, AdviceTape};

/// Traces RV64I source instructions without virtual-sequence expansion.
///
/// Decoding always uses RV64I and defaults to strict decoding. ECALL and
/// EBREAK are rejected when reached;
/// fetches must name decoded instructions, accesses must be aligned, control
/// cells are write-once and unreadable, and stores cannot overlap instructions.
/// Rows capture old operands and aligned RAM doublewords, including x0 loads.
/// The first PC stall emits one final row; termination stores do not stop tracing.
/// Text spans above 256 MiB are rejected before emulator construction. The
/// memory configuration must cover the loaded image, as for `TracerBackend`.
#[derive(Default, Debug, Clone)]
pub struct SourceTracerBackend {
    row_capacity: usize,
    decode_mode: DecodeMode,
}

impl SourceTracerBackend {
    /// Reserves space for this many rows before execution; otherwise capacity
    /// grows amortised. The default hint is zero.
    pub fn with_row_capacity(rows: usize) -> Self {
        Self {
            row_capacity: rows,
            decode_mode: DecodeMode::Strict,
        }
    }

    /// Selects how executable sections containing data words are decoded.
    ///
    /// Rows index the instruction list returned by `decode_elf_with_mode` with
    /// RV64I and this same mode. A fetch of an omitted word returns
    /// `PcOutsideProgram`; loads still read its original image bytes.
    pub fn with_decode_mode(mut self, mode: DecodeMode) -> Self {
        self.decode_mode = mode;
        self
    }
}

impl ExecutionBackend<SourceTraceRow> for SourceTracerBackend {
    type Trace = OwnedTrace<SourceTraceRow>;

    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError> {
        let mut execution = SourceExecution::new(program, inputs, self.decode_mode)?;
        let mut rows = Vec::with_capacity(self.row_capacity);
        while execution.step(&mut rows)? {}
        let (advice_tape, memory, device) = crate::finish_emulator(execution.emulator);
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            device,
            Some(MemoryImage {
                bytes: memory.materialized_nonzero_bytes(),
            }),
            Some(advice_tape.into_bytes()),
        ))
    }
}

struct InstructionTable {
    base: u64,
    slots: Vec<u32>,
}

impl InstructionTable {
    const EMPTY: u32 = u32::MAX;
    const MAX_SPAN: u64 = 1 << 28;

    fn new(instructions: &[Instruction]) -> Result<Self, SourceTraceError> {
        let base = instructions
            .iter()
            .map(Instruction::address)
            .min()
            .unwrap_or(0);
        let highest = instructions.iter().map(Instruction::address).max();
        let span = match highest {
            Some(highest) => highest
                .checked_add(4)
                .and_then(|end| end.checked_sub(base))
                .unwrap_or(u64::MAX),
            None => 0,
        };
        if span > Self::MAX_SPAN {
            return Err(SourceTraceError::ProgramTextTooLarge { span });
        }
        let mut slots = vec![Self::EMPTY; (span / 4) as usize];
        for (index, instruction) in instructions.iter().enumerate() {
            let slot = u32::try_from((instruction.address() - base) / 4)
                .map_err(|_| SourceTraceError::ProgramTextTooLarge { span })?;
            let index =
                u32::try_from(index).map_err(|_| SourceTraceError::ProgramTextTooLarge { span })?;
            slots[slot as usize] = index;
        }
        Ok(Self { base, slots })
    }

    #[inline]
    fn lookup(&self, pc: u64) -> Option<u32> {
        let offset = pc.checked_sub(self.base)?;
        if offset & 3 != 0 {
            return None;
        }
        self.slots
            .get((offset / 4) as usize)
            .copied()
            .filter(|index| *index != Self::EMPTY)
    }

    fn overlaps_store(&self, address: u64, width: u8) -> bool {
        let end = self.base + self.slots.len() as u64 * 4;
        if address >= end || address.saturating_add(u64::from(width)) <= self.base {
            return false;
        }
        let first = address.max(self.base) & !3;
        let last = address.saturating_add(u64::from(width) - 1).min(end - 1) & !3;
        self.lookup(first).is_some() || (last != first && self.lookup(last).is_some())
    }
}

#[derive(Clone, Copy)]
enum MemoryAccess {
    Load { width: u8 },
    Store { width: u8 },
}

struct DecodedInstruction {
    instruction: Instruction,
    kind: Kind,
    operands: NormalizedOperands,
    access: Option<MemoryAccess>,
}

impl DecodedInstruction {
    fn new(instruction: Instruction) -> Self {
        let source = instruction.source_instruction();
        let kind = source.kind();
        let access = match kind {
            Kind::LB | Kind::LBU => Some(MemoryAccess::Load { width: 1 }),
            Kind::LH | Kind::LHU => Some(MemoryAccess::Load { width: 2 }),
            Kind::LW | Kind::LWU => Some(MemoryAccess::Load { width: 4 }),
            Kind::LD => Some(MemoryAccess::Load { width: 8 }),
            Kind::SB => Some(MemoryAccess::Store { width: 1 }),
            Kind::SH => Some(MemoryAccess::Store { width: 2 }),
            Kind::SW => Some(MemoryAccess::Store { width: 4 }),
            Kind::SD => Some(MemoryAccess::Store { width: 8 }),
            _ => None,
        };
        Self {
            instruction,
            kind,
            operands: source.row().operands,
            access,
        }
    }
}

struct SourceExecution {
    instructions: Vec<DecodedInstruction>,
    table: InstructionTable,
    emulator: Emulator,
    control_written: [bool; 2],
}

impl SourceExecution {
    fn new(
        program: &JoltProgram,
        inputs: TraceInputs,
        mode: DecodeMode,
    ) -> Result<Self, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let (instructions, memory_init, _, _) =
            crate::decode_with_mode(program.elf_bytes(), RV64I, mode)?;
        let table = InstructionTable::new(&instructions)?;
        let mut emulator = crate::create_emulator(
            program.elf_bytes(),
            None,
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        let mmu = &mut emulator.get_mut_cpu().mmu;
        // Original positions break address ties so later sections win. Sorting
        // indices avoids allocating over gaps in the image's address space.
        let mut image_order: Vec<usize> = (0..memory_init.len()).collect();
        image_order.sort_unstable_by_key(|&index| (memory_init[index].0, index));
        for (position, &index) in image_order.iter().enumerate() {
            let (address, expected) = memory_init[index];
            if image_order
                .get(position + 1)
                .is_some_and(|&next| memory_init[next].0 == address)
            {
                continue;
            }
            let actual = if mmu.memory.validate_address(address) {
                mmu.memory.memory.get_byte(address - RAM_START_ADDRESS)
            } else {
                0
            };
            if actual != expected {
                return Err(SourceTraceError::ImageMismatch { address }.into());
            }
        }
        mmu.set_access_recording(false);
        if let Some(device) = mmu.jolt_device.as_mut() {
            device
                .outputs
                .reserve(device.memory_layout.max_output_size as usize);
        }
        Ok(Self {
            instructions: instructions
                .into_iter()
                .map(DecodedInstruction::new)
                .collect(),
            table,
            emulator,
            control_written: [false; 2],
        })
    }

    #[inline]
    fn step(&mut self, rows: &mut Vec<SourceTraceRow>) -> Result<bool, SourceTraceError> {
        let cpu = self.emulator.get_mut_cpu();
        let pc = cpu.read_pc();
        let index = self
            .table
            .lookup(pc)
            .ok_or(SourceTraceError::PcOutsideProgram { pc })?;
        let instruction = &self.instructions[index as usize];
        let kind = instruction.kind;
        if matches!(kind, Kind::ECALL | Kind::EBREAK) {
            return Err(SourceTraceError::UnsupportedInstruction { pc, kind });
        }
        let operands = &instruction.operands;
        let mut registers = RegisterState {
            rs1: operands.rs1.map(|register| RegisterRead {
                register,
                value: cpu.read_register(register) as u64,
            }),
            rs2: operands.rs2.map(|register| RegisterRead {
                register,
                value: cpu.read_register(register) as u64,
            }),
            rd: operands.rd.map(|register| RegisterWrite {
                register,
                pre_value: cpu.read_register(register) as u64,
                post_value: 0,
            }),
        };
        let ram_access = if let Some(access) = instruction.access {
            let (width, store) = match access {
                MemoryAccess::Load { width } => (width, false),
                MemoryAccess::Store { width } => (width, true),
            };
            let ea = registers
                .rs1
                .map_or(0, |read| read.value)
                .wrapping_add(operands.imm as u64);
            if !ea.is_multiple_of(u64::from(width)) {
                return Err(SourceTraceError::MisalignedAccess {
                    pc,
                    address: ea,
                    width,
                });
            }
            let address = ea & !7;
            let control = cpu.mmu.jolt_device.as_ref().and_then(|device| {
                if address == device.memory_layout.panic {
                    Some(0)
                } else if address == device.memory_layout.termination {
                    Some(1)
                } else {
                    None
                }
            });
            if let Some(cell) = control {
                if !store || self.control_written[cell] {
                    return Err(SourceTraceError::DeviceRegisterAccess { pc, address: ea });
                }
            }
            if store && self.table.overlaps_store(ea, width) {
                return Err(SourceTraceError::StoreToProgramText { pc, address: ea });
            }
            let pre_value = if control.is_some() {
                0
            } else {
                cpu.mmu.load_doubleword_raw(address)
            };
            if store {
                let shift = (ea & 7) * 8;
                let mask = (u64::MAX >> ((8 - width) * 8)) << shift;
                let value = registers.rs2.map_or(0, |read| read.value);
                let post_value = (pre_value & !mask) | ((value << shift) & mask);
                if let Some(cell) = control {
                    self.control_written[cell] = true;
                }
                RamAccess::Write(RamWrite {
                    address,
                    pre_value,
                    post_value,
                })
            } else {
                RamAccess::Read(RamRead {
                    address,
                    value: pre_value,
                })
            }
        } else {
            RamAccess::NoOp
        };
        cpu.pc = pc.wrapping_add(4);
        instruction.instruction.execute_direct(cpu);
        cpu.x[0] = 0;
        if let Some(rd) = registers.rd.as_mut() {
            rd.post_value = cpu.read_register(rd.register) as u64;
        }
        rows.push(SourceTraceRow::new(
            index, pc, cpu.pc, registers, ram_access,
        ));
        cpu.trace_len += 1;
        Ok(cpu.pc != pc)
    }
}

#[cfg(test)]
mod tests;
