use std::collections::BTreeMap;

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_program::{
    execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead,
        RegisterState, RegisterWrite, SourceTraceError, SourceTraceRow, TraceError, TraceInputs,
        TraceOutput,
    },
    image::{decode_elf, decode_elf_with_mode, DecodeMode},
    ProgramError,
};
use jolt_riscv::{SourceInstructionKind as Kind, RV64I};

use super::{InstructionTable, SourceExecution, SourceTracerBackend};
use crate::{
    create_emulator,
    emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder},
    instruction::{Cycle, Instruction},
};

const ENTRY: u64 = 0x8000_0000;
const HALT: u32 = 0x0000_006f;

// Encodings and expected values follow Volume I, RV64I: I/R/S/B/U/J formats,
// sign extension, the low six (five for W) shift bits, and little-endian memory.
fn i(opcode: u32, funct3: u32, rd: u8, rs1: u8, immediate: i32) -> u32 {
    ((immediate as u32 & 0xfff) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | (u32::from(rd) << 7)
        | opcode
}

fn r(opcode: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25)
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | (u32::from(rd) << 7)
        | opcode
}

fn s(funct3: u32, rs1: u8, rs2: u8, immediate: i32) -> u32 {
    let immediate = immediate as u32 & 0xfff;
    ((immediate >> 5) << 25)
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | ((immediate & 31) << 7)
        | 0x23
}

fn b(funct3: u32, rs1: u8, rs2: u8, immediate: i32) -> u32 {
    let immediate = immediate as u32;
    ((immediate & 0x1000) << 19)
        | ((immediate & 0x7e0) << 20)
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | ((immediate & 0x1e) << 7)
        | ((immediate & 0x800) >> 4)
        | 0x63
}

fn j(rd: u8, immediate: i32) -> u32 {
    let immediate = immediate as u32;
    ((immediate & 0x100000) << 11)
        | ((immediate & 0x7fe) << 20)
        | ((immediate & 0x800) << 9)
        | (immediate & 0xff000)
        | (u32::from(rd) << 7)
        | 0x6f
}

fn registers(
    rs1: Option<(u8, u64)>,
    rs2: Option<(u8, u64)>,
    rd: Option<(u8, u64, u64)>,
) -> RegisterState {
    RegisterState {
        rs1: rs1.map(|(register, value)| RegisterRead { register, value }),
        rs2: rs2.map(|(register, value)| RegisterRead { register, value }),
        rd: rd.map(|(register, pre_value, post_value)| RegisterWrite {
            register,
            pre_value,
            post_value,
        }),
    }
}

fn read(address: u64, value: u64) -> RamAccess {
    RamAccess::Read(RamRead { address, value })
}

fn write(address: u64, pre_value: u64, post_value: u64) -> RamAccess {
    RamAccess::Write(RamWrite {
        address,
        pre_value,
        post_value,
    })
}

struct TextSections {
    first_address: u64,
    split: usize,
    second_address: u64,
}

struct Fixture {
    words: Vec<u32>,
    expected: Vec<SourceTraceRow>,
    inputs: TraceInputs,
    text_sections: Option<TextSections>,
    decode_mode: DecodeMode,
}

impl Fixture {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            expected: Vec::new(),
            text_sections: None,
            decode_mode: DecodeMode::Strict,
            inputs: TraceInputs::new(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                MemoryConfig {
                    program_size: Some(0x1000),
                    heap_size: 0x2000,
                    stack_size: 0x1000,
                    max_input_size: 8,
                    max_output_size: 0x2000,
                    max_trusted_advice_size: 8,
                    max_untrusted_advice_size: 8,
                },
            ),
        }
    }

    fn layout(&self) -> MemoryLayout {
        MemoryLayout::new(&self.inputs.memory_config)
    }

    fn pc(&self) -> u64 {
        match &self.text_sections {
            Some(sections) if self.words.len() >= sections.split => {
                sections.second_address + 4 * (self.words.len() - sections.split) as u64
            }
            Some(sections) => sections.first_address + 4 * self.words.len() as u64,
            None => ENTRY + 4 * self.words.len() as u64,
        }
    }

    fn emit(&mut self, word: u32, registers: RegisterState, ram: RamAccess) {
        self.emit_to(word, registers, ram, self.pc() + 4);
    }

    fn emit_to(&mut self, word: u32, registers: RegisterState, ram: RamAccess, next_pc: u64) {
        self.expected.push(SourceTraceRow::new(
            self.words.len() as u32,
            self.pc(),
            next_pc,
            registers,
            ram,
        ));
        self.words.push(word);
    }

    fn addi(&mut self, rd: u8, rs1: u8, pre: u64, old: u64, imm: i32, post: u64) {
        self.emit(
            i(0x13, 0, rd, rs1, imm),
            registers(Some((rs1, old)), None, Some((rd, pre, post))),
            RamAccess::NoOp,
        );
    }

    fn address(&mut self, rd: u8, address: u64) {
        let upper = (address + 0x800) & !0xfff;
        // The fixtures use addresses around 2^31. LUI's negative value is
        // 0xffff_ffff_0000_0000 + upper; SLLI 32 and SRLI 32 remove that prefix.
        let signed_upper = if upper < 0x8000_0000 {
            upper
        } else {
            0xffff_ffff_0000_0000 + upper
        };
        self.emit(
            upper as u32 | (u32::from(rd) << 7) | 0x37,
            registers(None, None, Some((rd, 0, signed_upper))),
            RamAccess::NoOp,
        );
        self.emit(
            i(0x13, 1, rd, rd, 32),
            registers(
                Some((rd, signed_upper)),
                None,
                Some((rd, signed_upper, upper << 32)),
            ),
            RamAccess::NoOp,
        );
        self.emit(
            i(0x13, 5, rd, rd, 32),
            registers(
                Some((rd, upper << 32)),
                None,
                Some((rd, upper << 32, upper)),
            ),
            RamAccess::NoOp,
        );
        self.addi(
            rd,
            rd,
            upper,
            upper,
            (address as i64 - upper as i64) as i32,
            address,
        );
    }

    fn halt(&mut self) {
        self.emit_to(
            HALT,
            registers(None, None, Some((0, 0, 0))),
            RamAccess::NoOp,
            self.pc(),
        );
    }

    fn program(&self) -> JoltProgram {
        let mut elf = build_elf64(&self.words, &[], StrtabOrder::GnuLd);
        if let Some(sections) = &self.text_sections {
            // System V gABI ELF64: e_entry/e_shoff/e_shnum at 24/40/60;
            // sh_addr/sh_offset/sh_size at 16/24/32 in each 64-byte header.
            let shoff = u64::from_le_bytes(elf[40..48].try_into().unwrap()) as usize;
            let text_header = shoff + 64;
            let mut second_header = elf[text_header..text_header + 64].to_vec();
            let text_offset = u64::from_le_bytes(second_header[24..32].try_into().unwrap());
            let first_size = 4 * sections.split as u64;
            let second_size = 4 * (self.words.len() - sections.split) as u64;
            elf[24..32].copy_from_slice(&sections.first_address.to_le_bytes());
            elf[text_header + 16..text_header + 24]
                .copy_from_slice(&sections.first_address.to_le_bytes());
            elf[text_header + 32..text_header + 40].copy_from_slice(&first_size.to_le_bytes());
            second_header[16..24].copy_from_slice(&sections.second_address.to_le_bytes());
            second_header[24..32].copy_from_slice(&(text_offset + first_size).to_le_bytes());
            second_header[32..40].copy_from_slice(&second_size.to_le_bytes());
            let count = u16::from_le_bytes(elf[60..62].try_into().unwrap());
            elf[60..62].copy_from_slice(&(count + 1).to_le_bytes());
            elf.extend_from_slice(&second_header);
        }
        JoltProgram::from_elf_bytes_with_profile(elf, RV64I)
    }

    fn complete(self) -> TraceOutput<OwnedTrace<SourceTraceRow>> {
        self.complete_with_reference_panic(false)
    }

    fn complete_with_reference_panic(
        self,
        expected_reference_panic: bool,
    ) -> TraceOutput<OwnedTrace<SourceTraceRow>> {
        let program = self.program();
        let output = SourceTracerBackend::with_row_capacity(self.expected.len())
            .with_decode_mode(self.decode_mode)
            .trace(&program, self.inputs.clone())
            .unwrap();
        assert_eq!(output.trace.rows(), self.expected);
        check_replay_with_mode(&program, &self.inputs, &output, self.decode_mode);
        if !output.device.panic {
            check_lockstep_with_reference_panic(
                &program,
                &self.inputs,
                output.trace.rows(),
                expected_reference_panic,
                self.decode_mode,
            );
        }
        output
    }

    fn error(self, expected: SourceTraceError) {
        let program = self.program();
        let error = SourceTracerBackend::default()
            .with_decode_mode(self.decode_mode)
            .trace(&program, self.inputs.clone())
            .unwrap_err();
        assert!(matches!(error, TraceError::SourceTrace(actual) if actual == expected));
        let mut execution = SourceExecution::new(&program, self.inputs, self.decode_mode).unwrap();
        let mut rows = Vec::with_capacity(self.words.len());
        for _ in 0..=self.words.len() {
            let cpu = execution.emulator.get_cpu();
            let registers = cpu.x;
            let pc = cpu.read_pc();
            let memory = cpu.mmu.memory.memory.materialized_nonzero_bytes();
            let device = cpu.mmu.jolt_device.as_ref().unwrap().clone();
            let trace_len = cpu.trace_len;
            let row_count = rows.len();
            match execution.step(&mut rows) {
                Ok(more) => assert!(more, "error fixture terminated before its error"),
                Err(error) => {
                    assert_eq!(error, expected);
                    let cpu = execution.emulator.get_cpu();
                    assert_eq!(cpu.x, registers);
                    assert_eq!(cpu.read_pc(), pc);
                    assert_eq!(cpu.mmu.memory.memory.materialized_nonzero_bytes(), memory);
                    assert_eq!(cpu.mmu.jolt_device.as_ref().unwrap(), &device);
                    assert_eq!(cpu.trace_len, trace_len);
                    assert_eq!(rows.len(), row_count);
                    return;
                }
            }
        }
        panic!("error fixture did not reach its error");
    }
}

fn memory_word(memory: &BTreeMap<u64, u8>, address: u64) -> u64 {
    u64::from_le_bytes(std::array::from_fn(|byte| {
        memory.get(&(address + byte as u64)).copied().unwrap_or(0)
    }))
}

fn check_replay(
    program: &JoltProgram,
    inputs: &TraceInputs,
    output: &TraceOutput<OwnedTrace<SourceTraceRow>>,
) {
    check_replay_with_mode(program, inputs, output, DecodeMode::Strict);
}

fn check_replay_with_mode(
    program: &JoltProgram,
    inputs: &TraceInputs,
    output: &TraceOutput<OwnedTrace<SourceTraceRow>>,
    mode: DecodeMode,
) {
    let image = decode_elf_with_mode(program.elf_bytes(), RV64I, mode).unwrap();
    let mut memory = BTreeMap::new();
    for (address, byte) in image.memory_init {
        memory.insert(address, byte);
    }
    let layout = MemoryLayout::new(&inputs.memory_config);
    for (start, bytes) in [
        (layout.input_start, &inputs.inputs),
        (layout.trusted_advice_start, &inputs.trusted_advice),
        (layout.untrusted_advice_start, &inputs.untrusted_advice),
    ] {
        memory.extend(
            bytes
                .iter()
                .enumerate()
                .map(|(offset, byte)| (start + offset as u64, *byte)),
        );
    }
    let mut x = [0u64; 32];
    let mut pc = image.entry_address;
    for row in output.trace.rows() {
        assert_eq!(row.pc(), pc);
        assert_eq!(
            image.instructions[row.instruction_index() as usize]
                .row()
                .address,
            pc as usize
        );
        let state = row.registers();
        for read in [state.rs1, state.rs2].into_iter().flatten() {
            assert_eq!(read.value, x[usize::from(read.register)], "read at {pc:#x}");
        }
        if let Some(rd) = state.rd {
            assert_eq!(rd.pre_value, x[usize::from(rd.register)], "rd at {pc:#x}");
            x[usize::from(rd.register)] = rd.post_value;
        }
        match row.ram_access() {
            RamAccess::Read(read) => assert_eq!(read.value, memory_word(&memory, read.address)),
            RamAccess::Write(write) => {
                assert_eq!(write.pre_value, memory_word(&memory, write.address));
                memory.extend(
                    write
                        .post_value
                        .to_le_bytes()
                        .into_iter()
                        .enumerate()
                        .map(|(offset, byte)| (write.address + offset as u64, byte)),
                );
            }
            RamAccess::NoOp => assert_eq!(
                (row.ram_address(), row.ram_pre_value(), row.ram_post_value()),
                (0, 0, 0)
            ),
        }
        assert_eq!(x[0], 0);
        pc = row.next_pc();
    }
    let actual: Vec<_> = memory
        .iter()
        .filter(|(address, byte)| **address >= ENTRY && **byte != 0)
        .map(|(address, byte)| (*address - ENTRY, *byte))
        .collect();
    assert_eq!(actual, output.final_memory.as_ref().unwrap().bytes);
    for address in layout.output_start..layout.output_end {
        let offset = (address - layout.output_start) as usize;
        assert_eq!(
            memory.get(&address).copied().unwrap_or(0),
            output.device.outputs.get(offset).copied().unwrap_or(0)
        );
    }
}

fn check_lockstep(program: &JoltProgram, inputs: &TraceInputs, expected: &[SourceTraceRow]) {
    check_lockstep_with_reference_panic(program, inputs, expected, false, DecodeMode::Strict);
}

fn check_lockstep_with_reference_panic(
    program: &JoltProgram,
    inputs: &TraceInputs,
    expected: &[SourceTraceRow],
    expected_reference_panic: bool,
    mode: DecodeMode,
) {
    let mut source = SourceExecution::new(program, inputs.clone(), mode).unwrap();
    let mut reference = create_emulator(
        program.elf_bytes(),
        None,
        &inputs.inputs,
        &inputs.untrusted_advice,
        &inputs.trusted_advice,
        &inputs.memory_config,
        None,
    );
    let mut source_rows = Vec::with_capacity(expected.len());
    let mut cycles: Vec<Cycle> = Vec::new();
    for row in expected {
        cycles.clear();
        reference.tick(Some(&mut cycles));
        let more = source.step(&mut source_rows).unwrap();
        assert_eq!(source_rows.last(), Some(row));
        assert!(!cycles.is_empty());
        let first = cycles.first().unwrap().instruction();
        assert_eq!(first.address(), row.pc());
        match first.virtual_sequence_remaining() {
            None => assert_eq!(cycles.len(), 1),
            Some(remaining) => {
                assert!(
                    first
                        .try_jolt_instruction_row()
                        .unwrap()
                        .is_first_in_sequence
                );
                assert_eq!(cycles.len(), usize::from(remaining) + 1);
                for (index, cycle) in cycles.iter().enumerate() {
                    let instruction = cycle.instruction();
                    assert_eq!(instruction.address(), row.pc());
                    assert_eq!(
                        instruction.virtual_sequence_remaining(),
                        Some(remaining - index as u16)
                    );
                    assert_eq!(
                        instruction
                            .try_jolt_instruction_row()
                            .unwrap()
                            .is_first_in_sequence,
                        index == 0
                    );
                }
            }
        }
        for register in 0..32 {
            assert_eq!(
                source.emulator.get_cpu().x[register],
                reference.get_cpu().x[register],
                "x{register} after {:#x}",
                row.pc()
            );
        }
        assert_eq!(
            source.emulator.get_cpu().read_pc(),
            reference.get_cpu().read_pc()
        );
        assert_eq!(
            source
                .emulator
                .get_cpu()
                .mmu
                .memory
                .memory
                .materialized_nonzero_bytes(),
            reference
                .get_cpu()
                .mmu
                .memory
                .memory
                .materialized_nonzero_bytes(),
            "RAM after {:#x}",
            row.pc(),
        );
        assert_eq!(more, row.pc() != row.next_pc());
    }
    assert_eq!(
        source
            .emulator
            .get_cpu()
            .mmu
            .memory
            .memory
            .materialized_nonzero_bytes(),
        reference
            .get_cpu()
            .mmu
            .memory
            .memory
            .materialized_nonzero_bytes()
    );
    let source_device = source.emulator.get_cpu().mmu.jolt_device.as_ref().unwrap();
    let reference_device = reference.get_cpu().mmu.jolt_device.as_ref().unwrap();
    assert_eq!(source_device.outputs, reference_device.outputs);
    assert!(!source_device.panic);
    assert_eq!(reference_device.panic, expected_reference_panic);
    if !expected_reference_panic {
        assert_eq!(source_device.panic, reference_device.panic);
    }
}

#[test]
fn smoke() {
    let mut f = Fixture::new();
    f.addi(1, 0, 0, 0, 1, 1);
    f.addi(2, 1, 0, 1, 2, 3);
    f.halt();
    f.complete();
}

#[test]
fn row_type() {
    const {
        assert!(size_of::<SourceTraceRow>() == 80);
        assert!(align_of::<SourceTraceRow>() == 8);
    }
    let states = [
        RegisterState::default(),
        registers(Some((0, 0)), None, None),
        registers(None, None, Some((0, 0, 0))),
        registers(Some((31, 101)), Some((4, 103)), Some((5, 107, 109))),
        registers(Some((255, 163)), Some((63, 167)), Some((0, 173, 179))),
    ];
    let accesses = [
        (RamAccess::NoOp, (0, 0, 0)),
        (read(0, 0), (0, 0, 0)),
        (write(0, 0, 0), (0, 0, 0)),
        (read(113, 127), (113, 127, 127)),
        (write(131, 137, 139), (131, 137, 139)),
    ];
    let mut rows = Vec::new();
    for registers in states {
        for (access, words) in accesses {
            let row = SourceTraceRow::new(149, 151, 157, registers, access);
            assert_eq!(
                (row.instruction_index(), row.pc(), row.next_pc()),
                (149, 151, 157)
            );
            assert_eq!(row.registers(), registers);
            assert_eq!(row.ram_access(), access);
            assert_eq!(
                (row.ram_address(), row.ram_pre_value(), row.ram_post_value()),
                words
            );
            assert_eq!(row.rs1_value(), registers.rs1.map_or(0, |r| r.value));
            assert_eq!(row.rs2_value(), registers.rs2.map_or(0, |r| r.value));
            assert_eq!(row.rd_pre_value(), registers.rd.map_or(0, |r| r.pre_value));
            assert_eq!(
                row.rd_post_value(),
                registers.rd.map_or(0, |r| r.post_value)
            );
            assert!(!rows.contains(&row));
            rows.push(row);
        }
    }
}

#[test]
fn alu() {
    // x5 = -1 and x6 = 1: signed comparisons differ from unsigned, and
    // ADDW/SUBW truncate before sign extension (RV64I sections 4.2 and 4.3).
    let register_cases = [
        (0x33, 0, 0, 0),
        (0x33, 0, 0x20, 0xffff_ffff_ffff_fffe),
        (0x33, 1, 0, 0xffff_ffff_ffff_fffe),
        (0x33, 2, 0, 1),
        (0x33, 3, 0, 0),
        (0x33, 4, 0, 0xffff_ffff_ffff_fffe),
        (0x33, 5, 0, 0x7fff_ffff_ffff_ffff),
        (0x33, 5, 0x20, u64::MAX),
        (0x33, 6, 0, u64::MAX),
        (0x33, 7, 0, 1),
        (0x3b, 0, 0, 0),
        (0x3b, 0, 0x20, 0xffff_ffff_ffff_fffe),
        (0x3b, 1, 0, 0xffff_ffff_ffff_fffe),
        (0x3b, 5, 0, 0x7fff_ffff),
        (0x3b, 5, 0x20, u64::MAX),
    ];
    for (opcode, funct3, funct7, result) in register_cases {
        for (rd, pre) in [(7, 0), (5, u64::MAX), (6, 1), (0, 0)] {
            let mut f = Fixture::new();
            f.addi(5, 0, 0, 0, -1, u64::MAX);
            f.addi(6, 0, 0, 0, 1, 1);
            f.emit(
                r(opcode, funct3, funct7, rd, 5, 6),
                registers(
                    Some((5, u64::MAX)),
                    Some((6, 1)),
                    Some((rd, pre, if rd == 0 { 0 } else { result })),
                ),
                RamAccess::NoOp,
            );
            f.halt();
            f.complete();
        }
    }
    let immediate_cases = [
        (0x13, 0, -2, 0xffff_ffff_ffff_fffd),
        (0x13, 2, -2, 0),
        (0x13, 3, -2, 0),
        (0x13, 4, -2, 1),
        (0x13, 6, -2, u64::MAX),
        (0x13, 7, -2, 0xffff_ffff_ffff_fffe),
        (0x13, 1, 31, 0xffff_ffff_8000_0000),
        (0x13, 1, 32, 0xffff_ffff_0000_0000),
        (0x13, 1, 63, 0x8000_0000_0000_0000),
        (0x13, 5, 31, 0x1_ffff_ffff),
        (0x13, 5, 32, 0xffff_ffff),
        (0x13, 5, 63, 1),
        (0x13, 5, 0x41f, u64::MAX),
        (0x13, 5, 0x420, u64::MAX),
        (0x13, 5, 0x43f, u64::MAX),
        (0x1b, 0, -2, 0xffff_ffff_ffff_fffd),
        (0x1b, 1, 31, 0xffff_ffff_8000_0000),
        (0x1b, 5, 31, 1),
        (0x1b, 5, 0x41f, u64::MAX),
    ];
    for (opcode, funct3, immediate, result) in immediate_cases {
        for (rd, pre) in [(0, 0), (5, u64::MAX), (7, 0)] {
            let mut f = Fixture::new();
            f.addi(5, 0, 0, 0, -1, u64::MAX);
            f.emit(
                i(opcode, funct3, rd, 5, immediate),
                registers(
                    Some((5, u64::MAX)),
                    None,
                    Some((rd, pre, if rd == 0 { 0 } else { result })),
                ),
                RamAccess::NoOp,
            );
            f.halt();
            f.complete();
        }
    }
    for (shift, left, right, left_w, right_w) in [
        (
            31,
            0xffff_ffff_8000_0000,
            0x1_ffff_ffff,
            0xffff_ffff_8000_0000,
            1,
        ),
        (32, 0xffff_ffff_0000_0000, 0xffff_ffff, u64::MAX, u64::MAX),
        (63, 0x8000_0000_0000_0000, 1, 0xffff_ffff_8000_0000, 1),
        (64, u64::MAX, u64::MAX, u64::MAX, u64::MAX),
    ] {
        for (opcode, funct3, funct7, result) in [
            (0x33, 1, 0, left),
            (0x33, 5, 0, right),
            (0x33, 5, 0x20, u64::MAX),
            (0x3b, 1, 0, left_w),
            (0x3b, 5, 0, right_w),
            (0x3b, 5, 0x20, u64::MAX),
        ] {
            let mut f = Fixture::new();
            f.addi(5, 0, 0, 0, -1, u64::MAX);
            f.addi(6, 0, 0, 0, shift, shift as u64);
            f.emit(
                r(opcode, funct3, funct7, 7, 5, 6),
                registers(
                    Some((5, u64::MAX)),
                    Some((6, shift as u64)),
                    Some((7, 0, result)),
                ),
                RamAccess::NoOp,
            );
            f.halt();
            f.complete();
        }
    }
    for (opcode, funct3, funct7, result) in [
        (0x33, 0, 0, 6),
        (0x33, 0, 0x20, 0),
        (0x33, 1, 0, 24),
        (0x33, 2, 0, 0),
        (0x33, 3, 0, 0),
        (0x33, 4, 0, 0),
        (0x33, 5, 0, 0),
        (0x33, 5, 0x20, 0),
        (0x33, 6, 0, 3),
        (0x33, 7, 0, 3),
        (0x3b, 0, 0, 6),
        (0x3b, 0, 0x20, 0),
        (0x3b, 1, 0, 24),
        (0x3b, 5, 0, 0),
        (0x3b, 5, 0x20, 0),
    ] {
        let mut f = Fixture::new();
        f.addi(5, 0, 0, 0, 3, 3);
        f.emit(
            r(opcode, funct3, funct7, 5, 5, 5),
            registers(Some((5, 3)), Some((5, 3)), Some((5, 3, result))),
            RamAccess::NoOp,
        );
        f.halt();
        f.complete();
    }
    for (immediate, result) in [(1, 0xffff_ffff_8000_0000), (-1, 0x7fff_fffe)] {
        let mut f = Fixture::new();
        f.emit(
            0x8000_02b7,
            registers(None, None, Some((5, 0, 0xffff_ffff_8000_0000))),
            RamAccess::NoOp,
        );
        f.addi(
            5,
            5,
            0xffff_ffff_8000_0000,
            0xffff_ffff_8000_0000,
            -1,
            0xffff_ffff_7fff_ffff,
        );
        f.emit(
            i(0x1b, 0, 5, 5, immediate),
            registers(
                Some((5, 0xffff_ffff_7fff_ffff)),
                None,
                Some((5, 0xffff_ffff_7fff_ffff, result)),
            ),
            RamAccess::NoOp,
        );
        f.halt();
        f.complete();
    }
}

#[test]
fn upper() {
    let mut f = Fixture::new();
    for (word, value) in [
        (0x1234_52b7, 0x1234_5000),
        (0x8000_0337, 0xffff_ffff_8000_0000),
    ] {
        let rd = if word == 0x1234_52b7 { 5 } else { 6 };
        f.emit(
            word,
            registers(None, None, Some((rd, 0, value))),
            RamAccess::NoOp,
        );
    }
    f.emit(
        0x1234_5397,
        registers(None, None, Some((7, 0, 0x9234_5008))),
        RamAccess::NoOp,
    );
    f.emit(
        0x8000_0417,
        registers(None, None, Some((8, 0, 12))),
        RamAccess::NoOp,
    );
    f.halt();
    f.complete();
}

#[test]
fn branches() {
    for (left_imm, left, right_imm, right, cases) in [
        (
            -1,
            u64::MAX,
            1,
            1,
            [
                (0, false),
                (1, true),
                (4, true),
                (5, false),
                (6, false),
                (7, true),
            ],
        ),
        (
            0,
            0,
            -1,
            u64::MAX,
            [
                (0, false),
                (1, true),
                (4, false),
                (5, true),
                (6, true),
                (7, false),
            ],
        ),
        (
            1,
            1,
            1,
            1,
            [
                (0, true),
                (1, false),
                (4, false),
                (5, true),
                (6, false),
                (7, true),
            ],
        ),
    ] {
        for (funct3, taken) in cases {
            for backwards in [false, true] {
                let mut f = Fixture::new();
                f.addi(5, 0, 0, 0, left_imm, left);
                f.addi(6, 0, 0, 0, right_imm, right);
                if backwards {
                    f.emit_to(
                        j(0, 8),
                        registers(None, None, Some((0, 0, 0))),
                        RamAccess::NoOp,
                        ENTRY + 16,
                    );
                    f.words.push(HALT);
                    f.emit_to(
                        b(funct3, 5, 6, -4),
                        registers(Some((5, left)), Some((6, right)), None),
                        RamAccess::NoOp,
                        if taken { ENTRY + 12 } else { ENTRY + 20 },
                    );
                    if taken {
                        f.expected.push(SourceTraceRow::new(
                            3,
                            ENTRY + 12,
                            ENTRY + 12,
                            registers(None, None, Some((0, 0, 0))),
                            RamAccess::NoOp,
                        ));
                    } else {
                        f.halt();
                    }
                } else {
                    let pc = f.pc();
                    f.emit_to(
                        b(funct3, 5, 6, 8),
                        registers(Some((5, left)), Some((6, right)), None),
                        RamAccess::NoOp,
                        pc + if taken { 8 } else { 4 },
                    );
                    if taken {
                        f.words.push(i(0x13, 0, 7, 0, 99));
                    } else {
                        f.addi(7, 0, 0, 0, 99, 99);
                    }
                    f.halt();
                }
                f.complete();
            }
            let mut f = Fixture::new();
            f.addi(5, 0, 0, 0, left_imm, left);
            f.addi(6, 0, 0, 0, right_imm, right);
            let pc = f.pc();
            f.emit_to(
                b(funct3, 5, 6, 2),
                registers(Some((5, left)), Some((6, right)), None),
                RamAccess::NoOp,
                if taken { pc + 2 } else { pc + 4 },
            );
            f.halt();
            if taken {
                f.error(SourceTraceError::PcOutsideProgram { pc: pc + 2 });
            } else {
                f.complete();
            }
        }
    }
    for (funct3, taken) in [
        (0, true),
        (1, false),
        (4, false),
        (5, true),
        (6, false),
        (7, true),
    ] {
        let mut f = Fixture::new();
        f.addi(5, 0, 0, 0, -1, u64::MAX);
        let pc = f.pc();
        f.emit_to(
            b(funct3, 5, 5, 0),
            registers(Some((5, u64::MAX)), Some((5, u64::MAX)), None),
            RamAccess::NoOp,
            if taken { pc } else { pc + 4 },
        );
        if !taken {
            f.halt();
        }
        f.complete();
    }
    for (funct3, taken) in [
        (0, false),
        (1, true),
        (4, true),
        (5, false),
        (6, false),
        (7, true),
    ] {
        let mut f = Fixture::new();
        f.emit(
            0x8000_02b7,
            registers(None, None, Some((5, 0, 0xffff_ffff_8000_0000))),
            RamAccess::NoOp,
        );
        f.emit(
            i(0x13, 1, 5, 5, 32),
            registers(
                Some((5, 0xffff_ffff_8000_0000)),
                None,
                Some((5, 0xffff_ffff_8000_0000, 0x8000_0000_0000_0000)),
            ),
            RamAccess::NoOp,
        );
        f.addi(6, 0, 0, 0, -1, u64::MAX);
        f.emit(
            i(0x13, 5, 6, 6, 1),
            registers(
                Some((6, u64::MAX)),
                None,
                Some((6, u64::MAX, 0x7fff_ffff_ffff_ffff)),
            ),
            RamAccess::NoOp,
        );
        f.emit_to(
            b(funct3, 5, 6, 8),
            registers(
                Some((5, 0x8000_0000_0000_0000)),
                Some((6, 0x7fff_ffff_ffff_ffff)),
                None,
            ),
            RamAccess::NoOp,
            if taken { ENTRY + 24 } else { ENTRY + 20 },
        );
        if taken {
            f.words.push(i(0x13, 0, 7, 0, 99));
        } else {
            f.addi(7, 0, 0, 0, 99, 99);
        }
        f.halt();
        f.complete();
    }
}

#[test]
fn jumps() {
    for rd in [0, 7] {
        let mut f = Fixture::new();
        f.emit_to(
            j(rd, 8),
            registers(
                None,
                None,
                Some((rd, 0, if rd == 0 { 0 } else { ENTRY + 4 })),
            ),
            RamAccess::NoOp,
            ENTRY + 8,
        );
        f.words.push(0x0000_0073);
        f.halt();
        f.complete();
    }
    for (rd, odd) in [(0, false), (5, false), (5, true)] {
        let mut f = Fixture::new();
        let target = ENTRY + 24;
        f.address(5, target + u64::from(odd));
        f.emit_to(
            i(0x67, 0, rd, 5, 0),
            registers(
                Some((5, target + u64::from(odd))),
                None,
                Some((
                    rd,
                    if rd == 5 { target + u64::from(odd) } else { 0 },
                    if rd == 0 { 0 } else { ENTRY + 20 },
                )),
            ),
            RamAccess::NoOp,
            target,
        );
        f.words.push(0x0000_0073);
        f.halt();
        f.complete();
    }
    let mut f = Fixture::new();
    f.words.extend([j(0, 2), HALT]);
    f.error(SourceTraceError::PcOutsideProgram { pc: ENTRY + 2 });
    let mut f = Fixture::new();
    f.address(5, ENTRY + 22);
    f.words.extend([i(0x67, 0, 0, 5, 0), HALT]);
    f.error(SourceTraceError::PcOutsideProgram { pc: ENTRY + 22 });
}

const DATA: u64 = 0xf8e7_d6c5_b4a3_9281;

#[test]
fn loads() {
    let cases: &[(u32, &[u64])] = &[
        (
            0,
            &[
                0xffff_ffff_ffff_ff81,
                0xffff_ffff_ffff_ff92,
                0xffff_ffff_ffff_ffa3,
                0xffff_ffff_ffff_ffb4,
                0xffff_ffff_ffff_ffc5,
                0xffff_ffff_ffff_ffd6,
                0xffff_ffff_ffff_ffe7,
                0xffff_ffff_ffff_fff8,
            ],
        ),
        (4, &[0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8]),
        (
            1,
            &[
                0xffff_ffff_ffff_9281,
                0xffff_ffff_ffff_b4a3,
                0xffff_ffff_ffff_d6c5,
                0xffff_ffff_ffff_f8e7,
            ],
        ),
        (5, &[0x9281, 0xb4a3, 0xd6c5, 0xf8e7]),
        (2, &[0xffff_ffff_b4a3_9281, 0xffff_ffff_f8e7_d6c5]),
        (6, &[0xb4a3_9281, 0xf8e7_d6c5]),
        (3, &[DATA]),
    ];
    for &(funct3, values) in cases {
        let width = 8 / values.len();
        for (offset, &value) in values.iter().enumerate() {
            let mut f = Fixture::new();
            f.inputs.inputs = DATA.to_le_bytes().to_vec();
            let base = f.layout().input_start;
            f.address(5, base);
            f.emit(
                i(0x03, funct3, 6, 5, (offset * width) as i32),
                registers(Some((5, base)), None, Some((6, 0, value))),
                read(base, DATA),
            );
            f.halt();
            f.complete();
        }
        let mut f = Fixture::new();
        f.inputs.inputs = DATA.to_le_bytes().to_vec();
        let base = f.layout().input_start;
        f.address(5, base);
        f.emit(
            i(0x03, funct3, 0, 5, 0),
            registers(Some((5, base)), None, Some((0, 0, 0))),
            read(base, DATA),
        );
        f.halt();
        f.complete();
    }
    for (funct3, post) in [(0, 0xffff_ffff_ffff_ff81), (3, 0xf8e7_d6c5_b4a3_9281)] {
        let (mut f, base) = initialized_heap();
        assert_eq!(base, 0x8000_4070);
        f.emit(
            i(0x03, funct3, 5, 5, 0),
            registers(Some((5, 0x8000_4070)), None, Some((5, 0x8000_4070, post))),
            read(0x8000_4070, 0xf8e7_d6c5_b4a3_9281),
        );
        f.halt();
        f.complete();
    }
}

fn initialized_heap() -> (Fixture, u64) {
    let mut f = Fixture::new();
    f.inputs.inputs = DATA.to_le_bytes().to_vec();
    let layout = f.layout();
    let base = layout.heap_end - 16;
    f.address(5, base);
    f.address(6, layout.input_start);
    f.emit(
        i(0x03, 3, 7, 6, 0),
        registers(Some((6, layout.input_start)), None, Some((7, 0, DATA))),
        read(layout.input_start, DATA),
    );
    f.emit(
        s(3, 5, 7, 0),
        registers(Some((5, base)), Some((7, DATA)), None),
        write(base, 0, DATA),
    );
    f.addi(8, 0, 0, 0, -1, u64::MAX);
    (f, base)
}

#[test]
fn stores() {
    let cases: &[(u32, &[u64])] = &[
        (
            0,
            &[
                0xf8e7_d6c5_b4a3_92ff,
                0xf8e7_d6c5_b4a3_ff81,
                0xf8e7_d6c5_b4ff_9281,
                0xf8e7_d6c5_ffa3_9281,
                0xf8e7_d6ff_b4a3_9281,
                0xf8e7_ffc5_b4a3_9281,
                0xf8ff_d6c5_b4a3_9281,
                0xffe7_d6c5_b4a3_9281,
            ],
        ),
        (
            1,
            &[
                0xf8e7_d6c5_b4a3_ffff,
                0xf8e7_d6c5_ffff_9281,
                0xf8e7_ffff_b4a3_9281,
                0xffff_d6c5_b4a3_9281,
            ],
        ),
        (2, &[0xf8e7_d6c5_ffff_ffff, 0xffff_ffff_b4a3_9281]),
        (3, &[u64::MAX]),
    ];
    for &(funct3, posts) in cases {
        let width = 8 / posts.len();
        for (offset, &post) in posts.iter().enumerate() {
            let (mut f, base) = initialized_heap();
            f.emit(
                s(funct3, 5, 8, (offset * width) as i32),
                registers(Some((5, base)), Some((8, u64::MAX)), None),
                write(base, DATA, post),
            );
            f.halt();
            f.complete();
        }
    }
    let (mut f, base) = initialized_heap();
    f.emit(
        s(0, 5, 8, 1),
        registers(Some((5, base)), Some((8, u64::MAX)), None),
        write(base, DATA, 0xf8e7_d6c5_b4a3_ff81),
    );
    f.emit(
        s(1, 5, 8, 0),
        registers(Some((5, base)), Some((8, u64::MAX)), None),
        write(base, 0xf8e7_d6c5_b4a3_ff81, 0xf8e7_d6c5_b4a3_ffff),
    );
    f.halt();
    f.complete();
    for (funct3, offset, post) in [(0, 1, 0xf8e7_d6c5_b4a3_7081), (3, 0, 0x0000_0000_8000_4070)] {
        let (mut f, base) = initialized_heap();
        assert_eq!(base, 0x8000_4070);
        f.emit(
            s(funct3, 5, 5, offset),
            registers(Some((5, 0x8000_4070)), Some((5, 0x8000_4070)), None),
            write(0x8000_4070, 0xf8e7_d6c5_b4a3_9281, post),
        );
        f.halt();
        f.complete();
    }
}

#[test]
fn end_of_ram() {
    for (funct3, width) in [(0, 1), (4, 1), (1, 2), (5, 2), (2, 4), (6, 4), (3, 8)] {
        for rd in [6, 0] {
            let mut f = Fixture::new();
            let address = f.layout().heap_end - width;
            f.address(5, address);
            f.emit(
                i(0x03, funct3, rd, 5, 0),
                registers(Some((5, address)), None, Some((rd, 0, 0))),
                read(address & !7, 0),
            );
            f.halt();
            f.complete();
        }
    }
    for (funct3, width, post) in [
        (0, 1, 0x0100_0000_0000_0000),
        (1, 2, 0x0001_0000_0000_0000),
        (2, 4, 0x0000_0001_0000_0000),
        (3, 8, 1),
    ] {
        let mut f = Fixture::new();
        let address = f.layout().heap_end - width;
        f.address(5, address);
        f.addi(6, 0, 0, 0, 1, 1);
        f.emit(
            s(funct3, 5, 6, 0),
            registers(Some((5, address)), Some((6, 1)), None),
            write(address & !7, 0, post),
        );
        f.halt();
        f.complete();
    }
}

#[test]
fn misalignment() {
    for (opcode, funct3, width) in [
        (0x03, 1, 2),
        (0x03, 5, 2),
        (0x03, 2, 4),
        (0x03, 6, 4),
        (0x03, 3, 8),
        (0x23, 1, 2),
        (0x23, 2, 4),
        (0x23, 3, 8),
    ] {
        for offset in (1..8).filter(|offset| offset % width != 0) {
            let mut f = Fixture::new();
            let address = f.layout().heap_end - 16;
            f.address(5, address);
            let pc = f.pc();
            f.words.push(if opcode == 0x03 {
                i(opcode, funct3, 6, 5, offset)
            } else {
                s(funct3, 5, 6, offset)
            });
            f.halt();
            f.error(SourceTraceError::MisalignedAccess {
                pc,
                address: address + offset as u64,
                width: width as u8,
            });
        }
    }
}

#[test]
fn gate() {
    for (word, kind) in [(0x0000_0073, Kind::ECALL), (0x0010_0073, Kind::EBREAK)] {
        let mut f = Fixture::new();
        f.words.extend([word, HALT]);
        f.error(SourceTraceError::UnsupportedInstruction { pc: ENTRY, kind });
    }
    let mut f = Fixture::new();
    f.emit_to(
        j(0, 8),
        registers(None, None, Some((0, 0, 0))),
        RamAccess::NoOp,
        ENTRY + 8,
    );
    f.words.push(0x0000_0073);
    f.halt();
    f.complete();
    let mut f = Fixture::new();
    f.emit(0x0ff0_000f, RegisterState::default(), RamAccess::NoOp);
    f.halt();
    f.complete();
    for (word, compressed) in [(0x0262_83b3, false), (0x0000_0085, true)] {
        let mut f = Fixture::new();
        f.words.extend([word, HALT]);
        let program = JoltProgram::from_elf_bytes(f.program().elf_bytes().to_vec());
        let error = SourceTracerBackend::default()
            .trace(&program, f.inputs)
            .unwrap_err();
        if compressed {
            assert!(matches!(
                error,
                TraceError::Program(ProgramError::IllegalCompressedInstruction { address: ENTRY })
            ));
        } else {
            assert!(matches!(
                error,
                TraceError::Program(ProgramError::IllegalSourceInstruction(Kind::MUL))
            ));
        }
    }
    let error = SourceTracerBackend::default()
        .trace(&JoltProgram::default(), TraceInputs::default())
        .unwrap_err();
    assert!(matches!(error, TraceError::MissingElfBytes));
}

#[test]
fn fetch() {
    for (words, pc) in [
        (vec![0, HALT], ENTRY),
        (vec![0x0010_0093, 0, HALT], ENTRY + 4),
        (vec![0x0010_0093], ENTRY + 4),
        (vec![j(0, -4), HALT], ENTRY - 4),
        (vec![j(0, 12), HALT], ENTRY + 12),
    ] {
        let mut f = Fixture::new();
        f.words = words;
        f.error(SourceTraceError::PcOutsideProgram { pc });
    }
}

#[test]
fn text_stores() {
    // Four address-building words, then the store at +16 and last JAL at +24.
    for (funct3, address, rejected) in [
        (3, ENTRY + 24, true),
        (0, ENTRY + 27, true),
        (2, ENTRY, true),
        (2, ENTRY + 28, false),
        (0, ENTRY + 28, false),
    ] {
        let mut f = Fixture::new();
        f.address(5, address);
        let pc = f.pc();
        f.emit(
            s(funct3, 5, 0, 0),
            registers(Some((5, address)), Some((0, 0)), None),
            write(address & !7, 0x0000_006f, 0x0000_006f),
        );
        f.addi(6, 0, 0, 0, 1, 1);
        f.halt();
        if rejected {
            f.error(SourceTraceError::StoreToProgramText { pc, address });
        } else {
            f.complete();
        }
    }
}

#[test]
fn sparse_text_stores() {
    // B7 protects decoded instruction bytes, not the holes inside the text span.
    for (address, pre, rejected) in [
        (0x8000_0040, 0x0000_006f_0000_0000, true),
        (0x8000_0000, 0, false),
        (0x8000_0048, 0, false),
        (0x8000_0038, 0, false),
    ] {
        let mut f = Fixture::new();
        f.text_sections = Some(TextSections {
            first_address: 0x8000_0008,
            split: 7,
            second_address: 0x8000_0044,
        });
        f.address(5, address);
        f.addi(6, 0, 0, 0, 1, 1);
        f.emit(
            s(3, 5, 6, 0),
            registers(Some((5, address)), Some((6, 1)), None),
            write(address, pre, 1),
        );
        f.emit_to(
            0x0240_006f,
            registers(None, None, Some((0, 0, 0))),
            RamAccess::NoOp,
            0x8000_0044,
        );
        f.halt();
        let program = f.program();
        let image = decode_elf(program.elf_bytes(), RV64I).unwrap();
        assert_eq!(
            image
                .instructions
                .iter()
                .map(|instruction| instruction.row().address)
                .collect::<Vec<_>>(),
            [
                0x8000_0008,
                0x8000_000c,
                0x8000_0010,
                0x8000_0014,
                0x8000_0018,
                0x8000_001c,
                0x8000_0020,
                0x8000_0044,
            ],
        );
        if rejected {
            f.error(SourceTraceError::StoreToProgramText {
                pc: 0x8000_001c,
                address: 0x8000_0040,
            });
        } else {
            f.complete();
        }
    }
}

#[test]
fn termination() {
    for (word, state) in [
        (b(0, 0, 0, 0), registers(Some((0, 0)), Some((0, 0)), None)),
        (HALT, registers(None, None, Some((0, 0, 0)))),
        (j(1, 0), registers(None, None, Some((1, 0, ENTRY + 4)))),
    ] {
        let mut f = Fixture::new();
        f.emit_to(word, state, RamAccess::NoOp, ENTRY);
        f.complete();
    }
    let mut f = Fixture::new();
    f.address(5, ENTRY + 16);
    f.emit_to(
        i(0x67, 0, 0, 5, 0),
        registers(Some((5, ENTRY + 16)), None, Some((0, 0, 0))),
        RamAccess::NoOp,
        ENTRY + 16,
    );
    f.complete();
    let mut f = Fixture::new();
    let address = f.layout().termination;
    f.address(5, address);
    f.addi(6, 0, 0, 0, 1, 1);
    f.emit(
        s(0, 5, 6, 0),
        registers(Some((5, address)), Some((6, 1)), None),
        write(address, 0, 1),
    );
    f.addi(7, 0, 0, 0, 9, 9);
    f.halt();
    f.complete();
}

#[test]
fn control_cells() {
    for (panic_cell, offset, funct3, value, post, panic) in [
        (true, 0, 0, 1, 1, true),
        (true, 0, 0, 0, 0, true),
        (true, 1, 0, 1, 0x100, false),
        (false, 0, 3, 1, 1, false),
    ] {
        let mut f = Fixture::new();
        let layout = f.layout();
        let address = if panic_cell {
            layout.panic
        } else {
            layout.termination
        };
        f.address(5, address);
        f.addi(6, 0, 0, 0, value, value as u64);
        f.emit(
            s(funct3, 5, 6, offset),
            registers(Some((5, address)), Some((6, value as u64)), None),
            write(address, 0, post),
        );
        f.halt();
        // expand_narrow_store writes an aligned SD: only the reference's
        // SB at panic+1 covers panic's base and sets its device flag.
        let output = if (panic_cell, offset) == (true, 1) {
            f.complete_with_reference_panic(true)
        } else {
            f.complete()
        };
        assert_eq!(output.device.panic, panic);
    }
    for panic_cell in [false, true] {
        for (funct3, width) in [(0, 1), (1, 2), (2, 4), (3, 8)] {
            for offset in (0..8).step_by(width) {
                let mut f = Fixture::new();
                let layout = f.layout();
                let address = if panic_cell {
                    layout.panic
                } else {
                    layout.termination
                };
                f.address(5, address);
                f.words.push(s(0, 5, 0, 1));
                let pc = f.pc();
                f.words.push(s(funct3, 5, 0, offset));
                f.halt();
                f.error(SourceTraceError::DeviceRegisterAccess {
                    pc,
                    address: address + offset as u64,
                });
            }
        }
        for after_store in [false, true] {
            for (funct3, offset) in [(0, 3), (3, 0)] {
                let mut f = Fixture::new();
                let layout = f.layout();
                let address = if panic_cell {
                    layout.panic
                } else {
                    layout.termination
                };
                f.address(5, address);
                if after_store {
                    f.words.push(s(0, 5, 0, 1));
                }
                let pc = f.pc();
                f.words.push(i(0x03, funct3, 6, 5, offset));
                f.halt();
                f.error(SourceTraceError::DeviceRegisterAccess {
                    pc,
                    address: address + offset as u64,
                });
            }
        }
    }
    let mut f = Fixture::new();
    let address = f.layout().panic;
    f.address(5, address);
    f.words.push(i(0x03, 3, 6, 5, 1));
    f.error(SourceTraceError::MisalignedAccess {
        pc: ENTRY + 16,
        address: address + 1,
        width: 8,
    });
}

#[test]
fn io() {
    let mut f = Fixture::new();
    f.inputs.inputs = DATA.to_le_bytes().to_vec();
    f.inputs.trusted_advice = vec![17, 19];
    f.inputs.untrusted_advice = vec![23, 29];
    let layout = f.layout();
    f.address(5, layout.input_start);
    f.address(6, layout.output_start);
    f.address(7, layout.termination);
    f.emit(
        i(0x03, 3, 8, 5, 0),
        registers(Some((5, layout.input_start)), None, Some((8, 0, DATA))),
        read(layout.input_start, DATA),
    );
    f.emit(
        s(3, 6, 8, 0),
        registers(Some((6, layout.output_start)), Some((8, DATA)), None),
        write(layout.output_start, 0, DATA),
    );
    f.addi(9, 0, 0, 0, 1, 1);
    f.emit(
        s(0, 7, 9, 0),
        registers(Some((7, layout.termination)), Some((9, 1)), None),
        write(layout.termination, 0, 1),
    );
    f.halt();
    assert_eq!(f.complete().device.outputs, DATA.to_le_bytes());
}

#[test]
fn text_span() {
    let instructions = [
        Instruction::decode(HALT, ENTRY, false).unwrap(),
        Instruction::decode(HALT, ENTRY + (1 << 28), false).unwrap(),
    ];
    assert!(matches!(
        InstructionTable::new(&instructions),
        Err(SourceTraceError::ProgramTextTooLarge { span: 0x1000_0004 })
    ));
}

#[test]
fn data_words_inside_executable_sections() {
    let mut f = Fixture::new();
    f.decode_mode = DecodeMode::DataHoles;
    f.words = vec![
        0x0000_0297,
        0x0102_b303,
        0x0000_0013,
        0x00c0_006f,
        0x8000_0018,
        0,
        HALT,
    ];
    f.expected = vec![
        SourceTraceRow::new(
            0,
            ENTRY,
            ENTRY + 4,
            registers(None, None, Some((5, 0, ENTRY))),
            RamAccess::NoOp,
        ),
        SourceTraceRow::new(
            1,
            ENTRY + 4,
            ENTRY + 8,
            registers(Some((5, ENTRY)), None, Some((6, 0, ENTRY + 24))),
            read(ENTRY + 16, ENTRY + 24),
        ),
        SourceTraceRow::new(
            2,
            ENTRY + 8,
            ENTRY + 12,
            registers(Some((0, 0)), None, Some((0, 0, 0))),
            RamAccess::NoOp,
        ),
        SourceTraceRow::new(
            3,
            ENTRY + 12,
            ENTRY + 24,
            registers(None, None, Some((0, 0, 0))),
            RamAccess::NoOp,
        ),
        SourceTraceRow::new(
            4,
            ENTRY + 24,
            ENTRY + 24,
            registers(None, None, Some((0, 0, 0))),
            RamAccess::NoOp,
        ),
    ];
    f.complete();
}

#[test]
fn fetching_a_data_hole_fails() {
    let mut f = Fixture::new();
    f.decode_mode = DecodeMode::DataHoles;
    f.words = vec![
        0x0000_0297,
        0x0102_b303,
        0x0000_0013,
        0x0000_0013,
        0x8000_0018,
        0,
        HALT,
    ];
    f.error(SourceTraceError::PcOutsideProgram { pc: ENTRY + 16 });
}

#[test]
fn default_decode_mode_rejects_data_in_executable_sections() {
    let mut f = Fixture::new();
    f.words = vec![
        0x0000_0297,
        0x0102_b303,
        0x0000_0013,
        0x00c0_006f,
        0x8000_0018,
        0,
        HALT,
    ];
    let program = f.program();
    for mut backend in [
        SourceTracerBackend::default(),
        SourceTracerBackend::with_row_capacity(5),
    ] {
        assert!(matches!(
            backend.trace(&program, f.inputs.clone()),
            Err(TraceError::Program(ProgramError::IllegalCompressedInstruction {
                address
            })) if address == ENTRY + 16
        ));
    }
}

mod stress;
