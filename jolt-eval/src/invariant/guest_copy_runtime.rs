//! The guest copy runtime (`jolt-platform/src/memory.S`) must implement
//! `memcpy` and `memmove` as they are linked into normally built guests.
//!
//! Each input is one external call. It runs on the tracer emulator (the
//! prover's execution engine) in a `no_std` guest, where `--wrap` resolves
//! `__real_memcpy`/`__real_memmove` to compiler-builtins, and in a `std`
//! guest, where they resolve to musl. Every call is checked for:
//!
//! 1. Snapshot semantics: both ranges and a guard band around each end equal
//!    `memmove` applied to the initial contents.
//! 2. The call ABI: control returns to `ra` within a tick bound with
//!    `a0 == dst` and `sp`, `gp`, `tp`, `s0`-`s11` unchanged.
//! 3. Source-instruction memory safety of the runtime: before every
//!    instruction inside `__wrap_memcpy`, `__wrap_memmove` or
//!    `__jolt_copy_misaligned` executes, its decoded load (store) must lie in
//!    the source (destination) range and be naturally aligned. Jolt expands
//!    subword stores into aligned doubleword read-modify-writes that touch
//!    neighbouring bytes, so this is checked on the source instruction; the
//!    guard bands of check 1 cover the expansion.
//! 4. The dispatch documented in `memory.S`: the original implementation is
//!    reached only for differing source/destination word offsets and
//!    `12 <= len < 544`, with the caller's arguments, return address and
//!    callee-saved registers intact; `__jolt_copy_misaligned` is entered only
//!    for differing offsets and `len >= 544`.

use std::sync::{Mutex, PoisonError};

use arbitrary::{Arbitrary, Unstructured};
use common::constants::{RAM_START_ADDRESS, STACK_CANARY_SIZE};
use common::jolt_device::MemoryConfig;
use jolt_host::Program;
use object::{Object, ObjectSymbol};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracer::emulator::cpu::{get_register_name, Cpu};
use tracer::emulator::Emulator;
use tracer::instruction::format::format_load::FormatLoad;
use tracer::instruction::format::format_s::FormatS;
use tracer::instruction::{uncompress_instruction, Instruction};

use crate::invariant::{CheckError, Invariant, InvariantViolation};

/// Bytes checked on each side of both ranges.
const GUARD: u64 = 512;
const MAX_LEN: u64 = 8192;
/// Largest `|dst - src|` the exact placement class produces.
const MAX_DISTANCE: i64 = 1 << 20;
/// Largest `|dst - src|` after `memcpy` placements are moved apart.
const MAX_DELTA: u64 = MAX_DISTANCE as u64 + MAX_LEN + 8;
const PAGE: u64 = 4096;
/// Distance from the arena start to the page holding `src`; destinations fit
/// on either side with their guard bands.
const ARENA_HALF: u64 = 2 << 20;

/// `memory.S` copies differing-offset lengths up to this with byte moves.
const MISMATCHED_BYTES_MAX: u64 = 11;
/// `memory.S` sends differing-offset lengths from this up to the leaf.
const MISALIGNED_LEAF_MIN: u64 = 544;

const RA: usize = 1;
const SP: usize = 2;
const GP: usize = 3;
const TP: usize = 4;
const A0: usize = 10;
const A1: usize = 11;
const A2: usize = 12;
/// `t0`-`t6` and `a3`-`a7`.
const CALLER_SAVED_GARBAGE: [usize; 12] = [5, 6, 7, 13, 14, 15, 16, 17, 28, 29, 30, 31];
/// `s0`-`s11`.
const CALLEE_SAVED: [usize; 12] = [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

const LEAF: &str = "__jolt_copy_misaligned";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum CopyRoutine {
    Memcpy,
    Memmove,
}

impl CopyRoutine {
    const ALL: [Self; 2] = [Self::Memcpy, Self::Memmove];

    fn name(self) -> &'static str {
        match self {
            Self::Memcpy => "memcpy",
            Self::Memmove => "memmove",
        }
    }

    fn wrapper(self) -> &'static str {
        match self {
            Self::Memcpy => "__wrap_memcpy",
            Self::Memmove => "__wrap_memmove",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Memcpy => 0,
            Self::Memmove => 1,
        }
    }
}

/// One external call `routine(src + dst_delta, src, len)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GuestCopyRuntimeInput {
    pub routine: CopyRoutine,
    /// At most 8192.
    pub len: u64,
    /// Offset of `src` within its page (below 4096); `src % 8` is its word
    /// offset.
    pub src_page_offset: u16,
    /// `dst - src`. `memcpy` ranges must be identical or disjoint.
    pub dst_delta: i64,
    /// Selects the arena contents and the callee-saved register values.
    pub arena_seed: u64,
    /// Values left in `t0`-`t6` and `a3`-`a7` at the call.
    pub caller_saved: [u64; 12],
}

impl GuestCopyRuntimeInput {
    /// Lengths cluster on the thresholds in `memory.S`: the aligned table
    /// (`<= 88`), byte table (`<= 11`), 80- and 256-byte blocks, the leaf
    /// cutoff (544), and its 128-byte blocks; one class in nine is large.
    fn decode_len(class: u8, word: u16) -> u64 {
        let word = u64::from(word);
        match class % 9 {
            0 => word % 17,
            1 => [80, 88, 89][(word % 3) as usize] - 8 + (word / 3) % 17,
            2 => [256, 512][(word % 2) as usize] - 8 + (word / 2) % 17,
            3 => MISALIGNED_LEAF_MIN - 8 + word % 17,
            4 => 8 * (1 + word % 256) + (word / 256) % 15 - 7,
            5 => 128 * (1 + word % 48) + (word / 48) % 19 - 9,
            6 => word % 1024,
            7 => 2048 + word % (MAX_LEN - 2048 + 1),
            _ => word % (MAX_LEN + 1),
        }
    }

    /// Classes: identical pointers, overlap in either direction, small
    /// signed distances, adjacent ranges, far ranges, and an exact distance.
    fn decode_delta(class: u8, word: u32, len: u64) -> i64 {
        let len = len as i64;
        let sign = if word & 1 == 0 { 1 } else { -1 };
        let magnitude = i64::from(word >> 1);
        let page = PAGE as i64;
        match class % 6 {
            0 => 0,
            1 => sign * (1 + magnitude % (len - 1).max(1)),
            2 => i64::from(word % 129) - 64,
            3 => sign * (len + magnitude % 64),
            4 => sign * (len + page * (1 + magnitude % 64) + (magnitude / 64) % page),
            _ => i64::from(word as i32) % MAX_DISTANCE,
        }
    }

    /// The nearest disjoint placement on the same side with the same
    /// distance modulo 8, so the alignment relationship (and with it the
    /// `memory.S` dispatch) is kept.
    fn disjoint(delta: i64, len: u64) -> i64 {
        let distance = delta.unsigned_abs();
        if delta == 0 || distance >= len {
            return delta;
        }
        delta.signum() * (distance + (len - distance).next_multiple_of(8)) as i64
    }

    fn validate(&self) -> Result<(), CheckError> {
        if self.len > MAX_LEN {
            return Err(CheckError::InvalidInput(format!(
                "len {} exceeds {MAX_LEN}",
                self.len
            )));
        }
        if u64::from(self.src_page_offset) >= PAGE {
            return Err(CheckError::InvalidInput(format!(
                "src_page_offset {} is not below {PAGE}",
                self.src_page_offset
            )));
        }
        let distance = self.dst_delta.unsigned_abs();
        if distance > MAX_DELTA {
            return Err(CheckError::InvalidInput(format!(
                "|dst_delta| {distance} exceeds {MAX_DELTA}"
            )));
        }
        if self.routine == CopyRoutine::Memcpy && distance != 0 && distance < self.len {
            return Err(CheckError::InvalidInput(
                "memcpy ranges overlap without being identical".into(),
            ));
        }
        Ok(())
    }

    fn initial_byte(&self, address: u64) -> u8 {
        (splitmix64(self.arena_seed ^ (address >> 3)) >> ((address & 7) * 8)) as u8
    }
}

impl<'a> Arbitrary<'a> for GuestCopyRuntimeInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let routine = if u.arbitrary::<u8>()? & 1 == 0 {
            CopyRoutine::Memcpy
        } else {
            CopyRoutine::Memmove
        };
        let len = Self::decode_len(u.arbitrary()?, u.arbitrary()?);
        let src_page_offset = u.arbitrary::<u16>()? % PAGE as u16;
        let mut dst_delta = Self::decode_delta(u.arbitrary()?, u.arbitrary()?, len);
        // Overlapping memcpy is undefined; identical pointers are not (LLVM
        // emits `memcpy(p, p, n)`).
        if routine == CopyRoutine::Memcpy {
            dst_delta = Self::disjoint(dst_delta, len);
        }
        Ok(Self {
            routine,
            len,
            src_page_offset,
            dst_delta,
            arena_seed: u.arbitrary()?,
            caller_saved: u.arbitrary()?,
        })
    }
}

fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A call's addresses in one guest.
#[derive(Clone, Copy)]
struct Ranges {
    dst: u64,
    src: u64,
    len: u64,
}

impl Ranges {
    fn mismatched(&self) -> bool {
        (self.dst ^ self.src) & 7 != 0
    }

    fn expects_fallback(&self) -> bool {
        self.mismatched() && self.len > MISMATCHED_BYTES_MAX && self.len < MISALIGNED_LEAF_MIN
    }

    fn expects_leaf(&self) -> bool {
        self.mismatched() && self.len >= MISALIGNED_LEAF_MIN
    }

    fn contains(&self, inner: &Ranges) -> bool {
        inner.src >= self.src
            && inner.src + inner.len <= self.src + self.len
            && inner.dst >= self.dst
            && inner.dst + inner.len <= self.dst + self.len
    }

    /// Doubleword-aligned windows covering both ranges and their guard bands.
    fn windows(&self) -> Vec<(u64, u64)> {
        let window = |start: u64| {
            (
                (start - GUARD) & !7,
                (start + self.len + GUARD).next_multiple_of(8),
            )
        };
        let (low, high) = if self.src <= self.dst {
            (window(self.src), window(self.dst))
        } else {
            (window(self.dst), window(self.src))
        };
        if high.0 <= low.1 {
            vec![(low.0, high.1.max(low.1))]
        } else {
            vec![low, high]
        }
    }
}

#[derive(Clone, Copy)]
enum Op {
    Load { rs1: u8, imm: i64, width: u64 },
    Store { rs1: u8, imm: i64, width: u64 },
    Register,
}

impl Op {
    fn load(operands: FormatLoad, width: u64) -> Self {
        Self::Load {
            rs1: operands.rs1,
            imm: operands.imm,
            width,
        }
    }

    fn store(operands: FormatS, width: u64) -> Self {
        Self::Store {
            rs1: operands.rs1,
            imm: operands.imm,
            width,
        }
    }

    /// Memory instructions carry their access; any instruction outside the
    /// RV64I register, branch and jump set is rejected so a new memory
    /// operation cannot escape check 3.
    fn classify(instruction: &Instruction) -> Option<Self> {
        Some(match instruction {
            Instruction::LB(i) => Self::load(i.operands, 1),
            Instruction::LBU(i) => Self::load(i.operands, 1),
            Instruction::LH(i) => Self::load(i.operands, 2),
            Instruction::LHU(i) => Self::load(i.operands, 2),
            Instruction::LW(i) => Self::load(i.operands, 4),
            Instruction::LWU(i) => Self::load(i.operands, 4),
            Instruction::LD(i) => Self::load(i.operands, 8),
            Instruction::SB(i) => Self::store(i.operands, 1),
            Instruction::SH(i) => Self::store(i.operands, 2),
            Instruction::SW(i) => Self::store(i.operands, 4),
            Instruction::SD(i) => Self::store(i.operands, 8),
            Instruction::ADD(_)
            | Instruction::ADDI(_)
            | Instruction::ADDIW(_)
            | Instruction::ADDW(_)
            | Instruction::AND(_)
            | Instruction::ANDI(_)
            | Instruction::AUIPC(_)
            | Instruction::BEQ(_)
            | Instruction::BGE(_)
            | Instruction::BGEU(_)
            | Instruction::BLT(_)
            | Instruction::BLTU(_)
            | Instruction::BNE(_)
            | Instruction::JAL(_)
            | Instruction::JALR(_)
            | Instruction::LUI(_)
            | Instruction::OR(_)
            | Instruction::ORI(_)
            | Instruction::SLL(_)
            | Instruction::SLLI(_)
            | Instruction::SLLIW(_)
            | Instruction::SLLW(_)
            | Instruction::SLT(_)
            | Instruction::SLTI(_)
            | Instruction::SLTIU(_)
            | Instruction::SLTU(_)
            | Instruction::SRA(_)
            | Instruction::SRAI(_)
            | Instruction::SRAIW(_)
            | Instruction::SRAW(_)
            | Instruction::SRL(_)
            | Instruction::SRLI(_)
            | Instruction::SRLIW(_)
            | Instruction::SRLW(_)
            | Instruction::SUB(_)
            | Instruction::SUBW(_)
            | Instruction::XOR(_)
            | Instruction::XORI(_) => Self::Register,
            _ => return None,
        })
    }

    fn check(self, cpu: &Cpu, ranges: &Ranges) -> Result<(), String> {
        let (kind, rs1, imm, width, start) = match self {
            Self::Load { rs1, imm, width } => ("load", rs1, imm, width, ranges.src),
            Self::Store { rs1, imm, width } => ("store", rs1, imm, width, ranges.dst),
            Self::Register => return Ok(()),
        };
        let address = (cpu.read_register(rs1) as u64).wrapping_add(imm as u64);
        if !address.is_multiple_of(width) {
            return Err(format!("misaligned {width}-byte {kind} at {address:#x}"));
        }
        let offset = address.wrapping_sub(start);
        if offset >= ranges.len || ranges.len - offset < width {
            return Err(format!(
                "{width}-byte {kind} at {address:#x} leaves [{start:#x}, {:#x})",
                start + ranges.len
            ));
        }
        Ok(())
    }
}

/// One runtime routine, decoded with the tracer's decoder.
struct RoutineText {
    symbol: &'static str,
    start: u64,
    /// The instruction starting at each two-byte slot, if any.
    ops: Vec<Option<Op>>,
}

impl RoutineText {
    /// Linear decode is exact: the routines contain no data.
    fn decode(emulator: &mut Emulator, symbol: &'static str, start: u64, size: u64) -> Self {
        assert!(
            size > 0 && size.is_multiple_of(2),
            "{symbol} has size {size}"
        );
        let mmu = &mut emulator.get_mut_cpu().mmu;
        let mut ops = vec![None; (size / 2) as usize];
        let mut offset = 0;
        while offset < size {
            let pc = start + offset;
            let low = u32::from(mmu.load_raw(pc)) | u32::from(mmu.load_raw(pc + 1)) << 8;
            let (word, compressed) = if low & 3 == 3 {
                let high = u32::from(mmu.load_raw(pc + 2)) | u32::from(mmu.load_raw(pc + 3)) << 8;
                (low | high << 16, false)
            } else {
                (uncompress_instruction(low), true)
            };
            let instruction = Instruction::decode(word, pc, compressed)
                .unwrap_or_else(|error| panic!("{symbol}+{offset:#x}: {error}"));
            let op = Op::classify(&instruction).unwrap_or_else(|| {
                panic!("{symbol}+{offset:#x}: unclassified instruction {instruction:?}")
            });
            ops[(offset / 2) as usize] = Some(op);
            offset += if compressed { 2 } else { 4 };
        }
        assert_eq!(offset, size, "{symbol} does not end on an instruction");
        Self { symbol, start, ops }
    }

    fn slot(&self, pc: u64) -> Option<usize> {
        let offset = pc.checked_sub(self.start)?;
        let slot = (offset / 2) as usize;
        (offset.is_multiple_of(2) && slot < self.ops.len()).then_some(slot)
    }
}

/// `__wrap_memcpy`, `__wrap_memmove`, then the shared leaf.
const LEAF_INDEX: usize = 2;

struct GuestState {
    emulator: Emulator,
    /// Per routine, the instruction slots executed so far.
    executed: [Vec<bool>; 3],
}

/// A built guest whose copy runtime is located, decoded, and ready to call.
pub struct RuntimeGuest {
    label: &'static str,
    routines: [RoutineText; 3],
    /// Addresses `__real_memcpy` and `__real_memmove` resolve to.
    fallbacks: [u64; 2],
    return_address: u64,
    stack_pointer: u64,
    arena_start: u64,
    state: Mutex<GuestState>,
}

struct GuestBuild {
    label: &'static str,
    package: &'static str,
    func: Option<&'static str>,
    std: bool,
}

/// The guests `crates/jolt-verifier/tests/guest_runtime.rs` exercises.
const GUESTS: [GuestBuild; 2] = [
    GuestBuild {
        label: "no_std",
        package: "btreemap-guest",
        func: None,
        std: false,
    },
    GuestBuild {
        label: "std",
        package: "stdlib-guest",
        func: Some("string_concat"),
        std: true,
    },
];

impl RuntimeGuest {
    fn build(build: &GuestBuild) -> Self {
        let mut host = Program::new(build.package);
        host.set_std(build.std);
        if let Some(func) = build.func {
            host.set_func(func);
        }
        // Symbols locate the runtime. The routines are assembly, so the frame
        // pointers this mode also enables do not change them.
        host.set_backtrace("dwarf");
        let program = host.jolt_program().expect("guest builds");
        let elf = host.get_elf_contents().expect("built guest ELF");

        let object = object::File::parse(elf.as_slice()).expect("guest ELF parses");
        let symbol = |name: &str| {
            let mut matches = object
                .symbols()
                .filter(|symbol| symbol.name().is_ok_and(|symbol| symbol == name))
                .map(|symbol| (symbol.address(), symbol.size()));
            let found = matches
                .next()
                .unwrap_or_else(|| panic!("{} guest has no {name} symbol", build.label));
            assert!(
                matches.next().is_none(),
                "{} guest defines {name} more than once",
                build.label
            );
            found
        };

        let config = MemoryConfig {
            program_size: Some(program.program_end - RAM_START_ADDRESS),
            ..MemoryConfig::default()
        };
        let mut emulator = tracer::create_emulator(&elf, None, &[], &[], &[], &config, None);
        let layout = emulator
            .get_cpu()
            .mmu
            .jolt_device
            .as_ref()
            .expect("emulator has a device")
            .memory_layout
            .clone();
        let stack_top = layout.stack_end + STACK_CANARY_SIZE + layout.stack_size;
        let arena_start = (stack_top + PAGE).next_multiple_of(PAGE);
        assert!(arena_start + 2 * ARENA_HALF <= layout.heap_end);

        let mut decode = |name: &'static str| {
            let (start, size) = symbol(name);
            RoutineText::decode(&mut emulator, name, start, size)
        };
        let routines = [
            decode(CopyRoutine::Memcpy.wrapper()),
            decode(CopyRoutine::Memmove.wrapper()),
            decode(LEAF),
        ];
        let executed = routines.each_ref().map(|text| vec![false; text.ops.len()]);
        let return_address = emulator.get_cpu().read_pc();
        Self {
            label: build.label,
            routines,
            // `--wrap=memcpy` binds `__real_memcpy` to the `memcpy` definition.
            fallbacks: CopyRoutine::ALL.map(|routine| symbol(routine.name()).0),
            return_address,
            stack_pointer: stack_top & !15,
            arena_start,
            state: Mutex::new(GuestState { emulator, executed }),
        }
    }

    fn locate(&self, pc: u64) -> Option<(usize, usize)> {
        self.routines
            .iter()
            .enumerate()
            .find_map(|(routine, text)| text.slot(pc).map(|slot| (routine, slot)))
    }

    fn run(&self, input: &GuestCopyRuntimeInput) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let GuestState { emulator, executed } = &mut *state;
        let src = self.arena_start + ARENA_HALF + u64::from(input.src_page_offset);
        let ranges = Ranges {
            dst: src.wrapping_add_signed(input.dst_delta),
            src,
            len: input.len,
        };
        let windows = ranges.windows();

        let cpu = emulator.get_mut_cpu();
        for &(start, end) in &windows {
            for address in start..end {
                cpu.mmu.store_raw(address, input.initial_byte(address));
            }
        }
        for (register, value) in CALLER_SAVED_GARBAGE.into_iter().zip(input.caller_saved) {
            cpu.write_register(register, value as i64);
        }
        let mut value = input.arena_seed;
        for register in [GP, TP].into_iter().chain(CALLEE_SAVED) {
            value = splitmix64(value);
            cpu.write_register(register, value as i64);
        }
        cpu.write_register(RA, self.return_address as i64);
        cpu.write_register(SP, self.stack_pointer as i64);
        cpu.write_register(A0, ranges.dst as i64);
        cpu.write_register(A1, ranges.src as i64);
        cpu.write_register(A2, ranges.len as i64);
        cpu.update_pc(self.routines[input.routine.index()].start);

        self.execute(emulator, executed, input.routine, ranges)?;

        let cpu = emulator.get_mut_cpu();
        for &(start, end) in &windows {
            for address in start..end {
                let actual = cpu.mmu.load_raw(address);
                let expected = match address.checked_sub(ranges.dst) {
                    Some(offset) if offset < ranges.len => input.initial_byte(ranges.src + offset),
                    _ => input.initial_byte(address),
                };
                if actual != expected {
                    return Err(format!(
                        "byte at dst{:+} is {actual:#04x}, expected {expected:#04x}",
                        address as i64 - ranges.dst as i64
                    ));
                }
            }
        }
        Ok(())
    }

    /// Ticks the emulator from the call until it returns, checking the ABI,
    /// dispatch, and every runtime memory access. Calls the original
    /// implementation makes back into the runtime (musl's `memmove` calls
    /// `memcpy`) get frames of their own.
    fn execute(
        &self,
        emulator: &mut Emulator,
        executed: &mut [Vec<bool>; 3],
        routine: CopyRoutine,
        ranges: Ranges,
    ) -> Result<(), String> {
        let tick_limit = 4096 + 64 * ranges.len;
        let mut frames = vec![Frame::enter(emulator.get_cpu(), routine)];
        let mut ticks = 0;
        loop {
            let cpu = emulator.get_cpu();
            let pc = cpu.read_pc();
            let sp = cpu.read_register(SP as u8) as u64;
            while let Some(frame) =
                frames.pop_if(|frame| frame.registers[RA] == pc && frame.registers[SP] == sp)
            {
                frame.check_return(cpu)?;
            }
            let Some(caller) = frames.last() else {
                return Ok(());
            };
            if ticks == tick_limit {
                return Err(format!("no return within {tick_limit} instructions"));
            }

            match self.locate(pc) {
                Some((index, slot)) => {
                    let symbol = self.routines[index].symbol;
                    if caller.fallback {
                        let nested = CopyRoutine::ALL
                            .into_iter()
                            .find(|routine| pc == self.routines[routine.index()].start)
                            .ok_or_else(|| {
                                format!(
                                    "the original implementation jumped to {symbol}+{:#x}",
                                    2 * slot
                                )
                            })?;
                        let frame = Frame::enter(cpu, nested);
                        if !ranges.contains(&frame.ranges) {
                            return Err(format!(
                                "the original implementation called {symbol} on other ranges"
                            ));
                        }
                        frames.push(frame);
                    }
                    let current = frames.len() - 1;
                    let frame = &mut frames[current];
                    if index == LEAF_INDEX {
                        if !frame.ranges.expects_leaf() {
                            return Err(format!("{} entered {LEAF}", frame.routine.wrapper()));
                        }
                        frame.leaf = true;
                    } else if index != frame.routine.index() {
                        return Err(format!("{} reached {symbol}", frame.routine.wrapper()));
                    }
                    let op = self.routines[index].ops[slot].ok_or_else(|| {
                        format!("pc {symbol}+{:#x} is inside an instruction", 2 * slot)
                    })?;
                    op.check(cpu, &frame.ranges)
                        .map_err(|error| format!("{symbol}+{:#x}: {error}", 2 * slot))?;
                    executed[index][slot] = true;
                }
                None if caller.fallback => {}
                None if pc == self.fallbacks[caller.routine.index()] => {
                    let current = frames.len() - 1;
                    frames[current].enter_fallback(cpu)?;
                }
                None => {
                    return Err(format!(
                        "{} transferred control to {pc:#x}",
                        caller.routine.wrapper()
                    ));
                }
            }
            emulator.tick(None);
            ticks += 1;
        }
    }
}

struct Frame {
    routine: CopyRoutine,
    ranges: Ranges,
    registers: [u64; 32],
    fallback: bool,
    leaf: bool,
}

impl Frame {
    fn enter(cpu: &Cpu, routine: CopyRoutine) -> Self {
        let registers = std::array::from_fn(|register| cpu.read_register(register as u8) as u64);
        Self {
            routine,
            ranges: Ranges {
                dst: registers[A0],
                src: registers[A1],
                len: registers[A2],
            },
            registers,
            fallback: false,
            leaf: false,
        }
    }

    fn unchanged(&self, cpu: &Cpu, registers: &[usize], event: &str) -> Result<(), String> {
        for &register in registers {
            let value = cpu.read_register(register as u8) as u64;
            if value != self.registers[register] {
                return Err(format!(
                    "{} changed {} from {:#x} to {value:#x} {event}",
                    self.routine.wrapper(),
                    get_register_name(register),
                    self.registers[register]
                ));
            }
        }
        Ok(())
    }

    fn enter_fallback(&mut self, cpu: &Cpu) -> Result<(), String> {
        if !self.ranges.expects_fallback() {
            return Err(format!(
                "{} called the original implementation",
                self.routine.wrapper()
            ));
        }
        self.unchanged(
            cpu,
            &[[RA, SP, GP, TP, A0, A1, A2].as_slice(), &CALLEE_SAVED].concat(),
            "before the original implementation",
        )?;
        self.fallback = true;
        Ok(())
    }

    fn check_return(&self, cpu: &Cpu) -> Result<(), String> {
        let returned = cpu.read_register(A0 as u8) as u64;
        if returned != self.ranges.dst {
            return Err(format!(
                "{} returned {returned:#x} instead of dst {:#x}",
                self.routine.wrapper(),
                self.ranges.dst
            ));
        }
        self.unchanged(
            cpu,
            &[[SP, GP, TP].as_slice(), &CALLEE_SAVED].concat(),
            "across the call",
        )?;
        if self.fallback != self.ranges.expects_fallback() {
            return Err(format!(
                "{} did not call the original implementation",
                self.routine.wrapper()
            ));
        }
        if self.leaf != self.ranges.expects_leaf() {
            return Err(format!("{} did not enter {LEAF}", self.routine.wrapper()));
        }
        Ok(())
    }
}

#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct GuestCopyRuntimeInvariant;

impl Invariant for GuestCopyRuntimeInvariant {
    type Setup = Vec<RuntimeGuest>;
    type Input = GuestCopyRuntimeInput;

    fn name(&self) -> &str {
        "guest_copy_runtime"
    }

    fn description(&self) -> String {
        "Calling the guest runtime's __wrap_memcpy or __wrap_memmove \
         (jolt-platform/src/memory.S) in normally built no_std and std guests, \
         executed on the tracer emulator, must leave memory equal to memmove of \
         the initial snapshot (including guard bands around both ranges), return \
         dst to ra with sp, gp, tp and s0-s11 preserved, keep every load of the \
         runtime's own instructions inside the source range and every store \
         inside the destination range with natural alignment, and reach the \
         original implementation exactly for differing word offsets with \
         12 <= len < 544 (arguments and callee-saved state intact) and \
         __jolt_copy_misaligned exactly for differing offsets with len >= 544. \
         Overlapping memcpy ranges are undefined unless identical."
            .to_string()
    }

    fn setup(&self) -> Vec<RuntimeGuest> {
        // `cargo +<nightly> fuzz` exports RUSTUP_TOOLCHAIN to the fuzz binary.
        // Guests build with the toolchain `rust-toolchain.toml` pins, as `jolt
        // build` does from a shell; the fuzz nightly's std guest places data
        // in `.text`, which the bytecode decoder rejects.
        std::env::remove_var("RUSTUP_TOOLCHAIN");
        GUESTS.iter().map(RuntimeGuest::build).collect()
    }

    fn check(
        &self,
        setup: &Vec<RuntimeGuest>,
        input: GuestCopyRuntimeInput,
    ) -> Result<(), CheckError> {
        input.validate()?;
        for guest in setup {
            guest.run(&input).map_err(|error| {
                CheckError::Violation(InvariantViolation::with_details(
                    format!("{} guest: {error}", guest.label),
                    format!("{input:?}"),
                ))
            })?;
        }
        Ok(())
    }

    fn seed_corpus(&self) -> Vec<GuestCopyRuntimeInput> {
        seed_cases().into_iter().map(|(_, input)| input).collect()
    }
}

/// Inputs reaching every dispatch path and computed-jump entry of the
/// runtime, named after the path. `jolt-eval/fuzz/seeds/guest_copy_runtime`
/// holds their encodings.
fn seed_cases() -> Vec<(String, GuestCopyRuntimeInput)> {
    let mut seeds = Vec::new();
    for routine in CopyRoutine::ALL {
        let mut add = |label: String, len: u64, src_page_offset: u16, dst_delta: i64| {
            let arena_seed = splitmix64(seeds.len() as u64);
            let mut register = arena_seed;
            let caller_saved = std::array::from_fn(|_| {
                register = splitmix64(register);
                register
            });
            seeds.push((
                format!("{}-{label}", routine.name()),
                GuestCopyRuntimeInput {
                    routine,
                    len,
                    src_page_offset,
                    dst_delta,
                    arena_seed,
                    caller_saved,
                },
            ));
        };
        add("empty".into(), 0, 0, 4096);
        for len in (8..=88).step_by(8) {
            add(format!("aligned-table-{len}"), len, 0, 512);
        }
        for len in 2..=7 {
            add(format!("byte-table-{len}"), len, 0, 512);
        }
        for len in 8..=MISMATCHED_BYTES_MAX {
            add(format!("byte-table-{len}"), len, 0, 513);
        }
        add("one-byte".into(), 1, 5, 1024);
        add("aligned-general-forward".into(), 96, 0, -1024);
        add("aligned-general-backward".into(), 96, 0, 1024);
        // 351 bytes after the alignment edge reach a 256-byte block, an
        // 80-byte block, a word, and a 4/2/1-byte tail.
        for offset in 0..8 {
            let len = 351 + u64::from(offset);
            add(format!("general-forward-{offset}"), len, offset, -1024);
            add(format!("general-backward-{offset}"), len, offset, 1024);
        }
        add("identical".into(), 100, 3, 0);
        // Leaf calls at each nonzero `(src - dst) % 8`: in `a` the source is
        // unaligned once the destination is, so the first word is moved byte
        // by byte; in `b` the source starts aligned and that step is skipped.
        for word_offset in 1..8u16 {
            let shift = i64::from(word_offset);
            for (variant, src_page_offset) in [("a", word_offset), ("b", 0)] {
                add(
                    format!("leaf-forward-{word_offset}{variant}"),
                    712,
                    src_page_offset,
                    -1024 - shift,
                );
                add(
                    format!("leaf-backward-{word_offset}{variant}"),
                    712,
                    src_page_offset,
                    1024 - shift,
                );
            }
        }
        for len in [MISMATCHED_BYTES_MAX + 1, 100, MISALIGNED_LEAF_MIN - 1] {
            add(format!("fallback-{len}"), len, 0, 1025);
        }
        add("leaf-threshold".into(), MISALIGNED_LEAF_MIN, 0, 1025);
        if routine == CopyRoutine::Memmove {
            for (label, len, src_page_offset, dst_delta) in [
                ("table-forward", 64, 0, -8),
                ("bytes-backward", 7, 1, 1),
                ("general-forward", 400, 3, -8),
                ("general-backward", 400, 3, 8),
                ("leaf-forward", 700, 1, -3),
                ("leaf-backward", 700, 1, 3),
                ("fallback-forward", 100, 2, -5),
                ("fallback-backward", 100, 2, 5),
            ] {
                add(format!("overlap-{label}"), len, src_page_offset, dst_delta);
            }
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    /// The last (exact) classes of `decode_len` and `decode_delta`.
    const EXACT_LEN_CLASS: u8 = 8;
    const EXACT_DELTA_CLASS: u8 = 5;

    fn seed_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/guest_copy_runtime")
    }

    /// Bytes that decode to `input` through the exact length and placement
    /// classes.
    fn encode(input: &GuestCopyRuntimeInput) -> Vec<u8> {
        let mut bytes = vec![input.routine.index() as u8];
        bytes.push(EXACT_LEN_CLASS);
        bytes.extend((input.len as u16).to_le_bytes());
        bytes.extend(input.src_page_offset.to_le_bytes());
        bytes.push(EXACT_DELTA_CLASS);
        bytes.extend((input.dst_delta as i32).to_le_bytes());
        bytes.extend(input.arena_seed.to_le_bytes());
        for register in input.caller_saved {
            bytes.extend(register.to_le_bytes());
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> GuestCopyRuntimeInput {
        GuestCopyRuntimeInput::arbitrary(&mut Unstructured::new(bytes)).expect("seed decodes")
    }

    /// The checked-in seeds are exactly the encoded `seed_cases`. Replaying
    /// them through the fuzz decoder and harness executes every instruction
    /// of the runtime in both guests.
    #[test]
    fn seed_files_cover_every_runtime_instruction() {
        let cases = seed_cases();
        let mut files: Vec<_> = fs::read_dir(seed_directory())
            .expect("seed directory exists")
            .map(|entry| entry.expect("seed entry").file_name())
            .collect();
        files.sort();
        let mut names: Vec<_> = cases.iter().map(|(name, _)| OsString::from(name)).collect();
        names.sort();
        assert_eq!(
            files, names,
            "regenerate the seeds with the ignored write_seed_files test"
        );

        let invariant = GuestCopyRuntimeInvariant;
        let setup = invariant.setup();
        for (name, input) in &cases {
            let bytes = fs::read(seed_directory().join(name)).expect("seed is readable");
            assert_eq!(bytes, encode(input), "{name} is stale");
            let decoded = decode(&bytes);
            assert_eq!(&decoded, input, "{name} does not round-trip");
            if let Err(error) = invariant.check(&setup, decoded) {
                panic!("{name}: {error}");
            }
        }

        for guest in &setup {
            let state = guest.state.lock().expect("no check panicked");
            for (text, executed) in guest.routines.iter().zip(&state.executed) {
                let missed: Vec<_> = text
                    .ops
                    .iter()
                    .zip(executed)
                    .enumerate()
                    .filter(|(_, (op, hit))| op.is_some() && !**hit)
                    .map(|(slot, _)| format!("{}+{:#x}", text.symbol, 2 * slot))
                    .collect();
                let total = text.ops.iter().filter(|op| op.is_some()).count();
                eprintln!(
                    "{} guest {}: {}/{total} instructions executed",
                    guest.label,
                    text.symbol,
                    total - missed.len()
                );
                assert!(
                    missed.is_empty(),
                    "{} guest never executed {missed:?}",
                    guest.label
                );
            }
        }
    }

    #[test]
    #[ignore = "writes jolt-eval/fuzz/seeds/guest_copy_runtime; run explicitly and commit the output"]
    fn write_seed_files() {
        let directory = seed_directory();
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("remove old seeds");
        }
        fs::create_dir_all(&directory).expect("create seed directory");
        for (name, input) in seed_cases() {
            fs::write(directory.join(name), encode(&input)).expect("write seed");
        }
    }
}
