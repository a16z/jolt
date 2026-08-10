//! Test/bench harness: build tiny synthetic programs around individual rows
//! and execute them natively. Hidden API — consumed by the per-instruction
//! differential tests and the iai microbenchmarks.

use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_program::execution::{JoltProgram, TraceError};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};

/// The row-emission seam, re-exported so emitter A/B experiments (benches,
/// integration tests, the Alternative 11 stencil spike) can build their own
/// `EmitterSet` and compile with it.
pub use super::compile::emitter::{EmitOutcome, EmitterSet, RowEmitter};
pub use super::compile::CompiledProgram;
use super::memory::MemoryPlane;
use super::state::{ExitReason, GuestState, HostContext};

/// Source address of the row under test. Terminal rows (always-taken
/// self-branches with no register effects) catch the fall-through and the
/// ±8 branch targets.
pub const TEST_ADDR: u64 = RAM_START_ADDRESS + 0x1000;
/// RAM scratch region seeded identically on both sides for memory-op rows.
pub const SCRATCH_START: u64 = RAM_START_ADDRESS + 0x2000;
pub const SCRATCH_DWORDS: usize = 512;
/// Plane/interpreter memory capacity for harness runs.
pub const MEM_CAPACITY: u64 = 1 << 20;

pub fn memory_config() -> MemoryConfig {
    MemoryConfig {
        heap_size: MEM_CAPACITY,
        program_size: Some(1024),
        ..Default::default()
    }
}

/// An always-taken self-branch: `Beq x0, x0, 0`. Terminates via PC-stall
/// with no register or memory effects.
fn terminal_row(address: u64) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::BEQ,
        address: address as usize,
        operands: NormalizedOperands {
            rs1: Some(0),
            rs2: Some(0),
            rd: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: true,
        is_compressed: false,
    }
}

/// Synthetic program: terminals at `TEST_ADDR - 8`, `+ 4`, `+ 8` around the
/// row under test, so every generated control transfer lands on a compiled
/// group.
pub fn single_row_program(row: JoltInstructionRow) -> JoltProgram {
    assert_eq!(row.address as u64, TEST_ADDR, "row must sit at TEST_ADDR");
    let rows = vec![
        terminal_row(TEST_ADDR - 8),
        row,
        terminal_row(TEST_ADDR + 4),
        terminal_row(TEST_ADDR + 8),
    ];
    JoltProgram::from_parts(
        Vec::new(),
        rows,
        Vec::new(),
        RAM_START_ADDRESS + 1024,
        TEST_ADDR,
    )
}

/// A straight-line program: `count` copies of the row at consecutive
/// addresses followed by a fall-through terminal (for microbenchmarks).
/// The row must not transfer control (or must be a `Jal` with `imm = 4`).
pub fn straight_line_program(mut row: JoltInstructionRow, count: usize) -> JoltProgram {
    let mut rows = Vec::with_capacity(count + 1);
    for i in 0..count {
        row.address = (TEST_ADDR + 4 * i as u64) as usize;
        rows.push(row);
    }
    rows.push(terminal_row(TEST_ADDR + 4 * count as u64));
    JoltProgram::from_parts(
        Vec::new(),
        rows,
        Vec::new(),
        RAM_START_ADDRESS + 1024,
        TEST_ADDR,
    )
}

/// Compile without running (fail-fast coverage checks).
pub fn compile_only(program: &JoltProgram) -> Result<(), TraceError> {
    CompiledProgram::compile(program).map(|_| ())
}

/// Compile and hand back the artifact, so callers that inspect the live code
/// mapping (the safety tests) keep it alive while they look.
pub fn compile_program(program: &JoltProgram) -> Result<CompiledProgram, TraceError> {
    CompiledProgram::compile(program)
}

/// A compiled synthetic program plus its memory plane, reusable across runs
/// (for microbenchmarks: compilation stays outside the measured region).
pub struct Prepared {
    compiled: CompiledProgram,
    plane: MemoryPlane,
    entry: u64,
    pre_regs: [u64; REGISTER_COUNT as usize],
}

impl Prepared {
    pub fn new(
        program: &JoltProgram,
        pre_regs: [u64; REGISTER_COUNT as usize],
    ) -> Result<Self, TraceError> {
        Ok(Self {
            compiled: CompiledProgram::compile(program)?,
            plane: MemoryPlane::new(MEM_CAPACITY as usize)?,
            entry: program.entry_address,
            pre_regs,
        })
    }

    /// Run once from the initial register state; returns the row count.
    pub fn run_once(&mut self) -> Result<u64, TraceError> {
        let mut host = HostContext {
            device: JoltDevice::new(&memory_config()),
            advice_tape: Vec::new(),
            advice_cursor: 0,
            helper_error: None,
        };
        let mut guest = Box::new(GuestState {
            x: self.pre_regs,
            pc: self.entry,
            trace_len: 0,
            exit: ExitReason::Running as u64,
            fault_addr: 0,
            mem_base: self.plane.base() as u64,
            mem_size: self.plane.size() as u64,
            host: &raw mut host,
            advice_slots: [0; crate::native::state::ADVICE_SLOTS],
            advice_jobs: self.compiled.advice_jobs_ptr(),
            row_limit: u64::MAX,
            obs_cursor: core::ptr::null_mut(),
            obs_end: core::ptr::null_mut(),
        });
        guest.x[0] = 0;
        self.compiled.run(&mut guest)?;
        if guest.exit != ExitReason::Terminated as u64 {
            return Err(TraceError::Backend("benchmark program did not terminate"));
        }
        Ok(guest.trace_len)
    }
}

pub struct Outcome {
    pub regs: [u64; REGISTER_COUNT as usize],
    pub pc: u64,
    pub trace_len: u64,
    /// Scratch region contents after the run, as dwords.
    pub scratch: Vec<u64>,
    pub advice_tape: Vec<u8>,
    pub exit: u64,
    /// Faulting guest address when `exit` reports a memory fault.
    pub fault_addr: u64,
    pub helper_error: Option<String>,
}

/// Compile and run a synthetic program natively.
///
/// `pre_regs` seeds the guest registers; `scratch` seeds
/// `[SCRATCH_START, SCRATCH_START + 8 * SCRATCH_DWORDS)`; `advice` seeds the
/// runtime advice tape (cursor at 0).
pub fn run_program(
    program: &JoltProgram,
    pre_regs: &[u64; REGISTER_COUNT as usize],
    scratch: &[u64],
    advice: &[u8],
) -> Result<Outcome, TraceError> {
    let compiled = CompiledProgram::compile(program)?;

    let plane = MemoryPlane::new(MEM_CAPACITY as usize)?;
    let base = plane.base();
    for (i, &dword) in scratch.iter().enumerate() {
        let offset = (SCRATCH_START - RAM_START_ADDRESS) as usize + i * 8;
        // SAFETY: scratch region is far inside MEM_CAPACITY.
        unsafe {
            base.add(offset)
                .cast::<u8>()
                .copy_from(dword.to_le_bytes().as_ptr(), 8);
        }
    }

    let mut host = HostContext {
        device: JoltDevice::new(&memory_config()),
        advice_tape: advice.to_vec(),
        advice_cursor: 0,
        helper_error: None,
    };
    let mut guest = Box::new(GuestState {
        x: *pre_regs,
        pc: program.entry_address,
        trace_len: 0,
        exit: ExitReason::Running as u64,
        fault_addr: 0,
        mem_base: plane.base() as u64,
        mem_size: plane.size() as u64,
        host: &raw mut host,
        advice_slots: [0; crate::native::state::ADVICE_SLOTS],
        advice_jobs: compiled.advice_jobs_ptr(),
        row_limit: u64::MAX,
        obs_cursor: core::ptr::null_mut(),
        obs_end: core::ptr::null_mut(),
    });
    guest.x[0] = 0;

    compiled.run(&mut guest)?;

    let mut scratch_out = vec![0u64; SCRATCH_DWORDS];
    for (i, slot) in scratch_out.iter_mut().enumerate() {
        let offset = (SCRATCH_START - RAM_START_ADDRESS) as usize + i * 8;
        let mut bytes = [0u8; 8];
        // SAFETY: scratch region is far inside MEM_CAPACITY.
        unsafe {
            base.add(offset).cast::<u8>().copy_to(bytes.as_mut_ptr(), 8);
        }
        *slot = u64::from_le_bytes(bytes);
    }

    Ok(Outcome {
        regs: guest.x,
        pc: guest.pc,
        trace_len: guest.trace_len,
        scratch: scratch_out,
        advice_tape: host.advice_tape,
        exit: guest.exit,
        fault_addr: guest.fault_addr,
        helper_error: host.helper_error,
    })
}
