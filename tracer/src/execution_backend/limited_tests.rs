#![expect(clippy::unwrap_used, reason = "test module")]

use std::collections::VecDeque;

use common::jolt_device::MemoryConfig;
use jolt_platform::JOLT_ADVICE_WRITE_CALL_ID;
use jolt_program::expand::{
    ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOperands,
};
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::RV64IMAC_JOLT_ALL_INLINES;
use jolt_riscv::{instructions::VirtualHostIO, InlineExtension, JoltInstructionKind as Kind};
#[cfg(feature = "field-inline")]
use jolt_riscv::{FieldInlineOp, RV64IMAC_JOLT_FIELD_INLINE};

use super::*;
use crate::emulator::cpu::PrivilegeMode;
use crate::emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder};
use crate::emulator::mmu::{AddressingMode, DRAM_BASE};
use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
use crate::instruction::format::format_inline::FormatInline;
use crate::instruction::inline::{
    InlineAdviceContext, InlineAdviceError, InlineRegistration, INLINE,
};
use crate::instruction::{Instruction, RISCVInstruction};
use crate::TracerInlineExpansionProvider;

fn guest(words: &[u32]) -> (JoltProgram, TraceInputs) {
    let elf = build_elf64(words, &[], StrtabOrder::GnuLd);
    #[cfg(feature = "field-inline")]
    let profile = RV64IMAC_JOLT_FIELD_INLINE;
    #[cfg(not(feature = "field-inline"))]
    let profile = RV64IMAC_JOLT_ALL_INLINES;
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut TracerInlineExpansionProvider::new(),
        profile,
    )
    .unwrap();
    let inputs = TraceInputs {
        memory_config: MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        },
        ..Default::default()
    };
    (program, inputs)
}

fn limited(limit: usize) -> TracerBackend {
    TracerBackend {
        elf_path: None,
        trace_allocation_limit: Some(limit),
    }
}

fn storage(limit: usize, cycles: Vec<Cycle>, rows: Vec<JoltTraceRow>) -> LimitedTrace {
    LimitedTrace {
        limit,
        resident: 0,
        cycles,
        rows,
        chunks: Vec::new(),
        fields: 0,
    }
}

fn rejection<T>(result: Result<T, TraceError>) -> Option<(TraceAllocationPhase, usize)> {
    match result {
        Err(TraceError::TraceAllocation(TraceAllocationError::LimitExceeded {
            phase,
            required_bytes,
            ..
        })) => Some((phase, required_bytes)),
        _ => None,
    }
}

fn no_advice(
    _operands: FormatInline,
    _cpu: &mut dyn InlineAdviceContext,
) -> Result<Option<VecDeque<u64>>, InlineAdviceError> {
    Ok(None)
}

/// A registered inline that fills the `u16` sequence length: these rows plus
/// one reset row.
fn longest_inline(
    mut asm: InlineExpansionBuilder,
    _operands: InlineOperands,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let register = asm.allocate_for_inline()?;
    for _ in 1..u16::MAX {
        asm.emit_i(Kind::ADDI, *register, 0, 1);
    }
    asm.release(register);
    asm.finalize()
}

/// A registered inline whose host advice write precedes a write of x5.
fn host_write_then_increment(
    mut asm: InlineExpansionBuilder,
    _operands: InlineOperands,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    asm.emit_i(Kind::VirtualHostIO(VirtualHostIO(())), 0, 0, 0);
    asm.emit_i(Kind::ADDI, 5, 5, 1);
    asm.finalize()
}

// custom-1, funct3 6, rd x5; funct7 0x7d and 0x7c.
const LONGEST_INLINE_WORD: u32 = 0xfa00_62ab;
const HOST_WRITE_INLINE_WORD: u32 = 0xf800_62ab;

inventory::submit! {
    InlineRegistration {
        opcode: 0x2b,
        funct3: 0x6,
        funct7: 0x7d,
        extension: InlineExtension::Sha2,
        name: "TEST_LONGEST_INLINE",
        build_sequence: longest_inline,
        build_advice: no_advice,
    }
}

inventory::submit! {
    InlineRegistration {
        opcode: 0x2b,
        funct3: 0x6,
        funct7: 0x7c,
        extension: InlineExtension::Sha2,
        name: "TEST_HOST_WRITE_INLINE",
        build_sequence: host_write_then_increment,
        build_advice: no_advice,
    }
}

/// The setup check precedes every setup request: an unparsable ELF would
/// panic in emulator setup. Its bound is exact, and it charges at least the
/// flat memory, the initial row reserve and the pending window.
#[test]
fn setup_is_checked_before_any_setup_request() {
    let program = JoltProgram::from_elf_bytes(vec![0; 64]);
    let inputs = TraceInputs {
        memory_config: MemoryConfig {
            program_size: Some(64),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        rejection(limited(1 << 20).trace(&program, inputs)),
        Some((TraceAllocationPhase::Initial, _))
    ));

    // addi x1, x0, 1 ; j .
    let (program, inputs) = guest(&[0x0010_0093, 0x0000_006f]);
    let (_, required) = rejection(limited(0).trace(&program, inputs.clone())).unwrap();
    let layout = MemoryLayout::try_new(&inputs.memory_config).unwrap();
    let floor = layout.get_total_memory_size() as usize
        + crate::trace_capacity_reserve() * size_of::<JoltTraceRow>()
        + pending_window(SERIAL_BATCH_ROWS).unwrap() * size_of::<Cycle>();
    assert!(required >= floor, "{required} < {floor}");
    assert_eq!(
        rejection(limited(required - 1).trace(&program, inputs.clone())),
        Some((TraceAllocationPhase::Initial, required))
    );
    let mut backend = limited(required);
    assert!(!backend
        .trace(&program, inputs.clone())
        .unwrap()
        .trace
        .rows()
        .is_empty());
    assert!(matches!(
        backend.execute(&program, inputs, 100),
        Err(TraceError::UnsupportedTraceAllocationMode)
    ));
}

/// The pending window has room for the longest legal expansion on top of a
/// batch that is one cycle short of converting.
#[test]
fn pending_window_holds_the_longest_tick_without_growing() {
    let batch_rows = 2;
    let window = pending_window(batch_rows).unwrap();
    let mut cycles = Vec::with_capacity(window);
    cycles.resize(batch_rows - 1, Cycle::NoOp);
    let buffer = (cycles.as_ptr(), cycles.capacity());
    let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
    Instruction::from(INLINE::new(LONGEST_INLINE_WORD, 0x8000_0000, false, false))
        .trace(&mut cpu, Some(&mut cycles));
    assert_eq!(cycles.len(), window);
    assert_eq!((cycles.as_ptr(), cycles.capacity()), buffer);
}

/// Row growth is charged with the old and new row buffers, the held chunk
/// metadata and the pending cycles live together; chunk-metadata growth with
/// its old and new buffers, and a first metadata buffer, likewise, each before
/// it is allocated. A rejection changes nothing.
#[test]
fn conversion_growth_is_checked_before_it_is_made() {
    let row = size_of::<JoltTraceRow>();
    let chunk = size_of::<ConvertedChunk>();
    let bytecode = BytecodePCMapper::try_new(&[]).unwrap();
    let cycles = vec![Cycle::NoOp; 12];
    let pending = heap_bytes(&cycles);

    // The new 16 rows beside the old 4 and the chunk record already held.
    let grown = pending + 20 * row + chunk;
    let mut trace = storage(grown - 1, cycles.clone(), vec![JoltTraceRow::default(); 4]);
    trace.chunks = vec![ConvertedChunk::default()];
    assert_eq!(trace.chunks.capacity(), 1);
    assert_eq!(
        rejection(trace.convert_pending(&bytecode, 0)),
        Some((TraceAllocationPhase::RowGrowth, grown))
    );
    assert_eq!((trace.rows.len(), trace.rows.capacity()), (4, 4));
    assert_eq!((trace.chunks.capacity(), trace.cycles.len()), (1, 12));
    trace.limit = grown;
    trace.convert_pending(&bytecode, 0).unwrap();
    assert_eq!((trace.rows.len(), trace.rows.capacity()), (16, 16));
    assert_eq!((trace.chunks.len(), trace.cycles.len()), (2, 0));

    // Two new chunk records beside the one they replace; rows do not grow.
    let replaced = pending + 12 * row + 3 * chunk;
    let mut trace = storage(replaced - 1, cycles.clone(), Vec::with_capacity(12));
    trace.chunks = vec![ConvertedChunk::default()];
    assert_eq!(
        rejection(trace.convert_pending(&bytecode, 0)),
        Some((TraceAllocationPhase::Conversion, replaced))
    );
    assert_eq!((trace.rows.capacity(), trace.chunks.capacity()), (12, 1));
    trace.limit = replaced;
    trace.convert_pending(&bytecode, 0).unwrap();
    assert_eq!((trace.chunks.len(), trace.chunks.capacity()), (2, 2));

    let mut trace = storage(0, cycles, Vec::with_capacity(12));
    let held = pending + 12 * row;
    trace.limit = held + chunk - 1;
    assert_eq!(
        rejection(trace.convert_pending(&bytecode, 0)),
        Some((TraceAllocationPhase::Conversion, held + chunk))
    );
    assert_eq!((trace.rows.len(), trace.chunks.capacity()), (0, 0));
    trace.limit = held + chunk;
    trace.convert_pending(&bytecode, 0).unwrap();
    assert_eq!((trace.rows.len(), trace.chunks.len()), (12, 1));
}

/// `len` cycles with a field-inline event at each of `positions`.
#[cfg(feature = "field-inline")]
fn field_inline_cycles(positions: &[usize], len: usize) -> (Vec<Cycle>, BytecodePCMapper) {
    let instruction = Instruction::decode(
        super::tests::field_inline_word(FieldInlineOp::LoadAccumulateFromRegister, 2, 5, 0),
        0x8000_0000,
        false,
    )
    .unwrap();
    let bytecode =
        BytecodePCMapper::try_new(&[instruction.try_jolt_instruction_row().unwrap()]).unwrap();
    let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
    cpu.write_register(5, 11);
    let mut cycles = vec![Cycle::NoOp; len];
    for &position in positions {
        let mut captured = Vec::new();
        instruction.trace(&mut cpu, Some(&mut captured));
        cycles[position] = captured[0];
    }
    (cycles, bytecode)
}

/// Every conversion slice's field-event buffer is admitted together with
/// the chunk metadata, before any worker allocates one.
#[cfg(feature = "field-inline")]
#[test]
fn conversion_admits_every_slice_field_buffer_together() {
    let threshold = PARALLEL_ROW_CONVERSION_THRESHOLD;
    let (cycles, bytecode) = field_inline_cycles(&[0, threshold], threshold + 1);
    let event = size_of::<FieldEvent>();
    let held = heap_bytes(&cycles) + (threshold + 1) * size_of::<JoltTraceRow>();
    let both = held + 2 * size_of::<ConvertedChunk>() + 2 * event;
    let mut trace = storage(both - event, cycles, Vec::with_capacity(threshold + 1));
    assert_eq!(
        rejection(trace.convert_pending(&bytecode, 0)),
        Some((TraceAllocationPhase::Conversion, both))
    );
    trace.limit = both;
    trace.convert_pending(&bytecode, 0).unwrap();
    let lengths: Vec<_> = trace
        .chunks
        .iter()
        .map(|chunk| chunk.fields.len())
        .collect();
    assert_eq!(lengths, [1, 1]);
    assert_eq!(trace.fields, 2 * event);
    assert_eq!(
        (trace.rows.len(), trace.rows.capacity()),
        (threshold + 1, threshold + 1)
    );
}

/// The merged field-event buffer is charged beside its source buffers, the
/// pending window, the rows and the chunk metadata.
#[cfg(feature = "field-inline")]
#[test]
fn field_merge_is_checked_beside_its_sources() {
    let threshold = PARALLEL_ROW_CONVERSION_THRESHOLD;
    let (cycles, bytecode) = field_inline_cycles(&[0, threshold], threshold + 1);
    let mut memory = Memory::empty();
    memory.init(64);
    let converted = |limit| {
        let mut trace = storage(
            usize::MAX,
            cycles.clone(),
            Vec::with_capacity(threshold + 1),
        );
        trace.convert_pending(&bytecode, 0).unwrap();
        LimitedTrace { limit, ..trace }
    };
    let trace = converted(0);
    let sources = 2 * size_of::<FieldEvent>();
    assert_eq!(trace.fields, sources);
    let needed = heap_bytes(&trace.cycles)
        + heap_bytes(&trace.rows)
        + heap_bytes(&trace.chunks)
        + 2 * sources;
    assert_eq!(
        rejection(converted(needed - 1).finish(&memory, 0)),
        Some((TraceAllocationPhase::FieldMerge, needed))
    );
    let (data, _, _) = converted(needed).finish(&memory, 0).unwrap();
    let events: Vec<_> = data
        .field_events()
        .iter()
        .map(|event| event.cycle)
        .collect();
    assert_eq!(events, [0, threshold]);
}

/// A guest loop of a field-inline load (a no-op in other builds), an unsigned
/// division that expands into several rows, and a store: its limited trace
/// equals its original trace in rows, field events, dimensions, device, final
/// image and advice tape. Every division's tick overruns a three-row batch,
/// and the other batch spans two parallel conversion slices.
#[test]
fn limited_trace_matches_the_original_trace() {
    #[cfg(feature = "field-inline")]
    let field_load =
        super::tests::field_inline_word(FieldInlineOp::LoadAccumulateFromRegister, 2, 5, 0);
    #[cfg(not(feature = "field-inline"))]
    let field_load = 0x0000_0013; // addi x0, x0, 0
    let (program, inputs) = guest(&[
        0x00b0_0293, // addi x5, x0, 11
        0x0000_4337, // lui x6, 4: 16384 iterations
        0x0001_0397, // auipc x7, 0x10
        0x0030_0493, // addi x9, x0, 3
        field_load,
        0x0293_5433, // divu x8, x6, x9
        0x0083_b023, // sd x8, 0(x7)
        0x0083_8393, // addi x7, x7, 8
        0xfff3_0313, // addi x6, x6, -1
        0xfe03_16e3, // bne x6, x0, -20
        0x0000_006f, // j .
    ]);
    let inputs = inputs.with_advice_tape(Some(vec![1, 2, 3]));
    let original = TracerBackend::new()
        .trace(&program, inputs.clone())
        .unwrap();
    let rows = original.trace.rows();
    let wide = 2 * PARALLEL_ROW_CONVERSION_THRESHOLD - 100;
    assert!(rows.len() > 4 * wide);
    for batch_rows in [3, wide] {
        let limited = TracerBackend::new()
            .trace_limited(&program, inputs.clone(), usize::MAX, batch_rows)
            .unwrap();
        assert_eq!(limited.trace.rows(), rows);
        assert_eq!(limited.dimensions, original.dimensions);
        assert_eq!(limited.device, original.device);
        assert_eq!(limited.final_memory, original.final_memory);
        assert_eq!(limited.advice_tape, original.advice_tape);
        #[cfg(feature = "field-inline")]
        {
            let events = |output: &TraceOutput| {
                output
                    .trace
                    .field_events()
                    .iter()
                    .map(|event| (event.cycle, event.data))
                    .collect::<Vec<_>>()
            };
            assert_eq!(events(&limited), events(&original));
        }
    }
    // Events fall in the first batch's second slice and in later batches.
    #[cfg(feature = "field-inline")]
    {
        let events = original.trace.field_events();
        assert!(events
            .iter()
            .any(|event| (PARALLEL_ROW_CONVERSION_THRESHOLD..wide).contains(&event.cycle)));
        assert!(events.last().unwrap().cycle > 4 * wide);
    }
}

/// Within one tick, a registered inline's rows after a refused advice write do
/// not run, in trace and in execute mode; without a limit both rows run.
#[test]
fn refused_advice_write_stops_its_inline() {
    let inline = Instruction::from(INLINE::new(
        HOST_WRITE_INLINE_WORD,
        0x8000_0000,
        false,
        false,
    ));
    let writing_cpu = |limit: Option<usize>| {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.get_mut_mmu().init_memory(64);
        cpu.x[10] = i64::from(JOLT_ADVICE_WRITE_CALL_ID);
        cpu.x[11] = 0x8000_0000;
        cpu.x[12] = 8;
        if let Some(limit) = limit {
            cpu.set_advice_write_allowance(TraceAllocationAllowance::new(limit, 0));
        }
        cpu
    };
    // 8 scratch bytes and a new 8-byte tape.
    let refusal = Some(TraceAllocationError::LimitExceeded {
        phase: TraceAllocationPhase::AdviceWrite,
        limit_bytes: 15,
        required_bytes: 16,
    });

    let mut cpu = writing_cpu(Some(15));
    let mut cycles = Vec::new();
    inline.trace(&mut cpu, Some(&mut cycles));
    assert_eq!(cpu.advice_write_refusal(), refusal);
    assert_eq!((cpu.x[5], cycles.len(), cpu.advice_tape.len()), (0, 1, 0));

    let mut cpu = writing_cpu(Some(15));
    inline.execute(&mut cpu);
    assert_eq!(cpu.advice_write_refusal(), refusal);
    assert_eq!((cpu.x[5], cpu.trace_len, cpu.advice_tape.len()), (0, 1, 0));

    let mut cpu = writing_cpu(None);
    let mut cycles = Vec::new();
    inline.trace(&mut cpu, Some(&mut cycles));
    assert_eq!((cpu.x[5], cycles.len(), cpu.advice_tape.len()), (1, 2, 8));
}

/// A guest that writes 24 bytes of its own text to the advice tape, then
/// 64 KiB through the registered inline above, then 7 bytes, and stalls.
fn advice_writing_guest() -> (JoltProgram, TraceInputs) {
    guest(&[
        0x00ad_c537, // lui a0, 0xadc
        0xabe5_0513, // addi a0, a0, -0x542: the advice-write call
        0x0000_0597, // auipc a1, 0
        0x0180_0613, // addi a2, x0, 24
        0x0000_205b, // host I/O
        0x0001_0637, // lui a2, 0x10
        HOST_WRITE_INLINE_WORD,
        0x0070_0613, // addi a2, x0, 7
        0x0000_205b, // host I/O
        0x0000_006f, // j .
    ])
}

/// The tape-only run, which requests no trace storage, writes the tape the
/// original and the limited traces write, through several tape growths and
/// to the PC stall.
#[test]
fn tape_only_run_writes_the_traced_tape() {
    assert_eq!(JOLT_ADVICE_WRITE_CALL_ID, 0x00ad_babe);
    let (program, inputs) = advice_writing_guest();
    let original = TracerBackend::new()
        .trace(&program, inputs.clone())
        .unwrap();
    let tape = original.advice_tape.unwrap();
    assert_eq!(tape.len(), 24 + (1 << 16) + 7);
    let limited_trace = limited(usize::MAX).trace(&program, inputs.clone()).unwrap();
    assert_eq!(limited_trace.advice_tape.unwrap(), tape);
    let tape_only = TracerBackend::execute_advice_tape(&program, inputs, usize::MAX).unwrap();
    assert_eq!(tape_only, tape);
}

/// With each run's setup requirement as its limit, the 24-byte write fits
/// and the registered inline's 64 KiB write is refused: the limited trace and
/// the tape-only run return that error instead of running on to the stall
/// with a shorter tape. The tape-only setup requests no row reserve.
#[test]
fn refused_advice_write_fails_the_run() {
    let (program, inputs) = advice_writing_guest();
    let (_, setup) = rejection(limited(0).trace(&program, inputs.clone())).unwrap();
    let (phase, required) = rejection(limited(setup).trace(&program, inputs.clone())).unwrap();
    assert_eq!(phase, TraceAllocationPhase::AdviceWrite);
    assert!(required > setup + (1 << 16), "{required} {setup}");

    let tape_only = |limit| TracerBackend::execute_advice_tape(&program, inputs.clone(), limit);
    let (phase, setup) = rejection(tape_only(0)).unwrap();
    assert_eq!(phase, TraceAllocationPhase::Initial);
    assert!(setup < crate::trace_capacity_reserve() * size_of::<JoltTraceRow>());
    let (phase, required) = rejection(tape_only(setup)).unwrap();
    assert_eq!(phase, TraceAllocationPhase::AdviceWrite);
    assert!(required > setup + (1 << 16), "{required} {setup}");
}

/// The tape-only run stops where a trace stops: [`PassOne::step`], which it
/// runs, ends at the first tick that emits no row and never ticks again. No
/// guest can make such a tick (it needs paging, and satp writes do not
/// decode), so paging over a page table of zeros forces an instruction page
/// fault.
#[test]
fn tape_only_stop_ends_at_a_zero_row_tick() {
    let (program, inputs) = advice_writing_guest();
    let mut emulator = crate::create_emulator(
        program.elf_bytes(),
        None,
        &inputs.inputs,
        &inputs.untrusted_advice,
        &inputs.trusted_advice,
        &inputs.memory_config,
        None,
    );
    let mmu = emulator.get_mut_cpu().get_mut_mmu();
    mmu.update_addressing_mode(AddressingMode::SV39);
    mmu.update_privilege_mode(PrivilegeMode::User);
    mmu.update_ppn((DRAM_BASE + (1 << 16)) >> 12);
    let mut pass = PassOne::new(emulator);
    assert!(!pass.step());
    assert!(!pass.step());
    assert_eq!(pass.rows(), 0);
}
