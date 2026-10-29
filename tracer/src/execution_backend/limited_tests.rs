#![expect(clippy::unwrap_used, reason = "test module")]

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_platform::JOLT_ADVICE_WRITE_CALL_ID;
use jolt_program::expand::{
    ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOperands,
};
#[cfg(feature = "field-inline")]
use jolt_program::field_inline::FieldEncodedValue;
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::RV64IMAC_JOLT_ALL_INLINES;
use jolt_riscv::{
    instructions::VirtualHostIO, InlineExtension, JoltInstructionKind as Kind,
    JoltInstructionProfile, NormalizedOperands,
};
#[cfg(feature = "field-inline")]
use jolt_riscv::{FieldInlineOp, FIELD_REGISTER_COUNT, RV64IMAC_JOLT_FIELD_INLINE};
use rand::{rngs::StdRng, Rng, SeedableRng};
use strum::IntoEnumIterator;

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

#[cfg(feature = "field-inline")]
const PROFILE: JoltInstructionProfile = RV64IMAC_JOLT_FIELD_INLINE;
#[cfg(not(feature = "field-inline"))]
const PROFILE: JoltInstructionProfile = RV64IMAC_JOLT_ALL_INLINES;

fn guest(words: &[u32]) -> (JoltProgram, TraceInputs) {
    let elf = build_elf64(words, &[], StrtabOrder::GnuLd);
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut TracerInlineExpansionProvider::new(),
        PROFILE,
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

const ADDI: u32 = 0x0010_0093; // addi x1, x0, 1
const STALL: u32 = 0x0000_006f; // j .

/// Sets the initial cycle and row reserve of this test's process; nextest
/// runs each test in its own.
fn reserve_rows(rows: usize) {
    std::env::set_var("JOLT_TRACER_CAPACITY_ROWS", rows.to_string());
}

/// What a limited trace of a guest without input buffers holds after setup
/// besides its trace storage: the setup a refusal at limit 0 reports, less
/// its row reserve, its pending window and the ELF copy setup frees.
fn setup_bytes(program: &JoltProgram, inputs: &TraceInputs) -> usize {
    let (_, required) = rejection(limited(0).trace(program, inputs.clone())).unwrap();
    required
        - crate::trace_capacity_reserve() * size_of::<JoltTraceRow>()
        - pending_window(SERIAL_BATCH_ROWS).unwrap() * size_of::<Cycle>()
        - program.elf_bytes().len()
}

/// The row capacity of a limited trace, whose merged field events fill their
/// buffer.
fn row_capacity(output: &TraceOutput) -> usize {
    #[cfg(feature = "field-inline")]
    let events = size_of_val(output.trace.field_events());
    #[cfg(not(feature = "field-inline"))]
    let events = 0;
    (output.trace.heap_bytes() - events) / size_of::<JoltTraceRow>()
}

fn no_advice(
    _operands: FormatInline,
    _cpu: &mut dyn InlineAdviceContext,
) -> Result<Option<VecDeque<u64>>, InlineAdviceError> {
    Ok(None)
}

/// A registered inline of `ROWS` rows: these increments plus one reset row.
/// `u16::MAX` rows fill the sequence length.
fn addi_inline<const ROWS: u16>(
    mut asm: InlineExpansionBuilder,
    _operands: InlineOperands,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let register = asm.allocate_for_inline()?;
    for _ in 1..ROWS {
        asm.emit_i(Kind::ADDI, *register, 0, 1);
    }
    asm.release(register);
    asm.finalize()
}

/// A registered inline of two rows when first expanded and `u16::MAX` rows
/// after: outside the text, where no expansion is cached, its tick emits
/// more rows than its peek counted.
fn growing_inline(
    asm: InlineExpansionBuilder,
    operands: InlineOperands,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    static EXPANDED: AtomicBool = AtomicBool::new(false);
    if EXPANDED.swap(true, Ordering::Relaxed) {
        addi_inline::<{ u16::MAX }>(asm, operands)
    } else {
        addi_inline::<2>(asm, operands)
    }
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

// custom-1, funct3 6, rd x5; funct7 0x7d, 0x7c, 0x7a and 0x7b.
const LONGEST_INLINE_WORD: u32 = 0xfa00_62ab;
const HOST_WRITE_INLINE_WORD: u32 = 0xf800_62ab;
const MID_INLINE_WORD: u32 = 0xf400_62ab;
const MID_ROWS: u16 = 1 << 14;

inventory::submit! {
    InlineRegistration {
        opcode: 0x2b,
        funct3: 0x6,
        funct7: 0x7d,
        extension: InlineExtension::Sha2,
        name: "TEST_LONGEST_INLINE",
        build_sequence: addi_inline::<{ u16::MAX }>,
        build_advice: no_advice,
    }
}

inventory::submit! {
    InlineRegistration {
        opcode: 0x2b,
        funct3: 0x6,
        funct7: 0x7a,
        extension: InlineExtension::Sha2,
        name: "TEST_MID_INLINE",
        build_sequence: addi_inline::<MID_ROWS>,
        build_advice: no_advice,
    }
}

inventory::submit! {
    InlineRegistration {
        opcode: 0x2b,
        funct3: 0x6,
        funct7: 0x7b,
        extension: InlineExtension::Sha2,
        name: "TEST_GROWING_INLINE",
        build_sequence: growing_inline,
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
    let bytecode = BytecodePreprocessing::preprocess(Vec::new(), 0x8000_0000, PROFILE).unwrap();
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
fn field_inline_cycles(positions: &[usize], len: usize) -> (Vec<Cycle>, BytecodePreprocessing) {
    let instruction = Instruction::decode(
        super::tests::field_inline_word(FieldInlineOp::LoadAccumulateFromRegister, 2, 5, 0),
        0x8000_0000,
        false,
    )
    .unwrap();
    let bytecode = BytecodePreprocessing::preprocess(
        vec![instruction.try_jolt_instruction_row().unwrap()],
        0x8000_0000,
        PROFILE,
    )
    .unwrap();
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
        rejection(converted(needed - 1).finish(FinalImage::Flat(memory.flat_image()), 0)),
        Some((TraceAllocationPhase::FieldMerge, needed))
    );
    let (data, _, _) = converted(needed)
        .finish(FinalImage::Flat(memory.flat_image()), 0)
        .unwrap();
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
/// image and advice tape. It does so batching from the start, below the
/// default reserve's cycles, where every division's tick overruns a
/// three-row batch and the other batch spans two parallel conversion slices;
/// converting once, at the end; and, with a 2^20-cycle reserve whose
/// conversion fits for half of the run, batching from there.
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
    let matches = |limited: &TraceOutput| {
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
            assert_eq!(events(limited), events(&original));
        }
    };
    let (cycle, row) = (size_of::<Cycle>(), size_of::<JoltTraceRow>());
    let setup = setup_bytes(&program, &inputs);
    let reserve = crate::trace_capacity_reserve();
    for batch_rows in [3, wide] {
        let window = pending_window(batch_rows).unwrap();
        let limit = setup + reserve * row + window * cycle + (64 << 20);
        assert!(limit < setup + reserve * cycle);
        let limited = TracerBackend::new()
            .trace_limited(&program, inputs.clone(), limit, batch_rows)
            .unwrap();
        matches(&limited);
        assert_eq!(row_capacity(&limited), reserve);
    }
    let eager = limited(usize::MAX).trace(&program, inputs.clone()).unwrap();
    matches(&eager);
    assert_eq!(row_capacity(&eager), rows.len());
    reserve_rows(1 << 20);
    let setup = setup_bytes(&program, &inputs);
    let limit = setup + (cycle << 20) + rows.len() / 2 * row;
    assert!(limit < setup + (row << 20) + pending_window(SERIAL_BATCH_ROWS).unwrap() * cycle);
    let halfway = limited(limit).trace(&program, inputs.clone()).unwrap();
    matches(&halfway);
    assert_eq!(row_capacity(&halfway), 1 << 20);
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

/// The first tick commits to accumulating by its own emission, not by the
/// longest tick's. With a 2^15-cycle reserve, below the longest tick, the
/// guests Y = [addi; j .; MID] and X = [MID; j .; addi], MID a 2^14-row
/// inline, hold the same three words and so have equal setups. At the least
/// limit at which Y runs, below the batched setup, Y keeps exact rows and X,
/// whose first tick is MID, fails.
#[test]
fn exact_first_tick_admits_the_accumulation() {
    reserve_rows(1 << 15);
    assert!(usize::from(MID_ROWS) < 1 << 15 && 1 << 15 < MAX_EXPANDED_ROWS_PER_SOURCE);
    let y = guest(&[ADDI, STALL, MID_INLINE_WORD]);
    let x = guest(&[MID_INLINE_WORD, STALL, ADDI]);
    assert_eq!(y.0.elf_bytes().len(), x.0.elf_bytes().len());
    let run = |(program, inputs): &(JoltProgram, TraceInputs), limit| {
        limited(limit).trace(program, inputs.clone())
    };
    let (_, batched) = rejection(run(&y, 0)).unwrap();
    assert_eq!(rejection(run(&x, 0)).unwrap().1, batched);
    let (mut refused, mut least) = (0, batched);
    assert!(run(&y, least).is_ok());
    while least - refused > 1 {
        let middle = refused + (least - refused) / 2;
        if run(&y, middle).is_ok() {
            least = middle;
        } else {
            refused = middle;
        }
    }
    assert!(least < batched);
    let output = run(&y, least).unwrap();
    let original = TracerBackend::new().trace(&y.0, y.1.clone()).unwrap();
    assert_eq!(output.trace.rows(), original.trace.rows());
    assert_eq!(row_capacity(&output), output.trace.rows().len());
    assert!(matches!(
        run(&x, least),
        Err(TraceError::TraceAllocation(_))
    ));
    assert!(run(&x, batched).is_ok());
}

/// A run that accumulates a 2^14-row inline and must convert it before an
/// equal second tick continues after the conversion: it keeps every row of
/// the original trace, in rows grown to the 2^18-row reserve once the
/// accumulated cycles drop.
#[test]
fn transition_before_an_equal_tick_continues_the_run() {
    reserve_rows(1 << 18);
    let (program, inputs) = guest(&[MID_INLINE_WORD, MID_INLINE_WORD, ADDI, STALL]);
    let original = TracerBackend::new()
        .trace(&program, inputs.clone())
        .unwrap();
    let mid = usize::from(MID_ROWS);
    // Converting the first inline fits beside the reserved cycles; both do not.
    let limit = setup_bytes(&program, &inputs)
        + (size_of::<Cycle>() << 18)
        + mid * size_of::<JoltTraceRow>()
        + (64 << 10);
    let output = limited(limit).trace(&program, inputs).unwrap();
    assert_eq!(output.trace.rows(), original.trace.rows());
    assert!(output.trace.rows().len() > 2 * mid);
    assert_eq!(row_capacity(&output), 1 << 18);
}

/// A guest of `rows` one-row ticks, at least four: a countdown loop, an
/// increment where `rows` is even, and a self-jump that writes x1.
fn counted_guest(rows: usize) -> (JoltProgram, TraceInputs) {
    let pad = (rows - 3) % 2;
    let count = (rows - 3 - pad) / 2;
    let high = (count + 0x800) >> 12;
    let low = count.wrapping_sub(high << 12) as u32 & 0xfff;
    let mut words = vec![
        ((high as u32) << 12) | 0x2b7, // lui x5, high
        (low << 20) | 0x2_8293,        // addi x5, x5, low
        0xfff2_8293,                   // addi x5, x5, -1
        0xfe02_9ee3,                   // bne x5, x0, -4
    ];
    words.extend([ADDI].repeat(pad));
    words.push(0x0000_00ef); // jal x1, 0
    guest(&words)
}

/// Ending ticks are peeked, not bounded by the longest tick, and a grown
/// buffer converts what it holds. With a 2^16-cycle reserve, a run that
/// fills it, its PC stall peeked as no row, keeps exact rows one byte below
/// the reserve's growth, its old and new buffers together; a run two rows
/// past it grows once and keeps exact rows within 1 MiB above that growth,
/// below converting a full doubled buffer.
#[test]
fn ending_ticks_neither_grow_nor_collect_ahead() {
    let capacity = 1 << 16;
    reserve_rows(capacity);
    let growth = 3 * capacity * size_of::<Cycle>();
    for (rows, room) in [(capacity, 0), (capacity + 2, 1 << 20)] {
        let (program, inputs) = counted_guest(rows);
        let limit = setup_bytes(&program, &inputs) + growth + room - 1;
        let output = limited(limit).trace(&program, inputs).unwrap();
        assert_eq!(
            (output.trace.rows().len(), row_capacity(&output)),
            (rows, rows)
        );
    }
}

/// A 100_000-cycle reserve grows from its observed capacity by
/// `growth_target`, to 200_000 and then 400_000 cycles, as longest inlines
/// overrun it, and row anchors double from it. With room for the second
/// growth the run keeps exact rows; one byte short of it, the run converts
/// before the fourth inline into rows anchored at 400_000, not the 262_140
/// it needs.
#[test]
fn growth_and_anchors_start_from_an_irregular_reserve() {
    let initial = 100_000;
    reserve_rows(initial);
    let (program, inputs) = guest(&[[LONGEST_INLINE_WORD; 4].as_slice(), &[STALL]].concat());
    let original = TracerBackend::new()
        .trace(&program, inputs.clone())
        .unwrap();
    let rows = original.trace.rows().len();
    let growth = setup_bytes(&program, &inputs) + 6 * initial * size_of::<Cycle>();
    for (limit, capacity) in [(growth + (1 << 20), rows), (growth - 1, 4 * initial)] {
        let output = limited(limit).trace(&program, inputs.clone()).unwrap();
        assert_eq!(output.trace.rows(), original.trace.rows());
        assert_eq!(row_capacity(&output), capacity);
    }
}

/// A refused advice write fails the run at the tick that made it, before the
/// run converts for the next tick, refreshes the allowance for it, or ends
/// after a registered inline the refusal cut short. Each guest copies 1 MiB
/// of itself to the tape where its accumulated ticks fit with half of that
/// to spare; without a limit, each writes it all, as the tape-only run shows
/// (a trace that converts the registered inline's host I/O row fails main's
/// bytecode check; see [`advice_writing_guest`]). A write may not take the
/// conversion its tick reserved either: the last guest's write would fit
/// beside the longest tick's conversion, which the limit funds, but not
/// within it.
#[test]
fn refused_advice_write_outlives_the_next_tick() {
    reserve_rows(1 << 18);
    // The advice-write call: a0 its id, a1 this code, a2 1 MiB.
    let call = [0x00ad_c537, 0xabe5_0513, 0x0000_0597, 0x0010_0637];
    let host_io = 0x0000_205b;
    let longest = (size_of::<JoltTraceRow>() + FIELD_EVENT_BYTES) * MAX_EXPANDED_ROWS_PER_SOURCE;
    for (tail, room) in [
        ([host_io, MID_INLINE_WORD], 512 << 10),
        ([host_io, ADDI], 512 << 10),
        ([HOST_WRITE_INLINE_WORD, STALL], 512 << 10),
        ([host_io, STALL], longest + (1 << 20)),
    ] {
        let (program, inputs) = guest(&[&call[..], &tail, &[STALL]].concat());
        let unlimited = TracerBackend::execute_advice_tape(&program, inputs.clone(), usize::MAX);
        assert_eq!(unlimited.unwrap().len(), 1 << 20);
        let limit = setup_bytes(&program, &inputs) + (size_of::<Cycle>() << 18) + room;
        let refusal = rejection(limited(limit).trace(&program, inputs));
        assert_eq!(
            refusal.map(|(phase, _)| phase),
            Some(TraceAllocationPhase::AdviceWrite)
        );
    }
}

/// A tick that emits more cycles than its peek counted fails the run instead
/// of growing the cycle buffer unchecked: the guest stores the growing
/// inline past its text, where no expansion is cached, and jumps to it.
#[test]
fn uncounted_cycles_fail_the_run() {
    reserve_rows(1 << 15);
    let (program, inputs) = guest(&[
        0x0000_1317, // auipc x6, 1
        0xf600_63b7, // lui x7, 0xf6006
        0x2ab3_8393, // addi x7, x7, 0x2ab: x7 holds the growing inline, 0xf600_62ab
        0x0073_2023, // sw x7, 0(x6)
        0x0003_0067, // jr x6
    ]);
    assert!(matches!(
        limited(usize::MAX).trace(&program, inputs),
        Err(TraceError::UncountedCycles { capacity }) if capacity == 1 << 15
    ));
}

/// The transition converts the accumulated cycles into rows grown to the
/// anchor and leaves a one-tick window, a batch of one cycle. A plan whose
/// row growth does not fit holds a smaller batch and one tick within the
/// rows, and a limit below one tick refuses the window.
#[test]
fn planned_windows_hold_their_batch_and_one_tick() {
    let bytecode = BytecodePreprocessing::preprocess(Vec::new(), 0x8000_0000, PROFILE).unwrap();
    // A CPU at its PC stall, whose next tick emits no row.
    let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
    cpu.get_mut_mmu().init_memory(64);
    cpu.get_mut_mmu().store_word(DRAM_BASE, STALL).unwrap();
    cpu.update_pc(DRAM_BASE);
    cpu.tick(None);
    let mut trace = storage(usize::MAX, vec![Cycle::NoOp; 100], Vec::new());
    trace.start_batches(&bytecode, 0, 1 << 18, 0).unwrap();
    assert_eq!((trace.rows.len(), trace.rows.capacity()), (100, 1 << 18));
    assert_eq!(trace.cycles.capacity(), MAX_EXPANDED_ROWS_PER_SOURCE);
    let held = heap_bytes(&trace.rows) + heap_bytes(&trace.chunks);
    trace.limit = held + ((size_of::<Cycle>() + FIELD_EVENT_BYTES) << 17);
    let batch = trace.plan_batch(&mut cpu).unwrap();
    assert!(1 < batch && batch < SERIAL_BATCH_ROWS, "{batch}");
    assert_eq!(trace.cycles.capacity(), pending_window(batch).unwrap());
    assert!(trace.rows.len() + trace.cycles.capacity() <= trace.rows.capacity());
    let window = MAX_EXPANDED_ROWS_PER_SOURCE * size_of::<Cycle>();
    trace.limit = held + window - 1;
    assert_eq!(
        rejection(trace.plan_batch(&mut cpu)),
        Some((TraceAllocationPhase::CycleBuffer, held + window))
    );
}

/// Rows that a one-tick batch overruns grow before its window, not beside it.
/// With a one-cycle reserve, a longest inline converts into rows anchored at
/// 2^16 before the first addi, which fills them; the second addi needs them
/// doubled. The limit holds the doubled rows beside the old rows or beside
/// the window, but not beside both.
#[test]
fn one_tick_rows_grow_before_its_window() {
    reserve_rows(1);
    let (program, inputs) = guest(&[LONGEST_INLINE_WORD, ADDI, ADDI, STALL]);
    let original = TracerBackend::new()
        .trace(&program, inputs.clone())
        .unwrap();
    let window = MAX_EXPANDED_ROWS_PER_SOURCE * size_of::<Cycle>();
    let limit = setup_bytes(&program, &inputs) + (10 << 20) + window;
    let output = limited(limit).trace(&program, inputs).unwrap();
    assert_eq!(output.trace.rows(), original.trace.rows());
    assert_eq!(row_capacity(&output), 1 << 17);
}

/// An accumulated trace ends image first, as the original trace does, where
/// the image fits beside the flat memory and the cycles and the conversion
/// then fits with the flat memory released: there the rows first would not
/// fit. Otherwise it ends rows first, the cycles dropped before the field
/// merge and the image last, beside the rows instead of the cycles.
#[test]
fn accumulated_trace_ends_image_first_where_it_fits() {
    let bytecode = BytecodePreprocessing::preprocess(Vec::new(), 0x8000_0000, PROFILE).unwrap();
    let final_memory = |bytes| {
        let mut memory = Memory::empty();
        memory.init(1 << 20);
        for address in 0..bytes {
            memory.write_byte(address, 1);
        }
        memory
    };
    let flat = final_memory(0).flat_heap_bytes();
    let image = |bytes| final_memory(bytes).flat_image().bytes().unwrap();
    let cycles = vec![Cycle::NoOp; 100];
    let (pending, rows) = (heap_bytes(&cycles), 100 * size_of::<JoltTraceRow>());
    let finish = |bytes, limit| {
        let trace = LimitedTrace {
            resident: flat,
            ..storage(limit, cycles.clone(), Vec::new())
        };
        let finished = trace.finish_accumulated(&bytecode, final_memory(bytes), 0, 0);
        finished.map(|(data, _, image)| (data.len(), image.bytes.len()))
    };
    let image_first = flat + pending + image(1);
    assert_eq!(finish(1, image_first).unwrap(), (100, 1));
    assert_eq!(
        rejection(finish(1, image_first - 1)),
        Some((TraceAllocationPhase::RowGrowth, flat + pending + rows))
    );
    let rows_first = flat + rows + image(1 << 12);
    let limit = flat + pending + image(1 << 12) - 1;
    assert_eq!(finish(1 << 12, limit).unwrap(), (100, 1 << 12));
    assert_eq!(
        rejection(finish(1 << 12, rows_first - 1)),
        Some((TraceAllocationPhase::FinalMemory, rows_first))
    );
    #[cfg(feature = "field-inline")]
    {
        // The limit the last tick admitted: the merge fits once the cycles drop.
        let (cycles, bytecode) = field_inline_cycles(&[0, 1], 100);
        let events = 2 * size_of::<FieldEvent>();
        let limit = flat + heap_bytes(&cycles) + rows + size_of::<ConvertedChunk>() + events;
        let trace = LimitedTrace {
            resident: flat,
            ..storage(limit, cycles, Vec::new())
        };
        let (data, _, _) = trace
            .finish_accumulated(&bytecode, final_memory(1 << 10), 0, 2)
            .unwrap();
        assert_eq!(data.field_events().len(), 2);
    }
}

/// [`Instruction::emission`] is what the instruction's trace emits, in rows
/// and field events, for every variant with random operands, also with
/// `rd = x0` where it writes a register, and for the registered inlines
/// above; `NoOp` and `UNIMPL`, whose traces panic, are errors. Each runs with
/// its random register values, and with every integer register at `DRAM_BASE`
/// and every field register at one, beside a filled advice tape. Runs whose
/// trace panics are skipped, but every other variant traces.
#[test]
fn tick_emission_matches_trace_for_every_instruction() {
    let mut rng = StdRng::from_seed([5; 32]);
    let inlines = [LONGEST_INLINE_WORD, HOST_WRITE_INLINE_WORD, MID_INLINE_WORD];
    let fresh_cpu = || {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        let config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };
        cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&config));
        cpu.get_mut_mmu().init_memory(1 << 20);
        cpu.advice_tape = AdviceTape::from_bytes(vec![7; 64]);
        cpu
    };
    assert!(Instruction::UNIMPL.emission(&mut fresh_cpu()).is_err());
    let mut untraced = Vec::new();
    for template in Cycle::iter() {
        let mut traced = 0;
        for _ in 0..32 {
            let cycle = template.random(&mut rng);
            let random = match template {
                Cycle::INLINE(_) => {
                    let word = inlines[rng.gen_range(0..inlines.len())];
                    Instruction::from(INLINE::new(word, DRAM_BASE, false, false))
                }
                _ => cycle.instruction(),
            };
            // Decode admits only these CSRs, an advice load reads 1, 2, 4 or 8
            // bytes, and a field load writes an integer register: these
            // operands stand in.
            let mut operands = random.source_instruction().row().operands;
            match template {
                Cycle::CSRRS(_) | Cycle::CSRRW(_) => operands.imm = 0x300,
                Cycle::VirtualAdviceLoad(_) => operands.imm = 8,
                #[cfg(feature = "field-inline")]
                Cycle::FIELD_LOAD_ACCUMULATE_FROM_MEMORY(_) => {
                    (operands.rd, operands.rs1, operands.rs2) = (Some(5), Some(6), Some(1));
                }
                _ => {}
            }
            let rewrite = |operands| {
                let source = random.source_instruction().map_row(|mut row| {
                    row.operands = operands;
                    row
                });
                Instruction::try_from_source_instruction(source).unwrap()
            };
            let rd0 = operands.rd.map(|_| NormalizedOperands {
                rd: Some(0),
                ..operands
            });
            for instruction in std::iter::once(operands).chain(rd0).map(rewrite) {
                for random_registers in [true, false] {
                    let mut cpu = fresh_cpu();
                    if random_registers {
                        for (register, value) in
                            [cycle.rs1_read(), cycle.rs2_read()].into_iter().flatten()
                        {
                            cpu.write_register(usize::from(register), value as i64);
                        }
                    } else {
                        // Aligned with the immediate offset, within memory.
                        let imm = instruction.source_instruction().row().operands.imm;
                        let base = (i128::from(DRAM_BASE) + 4096 - imm.rem_euclid(8)) as i64;
                        for register in 1..32 {
                            cpu.write_register(register, base);
                        }
                        #[cfg(feature = "field-inline")]
                        for register in 0..FIELD_REGISTER_COUNT {
                            let one = FieldEncodedValue::from_u64(1);
                            cpu.field_registers.write(register, one);
                        }
                    }
                    if matches!(instruction, Instruction::NoOp) {
                        assert!(instruction.emission(&mut cpu).is_err());
                        continue;
                    }
                    let run = catch_unwind(AssertUnwindSafe(|| {
                        let emission = instruction.emission(&mut cpu).ok();
                        let mut cycles = Vec::new();
                        instruction.trace(&mut cpu, Some(&mut cycles));
                        (emission, cycles)
                    }));
                    let Ok((emission, cycles)) = run else {
                        continue;
                    };
                    #[cfg(feature = "field-inline")]
                    let field_events = field_event_count(&cycles);
                    #[cfg(not(feature = "field-inline"))]
                    let field_events = 0;
                    let emitted = Emission {
                        rows: cycles.len(),
                        field_events,
                    };
                    assert_eq!(emission, Some(emitted), "{instruction:?}");
                    traced += 1;
                }
            }
        }
        if traced == 0 && !matches!(template, Cycle::NoOp) {
            untraced.push(template);
        }
    }
    assert!(untraced.is_empty(), "{untraced:?}");
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

/// The guest's host I/O word: `.insn i 0x5B, 2, x0, x0, 0`.
const HOST_IO_WORD: u32 = 0x0000_205b;

/// A guest that writes 24 bytes of its own text to the advice tape, then
/// 64 KiB through `large_write` (its own host I/O or the registered inline
/// above), then 7 bytes, and stalls. A trace that converts the inline's rows
/// fails main's bytecode check: the inline builder keeps the host I/O row's
/// `x0` operands, which the executed row does not carry.
fn advice_writing_guest(large_write: u32) -> (JoltProgram, TraceInputs) {
    guest(&[
        0x00ad_c537, // lui a0, 0xadc
        0xabe5_0513, // addi a0, a0, -0x542: the advice-write call
        0x0000_0597, // auipc a1, 0
        0x0180_0613, // addi a2, x0, 24
        HOST_IO_WORD,
        0x0001_0637, // lui a2, 0x10
        large_write,
        0x0070_0613, // addi a2, x0, 7
        HOST_IO_WORD,
        0x0000_006f, // j .
    ])
}

/// The tape-only run, which requests no trace storage, writes the tape the
/// original and the limited traces write, through several tape growths and
/// to the PC stall.
#[test]
fn tape_only_run_writes_the_traced_tape() {
    assert_eq!(JOLT_ADVICE_WRITE_CALL_ID, 0x00ad_babe);
    let (program, inputs) = advice_writing_guest(HOST_IO_WORD);
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
    let (program, inputs) = advice_writing_guest(HOST_WRITE_INLINE_WORD);
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
    let (program, inputs) = advice_writing_guest(HOST_WRITE_INLINE_WORD);
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
