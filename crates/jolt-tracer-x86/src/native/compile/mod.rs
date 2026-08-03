//! AOT compilation driver: expanded bytecode rows → executable x86-64.
//!
//! Register plan (SysV callee-saved, so helper calls preserve them):
//!   r12 = &mut GuestState        r13 = guest RAM plane host base
//!   r14 = trace_len (row count)  r15 = jump table base
//! Scratch: rax rcx rdx rsi rdi r8–r11 (no guest state lives in scratch).
//! Guest registers live in the `GuestState::x` array — correctness first;
//! register-pinning optimizations belong to the performance slice, guarded
//! by the differential tests.

mod emit;
pub mod emitter;

use std::collections::BTreeMap;

use dynasmrt::{x64::Assembler, AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi};
use jolt_program::execution::{JoltProgram, TraceError};
use jolt_riscv::SourceInstructionKind;

use super::state::{AdviceJob, GuestState};
use emitter::EmitterSet;
use jolt_riscv::JoltInstructionRow;

/// One compiled code body (fast or record) with its dispatch table.
struct CompiledBody {
    buffer: dynasmrt::ExecutableBuffer,
    entry: AssemblyOffset,
    /// One host code address per halfword in `[text_base, text_end)`;
    /// non-group-start slots point at the bad-jump stub.
    jump_table: Vec<usize>,
}

/// A compiled program: the two code bodies the spec calls for (fast and
/// record, emitted from the same templates) plus the shared advice-job table.
///
/// The bodies are separate assemblies rather than two entry points in one:
/// branch targets are dynamic labels per guest address, and a branch in the
/// record body must land on the record body's copy of its target.
pub struct CompiledProgram {
    fast: CompiledBody,
    record: CompiledBody,
    /// Advice computations, one per group that needs them; generated code
    /// passes indices into this table. Identical for both bodies (same rows,
    /// same order), so it is collected once.
    advice_jobs: Vec<AdviceJob>,
}

impl CompiledProgram {
    pub fn advice_jobs_ptr(&self) -> *const AdviceJob {
        self.advice_jobs.as_ptr()
    }

    /// Run the fast body: execution only, no row materialization.
    pub fn run(&self, state: &mut GuestState) -> Result<(), TraceError> {
        Self::run_body(&self.fast, state)
    }

    /// Run the record body, filling the observation buffer described by
    /// `GuestState::obs_cursor`/`obs_end`.
    pub fn run_record(&self, state: &mut GuestState) -> Result<(), TraceError> {
        Self::run_body(&self.record, state)
    }

    fn run_body(body: &CompiledBody, state: &mut GuestState) -> Result<(), TraceError> {
        let entry = body.buffer.ptr(body.entry);
        // SAFETY: `entry` points into the finalized (read+execute) buffer at
        // the prologue emitted by `compile`; the generated code only touches
        // the GuestState plane, the RAM plane it carries, the observation
        // buffer, and the jump table, per the emitter's invariants.
        let f: extern "sysv64" fn(*mut GuestState, *const usize) =
            unsafe { core::mem::transmute(entry) };
        f(state, body.jump_table.as_ptr());
        Ok(())
    }
}

/// Which code body is being emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitMode {
    /// No row materialization: execute only, for the fast pass.
    Fast,
    /// Additionally capture each row's dynamic values (see `Observation`).
    Record,
}

/// Emission context handed to a [`RowEmitter`](emitter::RowEmitter): the
/// assembler plus the per-group state a row template may need.
pub struct Emitter {
    /// Which body this emission belongs to.
    pub mode: EmitMode,
    /// Index of the row being emitted into `expanded_bytecode`; recorded in
    /// each observation so the reassembly pass can recover the static half.
    pub row_index: usize,
    pub ops: Assembler,
    /// Advice jobs collected so far (index = the job id in generated code).
    pub advice_jobs: Vec<AdviceJob>,
    /// Next `VirtualAdvice` slot within the current group.
    pub advice_slot: usize,
    /// Whether the current group emitted an advice computation (i.e. its
    /// `VirtualAdvice` rows have values to read).
    pub advice_ready: bool,
    /// Dynamic label per group-start guest address (branch/jump targets).
    labels: BTreeMap<u64, DynamicLabel>,
    /// Guest addresses that start a compiled group, with their code offsets.
    group_offsets: Vec<(u64, AssemblyOffset)>,
    pub text_base: u64,
    pub text_span: u64,
}

impl Emitter {
    pub fn label_for(&mut self, address: u64) -> DynamicLabel {
        if let Some(&label) = self.labels.get(&address) {
            return label;
        }
        let label = self.ops.new_dynamic_label();
        let _ = self.labels.insert(address, label);
        label
    }
}

/// Compile with the production emitter set (dynasm templates).
pub fn compile(program: &JoltProgram) -> Result<CompiledProgram, TraceError> {
    compile_with(program, &EmitterSet::dynasm())
}

/// Compile with an explicit emitter set — the A/B entry point for
/// alternative row emitters (see `compile/emitter.rs`).
pub fn compile_with(
    program: &JoltProgram,
    emitters: &EmitterSet,
) -> Result<CompiledProgram, TraceError> {
    // VirtualSRL uses `tzcnt`; on pre-BMI1 CPUs it silently decodes as `bsf`
    // with different zero-input semantics, so refuse rather than mis-execute.
    if !std::arch::is_x86_feature_detected!("bmi1") {
        return Err(TraceError::Backend(
            "jolt-tracer-x86 requires BMI1 (tzcnt) support",
        ));
    }
    let rows = &program.expanded_bytecode;
    if rows.is_empty() {
        return Err(TraceError::Backend("program has no expanded bytecode"));
    }

    // Source rows keyed by address: the expanded bytecode erases the source
    // kind and inline key, which the per-group advice computations need.
    let sources = source_rows(program)?;

    let (fast, advice_jobs) = compile_body(rows, &sources, emitters, EmitMode::Fast)?;
    let (record, _) = compile_body(rows, &sources, emitters, EmitMode::Record)?;

    Ok(CompiledProgram {
        fast,
        record,
        advice_jobs,
    })
}

type SourceMap = BTreeMap<u64, jolt_riscv::SourceInstruction<jolt_riscv::SourceInstructionRow>>;

/// Emit one code body over every expanded row.
fn compile_body(
    rows: &[JoltInstructionRow],
    sources: &SourceMap,
    emitters: &EmitterSet,
    mode: EmitMode,
) -> Result<(CompiledBody, Vec<AdviceJob>), TraceError> {
    let text_base = rows.iter().map(|r| r.address as u64).min().unwrap_or(0);
    let text_end = rows.iter().map(|r| r.address as u64).max().unwrap_or(0) + 4;
    let text_span = text_end - text_base;

    let mut emitter = Emitter {
        mode,
        row_index: 0,
        ops: Assembler::new().map_err(|_| TraceError::Backend("failed to create x64 assembler"))?,
        advice_jobs: Vec::new(),
        advice_slot: 0,
        advice_ready: false,
        labels: BTreeMap::new(),
        group_offsets: Vec::new(),
        text_base,
        text_span,
    };

    let entry = emit::prologue(&mut emitter);

    let mut previous_address = None;
    for (row_index, row) in rows.iter().enumerate() {
        emitter.row_index = row_index;
        let address = row.address as u64;
        if previous_address != Some(address) {
            // Group start: define the branch-target label, record the jump
            // table offset, and emit this group's advice computation (which
            // must observe the pre-group register state) before its rows.
            let label = emitter.label_for(address);
            let offset = emitter.ops.offset();
            emitter.ops.dynamic_label(label);
            emitter.group_offsets.push((address, offset));
            previous_address = Some(address);
            emitter.advice_slot = 0;
            emitter.advice_ready = false;
            if let Some(source) = sources.get(&address) {
                if let Some(job) = advice_job(source)? {
                    emitter.advice_jobs.push(job);
                    let index = emitter.advice_jobs.len() - 1;
                    emitters.emit_advice_compute(&mut emitter, index)?;
                    emitter.advice_ready = true;
                }
            }
        }
        emitters.emit_row(&mut emitter, row)?;
    }

    // Execution falling off the end of the program is a bad jump.
    emit::jump_to_bad_jump(&mut emitter);
    let stubs = emit::stubs(&mut emitter);

    let group_offsets = core::mem::take(&mut emitter.group_offsets);
    let advice_jobs = core::mem::take(&mut emitter.advice_jobs);
    let buffer = emitter
        .ops
        .finalize()
        .map_err(|_| TraceError::Backend("x64 assembly finalize failed"))?;

    // Build the halfword-granular dispatch table with the bad-jump stub as
    // filler.
    let bad_jump = buffer.ptr(stubs.bad_jump) as usize;
    let slots = (text_span / 2) as usize;
    let mut jump_table = vec![bad_jump; slots];
    for (address, offset) in group_offsets {
        let slot = ((address - text_base) / 2) as usize;
        jump_table[slot] = buffer.ptr(offset) as usize;
    }

    Ok((
        CompiledBody {
            buffer,
            entry,
            jump_table,
        },
        advice_jobs,
    ))
}

/// Decode the program's source instructions and key them by address.
///
/// Programs assembled directly from rows (the test/bench harness) carry no
/// ELF; they get an empty map, and any group that actually needs advice then
/// fails at emission (see the `VirtualAdvice` arm) rather than here.
fn source_rows(
    program: &JoltProgram,
) -> Result<
    BTreeMap<u64, jolt_riscv::SourceInstruction<jolt_riscv::SourceInstructionRow>>,
    TraceError,
> {
    if program.elf_bytes().is_empty() {
        return Ok(BTreeMap::new());
    }
    let image = jolt_program::image::decode_elf(program.elf_bytes(), program.profile)
        .map_err(|_| TraceError::Backend("failed to decode program ELF for source recovery"))?;
    Ok(image
        .instructions
        .into_iter()
        .map(|instruction| (instruction.row().address as u64, instruction))
        .collect())
}

/// The advice computation a source instruction's group needs, if any.
fn advice_job(
    source: &jolt_riscv::SourceInstruction<jolt_riscv::SourceInstructionRow>,
) -> Result<Option<AdviceJob>, TraceError> {
    use SourceInstructionKind as S;
    let row = source.row();
    let (rs1, rs2) = (row.operands.rs1, row.operands.rs2);
    // Codes mirror the tracer's per-variant advice formulas (helpers.rs).
    let code = match source.kind() {
        S::Div(_) | S::Rem(_) => 0u8,
        S::DivW(_) | S::RemW(_) => 1,
        S::DivU(_) | S::RemU(_) => 2,
        S::DivUW(_) | S::RemUW(_) => 3,
        S::InlineDispatch(_) => {
            let inline = row
                .inline
                .ok_or(TraceError::Backend("inline source row without inline key"))?;
            let registration = tracer::instruction::inline::find_inline_registration(
                u32::from(inline.opcode),
                u32::from(inline.funct3),
                u32::from(inline.funct7),
            )
            .ok_or(TraceError::Backend("no registered inline for source row"))?;
            let operands =
                tracer::instruction::format::format_inline::FormatInline::from(row.operands);
            return Ok(Some(AdviceJob::Inline {
                registration,
                operands,
            }));
        }
        _ => return Ok(None),
    };
    let (Some(rs1), Some(rs2)) = (rs1, rs2) else {
        return Err(TraceError::Backend(
            "div-family source row missing operands",
        ));
    };
    Ok(Some(AdviceJob::Div { code, rs1, rs2 }))
}
