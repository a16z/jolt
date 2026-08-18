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

use super::state::{AdviceCompute, AdviceJob, GuestState};
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
    /// Same two bodies, with the chunk-boundary pause check emitted.
    fast_pausable: CompiledBody,
    record_pausable: CompiledBody,
    /// Advice computations, one per group that needs them; generated code
    /// passes indices into this table. Identical for both bodies (same rows,
    /// same order), so it is collected once.
    advice_jobs: Vec<AdviceJob>,
}

impl CompiledProgram {
    /// Compile with the production emitter set (dynasm templates).
    pub fn compile(program: &JoltProgram) -> Result<Self, TraceError> {
        Self::compile_with(program, &EmitterSet::dynasm())
    }

    /// Compile with an explicit emitter set — the A/B entry point for
    /// alternative row emitters (see `compile/emitter.rs`).
    pub fn compile_with(program: &JoltProgram, emitters: &EmitterSet) -> Result<Self, TraceError> {
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
        let sources = Self::source_rows(program)?;

        // Four bodies: {execute, record} x {eager, pausable}. Compilation is
        // ~10ms per body, so keeping the eager paths free of the chunk-boundary
        // check is worth the duplication.
        let (fast, advice_jobs) =
            CompiledBody::compile(rows, &sources, emitters, EmitMode::Fast, false)?;
        let (record, _) = CompiledBody::compile(rows, &sources, emitters, EmitMode::Record, false)?;
        let (fast_pausable, _) =
            CompiledBody::compile(rows, &sources, emitters, EmitMode::Fast, true)?;
        let (record_pausable, _) =
            CompiledBody::compile(rows, &sources, emitters, EmitMode::Record, true)?;

        Ok(Self {
            fast,
            record,
            fast_pausable,
            record_pausable,
            advice_jobs,
        })
    }

    pub fn advice_jobs_ptr(&self) -> *const AdviceJob {
        self.advice_jobs.as_ptr()
    }

    /// Host address of the fast body's entry point. Exposed so the safety
    /// tests can assert the finalized mapping's permissions (AC11): code is
    /// never writable and executable at the same time.
    pub fn code_address(&self) -> usize {
        self.fast.code_address()
    }

    /// Run the fast body: execution only, no row materialization.
    pub fn run(&self, state: &mut GuestState) -> Result<(), TraceError> {
        self.fast.run(state)
    }

    /// Run the record body, filling the observation buffer described by
    /// `GuestState::obs_cursor`/`obs_end`.
    pub fn run_record(&self, state: &mut GuestState) -> Result<(), TraceError> {
        self.record.run(state)
    }

    /// Run the pausable execute body: stops at the first group boundary at or
    /// past `GuestState::row_limit`, publishing the resume PC.
    pub fn run_pausable(&self, state: &mut GuestState) -> Result<(), TraceError> {
        self.fast_pausable.run(state)
    }

    /// Run the pausable record body, for chunk replay.
    pub fn run_record_pausable(&self, state: &mut GuestState) -> Result<(), TraceError> {
        self.record_pausable.run(state)
    }

    /// Decode the program's source instructions and key them by address.
    ///
    /// Programs assembled directly from rows (the test/bench harness) carry no
    /// ELF; they get an empty map, and any group that actually needs advice then
    /// fails at emission rather than here.
    fn source_rows(program: &JoltProgram) -> Result<SourceMap, TraceError> {
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
    /// Whether this body can pause at group boundaries. Only the chunked
    /// paths need that; the eager paths must not pay a per-group check.
    pub pausable: bool,
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
    /// Index of the current group's advice job, until the group ends and its
    /// consumed-slot count is patched in (see [`Self::finish_advice_group`]).
    current_advice_job: Option<usize>,
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

    /// A group's rows are all emitted: record how many `VirtualAdvice` slots
    /// it consumed on its job, so the runtime helper can check the computed
    /// value count against it. Div computations provide exactly two values,
    /// which makes over-consumption a compile-time error.
    fn finish_advice_group(&mut self) -> Result<(), TraceError> {
        let Some(index) = self.current_advice_job.take() else {
            return Ok(());
        };
        let job = &mut self.advice_jobs[index];
        job.advice_rows = self.advice_slot;
        if matches!(job.compute, AdviceCompute::Div { .. }) && job.advice_rows > 2 {
            return Err(TraceError::Backend(
                "DIV/REM group consumes more advice values than its computation provides",
            ));
        }
        Ok(())
    }
}

type SourceMap = BTreeMap<u64, jolt_riscv::SourceInstruction<jolt_riscv::SourceInstructionRow>>;

impl CompiledBody {
    fn code_address(&self) -> usize {
        self.buffer.ptr(self.entry) as usize
    }

    fn run(&self, state: &mut GuestState) -> Result<(), TraceError> {
        let entry = self.buffer.ptr(self.entry);
        // SAFETY: `entry` points into the finalized (read+execute) buffer at
        // the prologue emitted by `compile`; the generated code only touches
        // the GuestState plane, the RAM plane it carries, the observation
        // buffer, and the jump table, per the emitter's invariants.
        let f: extern "sysv64" fn(*mut GuestState, *const usize) =
            unsafe { core::mem::transmute(entry) };
        f(state, self.jump_table.as_ptr());
        Ok(())
    }

    /// Emit one code body over every expanded row.
    fn compile(
        rows: &[JoltInstructionRow],
        sources: &SourceMap,
        emitters: &EmitterSet,
        mode: EmitMode,
        pausable: bool,
    ) -> Result<(Self, Vec<AdviceJob>), TraceError> {
        let text_base = rows.iter().map(|r| r.address as u64).min().unwrap_or(0);
        let text_end = rows.iter().map(|r| r.address as u64).max().unwrap_or(0) + 4;
        let text_span = text_end - text_base;
        // The dispatch sequence compares the target's byte delta against the
        // span as an i32 immediate; a larger span would sign-extend negative
        // and let every in-range check pass vacuously.
        if text_span > i32::MAX as u64 {
            return Err(TraceError::Backend(
                "program text span exceeds the dispatch table's addressable range",
            ));
        }

        let mut emitter = Emitter {
            mode,
            pausable,
            row_index: 0,
            ops: Assembler::new()
                .map_err(|_| TraceError::Backend("failed to create x64 assembler"))?,
            advice_jobs: Vec::new(),
            advice_slot: 0,
            advice_ready: false,
            current_advice_job: None,
            labels: BTreeMap::new(),
            group_offsets: Vec::new(),
            text_base,
            text_span,
        };

        let entry = emitter.emit_prologue();

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
                if emitter.pausable {
                    emitter.emit_group_pause_check(address);
                }
                previous_address = Some(address);
                emitter.finish_advice_group()?;
                emitter.advice_slot = 0;
                emitter.advice_ready = false;
                if let Some(source) = sources.get(&address) {
                    if let Some(compute) = AdviceCompute::from_source(source)? {
                        emitter.advice_jobs.push(AdviceJob {
                            compute,
                            advice_rows: 0,
                        });
                        let index = emitter.advice_jobs.len() - 1;
                        emitter.current_advice_job = Some(index);
                        emitters.emit_advice_compute(&mut emitter, index)?;
                        emitter.advice_ready = true;
                    }
                }
            }
            emitters.emit_row(&mut emitter, row)?;
        }
        emitter.finish_advice_group()?;

        // Execution falling off the end of the program is a bad jump.
        emitter.emit_jump_to_bad_jump();
        let stubs = emitter.emit_stubs();

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
            Self {
                buffer,
                entry,
                jump_table,
            },
            advice_jobs,
        ))
    }
}

impl AdviceCompute {
    /// The advice computation a source instruction's group needs, if any.
    fn from_source(
        source: &jolt_riscv::SourceInstruction<jolt_riscv::SourceInstructionRow>,
    ) -> Result<Option<Self>, TraceError> {
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
                return Ok(Some(Self::Inline {
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
        Ok(Some(Self::Div { code, rs1, rs2 }))
    }
}
