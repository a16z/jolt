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

use std::collections::BTreeMap;

use dynasmrt::{x64::Assembler, AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi};
use jolt_program::execution::{JoltProgram, TraceError};

use super::state::GuestState;

/// A compiled program: executable buffer plus the indirect-dispatch table.
pub struct CompiledProgram {
    buffer: dynasmrt::ExecutableBuffer,
    entry: AssemblyOffset,
    /// One host code address per halfword in `[text_base, text_end)`;
    /// non-group-start slots point at the bad-jump stub.
    jump_table: Vec<usize>,
}

impl CompiledProgram {
    pub fn run(&self, state: &mut GuestState) -> Result<(), TraceError> {
        let entry = self.buffer.ptr(self.entry);
        // SAFETY: `entry` points into the finalized (read+execute) buffer at
        // the prologue emitted by `compile`; the generated code only touches
        // the GuestState plane, the RAM plane it carries, and the jump
        // table, per the emitter's invariants.
        let f: extern "sysv64" fn(*mut GuestState, *const usize) =
            unsafe { core::mem::transmute(entry) };
        f(state, self.jump_table.as_ptr());
        Ok(())
    }
}

pub(super) struct Emitter {
    pub ops: Assembler,
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

pub fn compile(program: &JoltProgram) -> Result<CompiledProgram, TraceError> {
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

    let text_base = rows.iter().map(|r| r.address as u64).min().unwrap_or(0);
    let text_end = rows.iter().map(|r| r.address as u64).max().unwrap_or(0) + 4;
    let text_span = text_end - text_base;

    let mut emitter = Emitter {
        ops: Assembler::new().map_err(|_| TraceError::Backend("failed to create x64 assembler"))?,
        labels: BTreeMap::new(),
        group_offsets: Vec::new(),
        text_base,
        text_span,
    };

    let entry = emit::prologue(&mut emitter);

    let mut previous_address = None;
    for row in rows {
        let address = row.address as u64;
        if previous_address != Some(address) {
            // Group start: define the branch-target label and record the
            // offset for the jump table.
            let label = emitter.label_for(address);
            let offset = emitter.ops.offset();
            emitter.ops.dynamic_label(label);
            emitter.group_offsets.push((address, offset));
            previous_address = Some(address);
        }
        emit::row(&mut emitter, row)?;
    }

    // Execution falling off the end of the program is a bad jump.
    emit::jump_to_bad_jump(&mut emitter);
    let stubs = emit::stubs(&mut emitter);

    let group_offsets = core::mem::take(&mut emitter.group_offsets);
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

    Ok(CompiledProgram {
        buffer,
        entry,
        jump_table,
    })
}
