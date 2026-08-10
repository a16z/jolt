//! The `RowEmitter` seam: how one expanded-bytecode row becomes machine code.
//!
//! Code generation is isolated behind this trait so an alternate emitter —
//! notably the copy-and-patch stencil route recorded in Alternative 11 of
//! `specs/x86-tracer-backend.md` — can be compared against the dynasm
//! templates per row kind, without touching the compile driver, the
//! differential tests, or the benches. Emitters are ordered: the first one
//! that claims a kind emits it, so a hybrid (stencils for some kinds, hand
//! templates for the rest) needs no further plumbing.
//!
//! Everything outside per-row emission — the prologue, dispatch stubs, jump
//! table, and group/chunk accounting — is shared infrastructure and stays in
//! the driver: those are properties of the execution model, not of how a row
//! is encoded.

use dynasmrt::DynasmApi as _;
use jolt_program::execution::TraceError;
use jolt_riscv::JoltInstructionRow;

use super::Emitter;

/// Whether an emitter handled a row.
///
/// Emitters report `Unsupported` instead of erroring so the next emitter in
/// the set gets a chance; the set raises the fail-fast error only when no
/// emitter claims the kind (spec invariant 7). This keeps the authoritative
/// list of supported kinds in one place: the emitter's own match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitOutcome {
    Emitted,
    Unsupported,
}

/// A code-generation strategy for expanded-bytecode rows.
pub trait RowEmitter {
    /// Emit code for one row, or report that this emitter does not handle
    /// the row's kind.
    fn emit_row(
        &self,
        cx: &mut Emitter,
        row: &JoltInstructionRow,
    ) -> Result<EmitOutcome, TraceError>;

    /// Emit a source-instruction group's advice computation, which must run
    /// before any of the group's rows.
    fn emit_advice_compute(&self, cx: &mut Emitter, job_index: usize);
}

/// An ordered set of emitters. The first that claims a kind emits it.
pub struct EmitterSet {
    emitters: Vec<Box<dyn RowEmitter>>,
}

impl EmitterSet {
    /// The production configuration: dynasm templates for every kind.
    pub fn dynasm() -> Self {
        Self {
            emitters: vec![Box::new(super::emit::DynasmEmitter)],
        }
    }

    /// Build a set from an explicit emitter list (front takes precedence).
    pub fn from_emitters(emitters: Vec<Box<dyn RowEmitter>>) -> Self {
        Self { emitters }
    }

    pub fn emit_row(&self, cx: &mut Emitter, row: &JoltInstructionRow) -> Result<(), TraceError> {
        for emitter in &self.emitters {
            let before = cx.ops.offset();
            if emitter.emit_row(cx, row)? == EmitOutcome::Emitted {
                return Ok(());
            }
            // Contract: declining must be side-effect-free. Partial emission
            // before a decline (a row prelude, say) would silently double the
            // trace-row count once a later emitter claims the kind.
            if cx.ops.offset() != before {
                return Err(TraceError::Backend(
                    "row emitter declined a kind after emitting code",
                ));
            }
        }
        // No emitter handles this kind: refuse to compile rather than
        // execute wrong semantics (spec invariant 7).
        Err(TraceError::Backend(
            "jolt-tracer-x86: unsupported instruction kind in bytecode",
        ))
    }

    /// Advice computation is emitted by the first emitter; it is execution
    /// -model plumbing (a helper call), identical across strategies.
    pub fn emit_advice_compute(
        &self,
        cx: &mut Emitter,
        job_index: usize,
    ) -> Result<(), TraceError> {
        let emitter = self
            .emitters
            .first()
            .ok_or(TraceError::Backend("no row emitter configured"))?;
        emitter.emit_advice_compute(cx, job_index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::harness::{single_row_program, TEST_ADDR};
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    /// An emitter that claims nothing — stands in for a partial emitter
    /// (e.g. stencils covering only some kinds) in the ordering tests.
    struct DeclineAll;

    impl RowEmitter for DeclineAll {
        fn emit_row(
            &self,
            _cx: &mut Emitter,
            _row: &JoltInstructionRow,
        ) -> Result<EmitOutcome, TraceError> {
            Ok(EmitOutcome::Unsupported)
        }

        fn emit_advice_compute(&self, _cx: &mut Emitter, _job_index: usize) {}
    }

    fn add_row() -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADD,
            address: TEST_ADDR as usize,
            operands: NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: true,
            is_compressed: false,
        }
    }

    /// A partial emitter ahead of the templates must fall through, not fail.
    #[test]
    fn declining_emitter_falls_through_to_the_next() {
        let program = single_row_program(add_row());
        let set = EmitterSet::from_emitters(vec![
            Box::new(DeclineAll),
            Box::new(super::super::emit::DynasmEmitter),
        ]);
        assert!(super::super::CompiledProgram::compile_with(&program, &set).is_ok());
    }

    /// When no emitter claims a kind, compilation fails fast rather than
    /// emitting nothing for the row (spec invariant 7).
    #[test]
    fn unclaimed_kind_fails_compilation() {
        let program = single_row_program(add_row());
        let set = EmitterSet::from_emitters(vec![Box::new(DeclineAll)]);
        assert!(super::super::CompiledProgram::compile_with(&program, &set).is_err());
    }

    /// An emitter that emits bytes and then declines would double-count rows
    /// once a later emitter claims the kind; the set must reject it.
    struct EmitThenDecline;

    impl RowEmitter for EmitThenDecline {
        fn emit_row(
            &self,
            cx: &mut Emitter,
            _row: &JoltInstructionRow,
        ) -> Result<EmitOutcome, TraceError> {
            use dynasmrt::DynasmApi as _;
            cx.ops.push(0x90); // stray nop
            Ok(EmitOutcome::Unsupported)
        }

        fn emit_advice_compute(&self, _cx: &mut Emitter, _job_index: usize) {}
    }

    #[test]
    fn partial_emission_before_decline_is_rejected() {
        let program = single_row_program(add_row());
        let set = EmitterSet::from_emitters(vec![
            Box::new(EmitThenDecline),
            Box::new(super::super::emit::DynasmEmitter),
        ]);
        let error = match super::super::CompiledProgram::compile_with(&program, &set) {
            Err(error) => format!("{error:?}"),
            Ok(_) => String::from("compiled successfully"),
        };
        assert!(
            error.contains("declined a kind after emitting"),
            "expected the partial-emission error, got: {error}"
        );
    }
}
