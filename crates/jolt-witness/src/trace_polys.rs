//! Lazy polynomial data access from an execution trace.
//!
//! [`TracePolynomials`] provides on-demand access to virtual polynomial
//! evaluation data backed by a `CycleRow` trace. Nothing is pre-materialized
//! until the consumer requests it.
//!
//! This is the sole interface between the proving layer (jolt-zkvm) and
//! trace-derived polynomial data. The prover never touches `CycleRow`
//! directly — all trace knowledge is encapsulated here.

use jolt_field::Field;
use jolt_host::CycleRow;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags};
use jolt_ir::PolynomialId;
use jolt_poly::EqPolynomial;

/// Lazy polynomial data backed by a `CycleRow` trace.
///
/// Provides three access modes:
/// - **Point evaluation**: `eval_at_cycle(poly_id, j)` — O(1), single cycle
/// - **Materialization**: `materialize(poly_id)` — O(T), full table on demand
/// - **MLE evaluation**: `eval_at_point(poly_id, point)` — O(T), eq-weighted sum
///
/// The prover uses `materialize()` for sumcheck witness buffers and
/// `eval_at_point()` for post-sumcheck claim evaluation.
pub struct TracePolynomials<'a, R> {
    trace: &'a [R],
    padded_len: usize,
}

impl<'a, R: CycleRow> TracePolynomials<'a, R> {
    pub fn new(trace: &'a [R]) -> Self {
        let padded_len = trace.len().next_power_of_two();
        Self { trace, padded_len }
    }

    /// Padded trace length (next power of 2).
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Raw trace length (before padding).
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    /// Extract a single polynomial value at one cycle.
    ///
    /// For "next-cycle" polynomials (NextPc, NextIsNoop, etc.), looks ahead
    /// to `trace[cycle + 1]` automatically.
    pub fn eval_at_cycle<F: Field>(&self, poly_id: PolynomialId, cycle: usize) -> F {
        if is_next_cycle_poly(poly_id) {
            let next = cycle + 1;
            if next < self.trace.len() {
                extract_value::<F, R>(poly_id, &self.trace[next])
            } else {
                extract_value::<F, R>(poly_id, &R::noop())
            }
        } else if cycle < self.trace.len() {
            extract_value::<F, R>(poly_id, &self.trace[cycle])
        } else {
            extract_value::<F, R>(poly_id, &R::noop())
        }
    }

    /// Materialize the full evaluation table for a polynomial.
    ///
    /// Returns a dense `Vec<F>` of length `padded_len`. Padding cycles
    /// use `CycleRow::noop()` values. Next-cycle polynomials use lookahead.
    pub fn materialize<F: Field>(&self, poly_id: PolynomialId) -> Vec<F> {
        (0..self.padded_len)
            .map(|j| {
                let v: F = self.eval_at_cycle(poly_id, j);
                v
            })
            .collect()
    }

    /// Evaluate the polynomial's MLE at a challenge point.
    ///
    /// Computes `Σ_j eq(point, j) · poly_value(j)` — the multilinear
    /// extension evaluated at `point`. Next-cycle polynomials use lookahead.
    pub fn eval_at_point<F: Field>(&self, poly_id: PolynomialId, point: &[F]) -> F {
        let eq = EqPolynomial::new(point.to_vec()).evaluations();
        let mut result = F::zero();
        for (j, &eq_val) in eq.iter().enumerate() {
            result += eq_val * self.eval_at_cycle(poly_id, j);
        }
        result
    }

    /// Build sparse RAM read-write entries for `SparseRwEvaluator`.
    ///
    /// Each cycle with a RAM access produces one entry sorted by cycle index.
    pub fn ram_entries<F: Field>(&self) -> Vec<crate::RwEntry<F>> {
        let mut entries = Vec::new();
        for (j, cycle) in self.trace.iter().enumerate() {
            if let Some(addr) = cycle.ram_access_address() {
                let read_val = cycle.ram_read_value().unwrap_or(0);
                let write_val = cycle.ram_write_value().unwrap_or(0);
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: addr as usize,
                    ra: F::one(),
                    val: F::from_u64(read_val),
                    prev_val: F::from_u64(read_val),
                    next_val: F::from_u64(write_val),
                });
            }
        }
        entries
    }

    /// Build sparse register read-write entries for `SparseRwEvaluator`.
    pub fn register_entries<F: Field>(&self) -> Vec<crate::RwEntry<F>> {
        let mut entries = Vec::new();
        for (j, cycle) in self.trace.iter().enumerate() {
            // RS1 read
            if let Some((idx, val)) = cycle.rs1_read() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(val),
                    prev_val: F::from_u64(val),
                    next_val: F::from_u64(val),
                });
            }
            // RS2 read
            if let Some((idx, val)) = cycle.rs2_read() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(val),
                    prev_val: F::from_u64(val),
                    next_val: F::from_u64(val),
                });
            }
            // RD write
            if let Some((idx, pre, post)) = cycle.rd_write() {
                entries.push(crate::RwEntry {
                    bind_pos: j,
                    free_pos: idx as usize,
                    ra: F::one(),
                    val: F::from_u64(pre),
                    prev_val: F::from_u64(pre),
                    next_val: F::from_u64(post),
                });
            }
        }
        // Sort by (bind_pos, free_pos) for sparse evaluator
        entries.sort_by_key(|e| (e.bind_pos, e.free_pos));
        entries
    }
}

/// Returns true for polynomial IDs that represent next-cycle values.
fn is_next_cycle_poly(poly_id: PolynomialId) -> bool {
    matches!(
        poly_id,
        PolynomialId::NextUnexpandedPc
            | PolynomialId::NextPc
            | PolynomialId::NextIsNoop
            | PolynomialId::NextIsVirtual
            | PolynomialId::NextIsFirstInSequence
    )
}

/// Extract a polynomial's per-cycle value from CycleRow methods.
///
/// This is the single source of truth for how PolynomialId maps to
/// trace-derived values. All ISA-specific knowledge flows through
/// CycleRow trait methods.
fn extract_value<F: Field, R: CycleRow>(poly_id: PolynomialId, cycle: &R) -> F {
    match poly_id {
        // Register values
        PolynomialId::RdWriteValue => {
            cycle.rd_write().map_or_else(F::zero, |(_, _, post)| F::from_u64(post))
        }
        PolynomialId::Rs1Value => {
            cycle.rs1_read().map_or_else(F::zero, |(_, v)| F::from_u64(v))
        }
        PolynomialId::Rs2Value => {
            cycle.rs2_read().map_or_else(F::zero, |(_, v)| F::from_u64(v))
        }
        PolynomialId::RegistersVal => {
            cycle.rd_write().map_or_else(F::zero, |(_, pre, _)| F::from_u64(pre))
        }

        // RAM values
        PolynomialId::RamAddress => {
            cycle.ram_access_address().map_or(F::zero(), F::from_u64)
        }
        PolynomialId::RamReadValue => {
            cycle.ram_read_value().map_or(F::zero(), F::from_u64)
        }
        PolynomialId::RamWriteValue => {
            cycle.ram_write_value().map_or(F::zero(), F::from_u64)
        }
        PolynomialId::RamVal => {
            cycle.ram_read_value().map_or(F::zero(), F::from_u64)
        }
        PolynomialId::RamValFinal => {
            cycle.ram_write_value().map_or(F::zero(), F::from_u64)
        }

        // Program counter and immediate
        PolynomialId::ExpandedPc | PolynomialId::UnexpandedPc => {
            F::from_u64(cycle.unexpanded_pc())
        }
        PolynomialId::NextUnexpandedPc | PolynomialId::NextPc => {
            // Caller provides the NEXT cycle via lookahead in eval_at_cycle.
            F::from_u64(cycle.unexpanded_pc())
        }
        PolynomialId::Imm => F::from_i128(cycle.imm()),

        // Lookup output
        PolynomialId::LookupOutput => {
            cycle.rd_write().map_or_else(F::zero, |(_, _, post)| F::from_u64(post))
        }
        PolynomialId::LeftLookupOperand | PolynomialId::RightLookupOperand => {
            // These are computed from instruction inputs — simplified here.
            F::zero()
        }
        PolynomialId::LeftInstructionInput => {
            let iflags = cycle.instruction_flags();
            if iflags[InstructionFlags::LeftOperandIsPC as usize] {
                F::from_u64(cycle.unexpanded_pc())
            } else if iflags[InstructionFlags::LeftOperandIsRs1Value as usize] {
                cycle.rs1_read().map_or_else(F::zero, |(_, v)| F::from_u64(v))
            } else {
                F::zero()
            }
        }
        PolynomialId::RightInstructionInput => {
            let iflags = cycle.instruction_flags();
            if iflags[InstructionFlags::RightOperandIsImm as usize] {
                F::from_i128(cycle.imm())
            } else if iflags[InstructionFlags::RightOperandIsRs2Value as usize] {
                cycle.rs2_read().map_or_else(F::zero, |(_, v)| F::from_u64(v))
            } else {
                F::zero()
            }
        }

        // Register addresses
        PolynomialId::Rs1Ra => {
            cycle.rs1_read().map_or_else(F::zero, |(idx, _)| F::from_u64(idx as u64))
        }
        PolynomialId::Rs2Ra => {
            cycle.rs2_read().map_or_else(F::zero, |(idx, _)| F::from_u64(idx as u64))
        }
        PolynomialId::RdWa => {
            cycle.rd_operand().map_or_else(F::zero, |rd| F::from_u64(rd as u64))
        }

        // Circuit flags (14 total)
        PolynomialId::JumpFlag => {
            bool_to_field(cycle.circuit_flags()[CircuitFlags::Jump as usize])
        }
        PolynomialId::WriteLookupToRdFlag => {
            bool_to_field(cycle.circuit_flags()[CircuitFlags::WriteLookupOutputToRD as usize])
        }
        PolynomialId::NextIsVirtual => {
            // Caller provides the NEXT cycle via lookahead in eval_at_cycle.
            bool_to_field(cycle.circuit_flags()[CircuitFlags::VirtualInstruction as usize])
        }
        PolynomialId::NextIsFirstInSequence => {
            bool_to_field(cycle.circuit_flags()[CircuitFlags::IsFirstInSequence as usize])
        }
        PolynomialId::OpFlag(i) => bool_to_field(cycle.circuit_flags()[i]),

        // Instruction flags
        PolynomialId::BranchFlag => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::Branch as usize])
        }
        PolynomialId::NextIsNoop => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::IsNoop as usize])
        }
        PolynomialId::IsRdNotZero => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::IsRdNotZero as usize])
        }
        PolynomialId::LeftIsRs1 => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::LeftOperandIsRs1Value as usize])
        }
        PolynomialId::LeftIsPc => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::LeftOperandIsPC as usize])
        }
        PolynomialId::RightIsRs2 => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::RightOperandIsRs2Value as usize])
        }
        PolynomialId::RightIsImm => {
            bool_to_field(cycle.instruction_flags()[InstructionFlags::RightOperandIsImm as usize])
        }

        // Derived flags
        PolynomialId::HammingWeight => {
            let flags = cycle.circuit_flags();
            bool_to_field::<F>(flags[CircuitFlags::Load as usize])
                + bool_to_field::<F>(flags[CircuitFlags::Store as usize])
        }
        PolynomialId::InstructionRafFlag => {
            let flags = cycle.circuit_flags();
            let interleaved = !flags[CircuitFlags::AddOperands as usize]
                && !flags[CircuitFlags::SubtractOperands as usize]
                && !flags[CircuitFlags::MultiplyOperands as usize]
                && !flags[CircuitFlags::Advice as usize];
            bool_to_field(interleaved)
        }

        // Committed polys — not trace-derived (come from WitnessStore)
        PolynomialId::RamInc
        | PolynomialId::RdInc
        | PolynomialId::InstructionRa(_)
        | PolynomialId::BytecodeRa(_)
        | PolynomialId::RamRa(_) => F::zero(),

        // Remaining (SpartanWitness, advice, RAF vals, LookupTableFlag, etc.)
        _ => F::zero(),
    }
}

#[inline]
fn bool_to_field<F: Field>(b: bool) -> F {
    if b { F::one() } else { F::zero() }
}

/// Sparse matrix entry for read-write checking evaluators.
///
/// Re-exported from jolt-witness for use by jolt-zkvm's evaluators.
#[derive(Clone, Debug)]
pub struct RwEntry<F> {
    pub bind_pos: usize,
    pub free_pos: usize,
    pub ra: F,
    pub val: F,
    pub prev_val: F,
    pub next_val: F,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require a concrete CycleRow impl (from jolt-host).
    // Integration tests in jolt-zkvm exercise the full path.
}
