//! The committed-bytecode claim-reduction kernel (stage 6b cycle phase →
//! stage 7 address phase): reduces the five staged `BytecodeValStage(i)`
//! claims into per-chunk `BytecodeChunk(i)` openings over the shared
//! precommitted schedule.
//!
//! The shared [`PrecommittedReductionKernel`](crate::precommitted_reduction)
//! core runs over:
//! - value table: the chunk-weight fold `Σ_c chunk_rbc_weight_c · chunk_c`
//!   of the committed chunk coefficient grids,
//! - eq table: `lane_weights[lane] · eq(r_bc)[cycle]` over the `(lane,
//!   cycle)` grid in the proof's trace order — the η/γ-folded lane weights
//!   carrying the whole input-claim algebra,
//! - aux tables: the raw per-chunk grids, bound alongside; their fully bound
//!   coefficients are the final per-chunk opening values.

use jolt_claims::protocols::jolt::{
    BytecodeClaimReductionLayout, JoltCommittedPolynomial, PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;

use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::BytecodeReductionCyclePhase;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::precommitted_reduction::{permute_tables, PrecommittedReductionKernel};
use super::views::eq_table;
use crate::committed_program::{build_committed_bytecode_chunk_coeffs, chunk_index_to_lane_cycle};
use crate::precommitted_reduction::{bytecode_reduction_cycle_kernel, PrecommittedReductionProver};
use crate::{
    KernelError, PrepareKernel, ProofSession, ReferenceBackend, RetainedProgram, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReductionCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>>, KernelError<F>>
    {
        let layout = inputs.relation.layout();
        // The chunk grids materialize from the session-resident retained
        // program (the prover keeps the full bytecode in committed mode).
        let program = session
            .state::<RetainedProgram>()
            .ok_or(KernelError::InvariantViolation {
                reason: "prover-retained program data was not parked in the proof session",
            })?
            .program
            .clone();
        let prover = bytecode_reduction_prover(
            layout,
            inputs.relation.weights(),
            &program.bytecode.bytecode,
        )?;
        Ok(bytecode_reduction_cycle_kernel(
            session,
            prover,
            layout.dimensions().has_address_phase(),
        ))
    }
}

/// The two-phase committed-bytecode reduction prover — see the module doc for
/// the value/eq/aux table construction.
fn bytecode_reduction_prover<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    weights: &BytecodeReductionWeights<F>,
    bytecode: &[JoltInstructionRow],
) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>> {
    let reduction = layout.precommitted().clone();
    let chunk_coeffs: Vec<Vec<F>> = build_committed_bytecode_chunk_coeffs(
        bytecode,
        layout.chunk_count(),
        layout.trace_order(),
    )?;
    let chunk_len = chunk_coeffs[0].len();
    if chunk_len != 1usize << reduction.poly_opening_round_permutation_be().len() {
        return Err(KernelError::TableSizeMismatch {
            table: "committed bytecode chunk grid".to_owned(),
            expected: 1usize << reduction.poly_opening_round_permutation_be().len(),
            got: chunk_len,
        });
    }
    if weights.chunk_rbc_weights.len() != chunk_coeffs.len() {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode chunk weights".to_owned(),
            expected: chunk_coeffs.len(),
            got: weights.chunk_rbc_weights.len(),
        });
    }

    let chunk_cycle_len = 1usize << layout.log_bytecode_chunk_size();
    let eq_cycle = eq_table(&weights.r_bc);
    let eq_template: Vec<F> = (0..chunk_len)
        .map(|index| {
            let (lane, cycle) =
                chunk_index_to_lane_cycle(index, chunk_cycle_len, layout.trace_order());
            weights.lane_weights[lane] * eq_cycle[cycle]
        })
        .collect();
    let value: Vec<F> = (0..chunk_len)
        .map(|index| {
            chunk_coeffs
                .iter()
                .zip(&weights.chunk_rbc_weights)
                .map(|(coeffs, weight)| coeffs[index] * *weight)
                .sum()
        })
        .collect();

    let mut tables = Vec::with_capacity(2 + chunk_coeffs.len());
    tables.push(value);
    tables.push(eq_template);
    tables.extend(chunk_coeffs);
    let mut permuted = permute_tables(&reduction, tables).into_iter();
    let (value, eq) = match (permuted.next(), permuted.next()) {
        (Some(value), Some(eq)) => (value, eq),
        _ => {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode reduction table permutation lost the value/eq tables",
            });
        }
    };
    Ok(Box::new(PrecommittedReductionKernel::new(
        reduction,
        value,
        eq,
        permuted.collect(),
    )?))
}

/// The final per-chunk opening ids, in chunk order — the wire order of the
/// reduction's produced claims.
pub fn bytecode_chunk_ids(chunk_count: usize) -> Vec<JoltCommittedPolynomial> {
    (0..chunk_count)
        .map(JoltCommittedPolynomial::BytecodeChunk)
        .collect()
}
