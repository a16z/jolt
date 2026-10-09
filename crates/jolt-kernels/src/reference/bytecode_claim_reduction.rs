//! Reduce staged bytecode-value claims to one whole-bytecode opening. The
//! summand is the committed row/lane grid times the lane-weight/address-equality
//! grid, bound over the canonical two-phase precommitted schedule.

use jolt_claims::protocols::jolt::{BytecodeClaimReductionLayout, PrecommittedReductionLayout};
use jolt_field::JoltField;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;

use crate::ProverInputs;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::BytecodeReductionCyclePhase;
use jolt_witness::JoltWitnessPlane;

use super::views::eq_table;

use crate::committed_program::build_committed_bytecode_coeffs;
use crate::committed_program::bytecode_index_to_lane_row;
use crate::precommitted_reduction::{permute_tables, CycleReductionKernel};
use crate::{KernelError, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel};

impl<F: JoltField> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReductionCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>>, KernelError<F>>
    {
        let layout = inputs.relation.layout();
        let program = witness.program_preprocessing();
        Ok(Box::new(bytecode_reduction_kernel(
            layout,
            inputs.relation.weights(),
            &program.bytecode.bytecode,
        )?))
    }
}

pub(crate) fn bytecode_reduction_kernel<F: JoltField>(
    layout: &BytecodeClaimReductionLayout,
    weights: &BytecodeReductionWeights<F>,
    bytecode: &[JoltInstructionRow],
) -> Result<CycleReductionKernel<F, BytecodeReductionCyclePhase<F>>, KernelError<F>> {
    let reduction = layout.precommitted().clone();
    let value = build_committed_bytecode_coeffs(bytecode, layout.trace_order())?;
    let eq_rows = eq_table(&weights.r_bc);
    let rows = 1usize << layout.log_rows();
    let entry = |index| {
        let (lane, row) = bytecode_index_to_lane_row(index, rows, layout.trace_order());
        weights.lane_weights[lane] * eq_rows[row]
    };
    #[cfg(feature = "parallel")]
    let eq = {
        use rayon::prelude::*;
        if value.len() >= 1 << 10 {
            (0..value.len()).into_par_iter().map(entry).collect()
        } else {
            (0..value.len()).map(entry).collect()
        }
    };
    #[cfg(not(feature = "parallel"))]
    let eq = (0..value.len()).map(entry).collect();
    let mut tables = permute_tables(&reduction, vec![value, eq]).into_iter();
    let (Some(value), Some(eq)) = (tables.next(), tables.next()) else {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode permutation lost the value/eq tables",
        });
    };
    CycleReductionKernel::new(reduction, value, eq)
}
