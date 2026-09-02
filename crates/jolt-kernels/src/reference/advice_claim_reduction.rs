//! The advice claim-reduction kernel: the cycle-phase precommitted reduction
//! of a trusted/untrusted advice opening (stage 6b; stage 7's address phase
//! resumes from the parked carry), plus the stage-4 advice opening evaluation
//! it reduces.
//!
//! The reduction member is the shared
//! [`CycleReductionKernel`](crate::precommitted_reduction): the advice
//! polynomial as the value table and the eq table of the staged RAM
//! value-check point, both permuted into Dory opening-round order, so the
//! fully bound value coefficient IS the final `@AdviceClaimReduction` opening
//! value.

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
};
use jolt_field::JoltField;
use jolt_witness::JoltWitnessOracle;

use crate::ProverInputs;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase,
};
use jolt_witness::JoltWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::opening::AdviceOpeningEvaluation;
use crate::precommitted_reduction::{
    lsb_permutation, permute_challenges, permute_coefficients, CycleReductionKernel,
};
use crate::{KernelError, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel};

impl<F: JoltField> AdviceOpeningEvaluation<F> for ReferenceBackend {
    // The backend-neutral `AdviceOpeningEvaluation::evaluate` span lives at
    // the stage-4 call boundary (`crates/jolt-prover/src/stages/stage4.rs`),
    // so every implementation inherits it — see the taxonomy's kernel-seam
    // contract.
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<F, KernelError<F>> {
        let table = advice_table(witness, kind, point.len())?;
        let eq = eq_table(point);
        Ok(table
            .iter()
            .zip(&eq)
            .map(|(value, weight)| *value * *weight)
            .sum())
    }
}

impl<F: JoltField> PrepareKernel<F, TrustedAdviceCyclePhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, TrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let layout = inputs.relation.layout();
        let r_val =
            inputs
                .relation
                .reference_opening_point()
                .ok_or(KernelError::InvariantViolation {
                    reason: "trusted-advice cycle phase carries no reference opening point",
                })?;
        Ok(Box::new(advice_reduction_kernel::<
            F,
            TrustedAdviceCyclePhase<F>,
        >(
            JoltAdviceKind::Trusted, layout, r_val, witness
        )?))
    }
}

impl<F: JoltField> PrepareKernel<F, UntrustedAdviceCyclePhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, UntrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let layout = inputs.relation.layout();
        let r_val =
            inputs
                .relation
                .reference_opening_point()
                .ok_or(KernelError::InvariantViolation {
                    reason: "untrusted-advice cycle phase carries no reference opening point",
                })?;
        Ok(Box::new(advice_reduction_kernel::<
            F,
            UntrustedAdviceCyclePhase<F>,
        >(
            JoltAdviceKind::Untrusted,
            layout,
            r_val,
            witness,
        )?))
    }
}

/// The advice reduction's cycle-phase kernel: the advice polynomial as the
/// value table and the eq table of the staged RAM value-check point, both
/// permuted into Dory opening-round order.
fn advice_reduction_kernel<F: JoltField, R>(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    r_val: &[F],
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<CycleReductionKernel<F, R>, KernelError<F>> {
    let reduction = layout.precommitted().clone();
    let permutation = reduction.poly_opening_round_permutation_be();
    if r_val.len() != permutation.len() {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "advice reference point has {} variables, schedule expects {}",
                r_val.len(),
                permutation.len()
            ),
        });
    }
    let table = advice_table(witness, kind, permutation.len())?;

    // Both tables in Dory opening-round order: the coefficient permute and
    // the challenge permute are the same LSB relabeling, so
    // `permuted_table[i] · permuted_eq[i]` pairs exactly as the unpermuted
    // product did and the sum (the input claim) is preserved.
    let (value, eq) = match lsb_permutation(permutation) {
        Some(old_lsb_to_new_lsb) => (
            permute_coefficients(&table, &old_lsb_to_new_lsb),
            eq_table(&permute_challenges(r_val, &old_lsb_to_new_lsb)),
        ),
        None => (table, eq_table(r_val)),
    };
    CycleReductionKernel::new(reduction, value, eq, Vec::new())
}

fn advice_table<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    kind: JoltAdviceKind,
    expected_vars: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let table = dense_view(witness, ram_val_check_advice_opening(kind))?;
    if table.len() != 1usize << expected_vars {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{kind:?} advice"),
            expected: 1usize << expected_vars,
            got: table.len(),
        });
    }
    Ok(table)
}
