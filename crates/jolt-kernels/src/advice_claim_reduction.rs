//! The advice claim-reduction kernel: the two-phase precommitted reduction of
//! a trusted/untrusted advice opening (stage 6b cycle phase → stage 7 address
//! phase), plus the stage-4 advice opening evaluation it reduces.
//!
//! The reduction member is the shared
//! [`PrecommittedReductionKernel`](crate::precommitted_reduction) core: the
//! advice polynomial as the value table and the eq table of the staged RAM
//! value-check point, both permuted into Dory opening-round order, so the
//! fully bound value coefficient IS the final `@AdviceClaimReduction` opening
//! value.

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::precommitted_reduction::{
    lsb_permutation, permute_challenges, permute_coefficients, PrecommittedReductionKernel,
    PrecommittedReductionProver,
};
use crate::views::{dense_view, eq_table};
use crate::{KernelError, ProofSession, ReferenceBackend};

/// The advice claim-reduction slot: the stage-4 opening evaluation and the
/// stage-6b/7 reduction member share it because both are the advice
/// polynomial's protocol duties (there is exactly one advice oracle read
/// path).
pub trait AdviceClaimReduction<F: Field> {
    /// Evaluate the advice polynomial at `point` (big-endian) — the value the
    /// stage-4 RAM value-check stages under `@RamValCheck` for this kind.
    fn evaluate(
        &self,
        session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<F, KernelError<F>>;

    /// Build the two-phase reduction member for `kind`. `r_val` is the staged
    /// stage-4 opening point (big-endian, `advice_vars` long) the eq table is
    /// built from.
    fn prepare(
        &self,
        session: &mut ProofSession,
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        r_val: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>>;
}

impl<F: Field> AdviceClaimReduction<F> for ReferenceBackend {
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<F, KernelError<F>> {
        let table = advice_table(witness, kind, point.len())?;
        let eq = eq_table(point);
        Ok(table
            .iter()
            .zip(&eq)
            .map(|(value, weight)| *value * *weight)
            .sum())
    }

    fn prepare(
        &self,
        _session: &mut ProofSession,
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        r_val: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn PrecommittedReductionProver<F>>, KernelError<F>> {
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
        Ok(Box::new(PrecommittedReductionKernel::new(
            reduction,
            value,
            eq,
            Vec::new(),
        )?))
    }
}

fn advice_table<F: Field>(
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
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
