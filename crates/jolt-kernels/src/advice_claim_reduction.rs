//! The advice claim-reduction kernel: the two-phase precommitted reduction of
//! a trusted/untrusted advice opening (stage 6b cycle phase → stage 7 address
//! phase), plus the stage-4 advice opening evaluation it reduces.
//!
//! The summand in both phases is `Σ_j advice(j) · eq(j)` over the advice
//! polynomial PERMUTED into Dory opening-round order and the eq table of the
//! staged RAM value-check point permuted the same way — so binding the batch
//! challenges low-to-high walks the polynomial's committed-matrix opening
//! order, and the fully bound coefficient IS the final `@AdviceClaimReduction`
//! opening value. The member is head-aligned (batch offset 0) and only
//! participates in its schedule's active rounds; on an inactive round inside
//! the window it emits the constant `claim/2` polynomial and halves its
//! running `scale`.
//!
//! Byte-parity subtlety: on active rounds the round polynomial is built from
//! the TRUE evaluations at 0 and 2 plus the hint `s(1) = claim/scale − s(0)`
//! (legacy `UniPoly::from_evals_and_hint`). Under the batch's `2^(max−rounds)`
//! dummy-round padding the incoming claim is the padded one, so the hinted
//! `s(1)` — and through it the emitted quadratic coefficient — carries the
//! padding factor. This is the wire format legacy emits and the verifier's
//! expected-output algebra encodes; reproduce it exactly, do not "fix" it to
//! the unpadded true polynomial.

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedClaimReduction,
    PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, ProofSession, ReferenceBackend};

/// One advice reduction member, spanning stages 6b and 7: the stage-6b recipe
/// drives the cycle phase and stages the handoff claim, then hands the SAME
/// object to stage 7, which calls [`Self::transition_to_address_phase`] and
/// drives the address phase (only when the schedule has active address
/// rounds).
pub trait AdviceReductionProver<F: Field>: ProveRounds<F> {
    /// Flip the member from the cycle phase to the address phase. Call between
    /// the stage-6b and stage-7 batches.
    fn transition_to_address_phase(&mut self);

    /// The intermediate claim staged at the cycle→address handoff:
    /// `Σ_i advice(i) · eq(i) · scale` over the cycle-bound tables. Meaningful
    /// only when the schedule has an address phase.
    fn cycle_intermediate_claim(&self) -> F;

    /// The fully bound advice coefficient — the final
    /// `@AdviceClaimReduction` opening value. Errors while any variable
    /// remains unbound.
    fn final_claim(&self) -> Result<F, KernelError<F>>;
}

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
    ) -> Result<Box<dyn AdviceReductionProver<F>>, KernelError<F>>;
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
    ) -> Result<Box<dyn AdviceReductionProver<F>>, KernelError<F>> {
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

        #[expect(
            clippy::expect_used,
            reason = "2 is invertible in any Jolt field (large-prime characteristic)"
        )]
        let two_inv = F::from_u64(2).inverse().expect("2 is invertible");
        Ok(Box::new(ReferenceAdviceReduction {
            reduction,
            in_cycle_phase: true,
            value,
            eq,
            scale: F::one(),
            scale_inv: F::one(),
            two_inv,
        }))
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

/// The LSB-index relabeling implied by the big-endian opening-round
/// permutation: variables sorted by their global opening round become the new
/// LSB order. Returns `None` when the relabeling is the identity.
fn lsb_permutation(poly_opening_round_permutation_be: &[usize]) -> Option<Vec<usize>> {
    let num_vars = poly_opening_round_permutation_be.len();
    let mut be_var_by_round: Vec<usize> = (0..num_vars).collect();
    be_var_by_round.sort_unstable_by_key(|&be_index| poly_opening_round_permutation_be[be_index]);

    let mut old_lsb_to_new_lsb = vec![0usize; num_vars];
    for (new_lsb, be_index) in be_var_by_round.into_iter().enumerate() {
        old_lsb_to_new_lsb[num_vars - 1 - be_index] = new_lsb;
    }
    old_lsb_to_new_lsb
        .iter()
        .enumerate()
        .any(|(old_lsb, &new_lsb)| old_lsb != new_lsb)
        .then_some(old_lsb_to_new_lsb)
}

/// Out-of-place coefficient permute: `out[new_index] = table[old_index]` where
/// each of `new_index`'s bits moves to its pre-image LSB position.
fn permute_coefficients<F: Copy>(table: &[F], old_lsb_to_new_lsb: &[usize]) -> Vec<F> {
    let num_vars = old_lsb_to_new_lsb.len();
    let mut new_lsb_to_old_lsb = vec![0usize; num_vars];
    for (old_lsb, &new_lsb) in old_lsb_to_new_lsb.iter().enumerate() {
        new_lsb_to_old_lsb[new_lsb] = old_lsb;
    }
    (0..table.len())
        .map(|new_index| {
            let mut old_index = 0usize;
            for (new_lsb, &old_lsb) in new_lsb_to_old_lsb.iter().enumerate() {
                old_index |= ((new_index >> new_lsb) & 1) << old_lsb;
            }
            table[old_index]
        })
        .collect()
}

/// The challenge-vector counterpart of [`permute_coefficients`]: relabel the
/// big-endian challenge positions so `eq(permuted_challenges)` indexes the
/// permuted coefficient table.
fn permute_challenges<F: Copy>(challenges_be: &[F], old_lsb_to_new_lsb: &[usize]) -> Vec<F> {
    let num_vars = challenges_be.len();
    let mut permuted = challenges_be.to_vec();
    for (old_be, &challenge) in challenges_be.iter().enumerate() {
        let new_lsb = old_lsb_to_new_lsb[num_vars - 1 - old_be];
        permuted[num_vars - 1 - new_lsb] = challenge;
    }
    permuted
}

struct ReferenceAdviceReduction<F> {
    reduction: PrecommittedClaimReduction,
    in_cycle_phase: bool,
    value: Vec<F>,
    eq: Vec<F>,
    /// `(1/2)^k` over the `k` inactive rounds ingested so far — the factor the
    /// running claim accumulated relative to the true bound product.
    scale: F,
    scale_inv: F,
    two_inv: F,
}

impl<F: Field> ReferenceAdviceReduction<F> {
    fn is_active_round(&self, round: usize) -> bool {
        let active = if self.in_cycle_phase {
            self.reduction.cycle_phase_rounds()
        } else {
            self.reduction.address_phase_rounds()
        };
        active.binary_search(&round).is_ok()
    }
}

impl<F: Field> ProveRounds<F> for ReferenceAdviceReduction<F> {
    fn num_rounds(&self) -> usize {
        if self.in_cycle_phase {
            self.reduction.cycle_phase_total_rounds()
        } else {
            self.reduction.address_phase_total_rounds()
        }
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if !self.is_active_round(round) {
            return Ok(UnivariatePoly::new(vec![previous_claim * self.two_inv]));
        }
        let half = self.value.len() / 2;
        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();
        for j in 0..half {
            let value_0 = self.value[2 * j];
            let value_1 = self.value[2 * j + 1];
            let eq_0 = self.eq[2 * j];
            let eq_1 = self.eq[2 * j + 1];
            eval_0 += value_0 * eq_0;
            eval_2 += (value_1 + value_1 - value_0) * (eq_1 + eq_1 - eq_0);
        }
        // The hinted middle evaluation (see the module doc for why the padded
        // claim, not the true sum, feeds it), then the {0,1,2} interpolation.
        let eval_1 = previous_claim * self.scale_inv - eval_0;
        let c2 = (eval_0 - eval_1 - eval_1 + eval_2) * self.two_inv;
        let c1 = eval_1 - eval_0 - c2;
        Ok(UnivariatePoly::new(vec![
            eval_0 * self.scale,
            c1 * self.scale,
            c2 * self.scale,
        ]))
    }

    fn ingest_challenge(&mut self, challenge: F, round: usize) -> Result<(), SumcheckError<F>> {
        if !self.is_active_round(round) {
            self.scale *= self.two_inv;
            self.scale_inv += self.scale_inv;
            return Ok(());
        }
        let half = self.value.len() / 2;
        for j in 0..half {
            self.value[j] =
                self.value[2 * j] + challenge * (self.value[2 * j + 1] - self.value[2 * j]);
            self.eq[j] = self.eq[2 * j] + challenge * (self.eq[2 * j + 1] - self.eq[2 * j]);
        }
        self.value.truncate(half);
        self.eq.truncate(half);
        Ok(())
    }
}

impl<F: Field> AdviceReductionProver<F> for ReferenceAdviceReduction<F> {
    fn transition_to_address_phase(&mut self) {
        self.in_cycle_phase = false;
    }

    fn cycle_intermediate_claim(&self) -> F {
        let product: F = self
            .value
            .iter()
            .zip(&self.eq)
            .map(|(value, eq)| *value * *eq)
            .sum();
        product * self.scale
    }

    fn final_claim(&self) -> Result<F, KernelError<F>> {
        if self.value.len() != 1 {
            return Err(KernelError::Unsupported {
                reason:
                    "advice reduction final claim requested before the polynomial is fully bound",
            });
        }
        Ok(self.value[0])
    }
}
