//! The shared two-phase precommitted claim-reduction kernel core, backing the
//! advice, committed-bytecode, and program-image reductions (stage 6b cycle
//! phase → stage 7 address phase).
//!
//! The summand in both phases is `Σ_j value(j) · eq(j)` over tables permuted
//! into Dory opening-round order, bound low-to-high; `aux` tables (the
//! per-chunk bytecode polynomials) bind alongside without joining the summand
//! — their fully bound coefficients are the reduction's final per-chunk
//! openings. The member is head-aligned (batch offset 0) and participates
//! only in its schedule's active rounds; on an inactive round inside the
//! window it emits the constant `claim/2` polynomial and halves its running
//! `scale`.
//!
//! Byte-parity subtlety: on active rounds the round polynomial is built from
//! the TRUE evaluations at 0 and 2 plus the hint `s(1) = claim/scale − s(0)`
//! (legacy `UniPoly::from_evals_and_hint`). Under the batch's `2^(max−rounds)`
//! dummy-round padding the incoming claim is the padded one, so the hinted
//! `s(1)` — and through it the emitted quadratic coefficient — carries the
//! padding factor. This is the wire format legacy emits and the verifier's
//! expected-output algebra encodes; reproduce it exactly, do not "fix" it to
//! the unpadded true polynomial.

use jolt_claims::protocols::jolt::PrecommittedClaimReduction;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};

use crate::KernelError;

/// One precommitted reduction member, spanning stages 6b and 7: the stage-6b
/// recipe drives the cycle phase and stages the handoff claim, then hands the
/// SAME object to stage 7, which calls
/// [`Self::transition_to_address_phase`] and drives the address phase (only
/// when the schedule has active address rounds).
pub trait PrecommittedReductionProver<F: Field>: ProveRounds<F> {
    /// Flip the member from the cycle phase to the address phase. Call between
    /// the stage-6b and stage-7 batches.
    fn transition_to_address_phase(&mut self);

    /// The intermediate claim staged at the cycle→address handoff:
    /// `Σ_i value(i) · eq(i) · scale` over the bound tables. Meaningful only
    /// when the schedule has an address phase.
    fn cycle_intermediate_claim(&self) -> F;

    /// The fully bound value coefficient — the reduction's final opening
    /// value (the advice/program-image polynomial's own opening; for the
    /// bytecode reduction, the chunk-weighted fold the per-chunk claims sum
    /// to). Errors while any variable remains unbound.
    fn final_claim(&self) -> Result<F, KernelError<F>>;

    /// The fully bound `aux` coefficients — the per-chunk `BytecodeChunk(i)`
    /// opening values. Empty for reductions without aux tables. Errors while
    /// any variable remains unbound.
    fn final_aux_claims(&self) -> Result<Vec<F>, KernelError<F>>;
}

pub(crate) struct PrecommittedReductionKernel<F> {
    reduction: PrecommittedClaimReduction,
    in_cycle_phase: bool,
    value: Vec<F>,
    eq: Vec<F>,
    aux: Vec<Vec<F>>,
    /// `(1/2)^k` over the `k` inactive rounds ingested so far — the factor the
    /// running claim accumulated relative to the true bound product.
    scale: F,
    scale_inv: F,
    two_inv: F,
}

impl<F: Field> PrecommittedReductionKernel<F> {
    /// Build a member from tables ALREADY permuted into Dory opening-round
    /// order (see [`lsb_permutation`] / [`permute_coefficients`]).
    pub(crate) fn new(
        reduction: PrecommittedClaimReduction,
        value: Vec<F>,
        eq: Vec<F>,
        aux: Vec<Vec<F>>,
    ) -> Result<Self, KernelError<F>> {
        let expected = 1usize << reduction.poly_opening_round_permutation_be().len();
        for (name, len) in std::iter::once(("value", value.len()))
            .chain(std::iter::once(("eq", eq.len())))
            .chain(aux.iter().map(|table| ("aux", table.len())))
        {
            if len != expected {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("precommitted reduction {name}"),
                    expected,
                    got: len,
                });
            }
        }
        #[expect(
            clippy::expect_used,
            reason = "2 is invertible in any Jolt field (large-prime characteristic)"
        )]
        let two_inv = F::from_u64(2).inverse().expect("2 is invertible");
        Ok(Self {
            reduction,
            in_cycle_phase: true,
            value,
            eq,
            aux,
            scale: F::one(),
            scale_inv: F::one(),
            two_inv,
        })
    }

    fn is_active_round(&self, round: usize) -> bool {
        let active = if self.in_cycle_phase {
            self.reduction.cycle_phase_rounds()
        } else {
            self.reduction.address_phase_rounds()
        };
        active.binary_search(&round).is_ok()
    }

    fn require_fully_bound(&self) -> Result<(), KernelError<F>> {
        if self.value.len() != 1 {
            return Err(KernelError::Unsupported {
                reason:
                    "precommitted reduction final claim requested before the polynomial is fully bound",
            });
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for PrecommittedReductionKernel<F> {
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
        let bind = |table: &mut Vec<F>| {
            for j in 0..half {
                table[j] = table[2 * j] + challenge * (table[2 * j + 1] - table[2 * j]);
            }
            table.truncate(half);
        };
        bind(&mut self.value);
        bind(&mut self.eq);
        for table in &mut self.aux {
            bind(table);
        }
        Ok(())
    }
}

impl<F: Field> PrecommittedReductionProver<F> for PrecommittedReductionKernel<F> {
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
        self.require_fully_bound()?;
        Ok(self.value[0])
    }

    fn final_aux_claims(&self) -> Result<Vec<F>, KernelError<F>> {
        self.require_fully_bound()?;
        Ok(self.aux.iter().map(|table| table[0]).collect())
    }
}

/// The LSB-index relabeling implied by the big-endian opening-round
/// permutation: variables sorted by their global opening round become the new
/// LSB order. Returns `None` when the relabeling is the identity.
pub(crate) fn lsb_permutation(poly_opening_round_permutation_be: &[usize]) -> Option<Vec<usize>> {
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
pub(crate) fn permute_coefficients<F: Copy>(table: &[F], old_lsb_to_new_lsb: &[usize]) -> Vec<F> {
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
pub(crate) fn permute_challenges<F: Copy>(
    challenges_be: &[F],
    old_lsb_to_new_lsb: &[usize],
) -> Vec<F> {
    let num_vars = challenges_be.len();
    let mut permuted = challenges_be.to_vec();
    for (old_be, &challenge) in challenges_be.iter().enumerate() {
        let new_lsb = old_lsb_to_new_lsb[num_vars - 1 - old_be];
        permuted[num_vars - 1 - new_lsb] = challenge;
    }
    permuted
}

/// Permute a batch of coefficient tables into the reduction's Dory
/// opening-round order (identity-permutation short-circuit included).
pub(crate) fn permute_tables<F: Copy>(
    reduction: &PrecommittedClaimReduction,
    tables: Vec<Vec<F>>,
) -> Vec<Vec<F>> {
    match lsb_permutation(reduction.poly_opening_round_permutation_be()) {
        Some(old_lsb_to_new_lsb) => tables
            .into_iter()
            .map(|table| permute_coefficients(&table, &old_lsb_to_new_lsb))
            .collect(),
        None => tables,
    }
}
