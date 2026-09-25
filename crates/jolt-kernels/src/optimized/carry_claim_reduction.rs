//! Optimized carry claim reduction (stage 6b, implicit-carry).
//!
//! The three eq leaves collapse into one weight table
//!
//! `W(j) = eq(r_product, j) + γ·eq(r_shift, j) + γ²·eq(0, j)`
//!
//! and the summand is `W·Carry`. [`CarryWeights`] keeps the two eq terms in
//! split (√T) tables until their low halves are exhausted, and the
//! `carry_init` point mass `γ²·eq(0, ·)` as one scalar riding on index 0:
//! under low-to-high binding `eq(0, (j', b))` restricts to `(1 − r)·eq(0, j')`,
//! so the mass never needs a table. The carry column stays in trace rows
//! until the first bind, avoiding a `T`-length field table before then.

use jolt_claims::protocols::jolt::geometry::spartan::carry_reduced;
use jolt_field::JoltField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckInputClaims};
use jolt_verifier::stages::stage6b::carry_claim_reduction::{
    CarryClaimReduction, CarryClaimReductionOutputClaims,
};
use jolt_witness::witnesses::{Carry, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{
    bind_all, bind_pairs, eq_table, pair, par_sum_pair_groups, round_poly_from_skipped_evals,
    scaled_eq_table, BundleAccess, BundleStore, RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Stage-6b carry claim reduction.
pub struct OptimizedCarryClaimReduction;

/// The committed carry column of one cycle.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct CarryRow {
    #[opening(committed = Carry)]
    carry: Carry,
}

impl<F: JoltField> PrepareKernel<F, CarryClaimReduction<F>> for OptimizedCarryClaimReduction {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, CarryClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = CarryClaimReduction<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let [product_cycle, shift_cycle] = relation.cycle_points();
        for point in [product_cycle, shift_cycle] {
            if point.len() != relation.rounds() {
                return Err(KernelError::InvariantViolation {
                    reason: "carry reduction cycle point has the wrong variable count",
                });
            }
        }
        let cycles = 1usize << relation.rounds();
        let rows = witness.shape(carry_reduced().polynomial_id())?.rows();
        if rows != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", carry_reduced()),
                expected: cycles,
                got: rows,
            });
        }

        let weights = CarryWeights::new(product_cycle, shift_cycle, inputs.challenges.gamma);
        let carry = if relation.rounds() == 0 {
            // No bind occurs on a single-cycle domain.
            CarryState::Dense(Polynomial::new(
                witness.oracle_table(carry_reduced().polynomial_id())?,
            ))
        } else {
            CarryState::Rows(BundleStore::resolve(witness, cycles)?)
        };

        Ok(Box::new(CarryKernel {
            progress: RoundProgress::new(relation.rounds()),
            carry,
            weights,
        }))
    }
}

/// `eq(r_product, ·) + γ·eq(r_shift, ·) + γ²·eq(0, ·)` under low-to-high
/// binding. The eq terms live in four ~√T split tables; the zero-selector
/// point mass is the scalar `zero_mass` at index 0, scaled by `(1 − r)` per
/// bind. Binding folds the exhausted low scalars into one dense high table
/// with the mass added in.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CarryWeights<F> {
    Split {
        product_lo: Vec<F>,
        product_hi: Vec<F>,
        shift_lo: Vec<F>,
        /// Pre-scaled by γ.
        shift_hi: Vec<F>,
        zero_mass: F,
    },
    Dense(Vec<F>),
}

impl<F: JoltField> CarryWeights<F> {
    fn new(product: &[F], shift: &[F], gamma: F) -> Self {
        debug_assert_eq!(product.len(), shift.len());
        let zero_mass = gamma * gamma;
        let mid = product.len() / 2;
        if mid == 0 {
            // At most two entries for zero or one variable.
            let mut table = eq_table(product);
            for (acc, term) in table.iter_mut().zip(scaled_eq_table(shift, gamma)) {
                *acc += term;
            }
            table[0] += zero_mass;
            return Self::Dense(table);
        }
        let (product_hi, product_lo) = product.split_at(product.len() - mid);
        let (shift_hi, shift_lo) = shift.split_at(shift.len() - mid);
        Self::Split {
            product_lo: eq_table(product_lo),
            product_hi: eq_table(product_hi),
            shift_lo: eq_table(shift_lo),
            shift_hi: scaled_eq_table(shift_hi, gamma),
            zero_mass,
        }
    }

    /// The weight table's `(lo, hi)` sumcheck pair at group `y` under
    /// low-to-high pairing.
    #[inline]
    fn pair(&self, y: usize) -> (F, F) {
        match self {
            Self::Split {
                product_lo,
                product_hi,
                shift_lo,
                shift_hi,
                zero_mass,
            } => {
                let lo_len = product_lo.len();
                let j = 2 * y;
                let hi = j / lo_len;
                debug_assert!(lo_len >= 2, "adjacent lo indices share the hi part");
                let (a, b) = (product_hi[hi], shift_hi[hi]);
                let mut even = a * product_lo[j % lo_len] + b * shift_lo[j % lo_len];
                // eq(0, ·) selects index 0 alone: the even slot of group 0.
                if y == 0 {
                    even += *zero_mass;
                }
                (
                    even,
                    a * product_lo[(j + 1) % lo_len] + b * shift_lo[(j + 1) % lo_len],
                )
            }
            Self::Dense(table) => (table[2 * y], table[2 * y + 1]),
        }
    }

    fn bind(&mut self, r: F) {
        match self {
            Self::Split {
                product_lo,
                product_hi,
                shift_lo,
                shift_hi,
                zero_mass,
            } => {
                bind_pairs(product_lo, r);
                bind_pairs(shift_lo, r);
                // eq(0, (j', b)) bound at b = r is (1 − r)·eq(0, j').
                *zero_mass *= F::one() - r;
                if product_lo.len() == 1 {
                    // Fold the exhausted low scalars into the high tables.
                    let (s_product, s_shift) = (product_lo[0], shift_lo[0]);
                    let mut dense: Vec<F> = product_hi
                        .iter()
                        .zip(shift_hi.iter())
                        .map(|(&a, &b)| a * s_product + b * s_shift)
                        .collect();
                    dense[0] += *zero_mass;
                    *self = Self::Dense(dense);
                }
            }
            Self::Dense(table) => bind_pairs(table, r),
        }
    }
}

/// Trace rows before the first bind; the dense bound table afterward.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CarryState<F: JoltField> {
    Rows(BundleStore<CarryRow>),
    Dense(Polynomial<F>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct CarryKernel<F: JoltField> {
    progress: RoundProgress,
    carry: CarryState<F>,
    weights: CarryWeights<F>,
}

fn row_unavailable<F: JoltField>() -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "carry claim-reduction trace rows",
    }
}

/// The carry values of the row pair `(2y, 2y + 1)`.
#[inline]
fn carry_pair<F: JoltField>(
    access: &BundleAccess<'_, CarryRow>,
    y: usize,
) -> Result<(F, F), SumcheckError<F>> {
    let even: CarryRow = access.row(2 * y).map_err(|_| row_unavailable())?;
    let odd: CarryRow = access.row(2 * y + 1).map_err(|_| row_unavailable())?;
    Ok((even.carry.to_field(), odd.carry.to_field()))
}

impl<F: JoltField> CarryKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if let CarryState::Dense(table) = &mut self.carry {
            bind_all([table], challenge);
        } else {
            self.carry = CarryState::Dense(self.materialize_bound(challenge)?);
        }
        self.weights.bind(challenge);
        self.progress.advance();
        Ok(())
    }

    /// Binds trace-row pairs directly into a half-length field table.
    fn materialize_bound(&self, challenge: F) -> Result<Polynomial<F>, SumcheckError<F>> {
        let CarryState::Rows(store) = &self.carry else {
            unreachable!("materialize_bound is only called in the rows state");
        };
        debug_assert_eq!(self.progress.bound(), 0);
        let half = (1usize << self.progress.total()) / 2;
        let access = store.access();
        let bound = |y: usize| -> Result<F, SumcheckError<F>> {
            let (even, odd) = carry_pair(&access, y)?;
            Ok(even + challenge * (odd - even))
        };
        #[cfg(feature = "parallel")]
        let table = (0..half)
            .into_par_iter()
            .map(bound)
            .collect::<Result<Vec<F>, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let table = (0..half).map(bound).collect::<Result<Vec<F>, _>>()?;
        Ok(Polynomial::new(table))
    }

    /// Summand evaluations at `t ∈ {0, 2}` for group `y`.
    #[inline]
    fn group_evals(&self, y: usize, carry: (F, F)) -> [F; 2] {
        let (w_lo, w_hi) = self.weights.pair(y);
        [
            w_lo * carry.0,
            (w_hi + w_hi - w_lo) * (carry.1 + carry.1 - carry.0),
        ]
    }
}

impl<F: JoltField> ProveRounds<F> for CarryKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        let evals = match &self.carry {
            CarryState::Rows(store) => {
                debug_assert_eq!(self.progress.bound(), 0);
                let half = (1usize << self.progress.total()) / 2;
                let access = store.access();
                let group = |y: usize| -> Result<[F; 2], SumcheckError<F>> {
                    Ok(self.group_evals(y, carry_pair(&access, y)?))
                };
                #[cfg(feature = "parallel")]
                let evals = (0..half)
                    .into_par_iter()
                    .try_fold(
                        || vec![F::zero(); 2],
                        |mut acc, y| {
                            let group = group(y)?;
                            acc[0] += group[0];
                            acc[1] += group[1];
                            Ok(acc)
                        },
                    )
                    .try_reduce(|| vec![F::zero(); 2], |a, b| Ok(merge_evals(a, b)))?;
                #[cfg(not(feature = "parallel"))]
                let evals = {
                    let mut acc = vec![F::zero(); 2];
                    for y in 0..half {
                        let group = group(y)?;
                        acc[0] += group[0];
                        acc[1] += group[1];
                    }
                    acc
                };
                evals
            }
            CarryState::Dense(table) => {
                let half = table.len() / 2;
                par_sum_pair_groups(half, 2, |acc, y| {
                    let group = self.group_evals(y, pair(table, y));
                    acc[0] += group[0];
                    acc[1] += group[1];
                })
            }
        };

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for CarryKernel<F> {
    type Relation = CarryClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<CarryClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let CarryState::Dense(carry) = &self.carry else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "carry table absent after full binding",
            });
        };
        Ok(CarryClaimReductionOutputClaims {
            carry: carry.evals()[0],
        })
    }
}

/// Byte parity with the reference kernel on a trace with live carries (the
/// canned sample trace carries none, which would make the parity vacuous).
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage6b::carry_claim_reduction::{
        CarryClaimReductionChallenges, CarryClaimReductionInputClaims,
    };

    use super::*;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::optimized::testing::with_carry_fixture;
    use crate::ReferenceBackend;

    const LOG_T: usize = 3;

    #[test]
    fn carry_claim_reduction_matches_reference() {
        with_carry_fixture(LOG_T, &[0, 7, 1 << 40, 3, u64::MAX], |witness| {
            let relation = CarryClaimReduction::new(
                TraceDimensions::new(LOG_T),
                synthetic_point(LOG_T, 3),
                synthetic_point(LOG_T, 5),
            );
            let challenges = CarryClaimReductionChallenges {
                gamma: Fr::from_u64(29),
            };
            let claims = CarryClaimReductionInputClaims::<Fr>::default();
            let input_points = CarryClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, CarryClaimReduction<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedCarryClaimReduction
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let sumcheck_challenges = synthetic_point(LOG_T, 401);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &sumcheck_challenges,
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
        });
    }
}
