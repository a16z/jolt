//! Packed `FieldRdInc` reconstruction: on the packed axis only the balanced
//! limb columns are committed ([`super::packing`]), so the reduced
//! `FieldRdInc` claim the stage-6b FR increment reduction produces is settled
//! against them by a reconstruction sumcheck — the full-width polynomial is
//! never PCS-opened. The FR analog of the advice word virtualization
//! (`protocols::jolt::lattice`'s advice reconstruction), with the byte decode
//! replaced by the balanced-digit recomposition.
//!
//! The decode identity ([`super::geometry::recomposition_coefficient`]):
//! `FieldRdInc(r_cycle) = Σ_column coeff · Σ_z value(digit) · eq(r_cycle,
//! cycle) · Col(z)` over the `(digit-value ‖ cycle)` domain, with `value` the
//! centered digit map ([`crate::lattice::balanced_inc_value`]). The map sends
//! the digit-zero row to zero, so the committed columns' omitted digit-zero
//! rows drop out of the decode exactly — no recentering term.
//!
//! The limb columns are prover-supplied, so the same sumcheck also proves
//! every committed cell is boolean, one γ-batched leg per column against a
//! fresh reference point (per-column powers: a shared power would let
//! booleanity defects cancel across columns):
//!
//! - **booleanity** (column `c`, at `γ^c`): `Σ_z eq(z, r_reference) ·
//!   (Col_c(z)² − Col_c(z)) = 0`,
//! - **decode** (at `γ^C`): sums to the consumed reduced `FieldRdInc` claim.
//!
//! No hamming leg: the columns are a representation of `FieldRdInc`, which
//! the decode leg pins in full; one-hotness beyond booleanity is not load
//! bearing (compare the untrusted-advice bytes, where hamming *is* the range
//! check). The legs share one sumcheck because the packed witness admits
//! exactly one claim per slot.

use jolt_field::{JoltField, Ring};
use serde::{Deserialize, Serialize};

use crate::lattice::BalancedIncChunking;
use crate::protocols::field_inline::{
    FieldIncLimbReconstructionChallenge, FieldIncLimbReconstructionPublic, FieldInlineChallengeId,
    FieldInlineCommittedPolynomial, FieldInlineExpr, FieldInlineOpeningId, FieldInlineRelationId,
};
use crate::{
    challenge, constant, derived, opening, ChallengeDrawError, InputClaims, OutputClaims,
    SumcheckChallenges, SymbolicSumcheck,
};

use super::geometry::recomposition_coefficient;
use super::packing::{field_inc_limb_columns, FieldIncLimbShape};

/// The per-column openings at the bound `(digit-value ‖ cycle)` point — the
/// final claims the packed limb object's opening consumes, in canonical
/// column order.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldIncLimbReconstruction)]
pub struct FieldIncLimbReconstructionOutputClaims<C> {
    #[opening(committed = FieldIncLimbColumn)]
    pub columns: Vec<C>,
}

/// The consumed claim: the stage-6b FR increment reduction's reduced
/// `FieldRdInc` terminus.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldIncLimbReconstructionInputClaims<C> {
    #[opening(committed = FieldRdInc, from = FieldRegistersIncClaimReduction)]
    pub rd_inc: C,
}

/// The vector field rules out the `SumcheckChallenges` derive, so the impl is
/// hand-written (the advice reconstruction pattern): the reference point is
/// not challenge-id-resolvable and the struct cannot be built from a
/// per-field scalar stream.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct FieldIncLimbReconstructionChallenges<F> {
    /// The fresh reference point the booleanity legs compare against, drawn
    /// over the full `(digit-value ‖ cycle)` domain before the gamma.
    pub r_reference: Vec<F>,
    pub gamma: F,
}

impl<F: JoltField> SumcheckChallenges<F, FieldInlineChallengeId>
    for FieldIncLimbReconstructionChallenges<F>
{
    fn from_transcript_values<I: Iterator<Item = F>>(
        _values: I,
    ) -> Result<Self, ChallengeDrawError> {
        Err(ChallengeDrawError::NotStreamConstructible)
    }

    fn resolve_challenge(&self, id: &FieldInlineChallengeId) -> Option<F> {
        (*id == FieldInlineChallengeId::from(FieldIncLimbReconstructionChallenge::Gamma))
            .then_some(self.gamma)
    }
}

#[derive(Clone)]
pub struct FieldIncLimbReconstruction {
    shape: FieldIncLimbShape,
}

impl FieldIncLimbReconstruction {
    pub const fn shape(&self) -> &FieldIncLimbShape {
        &self.shape
    }

    /// The canonical column count `C` — the decode leg's gamma power.
    fn column_count(&self) -> usize {
        // The shape was validated at plan construction; a malformed chunk
        // width would have failed there, so fall back to zero columns rather
        // than panic in an expression builder.
        BalancedIncChunking::new(self.shape.log_k_chunk).map_or(0, |chunking| {
            self.shape.limbs * (chunking.chunk_count() + 1)
        })
    }
}

impl SymbolicSumcheck for FieldIncLimbReconstruction {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type DerivedId = crate::protocols::field_inline::FieldInlineDerivedId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldIncLimbShape;
    type Challenges<F> = FieldIncLimbReconstructionChallenges<F>;
    type Inputs<C> = FieldIncLimbReconstructionInputClaims<C>;
    type Outputs<C> = FieldIncLimbReconstructionOutputClaims<C>;

    fn new(shape: FieldIncLimbShape) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldIncLimbReconstruction
    }

    fn rounds(&self) -> usize {
        self.shape.log_k_chunk + self.shape.log_t
    }

    fn degree(&self) -> usize {
        3
    }

    /// The booleanity legs sum to zero; the decode leg to the consumed
    /// reduced `FieldRdInc` claim.
    fn input_expression<F: Ring>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge(FieldIncLimbReconstructionChallenge::Gamma);
        gamma.pow(self.column_count())
            * opening(
                crate::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced(),
            )
    }

    fn output_expression<F: Ring>(&self) -> FieldInlineExpr<F> {
        let gamma = challenge::<F, _, _, _>(FieldIncLimbReconstructionChallenge::Gamma);
        let decode_power = self.column_count();
        let chunking = match BalancedIncChunking::new(self.shape.log_k_chunk) {
            Ok(chunking) => chunking,
            Err(_) => return FieldInlineExpr::zero(),
        };
        let columns = match field_inc_limb_columns(&self.shape) {
            Ok(columns) => columns,
            Err(_) => return FieldInlineExpr::zero(),
        };
        columns
            .into_iter()
            .enumerate()
            .map(|(position, column)| {
                let col = opening::<F, _, _, _>(field_inc_limb_column_opening_for(column));
                let coefficient =
                    recomposition_coefficient::<F>(chunking, self.shape.limbs, column)
                        .unwrap_or_else(F::zero);
                gamma.clone().pow(position)
                    * derived(FieldIncLimbReconstructionPublic::EqReference)
                    * (col.clone() * col.clone() - col.clone())
                    + gamma.clone().pow(decode_power)
                        * constant(coefficient)
                        * derived(FieldIncLimbReconstructionPublic::DigitValue)
                        * derived(FieldIncLimbReconstructionPublic::EqCycle)
                        * col
            })
            .fold(FieldInlineExpr::zero(), |acc, term| acc + term)
    }
}

/// The opening id of flat limb column `index` at this relation — the packed
/// limb object's slot claims.
pub fn field_inc_limb_column_opening(index: usize) -> FieldInlineOpeningId {
    field_inc_limb_column_opening_for(FieldInlineCommittedPolynomial::FieldIncLimbColumn(index))
}

fn field_inc_limb_column_opening_for(
    polynomial: FieldInlineCommittedPolynomial,
) -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        polynomial,
        FieldInlineRelationId::FieldIncLimbReconstruction,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::lattice::balanced_inc_value;
    use crate::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;
    use crate::protocols::field_inline::lattice::{canonical_limbs, column_selected_row};
    use crate::protocols::field_inline::FieldInlineDerivedId;
    use jolt_field::{Field, Fr, Ring};
    use jolt_poly::boolean_point_msb;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn shape() -> FieldIncLimbShape {
        FieldIncLimbShape {
            limbs: 4,
            log_t: 3,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn reconstruction_exposes_expected_dependencies() {
        let relation = FieldIncLimbReconstruction::new(shape());
        assert_eq!(
            FieldIncLimbReconstruction::id(),
            FieldInlineRelationId::FieldIncLimbReconstruction
        );
        assert_eq!(relation.rounds(), 8 + 3);
        assert_eq!(relation.degree(), 3);
        assert_eq!(relation.column_count(), 4 * 9);
        assert_eq!(
            relation.expected_output_openings::<Fr>().len(),
            relation.column_count()
        );
    }

    /// The claim struct's field order matches the packing plan's canonical
    /// column order, so the openings feed the packed slots one-for-one.
    #[test]
    fn output_claims_order_matches_the_packing_columns() {
        let relation = FieldIncLimbReconstruction::new(shape());
        let outputs = FieldIncLimbReconstructionOutputClaims::<Fr> {
            columns: vec![fr(1); relation.column_count()],
        };
        let expected: Vec<_> = field_inc_limb_columns(&shape())
            .unwrap()
            .into_iter()
            .map(field_inc_limb_column_opening_for)
            .collect();
        assert_eq!(outputs.canonical_order(), expected);
    }

    /// The whole relation over honest columns: Σ_z output_expression(z)
    /// equals input_expression with the consumed claim at a random cycle
    /// point — booleanity legs vanish, the decode leg recomposes.
    #[test]
    fn output_expression_sums_to_the_input_expression_over_honest_columns() {
        let shape = FieldIncLimbShape {
            limbs: 4,
            log_t: 2,
            log_k_chunk: 8,
        };
        let relation = FieldIncLimbReconstruction::new(shape);
        let chunking = BalancedIncChunking::new(shape.log_k_chunk).unwrap();
        let columns = field_inc_limb_columns(&shape).unwrap();
        let values = [
            fr(5),
            Fr::pow2(200) + fr(0x0123_4567_89ab_cdef),
            fr(0),
            fr(2).inverse().unwrap(),
        ];
        let gamma = fr(7);
        let r_reference: Vec<Fr> = (0..relation.rounds()).map(|i| fr(31 + i as u64)).collect();
        let r_cycle: Vec<Fr> = (0..shape.log_t).map(|i| fr(101 + i as u64)).collect();

        // The committed cells: col(digit-row, cycle) with digit-zero rows
        // omitted, exactly the packed witness.
        let cells = |column: FieldInlineCommittedPolynomial, digit: usize, cycle: usize| -> Fr {
            let hot =
                column_selected_row(chunking, &canonical_limbs(&values[cycle]), column).unwrap();
            fr(u64::from(hot == digit && hot != 0))
        };
        let eq =
            |point: &[Fr], index: usize| -> Fr { jolt_poly::eq_index_msb(point, index as u128) };

        let rounds = relation.rounds();
        let mut output_sum = fr(0);
        for z in 0..1usize << rounds {
            let digit = z >> shape.log_t;
            let cycle = z & ((1 << shape.log_t) - 1);
            output_sum += relation.output_expression::<Fr>().evaluate(
                |id| {
                    columns
                        .iter()
                        .enumerate()
                        .find_map(|(index, column)| {
                            (*id == field_inc_limb_column_opening(index))
                                .then(|| cells(*column, digit, cycle))
                        })
                        .unwrap_or_else(|| fr(0))
                },
                |_| gamma,
                |id| match *id {
                    FieldInlineDerivedId::FieldIncLimbReconstruction(
                        FieldIncLimbReconstructionPublic::EqReference,
                    ) => eq(&r_reference, z),
                    FieldInlineDerivedId::FieldIncLimbReconstruction(
                        FieldIncLimbReconstructionPublic::EqCycle,
                    ) => eq(&r_cycle, cycle),
                    FieldInlineDerivedId::FieldIncLimbReconstruction(
                        FieldIncLimbReconstructionPublic::DigitValue,
                    ) => balanced_inc_value(&boolean_point_msb::<Fr>(shape.log_k_chunk, digit)),
                    _ => fr(0),
                },
            );
        }

        let claim: Fr = values
            .iter()
            .enumerate()
            .map(|(cycle, value)| eq(&r_cycle, cycle) * *value)
            .sum();
        let input = relation.input_expression::<Fr>().evaluate(
            |id| {
                if *id == field_rd_inc_reduced() {
                    claim
                } else {
                    fr(0)
                }
            },
            |_| gamma,
            |_| fr(0),
        );
        assert_eq!(output_sum, input);
        let gamma_to_c = (0..relation.column_count()).fold(Fr::from_u64(1), |acc, _| acc * gamma);
        assert_eq!(input, gamma_to_c * claim);
    }
}
