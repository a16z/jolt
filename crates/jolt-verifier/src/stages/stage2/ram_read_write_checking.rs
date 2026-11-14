//! The stage 2 `RamReadWriteChecking` sumcheck instance.
//!
//! Owns the RAM read-write opening-point derivation and the `EqCycle`,
//! `CodeEqCycle` and `UnmappedEqCycle` public-value computations, in lockstep
//! with the BlindFold constraint's `ram::read_write_checking` formula. The
//! output check evaluates the cycle equality once and each mask once, in
//! O(log T + s·log K) field operations for `s` mask intervals.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamReadWriteChallenges, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::ReadWriteDimensions, JoltDerivedId, JoltRelationId, RamReadWritePublic,
};
use jolt_claims::{OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::{interval_mask_mle_msb, try_eq_mle};
use jolt_program::preprocess::ram::RamWordMasks;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM read/write value opening *values* from stage 1's outer
/// sumcheck. (Verifier-side constructor for the moved [`RamReadWriteInputClaims`].)
pub fn ram_read_write_input_values_from_upstream<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
) -> RamReadWriteInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    RamReadWriteInputClaims {
        ram_read_value: outer.ram_read_value,
        ram_write_value: outer.ram_write_value,
        load: outer.load,
        store: outer.store,
    }
}

#[derive(Clone)]
pub struct RamReadWriteChecking<F: JoltField> {
    symbolic: relations::ram::ReadWriteChecking,
    dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    product_tau_low: Vec<F>,
    masks: RamWordMasks,
}

impl<F: JoltField> RamReadWriteChecking<F> {
    pub fn new(
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        product_tau_low: Vec<F>,
        masks: RamWordMasks,
    ) -> Self {
        Self {
            symbolic: relations::ram::ReadWriteChecking::new(dimensions),
            dimensions,
            ram_log_k,
            product_tau_low,
            masks,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamReadWriteChecking,
        reason: reason.to_string(),
    }
}

/// A mask's multilinear extension at the address point, in O(log K) per
/// interval.
fn mask_mle<F: JoltField>(
    intervals: &[(usize, usize)],
    r_address: &[F],
) -> Result<F, VerifierError> {
    interval_mask_mle_msb(intervals, r_address).map_err(public_input_failed)
}

impl<F: JoltField> RamReadWriteChecking<F> {
    /// Splits the opening point `[r_address(log_k) || r_cycle(log_t)]` and
    /// evaluates the `EqCycle` public `eq(τ_low, r_cycle)`.
    fn address_and_eq_cycle<'a>(&self, point: &'a [F]) -> Result<(&'a [F], F), VerifierError> {
        if point.len() < self.ram_log_k {
            return Err(public_input_failed(format!(
                "RAM read-write opening point is too short: expected at least {}, got {}",
                self.ram_log_k,
                point.len()
            )));
        }
        let (r_address, r_cycle) = point.split_at(self.ram_log_k);
        let eq_cycle = try_eq_mle(&self.product_tau_low, r_cycle).map_err(public_input_failed)?;
        Ok((r_address, eq_cycle))
    }

    /// The public weight `id` at the opening point, from its address half and
    /// the cycle equality there: `EqCycle` itself, or that equality times the
    /// mask's multilinear extension at `r_address`. Evaluates at most one mask.
    fn public_value(
        &self,
        id: RamReadWritePublic,
        r_address: &[F],
        eq_cycle: F,
    ) -> Result<F, VerifierError> {
        let intervals = match id {
            RamReadWritePublic::EqCycle => return Ok(eq_cycle),
            RamReadWritePublic::CodeEqCycle => &self.masks.write_protected,
            RamReadWritePublic::UnmappedEqCycle => &self.masks.unmapped,
        };
        Ok(eq_cycle * mask_mle(intervals, r_address)?)
    }

    pub fn write_protected_word_intervals(&self) -> &[(usize, usize)] {
        &self.masks.write_protected
    }

    pub fn unmapped_word_intervals(&self) -> &[(usize, usize)] {
        &self.masks.unmapped
    }

    pub fn dimensions(&self) -> ReadWriteDimensions {
        self.dimensions
    }

    pub fn ram_log_k(&self) -> usize {
        self.ram_log_k
    }

    pub fn product_tau_low(&self) -> &[F] {
        &self.product_tau_low
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RamReadWriteChecking<F> {
    type Symbolic = relations::ram::ReadWriteChecking;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamReadWriteInputClaims<Vec<F>>,
    ) -> Result<RamReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(RamReadWriteOutputClaims {
            val: opening_point.clone(),
            ra: opening_point.clone(),
            inc: opening_point,
        })
    }

    /// Evaluates the cycle equality once, and each mask once because each
    /// masked weight occurs once in the output expression. The generic
    /// evaluator calls [`derive_output_term`](Self::derive_output_term) per
    /// leaf, and `EqCycle` weights three of the five claims, so it would
    /// evaluate the cycle equality five times.
    fn expected_output(
        &self,
        _input_points: &RamReadWriteInputClaims<Vec<F>>,
        output_values: &RamReadWriteOutputClaims<F>,
        output_points: &RamReadWriteOutputClaims<Vec<F>>,
        challenges: &RamReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let (r_address, eq_cycle) = self.address_and_eq_cycle(output_points.val())?;
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| {
                output_values
                    .resolve_output(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| {
                let JoltDerivedId::RamReadWrite(public_id) = id else {
                    return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
                };
                self.public_value(*public_id, r_address, eq_cycle)
            },
        )
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamReadWriteInputClaims<Vec<F>>,
        output_points: &RamReadWriteOutputClaims<Vec<F>>,
        _challenges: &RamReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let (r_address, eq_cycle) = self.address_and_eq_cycle(output_points.val())?;
        self.public_value(*public_id, r_address, eq_cycle)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    reason = "brute-force field sums over test fixtures"
)]
mod tests {
    use super::*;
    use jolt_field::{Fr, One, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// `eq(bits, r)` for an index whose most significant bit pairs with `r[0]`.
    fn eq_at_index(index: usize, r: &[Fr]) -> Fr {
        r.iter()
            .enumerate()
            .map(|(j, &r_j)| {
                if (index >> (r.len() - 1 - j)) & 1 == 1 {
                    r_j
                } else {
                    Fr::one() - r_j
                }
            })
            .product()
    }

    /// A mask's multilinear extension as a sum over the Boolean hypercube.
    fn mask_by_hypercube_sum(intervals: &[(usize, usize)], r_address: &[Fr]) -> Fr {
        (0..1 << r_address.len())
            .filter(|&k| {
                intervals
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&k))
            })
            .map(|k| eq_at_index(k, r_address))
            .sum()
    }

    /// The output check equals the relation's output expression with each
    /// public term computed by brute force: the cycle equality as a product of
    /// per-variable factors and each mask as a sum over all `2^log_k` addresses.
    #[test]
    fn expected_output_matches_brute_force_public_terms() {
        let (log_t, log_k) = (4, 5);
        let tau: Vec<Fr> = (11..).take(log_t).map(fr).collect();
        let masks = RamWordMasks {
            write_protected: vec![(0, 2), (3, 6)],
            unmapped: vec![(6, 9), (20, 32)],
        };
        let relation = RamReadWriteChecking::new(
            ReadWriteDimensions::new(log_t, log_k, 2, 1),
            log_k,
            tau.clone(),
            masks.clone(),
        );
        let point: Vec<Fr> = (0..).take(log_k + log_t).map(|i| fr(3 + 2 * i)).collect();
        let (r_address, r_cycle) = point.split_at(log_k);
        let eq_cycle: Fr = tau
            .iter()
            .zip(r_cycle)
            .map(|(&t, &r)| t * r + (Fr::one() - t) * (Fr::one() - r))
            .product();
        let code = mask_by_hypercube_sum(&masks.write_protected, r_address);
        let unmapped = mask_by_hypercube_sum(&masks.unmapped, r_address);

        let output_points = RamReadWriteOutputClaims {
            val: point.clone(),
            ra: point.clone(),
            inc: point.clone(),
        };
        let output_values = RamReadWriteOutputClaims {
            val: fr(5),
            ra: fr(7),
            inc: fr(9),
        };
        let challenges = RamReadWriteChallenges { gamma: fr(13) };
        let expected = relation
            .symbolic()
            .output_expression::<Fr>()
            .try_evaluate(
                |id| Ok::<_, VerifierError>(output_values.resolve_output(id).unwrap()),
                |id| Ok(challenges.resolve_challenge(id).unwrap()),
                |id| {
                    let JoltDerivedId::RamReadWrite(public_id) = id else {
                        return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
                    };
                    Ok(match public_id {
                        RamReadWritePublic::EqCycle => eq_cycle,
                        RamReadWritePublic::CodeEqCycle => eq_cycle * code,
                        RamReadWritePublic::UnmappedEqCycle => eq_cycle * unmapped,
                    })
                },
            )
            .unwrap();
        let actual = relation
            .expected_output(
                &RamReadWriteInputClaims::default(),
                &output_values,
                &output_points,
                &challenges,
            )
            .unwrap();
        assert_eq!(actual, expected);
    }
}
