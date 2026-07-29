//! The optimized RAM RA claim-reduction (stage 5) kernel.
//!
//! Same naive round loop as the reference over the same `T`-sized tables —
//! the summand
//! `(eq(r_cycle_raf, j) + γ·eq(r_cycle_rw, j) + γ²·eq(r_cycle_val, j)) · ra(r_address, j)`
//! — but the address-bound `ra` slice is computed in `O(T + K)` from the
//! session-shared RAM access columns instead of materializing and folding
//! the `(K × T)` grid: `ra_folded(j) = eq(r_address, addresses[j])`.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::ram_trace::RamAccessColumns;
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamRaClaimReduction<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let input_points = inputs.points;
        let expected_len = ram_log_k + log_t;
        for point in [
            input_points.raf(),
            input_points.read_write(),
            input_points.val_check(),
        ] {
            if point.len() != expected_len {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        // The shared address prefix (the relation's `derive_opening_points`
        // hard-checks that all three inputs agree on it).
        let r_address = &input_points.read_write()[..ram_log_k];

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << ram_log_k)?;
        let ra_folded = columns.fold_addresses(&eq_table(r_address));

        let opening_tables =
            BTreeMap::from([(ram_ra_claim_reduction(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleRaf),
                Polynomial::new(eq_table(&input_points.raf()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                Polynomial::new(eq_table(&input_points.read_write()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
                Polynomial::new(eq_table(&input_points.val_check()[ram_log_k..])),
            ),
        ]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    #[test]
    fn matches_reference_on_mixed_traffic() {
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        let ops = vec![
            RamOp::Write { word: 2, post: 7 },
            RamOp::Read { word: 2 },
            RamOp::Write { word: 5, post: 1 },
            RamOp::None,
            RamOp::Read { word: 5 },
            RamOp::Write { word: 2, post: 3 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let r_address = random_scalars(shape.log_k(), 53);
            let r_cycle_raf = random_scalars(shape.log_t, 59);
            let r_cycle_rw = random_scalars(shape.log_t, 61);
            let r_cycle_val = random_scalars(shape.log_t, 67);
            let gamma = random_scalars(1, 71)[0];

            let relation =
                RamRaClaimReduction::<Fr>::new(TraceDimensions::new(shape.log_t), shape.log_k());
            let claims = RamRaClaimReductionInputClaims {
                raf: Fr::from_u64(0),
                read_write: Fr::from_u64(0),
                val_check: Fr::from_u64(0),
            };
            let points = RamRaClaimReductionInputClaims::<Vec<Fr>> {
                raf: [r_address.clone(), r_cycle_raf.clone()].concat(),
                read_write: [r_address.clone(), r_cycle_rw.clone()].concat(),
                val_check: [r_address.clone(), r_cycle_val.clone()].concat(),
            };
            let challenges = RamRaClaimReductionChallenges { gamma };

            let mut reference_session = ProofSession::default();
            let reference = PrepareKernel::<Fr, _>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
            let mut session = ProofSession::default();
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();

            // The independently folded true input claim:
            // `Σ_j (eq_raf(j) + γ·eq_rw(j) + γ²·eq_val(j)) · ra_folded(j)`.
            let ra_folded =
                address_fold::<Fr>(witness, ram_ra_claim_reduction(), shape.log_t, &r_address)
                    .unwrap();
            let eq_raf = eq_table::<Fr>(&r_cycle_raf);
            let eq_rw = eq_table::<Fr>(&r_cycle_rw);
            let eq_val = eq_table::<Fr>(&r_cycle_val);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| (eq_raf[j] + gamma * eq_rw[j] + gamma * gamma * eq_val[j]) * ra_folded[j])
                .sum();

            assert_parity(
                reference,
                optimized,
                input_claim,
                &ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
                73,
            );
        });
    }
}
