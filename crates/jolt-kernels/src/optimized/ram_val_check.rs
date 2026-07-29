//! The optimized RAM value-check (stage 4) kernel.
//!
//! The round loop is unchanged from the reference — the summand
//! `inc(j) · ra(j) · (LT(j, r_cycle) + γ)` runs over `T`-sized tables
//! through the naive prover, so the round polynomials are byte-identical by
//! construction. Only `prepare` changes: the reference materializes the full
//! `(K × T)` RAM `ra` grid and folds it (`views::address_fold`); this kernel
//! computes the identical fold in `O(T + K)` from the session-shared RAM
//! access columns — the legacy `RaPolynomial` idea (one-hot `ra` kept as
//! per-cycle address indices) with the address point already fixed:
//! `ra_folded(j) = eq(r_address, addresses[j])` (0 on no-access cycles).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::{ram_inc_val_check, ram_ra_val_check};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, LtPolynomial, Polynomial};
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_witness::JoltWitnessPlane;

use super::ram_trace::RamAccessColumns;
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamValCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let ram_val_point: &[F] = &inputs.points.ram_val;
        if ram_val_point.len() != ram_log_k + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = ram_val_point.split_at(ram_log_k);

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << ram_log_k)?;
        let ra_folded = columns.fold_addresses(&eq_table(r_address));

        let lt_plus_gamma: Vec<F> = LtPolynomial::evaluations(r_cycle)
            .into_iter()
            .map(|lt| lt + inputs.challenges.gamma)
            .collect();

        let opening_tables = BTreeMap::from([
            (ram_ra_val_check(), Polynomial::new(ra_folded)),
            (
                ram_inc_val_check(),
                Polynomial::new(witness.oracle_table(ram_inc_val_check().polynomial_id())?),
            ),
        ]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma),
            Polynomial::new(lt_plus_gamma),
        )]);

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
    use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
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
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 9, post: 4 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 3, post: 8 },
            RamOp::Read { word: 15 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let r_address = random_scalars(shape.log_k(), 31);
            let r_cycle = random_scalars(shape.log_t, 37);
            let gamma = random_scalars(1, 41)[0];
            let relation = RamValCheck::<Fr>::new(
                TraceDimensions::new(shape.log_t),
                shape.log_k(),
                RamValCheckInit::full(Fr::from_u64(0)),
            );
            let claims = RamValCheckInputClaims {
                ram_val: Fr::from_u64(0),
                ram_val_final: Fr::from_u64(0),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims::<Vec<Fr>> {
                ram_val: [r_address.clone(), r_cycle.clone()].concat(),
                ram_val_final: r_address.clone(),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let challenges = RamValCheckChallenges { gamma };

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
            // `Σ_j inc(j) · ra_folded(j) · (LT(j, r_cycle) + γ)`.
            let ra_folded =
                address_fold::<Fr>(witness, ram_ra_val_check(), shape.log_t, &r_address).unwrap();
            let inc: Vec<Fr> = witness
                .oracle_table(ram_inc_val_check().polynomial_id())
                .unwrap();
            let lt = LtPolynomial::evaluations(&r_cycle);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| inc[j] * ra_folded[j] * (lt[j] + gamma))
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
                43,
            );
        });
    }
}
