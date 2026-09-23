//! The optimized RAM RAF-evaluation (stage 2) kernel.
//!
//! Same naive round loop as the reference over the same `K`-sized tables —
//! the summand `unmap(k) · ra_folded(k)` — but the cycle-folded `ra` is
//! computed in `O(T + K)` from the session-shared RAM access columns
//! instead of materializing and folding the `(K × T)` grid:
//! `ra_folded(k) = Σ_{j : addresses[j] = k} eq(τ_low, j)`.
//!
//! Only the default read-write config is supported (phase 1 = all cycle
//! rounds), where the relation's rounds equal `log_K` — same bar as the
//! reference kernel.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_ra_raf_evaluation;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::JoltWitnessPlane;

use super::ram_trace::SharedRamAddresses;
use super::OptimizedBackend;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, RamRafEvaluation<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRafEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRafEvaluation<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.read_write_dimensions();
        let ram_log_k = relation.ram_log_k();
        let lowest_address = relation.lowest_address();
        let tau_low = relation.tau_low();
        if dimensions.raf_evaluation_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "optimized RAM RAF evaluation supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }
        if tau_low.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM RAF evaluation tau_low disagrees with the trace geometry",
            });
        }

        let addresses = 1usize << ram_log_k;
        let address_column = SharedRamAddresses::shared(session, witness, dimensions.log_t())?;
        super::ram_trace::validate_addresses(&address_column, addresses)?;
        let ra_folded = super::ram_trace::fold_cycles(&address_column, tau_low, addresses);
        let unmap: Vec<F> = (0..addresses as u64)
            .map(|k| F::from_u64(8 * k + lowest_address))
            .collect();

        let opening_tables =
            BTreeMap::from([(ram_ra_raf_evaluation(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
            Polynomial::new(unmap),
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
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
    use jolt_claims::protocols::jolt::relations::ram::RamRafEvaluationInputClaims;
    use jolt_claims::NoChallenges;
    use jolt_field::{Fr, Ring};

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::cycle_fold;
    use crate::ReferenceBackend;

    #[test]
    fn matches_reference_on_mixed_traffic() {
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        let ops = vec![
            RamOp::Write { word: 6, post: 2 },
            RamOp::Read { word: 6 },
            RamOp::None,
            RamOp::Write { word: 3, post: 9 },
            RamOp::Read { word: 3 },
            RamOp::Read { word: 6 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let tau_low = random_scalars(shape.log_t, 83);
            let read_write_dimensions =
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k());
            let relation = RamRafEvaluation::<Fr>::new(
                read_write_dimensions,
                RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap(),
                shape.log_k(),
                super::super::testing::fixture_lowest_address(),
                tau_low.clone(),
            );
            let claims = RamRafEvaluationInputClaims {
                ram_address: Fr::from_u64(0),
            };
            let points = RamRafEvaluationInputClaims::<Vec<Fr>>::default();
            let challenges = NoChallenges::default();

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
            // `Σ_k unmap(k) · ra_folded(k)`.
            let ra_folded =
                cycle_fold::<Fr>(witness, ram_ra_raf_evaluation(), shape.log_k(), &tau_low)
                    .unwrap();
            let lowest = super::super::testing::fixture_lowest_address();
            let input_claim = (0..shape.ram_k as u64)
                .map(|k| Fr::from_u64(8 * k + lowest) * ra_folded[k as usize])
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
                89,
            );
        });
    }

    /// A non-default phase split (RAF rounds exceeding `log_K`) is rejected
    /// as `Unsupported` instead of misproving.
    #[test]
    fn rejects_non_default_phase_split() {
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        with_ram_fixture(shape, vec![RamOp::None; 3], |witness| {
            let read_write_dimensions = ReadWriteDimensions::new(
                shape.log_t,
                shape.log_k(),
                shape.log_t - 1,
                shape.log_k() + 1,
            );
            let relation = RamRafEvaluation::<Fr>::new(
                read_write_dimensions,
                RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap(),
                shape.log_k(),
                super::super::testing::fixture_lowest_address(),
                random_scalars(shape.log_t, 83),
            );
            let claims = RamRafEvaluationInputClaims {
                ram_address: Fr::from_u64(0),
            };
            let points = RamRafEvaluationInputClaims::<Vec<Fr>>::default();
            let challenges = NoChallenges::default();
            let result = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            );
            assert!(matches!(
                result.map(|_| ()),
                Err(KernelError::Unsupported { .. })
            ));
        });
    }
}
