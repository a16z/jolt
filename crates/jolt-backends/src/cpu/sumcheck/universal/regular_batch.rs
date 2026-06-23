use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck_prover::{
    BackendError as UniversalBackendError, BatchedSumcheckSpec,
    SumcheckBackend as UniversalSumcheckBackend,
};
use jolt_witness::WitnessNamespace;

use crate::{BackendError, SumcheckBackend, SumcheckRegularBatchState};

/// Regular-batch sumcheck over a pre-materialized [`SumcheckRegularBatchState`].
///
/// The universal handler owns front-loaded dummy rounds; this adapter filters the
/// legacy kernel output down to the active instance indices requested each round.
pub struct PreMaterializedRegularBatchBackend<F: Field, B, N: WitnessNamespace> {
    inner: B,
    state: SumcheckRegularBatchState<F>,
    max_rounds: usize,
    running_claims: Vec<F>,
    bound_round: Option<usize>,
    _marker: core::marker::PhantomData<(F, N)>,
}

impl<F, B, N> PreMaterializedRegularBatchBackend<F, B, N>
where
    F: Field,
    N: WitnessNamespace,
    B: SumcheckBackend<F, N>,
{
    pub fn new(inner: B, state: SumcheckRegularBatchState<F>, max_rounds: usize) -> Self {
        let running_claims = vec![F::zero(); state.instances.len()];
        Self {
            inner,
            state,
            max_rounds,
            running_claims,
            bound_round: None,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F, B, N> UniversalSumcheckBackend<F> for PreMaterializedRegularBatchBackend<F, B, N>
where
    F: Field,
    N: WitnessNamespace,
    B: SumcheckBackend<F, N>,
{
    type State = ();

    fn start(
        &mut self,
        spec: &BatchedSumcheckSpec<F>,
    ) -> Result<Self::State, UniversalBackendError> {
        if spec.instances.len() != self.state.instances.len() {
            return Err(UniversalBackendError::RoundPolynomialCountMismatch {
                expected: self.state.instances.len(),
                got: spec.instances.len(),
            });
        }
        if spec.num_rounds() != self.max_rounds {
            return Err(UniversalBackendError::UnsupportedRelation { label: spec.label });
        }
        Ok(())
    }

    fn round_polynomials(
        &mut self,
        _state: &Self::State,
        round: usize,
        active: &[usize],
        claims: &[F],
    ) -> Result<Vec<UnivariatePoly<F>>, UniversalBackendError> {
        if active.len() != claims.len() {
            return Err(UniversalBackendError::RoundPolynomialCountMismatch {
                expected: active.len(),
                got: claims.len(),
            });
        }

        let mut full_claims = self.running_claims.clone();
        for (&instance_index, &claim) in active.iter().zip(claims) {
            full_claims[instance_index] = claim;
        }
        self.running_claims = full_claims.clone();

        let rounds = self
            .inner
            .evaluate_sumcheck_regular_batch_round(
                &mut self.state,
                round,
                self.max_rounds,
                &full_claims,
            )
            .map_err(map_backend_error)?;

        active
            .iter()
            .map(|&instance_index| {
                rounds
                    .iter()
                    .find(|round| round.instance_index == instance_index)
                    .map(|round| round.polynomial.clone())
                    .ok_or(UniversalBackendError::InvalidActiveIndex {
                        index: instance_index,
                        batch_size: self.state.instances.len(),
                    })
            })
            .collect()
    }

    fn bind(
        &mut self,
        _state: &mut Self::State,
        round: usize,
        _instance: usize,
        challenge: F,
    ) -> Result<(), UniversalBackendError> {
        if self.bound_round == Some(round) {
            return Ok(());
        }
        self.inner
            .bind_sumcheck_regular_batch_state(&mut self.state, round, self.max_rounds, challenge)
            .map_err(map_backend_error)?;
        self.bound_round = Some(round);
        Ok(())
    }
}

fn map_backend_error(error: BackendError) -> UniversalBackendError {
    let _ = error;
    UniversalBackendError::UnsupportedRelation {
        label: "cpu_regular_batch",
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests may unwrap on assertion failures")]

    use std::{cell::RefCell, rc::Rc};

    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::{NamespaceId, WitnessNamespace};

    use super::*;
    use crate::{Backend, SumcheckRegularBatchInstance};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum TestNamespace {}

    impl WitnessNamespace for TestNamespace {
        type ChallengeId = u8;
        type CommittedId = u8;
        type OpeningId = u8;
        type PublicId = u8;
        type VirtualId = u8;

        const ID: NamespaceId = NamespaceId::new("regular_batch_adapter_test");
    }

    #[derive(Clone, Default)]
    struct CountingBackend {
        bound_rounds: Rc<RefCell<Vec<usize>>>,
    }

    impl Backend for CountingBackend {
        fn name(&self) -> &'static str {
            "counting_regular_batch"
        }
    }

    impl SumcheckBackend<Fr, TestNamespace> for CountingBackend {
        type Proof = ();

        fn bind_sumcheck_regular_batch_state(
            &mut self,
            _state: &mut SumcheckRegularBatchState<Fr>,
            round: usize,
            _max_rounds: usize,
            _challenge: Fr,
        ) -> Result<(), BackendError> {
            self.bound_rounds.borrow_mut().push(round);
            Ok(())
        }
    }

    #[test]
    fn binds_shared_regular_batch_state_once_per_round() {
        let counter = CountingBackend::default();
        let bound_rounds = counter.bound_rounds.clone();
        let state = SumcheckRegularBatchState::new(
            "test",
            vec![
                SumcheckRegularBatchInstance::new_products("a", Fr::from_u64(0), vec![], vec![]),
                SumcheckRegularBatchInstance::new_products("b", Fr::from_u64(0), vec![], vec![]),
            ],
        );
        let mut backend =
            PreMaterializedRegularBatchBackend::<Fr, _, TestNamespace>::new(counter, state, 2);

        backend.bind(&mut (), 0, 0, Fr::from_u64(1)).unwrap();
        backend.bind(&mut (), 0, 1, Fr::from_u64(1)).unwrap();
        backend.bind(&mut (), 1, 0, Fr::from_u64(1)).unwrap();

        assert_eq!(&*bound_rounds.borrow(), &[0, 1]);
    }
}
