//! A stage declared outside `jolt-prover` supplies its own witness plane.

#![expect(clippy::unwrap_used, reason = "test crate")]

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_claims::{opening, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck};
use jolt_field::{Fr, JoltField, Ring};
use jolt_kernels::{
    KernelError, KernelSlots, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
    SumcheckKernelError, WitnessPlane,
};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_prover::{impl_stage_prover, StageProver as _};
use jolt_sumcheck::{ClearSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckError};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use jolt_verifier::VerifierError;

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
struct ToyInputs<C> {
    #[opening(UnexpandedPC, from = RegistersValEvaluation)]
    claimed_sum: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[relation(RegistersValEvaluation)]
struct ToyOutputs<C> {
    #[opening(LookupOutput)]
    value: C,
}

#[derive(Clone)]
struct ToySymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for ToySymbolic {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = ToyInputs<C>;
    type Outputs<C> = ToyOutputs<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersValEvaluation
    }

    fn rounds(&self) -> usize {
        self.rounds
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::UnexpandedPC,
            Self::id(),
        ))
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::LookupOutput,
            Self::id(),
        ))
    }
}

#[derive(Clone)]
struct ToyRelation<F: JoltField> {
    symbolic: ToySymbolic,
    field: PhantomData<F>,
}

impl<F: JoltField> ConcreteSumcheck<F> for ToyRelation<F> {
    type Symbolic = ToySymbolic;

    fn symbolic(&self) -> &ToySymbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ToyInputs<Vec<F>>,
    ) -> Result<ToyOutputs<Vec<F>>, VerifierError> {
        Ok(ToyOutputs {
            value: sumcheck_point.to_vec(),
        })
    }
}

struct ToyTables<F: JoltField> {
    values: Vec<F>,
}

struct ToyPlane;

impl<F: JoltField> WitnessPlane<F> for ToyPlane {
    type Ref<'w>
        = &'w ToyTables<F>
    where
        F: 'w;
}

#[derive(SumcheckBatch)]
struct ToySumchecks<F: JoltField> {
    table: ToyRelation<F>,
}

toy_sumchecks_members!(impl_stage_prover plane = ToyPlane,);

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct DenseKernel<F: JoltField> {
    evals: Vec<F>,
    num_rounds: usize,
}

impl<F: JoltField> DenseKernel<F> {
    fn bind(&mut self, challenge: F) {
        let half = self.evals.len() / 2;
        let (low, high) = self.evals.split_at_mut(half);
        for (low, high) in low.iter_mut().zip(high) {
            *low += challenge * (*high - *low);
        }
        self.evals.truncate(half);
    }
}

impl<F: JoltField> ProveRounds<F> for DenseKernel<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: F = self.evals[..half].iter().copied().sum();
        let eval_1: F = self.evals[half..].iter().copied().sum();
        if eval_0 + eval_1 != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: eval_0 + eval_1,
            });
        }
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for DenseKernel<F> {
    type Relation = ToyRelation<F>;

    fn output_claims(
        &mut self,
        _inputs: &ToyInputs<F>,
    ) -> Result<ToyOutputs<F>, SumcheckKernelError<F>> {
        match self.evals.as_slice() {
            [value] => Ok(ToyOutputs { value: *value }),
            _ => Err(SumcheckKernelError::NotFullyBound {
                remaining: self.evals.len().trailing_zeros() as usize,
            }),
        }
    }
}

struct TablePrepare;

impl<F: JoltField> PrepareKernel<F, ToyRelation<F>, ToyPlane> for TablePrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &ToyTables<F>,
        inputs: ProverInputs<'_, F, ToyRelation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ToyRelation<F>>>, KernelError<F>> {
        let num_rounds = inputs.relation.rounds();
        if !witness.values.len().is_power_of_two()
            || witness.values.len().trailing_zeros() as usize != num_rounds
        {
            return Err(KernelError::InvariantViolation {
                reason: "the toy table length must match the relation's rounds",
            });
        }
        Ok(Box::new(DenseKernel {
            evals: witness.values.clone(),
            num_rounds,
        }))
    }
}

#[derive(KernelSlots)]
struct ToyKernels<F: JoltField> {
    table: Box<dyn PrepareKernel<F, ToyRelation<F>, ToyPlane>>,
}

#[test]
fn external_stage_with_own_plane_matches_verifier_twin() {
    let tables = ToyTables {
        values: [3, 7, 11, 17, 23, 31, 41, 53]
            .into_iter()
            .map(Fr::from_u64)
            .collect(),
    };
    let sumchecks = ToySumchecks {
        table: ToyRelation {
            symbolic: ToySymbolic::new(3),
            field: PhantomData,
        },
    };
    let inputs = ToyInputClaims {
        table: ToyInputs {
            claimed_sum: tables.values.iter().copied().sum(),
        },
    };
    let kernels = ToyKernels {
        table: Box::new(TablePrepare),
    };
    let input_points = sumchecks.empty_input_points();
    let mut prover_transcript = Blake2bTranscript::new(b"external-plane-stage");
    let challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();
    let proved = sumchecks
        .prove(
            &kernels,
            &mut ProofSession::default(),
            &mut SequentialRounds,
            &tables,
            &inputs,
            &input_points,
            &challenges,
            ClearSumcheckRecorder::<Fr, Fr>::new(),
            &mut prover_transcript,
        )
        .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(b"external-plane-stage");
    let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let verified_points = sumchecks
        .verify_clear(
            &inputs,
            &input_points,
            &verifier_challenges,
            &proved.output_claims,
            &proved.recorded.proof,
            &mut verifier_transcript,
            0,
        )
        .unwrap();
    sumchecks.append_output_claims(&mut verifier_transcript, &proved.output_claims);

    assert_eq!(proved.output_points, verified_points);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    assert_eq!(
        proved.output_claims.table.value,
        Polynomial::new(tables.values).evaluate(&proved.output_points.table.value),
    );
}
