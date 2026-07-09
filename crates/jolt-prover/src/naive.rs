//! The naive reference prover: a slow but correct prover for *any* relation,
//! derived from the observation that a relation's output `Expr` *is* its
//! sumcheck summand.
//!
//! Its state is one dense table per `Expr` leaf: `Opening` leaves resolve to
//! polynomials materialized from the witness, `Challenge` leaves to the drawn
//! scalars, and `Derived` leaves to the polynomial forms of what the
//! verifier's `derive_output_term` evaluates at a point (eq/LT/selector
//! tables, materialized over the domain by the caller). Each round message is
//! `Expr::try_evaluate` run pointwise over the remaining hypercube — cost
//! `O((degree+1) · 2^rounds · |Expr|)` per round, a **test oracle at harness
//! scale, never a performance path**.
//!
//! Two self-checks pin it: the engine's running-claim check
//! (`s(0) + s(1) == previous_claim`, re-checked per member here), and
//! [`validate_derived_tables`](NaiveSumcheckProver::validate_derived_tables) —
//! each bound `Derived` table's final value must equal `derive_output_term`
//! at the bound point, tying the hand-materialized tables to the verifier's
//! scalar path.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use jolt_claims::{InputClaims, OutputClaims, Source, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{ProveSumcheck, ProverError};

/// See the module docs. Construct with every leaf table the relation's output
/// expression references; [`new`](Self::new) validates coverage and sizes so
/// the round loop cannot miss a leaf.
pub struct NaiveSumcheckProver<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    relation: R,
    challenges: ConcreteSumcheckChallenges<F, R>,
    /// The expression's `Challenge` leaves pre-resolved to scalars at
    /// construction, so the round loop reads plain `Sync` data (the
    /// `Challenges` struct itself carries no `Sync` bound).
    challenge_values: BTreeMap<JoltChallengeId, F>,
    opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>>,
    derived_tables: BTreeMap<JoltDerivedId, Polynomial<F>>,
    rounds_bound: usize,
}

impl<F, R> NaiveSumcheckProver<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    /// Validate that every leaf of the relation's output expression is
    /// resolvable — each `Opening`/`Derived` factor has a table of exactly
    /// `2^rounds` evaluations, each `Challenge` factor a drawn scalar — and
    /// assemble the prover.
    pub fn new(
        relation: R,
        challenges: ConcreteSumcheckChallenges<F, R>,
        opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>>,
        derived_tables: BTreeMap<JoltDerivedId, Polynomial<F>>,
    ) -> Result<Self, ProverError<F>> {
        let expected_len = 1usize << relation.rounds();
        let check_len = |table: &Polynomial<F>, id: &dyn core::fmt::Debug| {
            if table.len() == expected_len {
                Ok(())
            } else {
                Err(ProverError::TableSizeMismatch {
                    table: format!("{id:?}"),
                    expected: expected_len,
                    got: table.len(),
                })
            }
        };
        let mut challenge_values = BTreeMap::new();
        for term in &relation.symbolic().output_expression::<F>().terms {
            for factor in &term.factors {
                match factor {
                    Source::Opening(id) => {
                        let table = opening_tables
                            .get(id)
                            .ok_or(ProverError::MissingOpeningTable { id: *id })?;
                        check_len(table, id)?;
                    }
                    Source::Derived(id) => {
                        let table = derived_tables
                            .get(id)
                            .ok_or(ProverError::MissingDerivedTable { id: *id })?;
                        check_len(table, id)?;
                    }
                    Source::Challenge(id) => {
                        let value = challenges
                            .resolve_challenge(id)
                            .ok_or(ProverError::MissingChallenge { id: *id })?;
                        let _ = challenge_values.insert(*id, value);
                    }
                }
            }
        }

        Ok(Self {
            relation,
            challenges,
            challenge_values,
            opening_tables,
            derived_tables,
            rounds_bound: 0,
        })
    }

    /// The drawn challenges this prover resolves `Challenge` leaves against.
    pub fn challenges(&self) -> &ConcreteSumcheckChallenges<F, R> {
        &self.challenges
    }

    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn require_fully_bound(&self) -> Result<(), ProverError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(ProverError::NotFullyBound { remaining }),
        }
    }

    /// The `Derived`-leaf cross-check: each bound table's final value must
    /// equal the verifier's `derive_output_term` at the bound point. This is
    /// what pins a hand-materialized eq/LT/selector table (orientation,
    /// binding order, contents) to the relation's scalar path.
    pub fn validate_derived_tables(
        &self,
        input_points: &SumcheckInputPoints<F, R>,
        output_points: &SumcheckOutputPoints<F, R>,
    ) -> Result<(), ProverError<F>> {
        self.require_fully_bound()?;
        for (id, table) in &self.derived_tables {
            let expected = self.relation.derive_output_term(
                id,
                input_points,
                output_points,
                &self.challenges,
            )?;
            let got = table.evals()[0];
            if got != expected {
                return Err(ProverError::DerivedTableDrift {
                    id: *id,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

impl<F, R> ProveRounds<F> for NaiveSumcheckProver<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let half = (1usize << self.remaining_rounds()) / 2;
        let degree = self.relation.degree();
        let expression = self.relation.symbolic().output_expression::<F>();
        let opening_tables = &self.opening_tables;
        let derived_tables = &self.derived_tables;
        let challenge_values = &self.challenge_values;

        // msg(t) = Σ_y Expr(leaf tables partially evaluated at (t, y)),
        // sampled at t = 0..=degree and interpolated.
        let mut evals = Vec::with_capacity(degree + 1);
        for t in 0..=degree {
            let point = F::from_u64(t as u64);
            let term = |y: usize| -> Result<F, SumcheckError<F>> {
                expression.try_evaluate(
                    |id| {
                        opening_tables
                            .get(id)
                            .map(|table| table.sumcheck_round_eval(y, point))
                            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })
                    },
                    |id| {
                        challenge_values
                            .get(id)
                            .copied()
                            .ok_or(SumcheckError::MissingEvaluationSource { kind: "challenge" })
                    },
                    |id| {
                        derived_tables
                            .get(id)
                            .map(|table| table.sumcheck_round_eval(y, point))
                            .ok_or(SumcheckError::MissingEvaluationSource { kind: "derived" })
                    },
                )
            };

            #[cfg(feature = "parallel")]
            let sum = (0..half)
                .into_par_iter()
                .map(term)
                .try_reduce(F::zero, |left, right| Ok(left + right))?;
            #[cfg(not(feature = "parallel"))]
            let sum = (0..half).try_fold(F::zero(), |acc, y| Ok(acc + term(y)?))?;

            evals.push(sum);
        }

        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn ingest_challenge(&mut self, challenge: F, _round: usize) -> Result<(), SumcheckError<F>> {
        for table in self.opening_tables.values_mut() {
            table.bind(challenge);
        }
        for table in self.derived_tables.values_mut() {
            table.bind(challenge);
        }
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F, R> ProveSumcheck<F> for NaiveSumcheckProver<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    type Relation = R;

    fn relation(&self) -> &R {
        &self.relation
    }

    fn output_claims(&mut self) -> Result<SumcheckOutputClaims<F, R>, ProverError<F>> {
        self.require_fully_bound()?;
        let opening_tables = &self.opening_tables;
        SumcheckOutputClaims::<F, R>::from_opening_values(|id| {
            opening_tables.get(id).map(|table| table.evals()[0])
        })
        .map_err(ProverError::from)
    }
}

/// A hand-built toy relation exercising every leaf kind (scalar, `Vec`
/// family, absent `Option`, `Challenge`, `Derived`) through the naive prover
/// against the relation's own algebra — the single-member rehearsal of a
/// stage recipe: head choreography → engine round loop → typed extraction →
/// `expected_output` fold → clear-verifier twin.
#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::{
        InstructionReadRafChallenge, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
        JoltVirtualPolynomial,
    };
    use jolt_claims::{
        challenge, derived, opening, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
    };
    use jolt_field::{Field, Fr, FromPrimitiveInt, RingCore};
    use jolt_poly::{EqPolynomial, Polynomial};
    use jolt_sumcheck::{
        append_sumcheck_claim, prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder,
        ProveRounds, SumcheckRecorder, OPENING_CLAIM_TRANSCRIPT_LABEL,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::VerifierError;

    use super::NaiveSumcheckProver;
    use crate::{ProveSumcheck, ProverError};

    const TOY_RELATION: JoltRelationId = JoltRelationId::RegistersValEvaluation;

    fn virt(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(polynomial, TOY_RELATION)
    }

    #[derive(jolt_claims::SumcheckChallenges)]
    struct ToyChallenges<F> {
        #[challenge(InstructionReadRafChallenge::Gamma)]
        gamma: F,
    }

    #[derive(jolt_claims::InputClaims)]
    struct ToyInputs<C> {
        #[opening(UnexpandedPC, from = RegistersValEvaluation)]
        total: C,
    }

    #[derive(jolt_claims::OutputClaims)]
    #[relation(RegistersValEvaluation)]
    struct ToyOutputs<C> {
        #[opening(LookupOutput)]
        a: C,
        #[opening(LeftLookupOperand)]
        b: C,
        #[opening(InstructionRa)]
        instruction_ra: Vec<C>,
        #[opening(RightLookupOperand)]
        c: C,
        #[opening(untrusted_advice)]
        untrusted: Option<C>,
    }

    struct ToySymbolic {
        rounds: usize,
    }

    impl SymbolicSumcheck for ToySymbolic {
        type RelationId = JoltRelationId;
        type OpeningId = JoltOpeningId;
        type DerivedId = JoltDerivedId;
        type ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId;
        type Shape = usize;
        type Challenges<F> = ToyChallenges<F>;
        type Inputs<C> = ToyInputs<C>;
        type Outputs<C> = ToyOutputs<C>;

        fn new(shape: usize) -> Self {
            Self { rounds: shape }
        }

        fn id() -> JoltRelationId {
            TOY_RELATION
        }

        fn rounds(&self) -> usize {
            self.rounds
        }

        fn degree(&self) -> usize {
            3
        }

        fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
            opening(virt(JoltVirtualPolynomial::UnexpandedPC))
        }

        fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
            opening(virt(JoltVirtualPolynomial::LookupOutput))
                * opening(virt(JoltVirtualPolynomial::LeftLookupOperand))
                * derived(JoltDerivedId::Test)
                + challenge(InstructionReadRafChallenge::Gamma)
                    * opening(virt(JoltVirtualPolynomial::InstructionRa(0)))
                    * opening(virt(JoltVirtualPolynomial::InstructionRa(1)))
                + opening(virt(JoltVirtualPolynomial::RightLookupOperand))
        }
    }

    struct ToyRelation<F: Field> {
        symbolic: ToySymbolic,
        reference_point: Vec<F>,
    }

    impl<F: Field> ConcreteSumcheck<F> for ToyRelation<F> {
        type Symbolic = ToySymbolic;

        fn symbolic(&self) -> &ToySymbolic {
            &self.symbolic
        }

        fn derive_opening_points(
            &self,
            sumcheck_point: &[F],
            _input_points: &ToyInputs<Vec<F>>,
        ) -> Result<ToyOutputs<Vec<F>>, VerifierError> {
            let point = sumcheck_point.to_vec();
            Ok(ToyOutputs {
                a: point.clone(),
                b: point.clone(),
                instruction_ra: vec![point.clone(), point.clone()],
                c: point,
                untrusted: None,
            })
        }

        fn derive_output_term(
            &self,
            id: &JoltDerivedId,
            _input_points: &ToyInputs<Vec<F>>,
            output_points: &ToyOutputs<Vec<F>>,
            _challenges: &ToyChallenges<F>,
        ) -> Result<F, VerifierError> {
            match id {
                JoltDerivedId::Test => {
                    Ok(EqPolynomial::new(self.reference_point.clone()).evaluate(output_points.a()))
                }
                _ => Err(VerifierError::MissingStageClaimDerived { id: *id }),
            }
        }
    }

    const ROUNDS: usize = 4;
    const SIZE: usize = 1 << ROUNDS;

    fn dense(seed: u64) -> Polynomial<Fr> {
        Polynomial::new(
            (0..SIZE as u64)
                .map(|i| Fr::from_u64(seed + 7 * i + 3))
                .collect::<Vec<_>>(),
        )
    }

    fn reference_point() -> Vec<Fr> {
        (0..ROUNDS).map(|i| Fr::from_u64(1000 + i as u64)).collect()
    }

    fn opening_tables() -> BTreeMap<JoltOpeningId, Polynomial<Fr>> {
        BTreeMap::from([
            (virt(JoltVirtualPolynomial::LookupOutput), dense(11)),
            (virt(JoltVirtualPolynomial::LeftLookupOperand), dense(22)),
            (virt(JoltVirtualPolynomial::InstructionRa(0)), dense(33)),
            (virt(JoltVirtualPolynomial::InstructionRa(1)), dense(44)),
            (virt(JoltVirtualPolynomial::RightLookupOperand), dense(55)),
        ])
    }

    fn derived_tables(reference_point: &[Fr]) -> BTreeMap<JoltDerivedId, Polynomial<Fr>> {
        BTreeMap::from([(
            JoltDerivedId::Test,
            Polynomial::new(EqPolynomial::new(reference_point.to_vec()).evaluations()),
        )])
    }

    /// Brute-force the output expression's sum over the hypercube — the true
    /// input claim the toy's `total` input opening carries.
    fn brute_force_sum(
        opening_tables: &BTreeMap<JoltOpeningId, Polynomial<Fr>>,
        derived_tables: &BTreeMap<JoltDerivedId, Polynomial<Fr>>,
        gamma: Fr,
    ) -> Fr {
        let expression = ToySymbolic::new(ROUNDS).output_expression::<Fr>();
        (0..SIZE)
            .map(|y| {
                expression.evaluate(
                    |id| opening_tables[id].evals()[y],
                    |_| gamma,
                    |id| derived_tables[id].evals()[y],
                )
            })
            .sum()
    }

    #[test]
    fn naive_prover_round_trips_against_relation_algebra() {
        let relation = ToyRelation {
            symbolic: ToySymbolic::new(ROUNDS),
            reference_point: reference_point(),
        };
        let mut prover_transcript = Blake2bTranscript::new(b"naive-toy");
        let challenges = relation.draw_challenges(&mut prover_transcript).unwrap();
        let gamma = challenges.gamma;

        let opening_tables = opening_tables();
        let derived_tables = derived_tables(&reference_point());
        let claimed_sum = brute_force_sum(&opening_tables, &derived_tables, gamma);

        // The one-member batch head (what the generated begin_batch performs).
        let inputs = ToyInputs { total: claimed_sum };
        let input_claim = relation.input_claim(&inputs, &challenges).unwrap();
        assert_eq!(input_claim, claimed_sum);
        let mut recorder = ClearSumcheckRecorder::<Fr, Fr>::new();
        recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
        let coefficient: Fr = prover_transcript.challenge_scalar();
        let prelude = BatchPrelude::new(
            vec![BatchMember {
                input_claim,
                coefficient,
                rounds: ROUNDS,
            }],
            ROUNDS,
            relation.degree(),
        );

        let mut naive =
            NaiveSumcheckProver::new(relation, challenges, opening_tables, derived_tables).unwrap();
        let mut members: Vec<&mut dyn ProveRounds<Fr>> = vec![&mut naive];
        let proved = prove_batch(
            &prelude,
            &mut members,
            &mut recorder,
            &mut prover_transcript,
        )
        .unwrap();

        // Typed extraction; the verifier's own algebra is the correctness check.
        let input_points = ToyInputs {
            total: vec![Fr::from_u64(9); ROUNDS],
        };
        let output_points = naive
            .relation()
            .derive_opening_points(&proved.challenges, &input_points)
            .unwrap();
        let output_claims = naive.output_claims().unwrap();
        naive
            .validate_derived_tables(&input_points, &output_points)
            .unwrap();

        // The assembled claims cover exactly the expression's openings (the
        // absent Option contributes nothing).
        assert_eq!(
            output_claims.canonical_order(),
            vec![
                virt(JoltVirtualPolynomial::LookupOutput),
                virt(JoltVirtualPolynomial::LeftLookupOperand),
                virt(JoltVirtualPolynomial::InstructionRa(0)),
                virt(JoltVirtualPolynomial::InstructionRa(1)),
                virt(JoltVirtualPolynomial::RightLookupOperand),
            ],
        );

        let expected = naive
            .relation()
            .expected_output(
                &input_points,
                &output_claims,
                &output_points,
                naive.challenges(),
            )
            .unwrap();
        assert_eq!(coefficient * expected, proved.final_claim);
        assert_eq!(proved.member_claims, vec![expected]);

        // Clear-verifier twin: same transcript schedule accepts the proof.
        let recorded = recorder
            .finish(&output_claims.opening_values(), &mut prover_transcript)
            .unwrap();
        let verifier_relation = ToyRelation {
            symbolic: ToySymbolic::new(ROUNDS),
            reference_point: reference_point(),
        };
        let mut verifier_transcript = Blake2bTranscript::new(b"naive-toy");
        let verifier_challenges = verifier_relation
            .draw_challenges(&mut verifier_transcript)
            .unwrap();
        let verifier_input_claim = verifier_relation
            .input_claim(&inputs, &verifier_challenges)
            .unwrap();
        append_sumcheck_claim(&mut verifier_transcript, &verifier_input_claim);
        let verifier_coefficient: Fr = verifier_transcript.challenge_scalar();
        let reduction = recorded
            .proof
            .verify_compressed_boolean(
                ROUNDS,
                verifier_relation.degree(),
                verifier_coefficient * verifier_input_claim,
                &mut verifier_transcript,
            )
            .unwrap();
        for value in output_claims.opening_values() {
            verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &value);
        }

        assert_eq!(reduction.value, proved.final_claim);
        assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    /// A derived table materialized at the wrong point survives the sumcheck
    /// (the prover's sum is self-consistent) but is caught by the
    /// `derive_output_term` cross-check — the drift detector that pins
    /// hand-written table resolvers to the verifier's scalar path.
    #[test]
    fn derived_table_drift_is_detected() {
        let relation = ToyRelation {
            symbolic: ToySymbolic::new(ROUNDS),
            reference_point: reference_point(),
        };
        let mut transcript = Blake2bTranscript::new(b"naive-drift");
        let challenges = relation.draw_challenges(&mut transcript).unwrap();
        let gamma = challenges.gamma;

        // Tables built against a DIFFERENT reference point than the relation's.
        let drifted_point: Vec<Fr> = (0..ROUNDS).map(|i| Fr::from_u64(77 + i as u64)).collect();
        let opening_tables = opening_tables();
        let derived_tables = derived_tables(&drifted_point);
        let claimed_sum = brute_force_sum(&opening_tables, &derived_tables, gamma);

        let coefficient: Fr = transcript.challenge_scalar();
        let prelude = BatchPrelude::new(
            vec![BatchMember {
                input_claim: claimed_sum,
                coefficient,
                rounds: ROUNDS,
            }],
            ROUNDS,
            relation.degree(),
        );
        let mut recorder = ClearSumcheckRecorder::<Fr, Fr>::new();
        let mut naive =
            NaiveSumcheckProver::new(relation, challenges, opening_tables, derived_tables).unwrap();
        let mut members: Vec<&mut dyn ProveRounds<Fr>> = vec![&mut naive];
        let proved = prove_batch(&prelude, &mut members, &mut recorder, &mut transcript).unwrap();

        let input_points = ToyInputs {
            total: vec![Fr::from_u64(9); ROUNDS],
        };
        let output_points = naive
            .relation()
            .derive_opening_points(&proved.challenges, &input_points)
            .unwrap();
        assert!(matches!(
            naive.validate_derived_tables(&input_points, &output_points),
            Err(ProverError::DerivedTableDrift {
                id: JoltDerivedId::Test,
                ..
            }),
        ));
    }

    #[test]
    fn construction_validates_leaf_coverage_and_sizes() {
        let challenges = || ToyChallenges::from_transcript_values([Fr::from_u64(5)].into_iter());
        let c_id = virt(JoltVirtualPolynomial::RightLookupOperand);

        // A missing opening table is rejected with its id.
        let mut incomplete = opening_tables();
        let _ = incomplete.remove(&c_id);
        let relation = ToyRelation {
            symbolic: ToySymbolic::new(ROUNDS),
            reference_point: reference_point(),
        };
        assert!(matches!(
            NaiveSumcheckProver::new(
                relation,
                challenges().unwrap(),
                incomplete,
                derived_tables(&reference_point()),
            ),
            Err(ProverError::MissingOpeningTable { id }) if id == c_id,
        ));

        // A mis-sized table is rejected.
        let mut mis_sized = opening_tables();
        let _ = mis_sized.insert(c_id, Polynomial::new(vec![Fr::from_u64(1); SIZE / 2]));
        let relation = ToyRelation {
            symbolic: ToySymbolic::new(ROUNDS),
            reference_point: reference_point(),
        };
        assert!(matches!(
            NaiveSumcheckProver::new(
                relation,
                challenges().unwrap(),
                mis_sized,
                derived_tables(&reference_point()),
            ),
            Err(ProverError::TableSizeMismatch { .. }),
        ));
    }
}
