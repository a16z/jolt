#![expect(
    clippy::panic,
    reason = "stage tests may panic when fixture construction fails"
)]

use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanOuterEvaluationClaims, WrapperRelationDimensions,
    WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL,
};
use jolt_crypto::Bn254;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::{CompressedPoly, Point};
use jolt_r1cs::ConstraintMatrices;
use jolt_sumcheck::CompressedSumcheckProof;
use jolt_transcript::{Blake2bTranscript, Label, Transcript as _};
use jolt_wrapper_verifier::{
    stages::{r1cs_relation, spartan},
    verify, CheckedInputs, Error, HyperKzgProof, R1csRelationStatement, SpartanProof, WrapperProof,
    WrapperVerifierConfig, WrapperVerifierInputs,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Transcript = Blake2bTranscript<Fr>;
type Pairing = Bn254;
type KzgPCS = HyperKZGScheme<Pairing>;

#[test]
fn verifier_rejects_dummy_hyperkzg_proof_after_spartan_accepts() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3), Fr::from_u64(5)], 8, 13);
    let inputs = fixture.inputs();
    let proof = proof_for_fixture(&fixture);
    let config = fixture.config();

    assert!(matches!(
        verify::<Pairing, Transcript>(&config, inputs, &proof),
        Err(Error::HyperKzgVerificationFailed { .. })
    ));
}

#[test]
fn verifier_rejects_relation_variable_mismatch_before_spartan() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let proof: WrapperProof<Pairing> = WrapperProof::new(WrapperRelationDimensions::new(9, 13, 1));

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::R1csRelationMismatch { .. })
    ));
}

#[test]
fn verifier_rejects_relation_constraint_mismatch_before_spartan() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let proof: WrapperProof<Pairing> = WrapperProof::new(WrapperRelationDimensions::new(8, 14, 1));

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::R1csRelationMismatch { .. })
    ));
}

#[test]
fn verifier_rejects_public_input_count_mismatch_before_spartan() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3), Fr::from_u64(5)], 8, 13);
    let inputs = fixture.inputs();
    let proof: WrapperProof<Pairing> = WrapperProof::new(WrapperRelationDimensions::new(8, 13, 1));

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::R1csRelationMismatch { .. })
    ));
}

#[test]
fn verifier_rejects_unpaddable_relation_before_spartan() {
    let relation = ConstraintMatrices::new(
        13,
        usize::MAX,
        vec![vec![]; 13],
        vec![vec![]; 13],
        vec![vec![]; 13],
    );
    let public_inputs = vec![Fr::from_u64(3)];
    let inputs = WrapperVerifierInputs {
        public_inputs: &public_inputs,
    };
    let proof: WrapperProof<Pairing> = WrapperProof::new(WrapperRelationDimensions::new(
        relation.num_vars,
        relation.num_constraints,
        public_inputs.len(),
    ));
    let (_, verifier) = make_setup(16);
    let config = WrapperVerifierConfig::for_relation(relation, public_inputs.len(), verifier);

    assert!(matches!(
        verify::<Pairing, Transcript>(&config, inputs, &proof),
        Err(Error::InvalidR1csRelationFacts { .. })
    ));
}

#[test]
fn verifier_rejects_spartan_outer_sumcheck_round_count_mismatch() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let relation = relation_for_fixture(&fixture);
    let proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            CompressedSumcheckProof {
                round_polynomials: vec![zero_round(); 3],
            },
            zero_outer_claims(),
            zero_inner_sumcheck(relation),
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    );

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::SpartanSumcheckFailed { .. })
    ));
}

#[test]
fn verifier_rejects_spartan_outer_sumcheck_degree_bound_violation() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let relation = relation_for_fixture(&fixture);
    let proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            CompressedSumcheckProof {
                round_polynomials: vec![
                    CompressedPoly::new(vec![
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                        Fr::from_u64(0),
                    ]);
                    4
                ],
            },
            zero_outer_claims(),
            zero_inner_sumcheck(relation),
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    );

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::SpartanSumcheckFailed { .. })
    ));
}

#[test]
fn verifier_rejects_spartan_outer_evaluation_claim_mismatch() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let relation = relation_for_fixture(&fixture);
    let proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            zero_spartan_sumcheck(relation),
            SpartanOuterEvaluationClaims::new(Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)),
            zero_inner_sumcheck(relation),
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    );

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::SpartanOuterReductionMismatch)
    ));
}

#[test]
fn verifier_rejects_outer_claims_that_only_satisfy_outer_relation() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let relation = relation_for_fixture(&fixture);
    let proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            zero_spartan_sumcheck(relation),
            SpartanOuterEvaluationClaims::new(Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(6)),
            zero_inner_sumcheck(relation),
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    );

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::SpartanInnerReductionMismatch)
    ));
}

#[test]
fn verifier_rejects_spartan_inner_sumcheck_round_count_mismatch() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3)], 8, 13);
    let inputs = fixture.inputs();
    let relation = relation_for_fixture(&fixture);
    let proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            zero_spartan_sumcheck(relation),
            zero_outer_claims(),
            CompressedSumcheckProof {
                round_polynomials: vec![zero_round(); 2],
            },
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    );

    assert!(matches!(
        verify::<Pairing, Transcript>(&fixture.config(), inputs, &proof),
        Err(Error::SpartanSumcheckFailed { .. })
    ));
}

#[test]
fn spartan_stage_derives_tau_before_outer_sumcheck() {
    let fixture = VerifierFixture::new(&[Fr::from_u64(3), Fr::from_u64(5)], 8, 13);
    let inputs = fixture.inputs();
    let checked = checked_inputs(&fixture);
    let proof = proof_for_fixture(&fixture);
    let mut transcript = Transcript::new(fixture.config().transcript_label);
    let relation = match r1cs_relation::verify(
        r1cs_relation::R1csRelationInputs {
            checked: &checked,
            relation: &fixture.relation,
            public_inputs: inputs.public_inputs,
            proof_relation: proof.relation,
        },
        &mut transcript,
    ) {
        Ok(relation) => relation,
        Err(error) => panic!("relation stage should accept fixture: {error}"),
    };

    let mut expected_transcript = transcript.clone();
    expected_transcript.append(&Label(WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL));
    let expected_tau = Point::high_to_low(
        expected_transcript
            .challenge_vector(relation.statement_facts.spartan.num_constraint_rounds()),
    );

    let mut wrong_order_transcript = transcript.clone();
    wrong_order_transcript.append(&Label(WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL));
    let wrong_order_tau = Point::high_to_low(
        wrong_order_transcript
            .challenge_vector(relation.statement_facts.spartan.num_constraint_rounds()),
    );

    let output = match spartan::verify(
        spartan::SpartanInputs {
            checked: &checked,
            proof: &proof,
            deps: spartan::deps(&relation),
        },
        &mut transcript,
    ) {
        Ok(output) => output,
        Err(error) => panic!("spartan stage should accept fixture: {error}"),
    };

    assert_eq!(output.outer.tau, expected_tau);
    assert_ne!(output.outer.tau, wrong_order_tau);
    assert_eq!(output.outer.rx, output.outer_reduction.point);
    assert_eq!(output.outer.final_claim, output.outer_reduction.value);
}

struct VerifierFixture {
    public_inputs: Vec<Fr>,
    relation: ConstraintMatrices<Fr>,
}

impl VerifierFixture {
    fn new(public_inputs: &[Fr], relation_variables: usize, relation_constraints: usize) -> Self {
        Self {
            public_inputs: public_inputs.to_vec(),
            relation: zero_relation(relation_variables, relation_constraints),
        }
    }

    fn inputs(&self) -> WrapperVerifierInputs<'_, Fr> {
        WrapperVerifierInputs {
            public_inputs: &self.public_inputs,
        }
    }

    fn config(&self) -> WrapperVerifierConfig<Pairing> {
        let (_, verifier) = make_setup(16);
        WrapperVerifierConfig::for_relation(
            self.relation.clone(),
            self.public_inputs.len(),
            verifier,
        )
    }
}

fn checked_inputs(fixture: &VerifierFixture) -> CheckedInputs {
    CheckedInputs {
        relation_variables: fixture.relation.num_vars,
        relation_constraints: fixture.relation.num_constraints,
        public_inputs: fixture.public_inputs.len(),
    }
}

fn proof_for_fixture(fixture: &VerifierFixture) -> WrapperProof<Pairing> {
    let relation = relation_for_fixture(fixture);
    WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::new(
            zero_spartan_sumcheck(relation),
            zero_outer_claims(),
            zero_inner_sumcheck(relation),
            Fr::from_u64(0),
        ),
        HyperKzgProof::default(),
    )
}

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Pairing>, HyperKZGVerifierSetup<Pairing>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5750_4150);
    let prover = KzgPCS::setup(
        &mut rng,
        max_degree,
        Pairing::g1_generator(),
        Pairing::g2_generator(),
    );
    let verifier = KzgPCS::verifier_setup(&prover);
    (prover, verifier)
}

fn relation_for_fixture(fixture: &VerifierFixture) -> WrapperRelationDimensions {
    WrapperRelationDimensions::new(
        fixture.relation.num_vars,
        fixture.relation.num_constraints,
        fixture.public_inputs.len(),
    )
}

fn zero_relation(variables: usize, constraints: usize) -> ConstraintMatrices<Fr> {
    ConstraintMatrices::new(
        constraints,
        variables,
        vec![vec![]; constraints],
        vec![vec![]; constraints],
        vec![vec![]; constraints],
    )
}

fn zero_spartan_sumcheck(relation: WrapperRelationDimensions) -> CompressedSumcheckProof<Fr> {
    let spartan_dimensions = match relation.spartan_dimensions() {
        Ok(dimensions) => dimensions,
        Err(error) => panic!("test relation dimensions are paddable: {error}"),
    };
    CompressedSumcheckProof {
        round_polynomials: vec![zero_round(); spartan_dimensions.num_constraint_rounds()],
    }
}

fn zero_inner_sumcheck(relation: WrapperRelationDimensions) -> CompressedSumcheckProof<Fr> {
    let spartan_dimensions = match relation.spartan_dimensions() {
        Ok(dimensions) => dimensions,
        Err(error) => panic!("test relation dimensions are paddable: {error}"),
    };
    CompressedSumcheckProof {
        round_polynomials: vec![zero_round(); spartan_dimensions.num_var_rounds()],
    }
}

fn zero_outer_claims() -> SpartanOuterEvaluationClaims<Fr> {
    SpartanOuterEvaluationClaims::new(Fr::from_u64(0), Fr::from_u64(0), Fr::from_u64(0))
}

fn zero_round() -> CompressedPoly<Fr> {
    CompressedPoly::new(vec![Fr::from_u64(0)])
}
