#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly on fixture construction errors"
)]

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanInnerBatchingCoefficients, SpartanOuterEvaluationClaims, WrapperRelationDimensions,
    WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
    WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
    WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL,
    WRAPPER_TRANSCRIPT_LABEL, WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL,
};
#[cfg(feature = "zk")]
use jolt_claims::{challenge, constant, opening, public};
#[cfg(feature = "zk")]
use jolt_crypto::Bn254G1;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitmentOpening;
use jolt_crypto::{
    r1cs::GrumpkinPointWithIdentityVar, Bn254, Grumpkin, GrumpkinPoint, JoltGroup, Pedersen,
    PedersenSetup, VectorCommitment,
};
use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt};
#[cfg(feature = "zk")]
use jolt_field::{FixedBytes, Invertible};
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup,
};
use jolt_hyrax::r1cs::{verify_opening, HyraxOpeningR1csInput};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
#[cfg(feature = "zk")]
use jolt_poly::CompressedPoly;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
#[cfg(feature = "zk")]
use jolt_r1cs::ClaimSourceTable;
use jolt_r1cs::{AssignedScalar, ConstraintMatrices, FqVar, LinearCombination, R1csBuilder};
use jolt_sumcheck::{
    allocate_sumcheck_r1cs_layout, append_sumcheck_r1cs_constraints, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, RoundMessage, SumcheckStatement, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
#[cfg(feature = "zk")]
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedRoundWitness, CommittedSumcheckConsistency,
    CommittedSumcheckProof, SumcheckDomainSpec, VerifiedCommittedRound,
};
use jolt_transcript::r1cs::{
    PoseidonR1csTranscript, R1csJoltByteTranscript, R1csJoltTranscript, R1csTranscript,
};
#[cfg(feature = "zk")]
use jolt_transcript::{AppendToTranscript, LabelWithCount};
use jolt_transcript::{Blake2bTranscript, Label, Transcript};
use jolt_wrapper_verifier::{
    stages::r1cs_relation, verify, HyperKzgProof, R1csRelationStatement, SpartanProof,
    WrapperProof, WrapperR1csBuilder, WrapperR1csProtocol, WrapperVerifierConfig,
    WrapperVerifierInputs,
};
#[cfg(feature = "zk")]
use jolt_wrapper_verifier::{
    stages::{
        hyperkzg::HyperKzgZkOutput,
        spartan::SpartanZkOutput,
        zk::outputs::{CommittedOutputClaimOutput, CommittedOutputClaimShape},
    },
    verify_zk, SpartanZkProof, WrapperZkProof, WrapperZkVerifierConfig,
};
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "zk")]
use rand_core::RngCore;
use rand_core::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

type Pairing = Bn254;
type KzgPCS = HyperKZGScheme<Pairing>;
type WrapperTranscript = Blake2bTranscript<Fr>;
type CircuitTranscript = PoseidonR1csTranscript;
type TestVc = Pedersen<GrumpkinPoint>;
#[cfg(feature = "zk")]
type WrapperBlindFoldVc = Pedersen<Bn254G1>;

const FIXTURE_MAGIC: &[u8] = b"jolt-wrapper-mini-protocol-v2";
#[cfg(feature = "zk")]
const ZK_FIXTURE_MAGIC: &[u8] = b"jolt-wrapper-mini-protocol-zk-v1";
const REGENERATE_FIXTURE_ENV: &str = "JOLT_WRAPPER_REGENERATE_FIXTURES";
#[cfg(feature = "zk")]
const ZK_STATISTICAL_SAMPLES_ENV: &str = "JOLT_WRAPPER_ZK_STATISTICAL_SAMPLES";
#[cfg(feature = "zk")]
const DEFAULT_ZK_STATISTICAL_SAMPLES: usize = 32;

#[test]
fn wrapper_mini_protocol_accepts_and_rejects_tampering() {
    let fixture = load_or_generate_fixture();
    let config = fixture.config();
    let inputs = fixture.inputs();

    verify::<Pairing, WrapperTranscript>(&config, inputs, &fixture.proof)
        .expect("honest mini protocol proof should verify");

    ProofTamper::all(&fixture)
        .into_par_iter()
        .for_each(|tamper| {
            let mut tampered = fixture.clone();
            tamper.apply(&mut tampered);

            assert!(
                verify::<Pairing, WrapperTranscript>(
                    &tampered.config(),
                    tampered.inputs(),
                    &tampered.proof
                )
                .is_err(),
                "accepted tampering target {tamper:?}"
            );
        });
}

#[cfg(feature = "zk")]
#[test]
fn wrapper_mini_protocol_zk_accepts_and_rejects_tampering() {
    let fixture = load_or_generate_zk_fixture(0x5a4b_5752_4150);
    verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
        &fixture.config(),
        fixture.inputs(),
        &fixture.proof,
    )
    .expect("honest ZK mini protocol proof should verify");

    ZkProofTamper::all(&fixture)
        .into_par_iter()
        .for_each(|tamper| {
            let mut tampered = fixture.clone();
            tamper.apply(&mut tampered);

            assert!(
                verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
                    &tampered.config(),
                    tampered.inputs(),
                    &tampered.proof
                )
                .is_err(),
                "accepted ZK tampering target {tamper:?}"
            );
        });
}

#[cfg(feature = "zk")]
#[test]
fn wrapper_mini_protocol_zk_rejects_incompatible_vc_setup() {
    let fixture = load_or_generate_zk_fixture(0x5a4b_5752_4150);
    verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
        &fixture.config(),
        fixture.inputs(),
        &fixture.proof,
    )
    .expect("baseline ZK fixture should verify before setup tampering");

    let mut wrong_blinding = fixture.config();
    wrong_blinding.vc_setup.blinding_generator += Pairing::g1_generator();
    assert!(
        verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
            &wrong_blinding,
            fixture.inputs(),
            &fixture.proof,
        )
        .is_err(),
        "accepted ZK proof with incompatible VC blinding generator"
    );

    let mut wrong_message_generator = fixture.config();
    wrong_message_generator.vc_setup.message_generators[0] += Pairing::g1_generator();
    assert!(
        verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
            &wrong_message_generator,
            fixture.inputs(),
            &fixture.proof,
        )
        .is_err(),
        "accepted ZK proof with incompatible VC message generator"
    );

    let mut wrong_capacity = fixture.config();
    wrong_capacity
        .vc_setup
        .message_generators
        .push(Pairing::g1_generator().scalar_mul(&Fr::from_u64(23)));
    assert!(
        verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
            &wrong_capacity,
            fixture.inputs(),
            &fixture.proof,
        )
        .is_err(),
        "accepted ZK proof with incompatible VC capacity"
    );
}

#[cfg(feature = "zk")]
#[test]
fn wrapper_mini_protocol_zk_rejects_hidden_eval_basis_mismatch() {
    let protocol = build_mini_protocol();
    let (prover_setup, verifier_setup, vc_setup) = make_zk_setup_with_hiding_scalars(
        protocol.witness.len().next_power_of_two(),
        Fr::from_u64(19),
        Fr::from_u64(17),
    );
    let mut rng = ChaCha20Rng::seed_from_u64(0x4849_4444_454e);
    let proof =
        prove_wrapper_zk_with_unchecked_final_eval(&protocol, &prover_setup, &vc_setup, &mut rng);
    let fixture = MiniProtocolZkFixture {
        public_inputs: protocol.public_inputs,
        relation: protocol.r1cs,
        proof,
        verifier_setup,
        vc_setup,
    };

    assert!(
        verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
            &fixture.config(),
            fixture.inputs(),
            &fixture.proof,
        )
        .is_err(),
        "accepted ZK proof whose HyperKZG hidden-eval basis does not match the BlindFold VC basis"
    );
}

#[cfg(feature = "zk")]
#[test]
fn wrapper_mini_protocol_zk_proofs_are_empirically_independent() {
    let samples = zk_statistical_samples();
    let mut projections = [
        StatisticalProjection::new("witness_commitment", samples),
        StatisticalProjection::new("hyperkzg_eval_commitment", samples),
        StatisticalProjection::new("blindfold_random_u", samples),
        StatisticalProjection::new("blindfold_auxiliary_commitment", samples),
        StatisticalProjection::new("blindfold_witness_opening", samples),
    ];

    let values = (0..samples)
        .into_par_iter()
        .map(|sample| {
            let fixture = load_or_generate_zk_fixture(0x9a4b_0000 + sample as u64);
            verify_zk::<Pairing, WrapperBlindFoldVc, WrapperTranscript>(
                &fixture.config(),
                fixture.inputs(),
                &fixture.proof,
            )
            .expect("ZK statistical sample should verify");
            zk_statistical_projection_values(&fixture)
        })
        .collect::<Vec<_>>();

    for sample in values {
        for (projection, value) in projections.iter_mut().zip(sample) {
            projection.push(value);
        }
    }

    for projection in &projections {
        assert_empirical_distribution(projection);
    }
    assert_empirical_pairwise_independence(&projections[0], &projections[1]);
    assert_empirical_pairwise_independence(&projections[1], &projections[2]);
    assert_empirical_pairwise_independence(&projections[2], &projections[4]);
}

#[derive(Clone, Serialize, Deserialize)]
struct MiniProtocolFixture {
    public_inputs: Vec<Fr>,
    relation: ConstraintMatrices<Fr>,
    proof: WrapperProof<Pairing>,
    verifier_setup: HyperKZGVerifierSetup<Pairing>,
}

impl MiniProtocolFixture {
    fn config(&self) -> WrapperVerifierConfig<Pairing> {
        WrapperVerifierConfig::for_relation(
            self.relation.clone(),
            self.public_inputs.len(),
            self.verifier_setup.clone(),
        )
    }

    fn inputs(&self) -> WrapperVerifierInputs<'_, Fr> {
        WrapperVerifierInputs {
            public_inputs: &self.public_inputs,
        }
    }
}

#[cfg(feature = "zk")]
#[derive(Clone, Serialize, Deserialize)]
struct MiniProtocolZkFixture {
    public_inputs: Vec<Fr>,
    relation: ConstraintMatrices<Fr>,
    proof: WrapperZkProof<Pairing, WrapperBlindFoldVc>,
    verifier_setup: HyperKZGVerifierSetup<Pairing>,
    vc_setup: PedersenSetup<Bn254G1>,
}

#[cfg(feature = "zk")]
impl MiniProtocolZkFixture {
    fn config(&self) -> WrapperZkVerifierConfig<Pairing, WrapperBlindFoldVc> {
        WrapperZkVerifierConfig::for_relation(
            self.relation.clone(),
            self.public_inputs.len(),
            self.verifier_setup.clone(),
            self.vc_setup.clone(),
        )
    }

    fn inputs(&self) -> WrapperVerifierInputs<'_, Fr> {
        WrapperVerifierInputs {
            public_inputs: &self.public_inputs,
        }
    }
}

fn load_or_generate_fixture() -> MiniProtocolFixture {
    let path = fixture_artifact_path();
    if env::var_os(REGENERATE_FIXTURE_ENV).is_none() && path.exists() {
        return read_fixture_artifact(&path);
    }

    let protocol = build_mini_protocol();
    let (prover_setup, verifier_setup) = make_setup(protocol.witness.len().next_power_of_two());
    let proof = prove_wrapper(&protocol, &prover_setup);
    let fixture = MiniProtocolFixture {
        public_inputs: protocol.public_inputs,
        relation: protocol.r1cs,
        proof,
        verifier_setup,
    };
    write_fixture_artifact(&path, &fixture);
    fixture
}

#[cfg(feature = "zk")]
fn generate_zk_fixture(seed: u64) -> MiniProtocolZkFixture {
    let protocol = build_mini_protocol();
    let (prover_setup, verifier_setup, vc_setup) =
        make_zk_setup(protocol.witness.len().next_power_of_two());
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let proof = prove_wrapper_zk(&protocol, &prover_setup, &vc_setup, &mut rng);
    MiniProtocolZkFixture {
        public_inputs: protocol.public_inputs,
        relation: protocol.r1cs,
        proof,
        verifier_setup,
        vc_setup,
    }
}

#[cfg(feature = "zk")]
fn load_or_generate_zk_fixture(seed: u64) -> MiniProtocolZkFixture {
    let path = zk_fixture_artifact_path(seed);
    if env::var_os(REGENERATE_FIXTURE_ENV).is_none() && path.exists() {
        return read_zk_fixture_artifact(&path);
    }

    let fixture = generate_zk_fixture(seed);
    write_zk_fixture_artifact(&path, &fixture);
    fixture
}

fn fixture_artifact_path() -> PathBuf {
    env::temp_dir()
        .join("jolt-wrapper-verifier-fixtures")
        .join("mini-protocol-v2.bin")
}

#[cfg(feature = "zk")]
fn zk_fixture_artifact_path(seed: u64) -> PathBuf {
    env::temp_dir()
        .join("jolt-wrapper-verifier-fixtures")
        .join(format!("mini-protocol-zk-v1-{seed:016x}.bin"))
}

fn read_fixture_artifact(path: &Path) -> MiniProtocolFixture {
    let bytes = fs::read(path).expect("read wrapper mini-protocol fixture");
    let Some(payload) = bytes.strip_prefix(FIXTURE_MAGIC) else {
        panic!(
            "invalid wrapper mini-protocol fixture magic at {}",
            path.display()
        );
    };
    let (fixture, consumed): (MiniProtocolFixture, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())
            .expect("decode wrapper mini-protocol fixture");
    assert_eq!(
        consumed,
        payload.len(),
        "trailing bytes in wrapper mini-protocol fixture"
    );
    fixture
}

fn write_fixture_artifact(path: &Path, fixture: &MiniProtocolFixture) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create wrapper fixture directory");
    }
    let mut bytes = FIXTURE_MAGIC.to_vec();
    bytes.extend(
        bincode::serde::encode_to_vec(fixture, bincode::config::standard())
            .expect("encode wrapper mini-protocol fixture"),
    );
    fs::write(path, bytes).expect("write wrapper mini-protocol fixture");
}

#[cfg(feature = "zk")]
fn read_zk_fixture_artifact(path: &Path) -> MiniProtocolZkFixture {
    let bytes = fs::read(path).expect("read wrapper ZK mini-protocol fixture");
    let Some(payload) = bytes.strip_prefix(ZK_FIXTURE_MAGIC) else {
        panic!(
            "invalid wrapper ZK mini-protocol fixture magic at {}",
            path.display()
        );
    };
    let (fixture, consumed): (MiniProtocolZkFixture, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())
            .expect("decode wrapper ZK mini-protocol fixture");
    assert_eq!(
        consumed,
        payload.len(),
        "trailing bytes in wrapper ZK mini-protocol fixture"
    );
    fixture
}

#[cfg(feature = "zk")]
fn write_zk_fixture_artifact(path: &Path, fixture: &MiniProtocolZkFixture) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create wrapper fixture directory");
    }
    let mut bytes = ZK_FIXTURE_MAGIC.to_vec();
    bytes.extend(
        bincode::serde::encode_to_vec(fixture, bincode::config::standard())
            .expect("encode wrapper ZK mini-protocol fixture"),
    );
    fs::write(path, bytes).expect("write wrapper ZK mini-protocol fixture");
}

#[derive(Clone, Copy, Debug)]
enum ProofTamper {
    PublicInput(usize),
    RelationCoefficient(Matrix, usize),
    RelationStatement(RelationStatementField),
    WitnessCommitment,
    OuterRound(usize),
    OuterEvaluationClaim(OuterClaimField),
    InnerRound(usize),
    WitnessOpeningClaim,
    HyperKzgFoldCommitment(usize),
    HyperKzgEvaluation(usize, usize),
    HyperKzgWitness(usize),
}

#[derive(Clone, Copy, Debug)]
enum Matrix {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug)]
enum RelationStatementField {
    Variables,
    Constraints,
    PublicInputs,
}

#[derive(Clone, Copy, Debug)]
enum OuterClaimField {
    A,
    B,
    C,
}

impl ProofTamper {
    fn all(fixture: &MiniProtocolFixture) -> Vec<Self> {
        let mut targets = Vec::new();
        targets.extend((0..fixture.public_inputs.len()).map(Self::PublicInput));
        for matrix in [Matrix::A, Matrix::B, Matrix::C] {
            targets.extend(
                interesting_indices(fixture.relation.num_constraints)
                    .into_iter()
                    .map(move |row| Self::RelationCoefficient(matrix, row)),
            );
        }
        targets.extend([
            Self::RelationStatement(RelationStatementField::Variables),
            Self::RelationStatement(RelationStatementField::Constraints),
            Self::RelationStatement(RelationStatementField::PublicInputs),
            Self::WitnessCommitment,
        ]);
        targets.extend(
            (0..fixture.proof.spartan.outer_sumcheck.round_polynomials.len()).map(Self::OuterRound),
        );
        targets.extend([
            Self::OuterEvaluationClaim(OuterClaimField::A),
            Self::OuterEvaluationClaim(OuterClaimField::B),
            Self::OuterEvaluationClaim(OuterClaimField::C),
        ]);
        targets.extend(
            (0..fixture.proof.spartan.inner_sumcheck.round_polynomials.len()).map(Self::InnerRound),
        );
        targets.push(Self::WitnessOpeningClaim);
        targets.extend(
            (0..fixture.proof.hyperkzg.witness_opening_proof.com.len())
                .map(Self::HyperKzgFoldCommitment),
        );
        if let Some(evaluations) = fixture
            .proof
            .hyperkzg
            .witness_opening_proof
            .clear_evaluations()
        {
            for (batch, values) in evaluations.iter().enumerate() {
                targets.extend(
                    interesting_indices(values.len())
                        .into_iter()
                        .map(move |index| Self::HyperKzgEvaluation(batch, index)),
                );
            }
        }
        targets.extend(
            (0..fixture.proof.hyperkzg.witness_opening_proof.w.len()).map(Self::HyperKzgWitness),
        );
        targets
    }

    fn apply(self, fixture: &mut MiniProtocolFixture) {
        match self {
            Self::PublicInput(index) => fixture.public_inputs[index] += Fr::from_u64(1),
            Self::RelationCoefficient(matrix, row) => {
                relation_row(&mut fixture.relation, matrix, row).push((0, Fr::from_u64(1)));
            }
            Self::RelationStatement(field) => match field {
                RelationStatementField::Variables => {
                    fixture.proof.relation.dimensions.variables += 1;
                }
                RelationStatementField::Constraints => {
                    fixture.proof.relation.dimensions.constraints += 1;
                }
                RelationStatementField::PublicInputs => {
                    fixture.proof.relation.dimensions.public_inputs += 1;
                }
            },
            Self::WitnessCommitment => {
                fixture.proof.hyperkzg.witness_commitment = HyperKZGCommitment::default();
            }
            Self::OuterRound(index) => {
                fixture.proof.spartan.outer_sumcheck.round_polynomials[index] =
                    fixture.proof.spartan.outer_sumcheck.round_polynomials[index]
                        .clone()
                        .decompress(Fr::from_u64(0))
                        .add_constant(Fr::from_u64(1))
                        .compress();
            }
            Self::OuterEvaluationClaim(field) => match field {
                OuterClaimField::A => {
                    fixture.proof.spartan.outer_evaluation_claims.a += Fr::from_u64(1);
                }
                OuterClaimField::B => {
                    fixture.proof.spartan.outer_evaluation_claims.b += Fr::from_u64(1);
                }
                OuterClaimField::C => {
                    fixture.proof.spartan.outer_evaluation_claims.c += Fr::from_u64(1);
                }
            },
            Self::InnerRound(index) => {
                fixture.proof.spartan.inner_sumcheck.round_polynomials[index] =
                    fixture.proof.spartan.inner_sumcheck.round_polynomials[index]
                        .clone()
                        .decompress(fixture.proof.spartan.outer_evaluation_claims.a)
                        .add_constant(Fr::from_u64(1))
                        .compress();
            }
            Self::WitnessOpeningClaim => {
                fixture.proof.spartan.witness_opening_claim += Fr::from_u64(1);
            }
            Self::HyperKzgFoldCommitment(index) => {
                fixture.proof.hyperkzg.witness_opening_proof.com[index] += Pairing::g1_generator();
            }
            Self::HyperKzgEvaluation(batch, index) => {
                let evaluations = fixture
                    .proof
                    .hyperkzg
                    .witness_opening_proof
                    .clear_evaluations_mut()
                    .expect("fixture uses transparent HyperKZG");
                evaluations[batch][index] += Fr::from_u64(1);
            }
            Self::HyperKzgWitness(index) => {
                fixture.proof.hyperkzg.witness_opening_proof.w[index] += Pairing::g1_generator();
            }
        }
    }
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
enum ZkProofTamper {
    PublicInput(usize),
    RelationCoefficient(Matrix, usize),
    RelationStatement(RelationStatementField),
    WitnessCommitment,
    OuterRound(usize),
    OuterOutputClaimCommitment(usize),
    InnerRound(usize),
    InnerOutputClaimCommitment(usize),
    HyperKzgFoldCommitment(usize),
    HyperKzgHiddenEvaluation(usize, usize),
    HyperKzgHiddenOutput,
    HyperKzgWitness(usize),
    BlindFoldRandomU,
    BlindFoldAuxiliaryCommitment(usize),
    BlindFoldRandomRoundCommitment(usize),
    BlindFoldRandomOutputClaimCommitment(usize),
    BlindFoldRandomAuxiliaryCommitment(usize),
    BlindFoldRandomErrorCommitment(usize),
    BlindFoldRandomEvalCommitment(usize),
    BlindFoldCrossTermCommitment(usize),
    BlindFoldOuterSumcheck(usize),
    BlindFoldInnerSumcheck(usize),
    BlindFoldWitnessOpening(usize),
    BlindFoldErrorOpening(usize),
    BlindFoldFoldedEvalOutput(usize),
    BlindFoldFoldedEvalBlinding(usize),
    BlindFoldFoldedEvalOutputOpening(usize, usize),
    BlindFoldFoldedEvalBlindingOpening(usize, usize),
}

#[cfg(feature = "zk")]
impl ZkProofTamper {
    fn all(fixture: &MiniProtocolZkFixture) -> Vec<Self> {
        let mut targets = Vec::new();
        targets.extend((0..fixture.public_inputs.len()).map(Self::PublicInput));
        for matrix in [Matrix::A, Matrix::B, Matrix::C] {
            targets.extend(
                interesting_indices(fixture.relation.num_constraints)
                    .into_iter()
                    .map(move |row| Self::RelationCoefficient(matrix, row)),
            );
        }
        targets.extend([
            Self::RelationStatement(RelationStatementField::Variables),
            Self::RelationStatement(RelationStatementField::Constraints),
            Self::RelationStatement(RelationStatementField::PublicInputs),
            Self::WitnessCommitment,
        ]);
        targets
            .extend((0..fixture.proof.spartan.outer_sumcheck.rounds.len()).map(Self::OuterRound));
        targets.extend(
            (0..fixture
                .proof
                .spartan
                .outer_sumcheck
                .output_claims
                .commitments
                .len())
                .map(Self::OuterOutputClaimCommitment),
        );
        targets
            .extend((0..fixture.proof.spartan.inner_sumcheck.rounds.len()).map(Self::InnerRound));
        targets.extend(
            (0..fixture
                .proof
                .spartan
                .inner_sumcheck
                .output_claims
                .commitments
                .len())
                .map(Self::InnerOutputClaimCommitment),
        );
        targets.extend(
            (0..fixture.proof.hyperkzg.witness_opening_proof.com.len())
                .map(Self::HyperKzgFoldCommitment),
        );
        if let jolt_hyperkzg::HyperKZGProofPayload::Zk { y, .. } =
            &fixture.proof.hyperkzg.witness_opening_proof.payload
        {
            for (batch, commitments) in y.iter().enumerate() {
                targets.extend(
                    interesting_indices(commitments.len())
                        .into_iter()
                        .map(move |index| Self::HyperKzgHiddenEvaluation(batch, index)),
                );
            }
        }
        targets.push(Self::HyperKzgHiddenOutput);
        targets.extend(
            (0..fixture.proof.hyperkzg.witness_opening_proof.w.len()).map(Self::HyperKzgWitness),
        );
        targets.push(Self::BlindFoldRandomU);
        targets.extend(
            (0..fixture.proof.blindfold.auxiliary_row_commitments.len())
                .map(Self::BlindFoldAuxiliaryCommitment),
        );
        targets.extend(
            (0..fixture.proof.blindfold.random_round_commitments.len())
                .map(Self::BlindFoldRandomRoundCommitment),
        );
        targets.extend(
            (0..fixture
                .proof
                .blindfold
                .random_output_claim_row_commitments
                .len())
                .map(Self::BlindFoldRandomOutputClaimCommitment),
        );
        targets.extend(
            (0..fixture
                .proof
                .blindfold
                .random_auxiliary_row_commitments
                .len())
                .map(Self::BlindFoldRandomAuxiliaryCommitment),
        );
        targets.extend(
            (0..fixture.proof.blindfold.random_error_row_commitments.len())
                .map(Self::BlindFoldRandomErrorCommitment),
        );
        targets.extend(
            (0..fixture.proof.blindfold.random_eval_commitments.len())
                .map(Self::BlindFoldRandomEvalCommitment),
        );
        targets.extend(
            (0..fixture
                .proof
                .blindfold
                .cross_term_error_row_commitments
                .len())
                .map(Self::BlindFoldCrossTermCommitment),
        );
        targets.extend(
            (0..fixture
                .proof
                .blindfold
                .outer_sumcheck
                .round_polynomials
                .len())
                .map(Self::BlindFoldOuterSumcheck),
        );
        targets.extend(
            (0..fixture
                .proof
                .blindfold
                .inner_sumcheck
                .round_polynomials
                .len())
                .map(Self::BlindFoldInnerSumcheck),
        );
        targets.extend(
            interesting_indices(
                fixture
                    .proof
                    .blindfold
                    .witness_opening
                    .combined_vector
                    .len(),
            )
            .into_iter()
            .map(Self::BlindFoldWitnessOpening),
        );
        targets.extend(
            interesting_indices(fixture.proof.blindfold.error_opening.combined_vector.len())
                .into_iter()
                .map(Self::BlindFoldErrorOpening),
        );
        targets.extend(
            (0..fixture.proof.blindfold.folded_eval_outputs.len())
                .map(Self::BlindFoldFoldedEvalOutput),
        );
        targets.extend(
            (0..fixture.proof.blindfold.folded_eval_blindings.len())
                .map(Self::BlindFoldFoldedEvalBlinding),
        );
        for (opening, proof) in fixture
            .proof
            .blindfold
            .folded_eval_output_openings
            .iter()
            .enumerate()
        {
            targets.extend(
                interesting_indices(proof.combined_vector.len())
                    .into_iter()
                    .map(move |index| Self::BlindFoldFoldedEvalOutputOpening(opening, index)),
            );
        }
        for (opening, proof) in fixture
            .proof
            .blindfold
            .folded_eval_blinding_openings
            .iter()
            .enumerate()
        {
            targets.extend(
                interesting_indices(proof.combined_vector.len())
                    .into_iter()
                    .map(move |index| Self::BlindFoldFoldedEvalBlindingOpening(opening, index)),
            );
        }
        targets
    }

    fn apply(self, fixture: &mut MiniProtocolZkFixture) {
        let generator = Pairing::g1_generator();
        match self {
            Self::PublicInput(index) => fixture.public_inputs[index] += Fr::from_u64(1),
            Self::RelationCoefficient(matrix, row) => {
                relation_row(&mut fixture.relation, matrix, row).push((0, Fr::from_u64(1)));
            }
            Self::RelationStatement(field) => match field {
                RelationStatementField::Variables => {
                    fixture.proof.relation.dimensions.variables += 1;
                }
                RelationStatementField::Constraints => {
                    fixture.proof.relation.dimensions.constraints += 1;
                }
                RelationStatementField::PublicInputs => {
                    fixture.proof.relation.dimensions.public_inputs += 1;
                }
            },
            Self::WitnessCommitment => {
                fixture.proof.hyperkzg.witness_commitment = HyperKZGCommitment::default();
            }
            Self::OuterRound(index) => {
                fixture.proof.spartan.outer_sumcheck.rounds[index].commitment += generator;
            }
            Self::OuterOutputClaimCommitment(index) => {
                fixture
                    .proof
                    .spartan
                    .outer_sumcheck
                    .output_claims
                    .commitments[index] += generator;
            }
            Self::InnerRound(index) => {
                fixture.proof.spartan.inner_sumcheck.rounds[index].commitment += generator;
            }
            Self::InnerOutputClaimCommitment(index) => {
                fixture
                    .proof
                    .spartan
                    .inner_sumcheck
                    .output_claims
                    .commitments[index] += generator;
            }
            Self::HyperKzgFoldCommitment(index) => {
                fixture.proof.hyperkzg.witness_opening_proof.com[index] += generator;
            }
            Self::HyperKzgHiddenEvaluation(batch, index) => {
                let jolt_hyperkzg::HyperKZGProofPayload::Zk { y, .. } =
                    &mut fixture.proof.hyperkzg.witness_opening_proof.payload
                else {
                    panic!("ZK fixture must use ZK HyperKZG payload");
                };
                y[batch][index] += generator;
            }
            Self::HyperKzgHiddenOutput => {
                let jolt_hyperkzg::HyperKZGProofPayload::Zk { y_out, .. } =
                    &mut fixture.proof.hyperkzg.witness_opening_proof.payload
                else {
                    panic!("ZK fixture must use ZK HyperKZG payload");
                };
                *y_out += generator;
            }
            Self::HyperKzgWitness(index) => {
                fixture.proof.hyperkzg.witness_opening_proof.w[index] += generator;
            }
            Self::BlindFoldRandomU => fixture.proof.blindfold.random_u += Fr::from_u64(1),
            Self::BlindFoldAuxiliaryCommitment(index) => {
                fixture.proof.blindfold.auxiliary_row_commitments[index] += generator;
            }
            Self::BlindFoldRandomRoundCommitment(index) => {
                fixture.proof.blindfold.random_round_commitments[index] += generator;
            }
            Self::BlindFoldRandomOutputClaimCommitment(index) => {
                fixture.proof.blindfold.random_output_claim_row_commitments[index] += generator;
            }
            Self::BlindFoldRandomAuxiliaryCommitment(index) => {
                fixture.proof.blindfold.random_auxiliary_row_commitments[index] += generator;
            }
            Self::BlindFoldRandomErrorCommitment(index) => {
                fixture.proof.blindfold.random_error_row_commitments[index] += generator;
            }
            Self::BlindFoldRandomEvalCommitment(index) => {
                fixture.proof.blindfold.random_eval_commitments[index] += generator;
            }
            Self::BlindFoldCrossTermCommitment(index) => {
                fixture.proof.blindfold.cross_term_error_row_commitments[index] += generator;
            }
            Self::BlindFoldOuterSumcheck(index) => {
                fixture.proof.blindfold.outer_sumcheck.round_polynomials[index] =
                    fixture.proof.blindfold.outer_sumcheck.round_polynomials[index]
                        .clone()
                        .decompress(Fr::from_u64(0))
                        .add_constant(Fr::from_u64(1))
                        .compress();
            }
            Self::BlindFoldInnerSumcheck(index) => {
                fixture.proof.blindfold.inner_sumcheck.round_polynomials[index] =
                    fixture.proof.blindfold.inner_sumcheck.round_polynomials[index]
                        .clone()
                        .decompress(Fr::from_u64(0))
                        .add_constant(Fr::from_u64(1))
                        .compress();
            }
            Self::BlindFoldWitnessOpening(index) => {
                fixture.proof.blindfold.witness_opening.combined_vector[index] += Fr::from_u64(1);
            }
            Self::BlindFoldErrorOpening(index) => {
                fixture.proof.blindfold.error_opening.combined_vector[index] += Fr::from_u64(1);
            }
            Self::BlindFoldFoldedEvalOutput(index) => {
                fixture.proof.blindfold.folded_eval_outputs[index] += Fr::from_u64(1);
            }
            Self::BlindFoldFoldedEvalBlinding(index) => {
                fixture.proof.blindfold.folded_eval_blindings[index] += Fr::from_u64(1);
            }
            Self::BlindFoldFoldedEvalOutputOpening(opening, index) => {
                fixture.proof.blindfold.folded_eval_output_openings[opening].combined_vector
                    [index] += Fr::from_u64(1);
            }
            Self::BlindFoldFoldedEvalBlindingOpening(opening, index) => {
                fixture.proof.blindfold.folded_eval_blinding_openings[opening].combined_vector
                    [index] += Fr::from_u64(1);
            }
        }
    }
}

fn interesting_indices(len: usize) -> Vec<usize> {
    match len {
        0 => Vec::new(),
        1 => vec![0],
        2 => vec![0, 1],
        _ => vec![0, len / 2, len - 1],
    }
}

#[cfg(feature = "zk")]
fn zk_statistical_samples() -> usize {
    env::var(ZK_STATISTICAL_SAMPLES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ZK_STATISTICAL_SAMPLES)
        .max(DEFAULT_ZK_STATISTICAL_SAMPLES)
}

#[cfg(feature = "zk")]
fn zk_statistical_projection_values(fixture: &MiniProtocolZkFixture) -> [u64; 5] {
    let hyperkzg_eval_commitment = match &fixture.proof.hyperkzg.witness_opening_proof.payload {
        jolt_hyperkzg::HyperKZGProofPayload::Zk { y_out, .. } => y_out,
        jolt_hyperkzg::HyperKZGProofPayload::Clear { .. } => {
            panic!("ZK fixture must use ZK HyperKZG payload")
        }
    };

    [
        transcript_projection(
            b"witness_commit",
            &fixture.proof.hyperkzg.witness_commitment,
        ),
        transcript_projection(b"hyper_eval", hyperkzg_eval_commitment),
        field_low_u64(fixture.proof.blindfold.random_u),
        transcript_projection(
            b"bf_aux",
            &fixture.proof.blindfold.auxiliary_row_commitments[0],
        ),
        opening_projection(b"bf_wit_open", &fixture.proof.blindfold.witness_opening),
    ]
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
struct StatisticalProjection {
    label: &'static str,
    values: Vec<u64>,
}

#[cfg(feature = "zk")]
impl StatisticalProjection {
    fn new(label: &'static str, capacity: usize) -> Self {
        Self {
            label,
            values: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, value: u64) {
        self.values.push(value);
    }
}

#[cfg(feature = "zk")]
fn field_low_u64(value: Fr) -> u64 {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("field byte prefix has length 8"),
    )
}

#[cfg(feature = "zk")]
fn transcript_projection<A: AppendToTranscript>(label: &'static [u8], value: &A) -> u64 {
    let mut transcript = WrapperTranscript::new(b"wrapper-zk-stat-proj");
    transcript.append(&Label(label));
    value.append_to_transcript(&mut transcript);
    field_low_u64(transcript.challenge())
}

#[cfg(feature = "zk")]
fn opening_projection(label: &'static [u8], opening: &VectorCommitmentOpening<Fr>) -> u64 {
    let mut transcript = WrapperTranscript::new(b"wrapper-zk-stat-proj");
    append_values(&mut transcript, label, &opening.combined_vector);
    transcript.append(&Label(b"opening_blinding"));
    opening
        .combined_blinding
        .append_to_transcript(&mut transcript);
    field_low_u64(transcript.challenge())
}

#[cfg(feature = "zk")]
fn assert_empirical_distribution(projection: &StatisticalProjection) {
    assert!(
        projection.values.len() >= 32,
        "{} needs enough samples for empirical checks",
        projection.label
    );
    assert_high_unique_ratio(projection);
    assert_low_bit_balance(projection);
    assert_low_bucket_chi_square(projection);
    assert_lag_one_correlation(projection);
}

#[cfg(feature = "zk")]
fn assert_empirical_pairwise_independence(
    lhs: &StatisticalProjection,
    rhs: &StatisticalProjection,
) {
    let correlation = pearson_correlation(&lhs.values, &rhs.values);
    assert!(
        correlation.abs() < 0.45,
        "{} and {} have suspicious pairwise correlation: {correlation}",
        lhs.label,
        rhs.label
    );
}

#[cfg(feature = "zk")]
fn assert_high_unique_ratio(projection: &StatisticalProjection) {
    let mut sorted = projection.values.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let minimum_unique = projection.values.len() * 95 / 100;
    assert!(
        sorted.len() >= minimum_unique,
        "{} reused too many projected samples: {} unique out of {}",
        projection.label,
        sorted.len(),
        projection.values.len()
    );
}

#[cfg(feature = "zk")]
fn assert_low_bit_balance(projection: &StatisticalProjection) {
    let ones = projection
        .values
        .iter()
        .map(|value| value.count_ones() as u64)
        .sum::<u64>();
    let bit_count = (projection.values.len() * u64::BITS as usize) as f64;
    let expected = bit_count / 2.0;
    let sigma = (bit_count / 4.0).sqrt();
    let z_score = ((ones as f64) - expected).abs() / sigma;
    assert!(
        z_score < 6.0,
        "{} low-bit balance failed: ones={ones}, z={z_score}",
        projection.label
    );
}

#[cfg(feature = "zk")]
fn assert_low_bucket_chi_square(projection: &StatisticalProjection) {
    const BUCKETS: usize = 16;
    let mut buckets = [0usize; BUCKETS];
    for &value in &projection.values {
        buckets[(value & (BUCKETS as u64 - 1)) as usize] += 1;
    }
    let expected = projection.values.len() as f64 / BUCKETS as f64;
    let chi_square = buckets
        .iter()
        .map(|&count| {
            let delta = count as f64 - expected;
            delta * delta / expected
        })
        .sum::<f64>();
    assert!(
        chi_square < 52.0,
        "{} bucket chi-square too high: {chi_square}",
        projection.label
    );
}

#[cfg(feature = "zk")]
fn assert_lag_one_correlation(projection: &StatisticalProjection) {
    let correlation = pearson_correlation(
        &projection.values[..projection.values.len() - 1],
        &projection.values[1..],
    );
    assert!(
        correlation.abs() < 0.45,
        "{} has suspicious lag-one correlation: {correlation}",
        projection.label
    );
}

#[cfg(feature = "zk")]
fn pearson_correlation(lhs: &[u64], rhs: &[u64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    assert!(lhs.len() >= 2);
    let lhs_values = lhs.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let rhs_values = rhs.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let lhs_mean = lhs_values.iter().sum::<f64>() / lhs_values.len() as f64;
    let rhs_mean = rhs_values.iter().sum::<f64>() / rhs_values.len() as f64;
    let mut numerator = 0.0;
    let mut lhs_variance = 0.0;
    let mut rhs_variance = 0.0;
    for (&lhs_value, &rhs_value) in lhs_values.iter().zip(&rhs_values) {
        let lhs_delta = lhs_value - lhs_mean;
        let rhs_delta = rhs_value - rhs_mean;
        numerator += lhs_delta * rhs_delta;
        lhs_variance += lhs_delta * lhs_delta;
        rhs_variance += rhs_delta * rhs_delta;
    }
    let denominator = (lhs_variance * rhs_variance).sqrt();
    assert!(denominator > 0.0);
    numerator / denominator
}

fn relation_row(
    relation: &mut ConstraintMatrices<Fr>,
    matrix: Matrix,
    row: usize,
) -> &mut Vec<(usize, Fr)> {
    match matrix {
        Matrix::A => &mut relation.a[row],
        Matrix::B => &mut relation.b[row],
        Matrix::C => &mut relation.c[row],
    }
}

trait AddConstant {
    fn add_constant(self, delta: Fr) -> Self;
}

impl AddConstant for UnivariatePoly<Fr> {
    fn add_constant(self, delta: Fr) -> Self {
        let value = self.evaluate(Fr::from_u64(0)) + delta;
        let shifted = self + UnivariatePoly::new(vec![delta, Fr::from_u64(0)]);
        debug_assert_eq!(shifted.evaluate(Fr::from_u64(0)), value);
        shifted
    }
}

fn build_mini_protocol() -> WrapperR1csProtocol<Fr> {
    let mut builder = WrapperR1csBuilder::<Fr, CircuitTranscript>::new(b"WrapperGoldStar");

    let public_seed = builder.alloc_public_scalar(Fr::from_u64(9));
    let private_seed = builder.alloc_witness_scalar(Fr::from_u64(21));
    let bytes = [0x77].map(|byte| builder.alloc_witness_byte(byte));
    builder
        .transcript
        .append_scalar(&mut builder.builder, b"public_seed", public_seed.clone());
    builder
        .transcript
        .append_scalar(&mut builder.builder, b"private_seed", private_seed.clone());
    builder
        .transcript
        .append_bytes(&mut builder.builder, b"protocol_bytes", &bytes);
    let challenge = builder.transcript.challenge_scalar(&mut builder.builder);

    let sumcheck_output = append_native_sumcheck(&mut builder.builder, &challenge);
    append_hyrax_opening(&mut builder.builder, &challenge);

    let public_output = builder.alloc_public_scalar(
        public_seed.value + private_seed.value * challenge.value + sumcheck_output,
    );
    let private_term = builder
        .builder
        .multiply(private_seed.lc.clone(), challenge.lc.clone());
    builder.builder.assert_equal(
        public_output.lc,
        public_seed.lc + private_term + LinearCombination::constant(sumcheck_output),
    );

    builder.finish().expect("mini protocol builds")
}

fn append_native_sumcheck(builder: &mut R1csBuilder<Fr>, challenge: &AssignedScalar<Fr>) -> Fr {
    let statement = SumcheckStatement::new(1, 2);
    let rounds = [VariableChallengeRound {
        degree: 2,
        challenge: challenge.lc.clone(),
    }];
    let layout =
        allocate_sumcheck_r1cs_layout(builder, statement, &rounds).expect("sumcheck layout");
    let c0 = Fr::from_u64(3);
    let c1 = Fr::from_u64(4);
    let c2 = Fr::from_u64(5);
    let output_claim = c0 + c1 * challenge.value + c2 * challenge.value * challenge.value;

    builder
        .assign(layout.input_claim, c0 + (c0 + c1 + c2))
        .expect("sumcheck input assignment");
    builder
        .assign(layout.rounds[0].coefficients[0], c0)
        .expect("sumcheck c0 assignment");
    builder
        .assign(layout.rounds[0].coefficients[1], c1)
        .expect("sumcheck c1 assignment");
    builder
        .assign(layout.rounds[0].coefficients[2], c2)
        .expect("sumcheck c2 assignment");
    builder
        .assign(layout.rounds[0].claim_out, output_claim)
        .expect("sumcheck output assignment");
    append_sumcheck_r1cs_constraints(builder, statement, &rounds, &layout)
        .expect("sumcheck constraints");

    output_claim
}

#[derive(Clone, Debug)]
struct VariableChallengeRound {
    degree: usize,
    challenge: LinearCombination<Fr>,
}

impl jolt_sumcheck::SumcheckR1csRound<Fr> for VariableChallengeRound {
    fn degree(&self) -> usize {
        self.degree
    }

    fn challenge(&self) -> LinearCombination<Fr> {
        self.challenge.clone()
    }
}

fn append_hyrax_opening(builder: &mut R1csBuilder<Fr>, challenge: &AssignedScalar<Fr>) {
    let row_challenge = FqVar::inject_fr_challenge(builder, challenge);
    let case = HyraxNativeCase::new(fr_to_fq(challenge.value));
    let input = HyraxOpeningR1csInput {
        row_commitments: case
            .row_commitments
            .iter()
            .map(|commitment| GrumpkinPointWithIdentityVar::alloc(builder, commitment))
            .collect(),
        row_point: vec![row_challenge],
        entry_point: case
            .entry_point
            .iter()
            .copied()
            .map(|value| FqVar::alloc(builder, value))
            .collect(),
        combined_row: case
            .combined_row
            .iter()
            .copied()
            .map(|value| FqVar::alloc(builder, value))
            .collect(),
        combined_blinding: FqVar::alloc(builder, case.combined_blinding),
        claimed_eval: FqVar::alloc(builder, case.claimed_eval),
    };
    verify_opening::<TestVc>(builder, &case.setup, &input).expect("valid Hyrax opening");
}

#[derive(Clone, Debug)]
struct HyraxNativeCase {
    setup: PedersenSetup<GrumpkinPoint>,
    row_commitments: Vec<GrumpkinPoint>,
    entry_point: Vec<Fq>,
    combined_row: Vec<Fq>,
    combined_blinding: Fq,
    claimed_eval: Fq,
}

impl HyraxNativeCase {
    fn new(row_point_head: Fq) -> Self {
        let generator = Grumpkin::generator();
        let setup = PedersenSetup::new(
            (1..=2)
                .map(|index| generator.scalar_mul(&Fq::from_u64(10 + index)))
                .collect(),
            generator.scalar_mul(&Fq::from_u64(99)),
        );
        let rows = [[2, 3], [5, 7]].map(|row| row.map(Fq::from_u64).to_vec());
        let row_blindings = [Fq::from_u64(101), Fq::from_u64(103)];
        let row_point = vec![row_point_head];
        let entry_point = vec![Fq::from_u64(7)];
        let row_commitments = rows
            .iter()
            .zip(row_blindings)
            .map(|(row, blinding)| TestVc::commit(&setup, row, &blinding))
            .collect::<Vec<_>>();
        let row_weights = EqPolynomial::new(row_point).evaluations();
        let entry_weights = EqPolynomial::new(entry_point.clone()).evaluations();
        let mut combined_row = vec![Fq::from_u64(0); 2];
        for (row, row_weight) in rows.iter().zip(&row_weights) {
            for (combined, row_entry) in combined_row.iter_mut().zip(row) {
                *combined += *row_weight * *row_entry;
            }
        }
        let combined_blinding = row_blindings
            .iter()
            .zip(&row_weights)
            .fold(Fq::from_u64(0), |acc, (blinding, row_weight)| {
                acc + *blinding * *row_weight
            });
        let claimed_eval = combined_row
            .iter()
            .zip(&entry_weights)
            .fold(Fq::from_u64(0), |acc, (entry, entry_weight)| {
                acc + *entry * *entry_weight
            });

        Self {
            setup,
            row_commitments,
            entry_point,
            combined_row,
            combined_blinding,
            claimed_eval,
        }
    }
}

fn prove_wrapper(
    protocol: &WrapperR1csProtocol<Fr>,
    setup: &HyperKZGProverSetup<Pairing>,
) -> WrapperProof<Pairing> {
    let relation = relation_dimensions(protocol);
    let witness_poly = witness_polynomial(protocol);
    let (witness_commitment, opening_hint) =
        <KzgPCS as CommitmentScheme>::commit(&witness_poly, setup);
    let mut proof = WrapperProof::from_parts(
        R1csRelationStatement::new(relation),
        SpartanProof::default(),
        HyperKzgProof::new(
            witness_commitment,
            jolt_hyperkzg::HyperKZGProof {
                com: Vec::new(),
                w: [Pairing::g1_generator(); 3],
                payload: jolt_hyperkzg::HyperKZGProofPayload::Clear {
                    v: [Vec::new(), Vec::new(), Vec::new()],
                },
            },
        ),
    );
    let checked = jolt_wrapper_verifier::CheckedInputs {
        relation_variables: protocol.r1cs.num_vars,
        relation_constraints: protocol.r1cs.num_constraints,
        public_inputs: protocol.public_inputs.len(),
    };
    let mut transcript = WrapperTranscript::new(WRAPPER_TRANSCRIPT_LABEL);
    let relation_output = r1cs_relation::verify(
        r1cs_relation::R1csRelationInputs {
            checked: &checked,
            relation: &protocol.r1cs,
            public_inputs: &protocol.public_inputs,
            proof_relation: proof.relation,
        },
        &mut transcript,
    )
    .expect("relation prelude should verify");
    transcript.append(&Label(WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL));
    transcript.append(&witness_commitment);

    let spartan = prove_spartan(
        &protocol.r1cs,
        &witness_poly,
        &relation_output,
        &mut transcript,
    );
    <KzgPCS as CommitmentScheme>::bind_opening_inputs(
        &mut transcript,
        spartan.witness_point.as_slice(),
        &spartan.witness_opening_claim,
    );
    let opening_proof = <KzgPCS as CommitmentScheme>::open(
        &witness_poly,
        spartan.witness_point.as_slice(),
        spartan.witness_opening_claim,
        setup,
        Some(opening_hint),
        &mut transcript,
    );

    proof.spartan = spartan.proof;
    proof.hyperkzg = HyperKzgProof::new(witness_commitment, opening_proof);
    proof
}

#[cfg(feature = "zk")]
fn prove_wrapper_zk(
    protocol: &WrapperR1csProtocol<Fr>,
    setup: &HyperKZGProverSetup<Pairing>,
    vc_setup: &PedersenSetup<Bn254G1>,
    rng: &mut ChaCha20Rng,
) -> WrapperZkProof<Pairing, WrapperBlindFoldVc> {
    prove_wrapper_zk_inner(protocol, setup, vc_setup, true, rng)
}

#[cfg(feature = "zk")]
fn prove_wrapper_zk_with_unchecked_final_eval(
    protocol: &WrapperR1csProtocol<Fr>,
    setup: &HyperKZGProverSetup<Pairing>,
    vc_setup: &PedersenSetup<Bn254G1>,
    rng: &mut ChaCha20Rng,
) -> WrapperZkProof<Pairing, WrapperBlindFoldVc> {
    prove_wrapper_zk_inner(protocol, setup, vc_setup, false, rng)
}

#[cfg(feature = "zk")]
fn prove_wrapper_zk_inner(
    protocol: &WrapperR1csProtocol<Fr>,
    setup: &HyperKZGProverSetup<Pairing>,
    vc_setup: &PedersenSetup<Bn254G1>,
    expect_final_eval_commitment_opening: bool,
    rng: &mut ChaCha20Rng,
) -> WrapperZkProof<Pairing, WrapperBlindFoldVc> {
    let relation = relation_dimensions(protocol);
    let witness_poly = witness_polynomial(protocol);
    let (witness_commitment, opening_hint) =
        <KzgPCS as ZkOpeningScheme>::commit_zk(witness_poly.evaluations(), setup);
    let checked = jolt_wrapper_verifier::CheckedInputs {
        relation_variables: protocol.r1cs.num_vars,
        relation_constraints: protocol.r1cs.num_constraints,
        public_inputs: protocol.public_inputs.len(),
    };
    let proof_relation = R1csRelationStatement::new(relation);
    let mut transcript = WrapperTranscript::new(WRAPPER_TRANSCRIPT_LABEL);
    let relation_output = r1cs_relation::verify(
        r1cs_relation::R1csRelationInputs {
            checked: &checked,
            relation: &protocol.r1cs,
            public_inputs: &protocol.public_inputs,
            proof_relation,
        },
        &mut transcript,
    )
    .expect("relation prelude should verify");
    transcript.append(&Label(WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL));
    transcript.append(&witness_commitment);

    let spartan = prove_spartan_zk(
        &protocol.r1cs,
        &witness_poly,
        &relation_output,
        vc_setup,
        &mut transcript,
        rng,
    );
    let (opening_proof, hiding_evaluation_commitment, evaluation_blind) = KzgPCS::open_zk(
        &witness_poly,
        spartan.witness_point.as_slice(),
        spartan.witness_opening_claim,
        setup,
        opening_hint,
        &mut transcript,
    );
    KzgPCS::bind_zk_opening_inputs(
        &mut transcript,
        spartan.witness_point.as_slice(),
        &hiding_evaluation_commitment,
    );
    if expect_final_eval_commitment_opening {
        assert!(WrapperBlindFoldVc::verify(
            vc_setup,
            &hiding_evaluation_commitment,
            &[spartan.witness_opening_claim],
            &evaluation_blind,
        ));
    }

    let hyperkzg_output = HyperKzgZkOutput {
        hiding_evaluation_commitment,
    };
    let blindfold_protocol =
        wrapper_blindfold_protocol(&spartan.output, &hyperkzg_output).expect("protocol builds");
    let blindfold_statement = wrapper_blindfold_statement(&spartan.output, &hyperkzg_output);
    let (witness_rows, witness_blindings) = wrapper_blindfold_witness_rows(
        &blindfold_protocol,
        &blindfold_statement,
        &spartan.output,
        [&spartan.outer_trace, &spartan.inner_trace],
        spartan.witness_opening_claim,
        evaluation_blind,
        rng,
    );
    let blindfold = prove_blindfold_from_rows(
        vc_setup,
        &blindfold_protocol,
        &mut transcript,
        BlindFoldWitnessRows {
            rows: &witness_rows,
            blindings: &witness_blindings,
            eval_outputs: &[spartan.witness_opening_claim],
            eval_blindings: &[evaluation_blind],
            expect_eval_commitment_openings: expect_final_eval_commitment_opening,
        },
        rng,
    );

    WrapperZkProof::from_parts(
        proof_relation,
        spartan.proof,
        HyperKzgProof::new(witness_commitment, opening_proof),
        blindfold,
    )
}

#[cfg(feature = "zk")]
struct ProvedZkSpartan {
    proof: SpartanZkProof<Bn254G1>,
    output: SpartanZkOutput<Fr, Bn254G1>,
    outer_trace: CommittedSumcheckTrace,
    inner_trace: CommittedSumcheckTrace,
    witness_point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Fr>,
    witness_opening_claim: Fr,
}

#[cfg(feature = "zk")]
fn prove_spartan_zk(
    relation: &ConstraintMatrices<Fr>,
    witness_poly: &Polynomial<Fr>,
    relation_output: &r1cs_relation::R1csRelationOutput<'_, Fr>,
    vc_setup: &PedersenSetup<Bn254G1>,
    transcript: &mut WrapperTranscript,
    rng: &mut ChaCha20Rng,
) -> ProvedZkSpartan {
    let dimensions = relation_output.statement_facts.spartan;
    let row_polys = row_value_polys(
        relation,
        witness_poly.evaluations(),
        dimensions.num_constraints_padded(),
    );
    transcript.append(&Label(WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL));
    let tau = transcript.challenge_vector(dimensions.num_constraint_rounds());
    let tau_point = jolt_poly::Point::high_to_low(tau.clone());

    transcript.append(&Label(WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL));
    let eq_tau = EqPolynomial::new(tau.as_slice().to_vec()).evaluations();
    let outer_statement = SumcheckStatement::new(
        dimensions.num_constraint_rounds(),
        WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
    );
    let (outer_trace, outer_reduction) = prove_committed_sum_of_products_sumcheck(
        vec![
            SumcheckProductTerm::new(
                Fr::from_u64(1),
                vec![
                    eq_tau.clone(),
                    row_polys.a.evaluations().to_vec(),
                    row_polys.b.evaluations().to_vec(),
                ],
            ),
            SumcheckProductTerm::new(
                -Fr::from_u64(1),
                vec![eq_tau, row_polys.c.evaluations().to_vec()],
            ),
        ],
        outer_statement,
        Fr::from_u64(0),
        vc_setup,
        transcript,
        rng,
    );
    let outer_claims = row_polys.evaluate(&outer_reduction.point);
    let outer_final_claim =
        EqPolynomial::<Fr>::mle(tau.as_slice(), outer_reduction.point.as_slice())
            * (outer_claims.a * outer_claims.b - outer_claims.c);
    assert_eq!(outer_reduction.value, outer_final_claim);
    let outer_trace = outer_trace.with_output_claim_row(
        vc_setup,
        vec![outer_claims.a, outer_claims.b, outer_claims.c],
        rng,
        transcript,
    );

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL));
    let batching = SpartanInnerBatchingCoefficients {
        a: transcript.challenge_scalar(),
        b: transcript.challenge_scalar(),
        c: transcript.challenge_scalar(),
    };
    let combined_matrix_poly = combined_matrix_column_poly(
        relation,
        &outer_reduction.point,
        batching,
        dimensions.num_vars_padded(),
    );

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL));
    let inner_statement = SumcheckStatement::new(
        dimensions.num_var_rounds(),
        WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
    );
    let inner_input_claim = batching.combine(outer_claims);
    let (inner_trace, inner_reduction) = prove_committed_sum_of_products_sumcheck(
        vec![SumcheckProductTerm::new(
            Fr::from_u64(1),
            vec![
                combined_matrix_poly.evaluations().to_vec(),
                witness_poly.evaluations().to_vec(),
            ],
        )],
        inner_statement,
        inner_input_claim,
        vc_setup,
        transcript,
        rng,
    );
    let witness_opening_claim = witness_poly.evaluate(&inner_reduction.point);
    let combined_matrix_eval = combined_matrix_poly.evaluate(&inner_reduction.point);
    assert_eq!(
        inner_reduction.value,
        combined_matrix_eval * witness_opening_claim
    );
    let inner_trace =
        inner_trace.with_output_claim_row(vc_setup, vec![witness_opening_claim], rng, transcript);

    let outer_consistency = outer_trace.consistency.clone();
    let inner_consistency = inner_trace.consistency.clone();
    let outer_output_claims = CommittedOutputClaimOutput {
        shape: CommittedOutputClaimShape {
            output_claim_count: 3,
            row_count: outer_trace.proof.output_claims.commitments.len(),
            row_len: WrapperBlindFoldVc::capacity(vc_setup),
        },
        commitments: outer_trace.proof.output_claims.clone(),
    };
    let inner_output_claims = CommittedOutputClaimOutput {
        shape: CommittedOutputClaimShape {
            output_claim_count: 1,
            row_count: inner_trace.proof.output_claims.commitments.len(),
            row_len: WrapperBlindFoldVc::capacity(vc_setup),
        },
        commitments: inner_trace.proof.output_claims.clone(),
    };
    let output = SpartanZkOutput {
        dimensions,
        tau: tau_point.clone(),
        outer_statement,
        outer_consistency,
        outer_rx: outer_reduction.point.clone(),
        eq_tau_rx: EqPolynomial::<Fr>::mle(tau_point.as_slice(), outer_reduction.point.as_slice()),
        outer_output_claims,
        inner_batching: batching,
        inner_statement,
        inner_consistency,
        inner_ry: inner_reduction.point.clone(),
        combined_matrix_eval,
        inner_output_claims,
    };

    ProvedZkSpartan {
        proof: SpartanZkProof::new(outer_trace.proof.clone(), inner_trace.proof.clone()),
        output,
        outer_trace,
        inner_trace,
        witness_point: inner_reduction.point,
        witness_opening_claim,
    }
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
struct CommittedSumcheckTrace {
    proof: CommittedSumcheckProof<Bn254G1>,
    consistency: CommittedSumcheckConsistency<Fr, Bn254G1>,
    coefficients: Vec<Vec<Fr>>,
    blindings: Vec<Fr>,
    output_claim_rows: Vec<Vec<Fr>>,
    output_claim_blindings: Vec<Fr>,
    input_claim: Fr,
    claim_outs: Vec<Fr>,
}

#[cfg(feature = "zk")]
impl CommittedSumcheckTrace {
    fn with_output_claim_row(
        mut self,
        setup: &PedersenSetup<Bn254G1>,
        mut row: Vec<Fr>,
        rng: &mut ChaCha20Rng,
        transcript: &mut WrapperTranscript,
    ) -> Self {
        let row_len = WrapperBlindFoldVc::capacity(setup);
        row.resize(row_len, Fr::from_u64(0));
        let blinding = rng_field(rng);
        let commitment = WrapperBlindFoldVc::commit(setup, &row, &blinding);
        self.proof.output_claims = CommittedOutputClaims {
            commitments: vec![commitment],
        };
        self.proof.output_claims.append_to_transcript(transcript);
        self.output_claim_rows = vec![row];
        self.output_claim_blindings = vec![blinding];
        self
    }
}

#[cfg(feature = "zk")]
fn prove_committed_sum_of_products_sumcheck(
    mut terms: Vec<SumcheckProductTerm>,
    statement: SumcheckStatement,
    input_claim: Fr,
    setup: &PedersenSetup<Bn254G1>,
    transcript: &mut WrapperTranscript,
    rng: &mut ChaCha20Rng,
) -> (CommittedSumcheckTrace, jolt_sumcheck::EvaluationClaim<Fr>) {
    let table_len = terms[0].factors[0].len();
    assert!(
        table_len.is_power_of_two(),
        "product tables must be over a Boolean hypercube"
    );
    assert_eq!(
        statement.num_vars,
        table_len.trailing_zeros() as usize,
        "sumcheck statement must match table length"
    );
    for term in &terms {
        assert!(
            term.factors.len() <= statement.degree,
            "term arity must fit the sumcheck degree"
        );
        for factor in &term.factors {
            assert_eq!(factor.len(), table_len, "product tables must align");
        }
    }

    let mut prefix = Vec::with_capacity(statement.num_vars);
    let mut rounds = Vec::with_capacity(statement.num_vars);
    let mut verified_rounds = Vec::with_capacity(statement.num_vars);
    let mut coefficients = Vec::with_capacity(statement.num_vars);
    let mut blindings = Vec::with_capacity(statement.num_vars);
    let mut claim_outs = Vec::with_capacity(statement.num_vars);
    let mut running_claim = input_claim;

    for _round in 0..statement.num_vars {
        let round_evals = sum_of_products_round_evals(&terms, statement.degree);
        let round_poly = UnivariatePoly::from_evals(&round_evals);
        assert_eq!(
            round_poly.evaluate(Fr::from_u64(0)) + round_poly.evaluate(Fr::from_u64(1)),
            running_claim
        );
        let round_coefficients = round_poly.coefficients().to_vec();
        let blinding = rng_field(rng);
        let round = CommittedRoundWitness {
            coefficients: round_coefficients.clone(),
            blinding,
        }
        .commit::<WrapperBlindFoldVc>(setup)
        .expect("round witness commits");
        round.append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = round_poly.evaluate(challenge);
        bind_sumcheck_terms(&mut terms, challenge);

        verified_rounds.push(VerifiedCommittedRound {
            commitment: round.commitment,
            degree: round.degree,
            challenge,
        });
        prefix.push(challenge);
        rounds.push(round);
        coefficients.push(round_coefficients);
        blindings.push(blinding);
        claim_outs.push(running_claim);
    }
    let value = final_sum_of_products_claim(&terms);
    assert_eq!(value, running_claim);
    let proof = CommittedSumcheckProof {
        rounds,
        output_claims: CommittedOutputClaims::default(),
    };
    (
        CommittedSumcheckTrace {
            proof,
            consistency: CommittedSumcheckConsistency {
                rounds: verified_rounds,
            },
            coefficients,
            blindings,
            output_claim_rows: Vec::new(),
            output_claim_blindings: Vec::new(),
            input_claim,
            claim_outs,
        },
        jolt_sumcheck::EvaluationClaim::new(prefix, value),
    )
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperZkOpening {
    OuterA,
    OuterB,
    OuterC,
    WitnessZ,
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperZkPublic {
    EqTauRx,
    CombinedMatrixEval,
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperZkChallenge {
    InnerA,
    InnerB,
    InnerC,
}

#[cfg(feature = "zk")]
type WrapperBlindFoldStatement = jolt_blindfold::BlindFoldStatement<
    Fr,
    WrapperZkOpening,
    Bn254G1,
    WrapperZkPublic,
    WrapperZkChallenge,
>;

#[cfg(feature = "zk")]
fn wrapper_blindfold_statement(
    spartan: &SpartanZkOutput<Fr, Bn254G1>,
    hyperkzg: &HyperKzgZkOutput<Bn254G1>,
) -> WrapperBlindFoldStatement {
    jolt_blindfold::BlindFoldStatement::new(
        vec![
            jolt_blindfold::BlindFoldStage::new(
                "wrapper-spartan-outer",
                spartan.outer_statement,
                SumcheckDomainSpec::BooleanHypercube,
                spartan.outer_consistency.clone(),
                jolt_blindfold::CommittedClaimRows::new(
                    vec![
                        WrapperZkOpening::OuterA,
                        WrapperZkOpening::OuterB,
                        WrapperZkOpening::OuterC,
                    ],
                    spartan.outer_output_claims.shape.row_len,
                    spartan.outer_output_claims.commitments.clone(),
                ),
                constant(Fr::from_u64(0)),
                public(WrapperZkPublic::EqTauRx)
                    * (opening(WrapperZkOpening::OuterA) * opening(WrapperZkOpening::OuterB)
                        - opening(WrapperZkOpening::OuterC)),
            ),
            jolt_blindfold::BlindFoldStage::new(
                "wrapper-spartan-inner",
                spartan.inner_statement,
                SumcheckDomainSpec::BooleanHypercube,
                spartan.inner_consistency.clone(),
                jolt_blindfold::CommittedClaimRows::new(
                    vec![WrapperZkOpening::WitnessZ],
                    spartan.inner_output_claims.shape.row_len,
                    spartan.inner_output_claims.commitments.clone(),
                ),
                challenge(WrapperZkChallenge::InnerA) * opening(WrapperZkOpening::OuterA)
                    + challenge(WrapperZkChallenge::InnerB) * opening(WrapperZkOpening::OuterB)
                    + challenge(WrapperZkChallenge::InnerC) * opening(WrapperZkOpening::OuterC),
                public(WrapperZkPublic::CombinedMatrixEval) * opening(WrapperZkOpening::WitnessZ),
            ),
        ],
        vec![jolt_blindfold::FinalOpeningBinding::new(
            vec![WrapperZkOpening::WitnessZ],
            vec![Fr::from_u64(1)],
            hyperkzg.hiding_evaluation_commitment,
        )],
    )
}

#[cfg(feature = "zk")]
fn wrapper_blindfold_protocol(
    spartan: &SpartanZkOutput<Fr, Bn254G1>,
    hyperkzg: &HyperKzgZkOutput<Bn254G1>,
) -> Result<jolt_blindfold::BlindFoldProtocol<Fr, Bn254G1>, jolt_blindfold::VerificationError<Fr>> {
    let statement = wrapper_blindfold_statement(spartan, hyperkzg);
    jolt_blindfold::BlindFoldProtocol::<Fr, Bn254G1>::builder::<
        WrapperZkOpening,
        WrapperZkPublic,
        WrapperZkChallenge,
    >()
    .public(WrapperZkPublic::EqTauRx, spartan.eq_tau_rx)
    .public(
        WrapperZkPublic::CombinedMatrixEval,
        spartan.combined_matrix_eval,
    )
    .challenge(WrapperZkChallenge::InnerA, spartan.inner_batching.a)
    .challenge(WrapperZkChallenge::InnerB, spartan.inner_batching.b)
    .challenge(WrapperZkChallenge::InnerC, spartan.inner_batching.c)
    .stage(statement.stages[0].name.clone())
    .sumcheck(statement.stages[0].statement)
    .domain(statement.stages[0].domain)
    .consistency(statement.stages[0].consistency.clone())
    .output_claim_rows(
        statement.stages[0].output_claim_rows.opening_ids.clone(),
        statement.stages[0].output_claim_rows.row_len,
        statement.stages[0].output_claim_rows.commitments.clone(),
    )
    .input_claim(statement.stages[0].input_claim.clone())
    .output_claim(statement.stages[0].output_claim.clone())
    .finish_stage()
    .expect("outer statement is complete")
    .stage(statement.stages[1].name.clone())
    .sumcheck(statement.stages[1].statement)
    .domain(statement.stages[1].domain)
    .consistency(statement.stages[1].consistency.clone())
    .output_claim_rows(
        statement.stages[1].output_claim_rows.opening_ids.clone(),
        statement.stages[1].output_claim_rows.row_len,
        statement.stages[1].output_claim_rows.commitments.clone(),
    )
    .input_claim(statement.stages[1].input_claim.clone())
    .output_claim(statement.stages[1].output_claim.clone())
    .finish_stage()
    .expect("inner statement is complete")
    .final_opening(
        statement.final_openings[0].opening_ids.clone(),
        statement.final_openings[0].coefficients.clone(),
        statement.final_openings[0].evaluation_commitment,
    )
    .build()
}

#[cfg(feature = "zk")]
fn wrapper_blindfold_witness_rows(
    protocol: &jolt_blindfold::BlindFoldProtocol<Fr, Bn254G1>,
    statement: &WrapperBlindFoldStatement,
    spartan: &SpartanZkOutput<Fr, Bn254G1>,
    traces: [&CommittedSumcheckTrace; 2],
    eval_output: Fr,
    eval_blinding: Fr,
    rng: &mut ChaCha20Rng,
) -> (Vec<Vec<Fr>>, Vec<Fr>) {
    let mut builder = R1csBuilder::<Fr>::new();
    let mut sources =
        ClaimSourceTable::<Fr, WrapperZkOpening, WrapperZkPublic, WrapperZkChallenge>::new();
    let layout =
        jolt_blindfold::r1cs::allocate_layout(&mut builder, statement).expect("layout allocates");
    for (stage, stage_layout) in statement.stages.iter().zip(&layout.stages) {
        let variables = stage_layout
            .output_claim_rows
            .iter()
            .flat_map(|row| row.variables.iter().take(stage.output_claim_rows.row_len));
        for (opening_id, &variable) in stage.output_claim_rows.opening_ids.iter().zip(variables) {
            sources.insert_opening(*opening_id, variable);
        }
    }
    sources.insert_public(WrapperZkPublic::EqTauRx, spartan.eq_tau_rx);
    sources.insert_public(
        WrapperZkPublic::CombinedMatrixEval,
        spartan.combined_matrix_eval,
    );
    sources.insert_challenge(WrapperZkChallenge::InnerA, spartan.inner_batching.a);
    sources.insert_challenge(WrapperZkChallenge::InnerB, spartan.inner_batching.b);
    sources.insert_challenge(WrapperZkChallenge::InnerC, spartan.inner_batching.c);

    for (stage, (stage_layout, trace)) in statement
        .stages
        .iter()
        .zip(layout.stages.iter().zip(traces))
    {
        assign_committed_trace(&mut builder, &stage_layout.sumcheck, trace);
        let variables = stage_layout
            .output_claim_rows
            .iter()
            .flat_map(|row| row.variables.iter().take(trace.output_claim_rows[0].len()));
        let values = trace
            .output_claim_rows
            .iter()
            .flat_map(|row| row.iter().copied());
        for (&variable, value) in variables
            .zip(values)
            .take(stage.output_claim_rows.opening_ids.len())
        {
            builder
                .assign(variable, value)
                .expect("output claim opening assigns");
        }
    }
    if let Some(evaluation) = layout.final_openings[0].evaluation {
        builder
            .assign(evaluation, eval_output)
            .expect("final opening evaluation assigns");
    }
    if let Some(blinding) = layout.final_openings[0].blinding {
        builder
            .assign(blinding, eval_blinding)
            .expect("final opening blinding assigns");
    }
    jolt_blindfold::r1cs::append(&mut builder, statement, &layout, &mut sources)
        .expect("constraints append");

    let witness = builder.witness().expect("witness is assigned");
    assert!(builder.into_matrices().check_witness(&witness).is_ok());

    let row_len = protocol.dimensions.witness.row_len;
    let mut rows = witness[1..=protocol.dimensions.coefficient_values]
        .chunks(row_len)
        .map(<[Fr]>::to_vec)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), protocol.dimensions.coefficient_rows);

    for trace in traces {
        for row in &trace.output_claim_rows {
            let mut row = row.clone();
            row.resize(row_len, Fr::from_u64(0));
            rows.push(row);
        }
    }
    assert_eq!(
        rows.len(),
        protocol.dimensions.witness_rows.output_claims.end
    );

    let output_claim_values = protocol.dimensions.output_claim_rows * row_len;
    let auxiliary_values =
        &witness[1 + protocol.dimensions.coefficient_values + output_claim_values..];
    let mut auxiliary_rows = auxiliary_values
        .chunks(row_len)
        .map(|chunk| {
            let mut row = chunk.to_vec();
            row.resize(row_len, Fr::from_u64(0));
            row
        })
        .collect::<Vec<_>>();
    auxiliary_rows.resize(
        protocol.dimensions.auxiliary_rows,
        vec![Fr::from_u64(0); row_len],
    );
    rows.extend(auxiliary_rows);
    rows.resize(
        protocol.dimensions.witness.row_count,
        vec![Fr::from_u64(0); row_len],
    );

    let mut blindings = traces
        .iter()
        .flat_map(|trace| trace.blindings.iter().copied())
        .collect::<Vec<_>>();
    blindings.extend(
        traces
            .iter()
            .flat_map(|trace| trace.output_claim_blindings.iter().copied()),
    );
    blindings.extend((0..protocol.dimensions.auxiliary_rows).map(|_| rng_field(rng)));
    blindings.resize(protocol.dimensions.witness.row_count, Fr::from_u64(0));

    assert_eq!(rows.len(), protocol.dimensions.witness.row_count);
    assert_eq!(blindings.len(), protocol.dimensions.witness.row_count);
    (rows, blindings)
}

#[cfg(feature = "zk")]
fn assign_committed_trace(
    builder: &mut R1csBuilder<Fr>,
    layout: &jolt_sumcheck::SumcheckR1csLayout,
    trace: &CommittedSumcheckTrace,
) {
    builder
        .assign(layout.input_claim, trace.input_claim)
        .expect("input claim assigns");
    for (round_layout, (round_coefficients, &claim_out)) in layout
        .rounds
        .iter()
        .zip(trace.coefficients.iter().zip(&trace.claim_outs))
    {
        for (&variable, &coefficient) in round_layout.coefficients.iter().zip(round_coefficients) {
            builder
                .assign(variable, coefficient)
                .expect("coefficient assigns");
        }
        builder
            .assign(round_layout.claim_out, claim_out)
            .expect("claim out assigns");
    }
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
struct BlindFoldWitnessRows<'a> {
    rows: &'a [Vec<Fr>],
    blindings: &'a [Fr],
    eval_outputs: &'a [Fr],
    eval_blindings: &'a [Fr],
    expect_eval_commitment_openings: bool,
}

#[cfg(feature = "zk")]
fn prove_blindfold_from_rows(
    setup: &PedersenSetup<Bn254G1>,
    protocol: &jolt_blindfold::BlindFoldProtocol<Fr, Bn254G1>,
    transcript: &mut WrapperTranscript,
    witness: BlindFoldWitnessRows<'_>,
    rng: &mut ChaCha20Rng,
) -> jolt_blindfold::BlindFoldProof<Fr, Bn254G1> {
    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let auxiliary_row_commitments = commit_rows(
        setup,
        &witness.rows[auxiliary_range.clone()],
        &witness.blindings[auxiliary_range],
    );
    let committed = protocol
        .committed_relaxed_instance(&auxiliary_row_commitments)
        .expect("committed relaxed instance builds");
    assert_eq!(
        committed.witness_row_commitments,
        commit_rows(setup, witness.rows, witness.blindings)
    );
    if witness.expect_eval_commitment_openings {
        for ((commitment, &output), &blinding) in protocol
            .eval_commitments
            .iter()
            .zip(witness.eval_outputs)
            .zip(witness.eval_blindings)
        {
            assert!(WrapperBlindFoldVc::verify(
                setup,
                commitment,
                &[output],
                &blinding
            ));
        }
    }

    let random_u = rng_field(rng);
    let mut random_witness_rows = random_rows(
        protocol.dimensions.witness.row_count,
        protocol.dimensions.witness.row_len,
        rng,
    );
    let mut random_witness_blindings = (0..protocol.dimensions.witness.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    for row in protocol.dimensions.witness_rows.padding.clone() {
        random_witness_rows[row].fill(Fr::from_u64(0));
        random_witness_blindings[row] = Fr::from_u64(0);
    }
    let random_eval_outputs = (0..protocol.eval_commitments.len())
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let random_eval_blindings = (0..protocol.eval_commitments.len())
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let final_coordinates = protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are in witness layout");
    let mut dedicated_rows = Vec::new();
    for coordinates in &final_coordinates {
        if let Some(coordinate) = coordinates.evaluation {
            dedicated_rows.push(coordinate.row);
        }
        if let Some(coordinate) = coordinates.blinding {
            dedicated_rows.push(coordinate.row);
        }
    }
    dedicated_rows.sort_unstable();
    dedicated_rows.dedup();
    for row in dedicated_rows {
        random_witness_rows[row].fill(Fr::from_u64(0));
    }
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_outputs[index];
        }
        if let Some(coordinate) = coordinates.blinding {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_blindings[index];
        }
    }
    let random_error_rows = error_rows_for(
        &protocol.r1cs,
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_len,
        protocol.dimensions.error.row_count,
    );
    let random_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let coefficient_range = protocol.dimensions.witness_rows.coefficients.clone();
    let output_claim_range = protocol.dimensions.witness_rows.output_claims.clone();
    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let random_round_commitments = commit_rows(
        setup,
        &random_witness_rows[coefficient_range.clone()],
        &random_witness_blindings[coefficient_range],
    );
    let random_output_claim_row_commitments = commit_rows(
        setup,
        &random_witness_rows[output_claim_range.clone()],
        &random_witness_blindings[output_claim_range],
    );
    let random_auxiliary_row_commitments = commit_rows(
        setup,
        &random_witness_rows[auxiliary_range.clone()],
        &random_witness_blindings[auxiliary_range],
    );
    let random_error_row_commitments =
        commit_rows(setup, &random_error_rows, &random_error_blindings);
    let random_eval_commitments = random_eval_outputs
        .iter()
        .zip(&random_eval_blindings)
        .map(|(&output, blinding)| WrapperBlindFoldVc::commit(setup, &[output], blinding))
        .collect::<Vec<_>>();
    let random_instance = protocol
        .random_relaxed_instance(
            &random_round_commitments,
            &random_output_claim_row_commitments,
            &random_auxiliary_row_commitments,
            &random_error_row_commitments,
            &random_eval_commitments,
            random_u,
        )
        .expect("random relaxed instance builds");
    assert_eq!(
        random_instance.witness_row_commitments,
        commit_rows(setup, &random_witness_rows, &random_witness_blindings)
    );

    let cross_term_error_rows = cross_term_error_rows_for(
        &protocol.r1cs,
        Fr::from_u64(1),
        &flatten(witness.rows),
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_len,
        protocol.dimensions.error.row_count,
    );
    let cross_term_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let cross_term_error_row_commitments =
        commit_rows(setup, &cross_term_error_rows, &cross_term_error_blindings);

    append_relaxed_instance(
        transcript,
        RelaxedInstanceLabels {
            u: b"bf_committed_u",
            witness: b"bf_committed_w",
            error: b"bf_committed_e",
            eval: b"bf_committed_eval",
        },
        committed.u,
        &committed.witness_row_commitments,
        &committed.error_row_commitments,
        &committed.eval_commitments,
    );
    append_relaxed_instance(
        transcript,
        RelaxedInstanceLabels {
            u: b"bf_random_u",
            witness: b"bf_random_w",
            error: b"bf_random_e",
            eval: b"bf_random_eval",
        },
        random_u,
        &random_instance.witness_row_commitments,
        &random_instance.error_row_commitments,
        &random_instance.eval_commitments,
    );
    append_values(transcript, b"bf_cross_e", &cross_term_error_row_commitments);
    let folding_challenge = transcript.challenge();

    let folded_u = Fr::from_u64(1) + folding_challenge * random_u;
    let folded_witness_rows = fold_rows(witness.rows, &random_witness_rows, folding_challenge);
    let folded_witness_blindings = fold_scalars(
        witness.blindings,
        &random_witness_blindings,
        folding_challenge,
    );
    let folded_error_rows = fold_error_rows(
        &zero_rows(
            protocol.dimensions.error.row_count,
            protocol.dimensions.error.row_len,
        ),
        &cross_term_error_rows,
        &random_error_rows,
        folding_challenge,
    );
    let folded_error_blindings = fold_error_scalars(
        &vec![Fr::from_u64(0); protocol.dimensions.error.row_count],
        &cross_term_error_blindings,
        &random_error_blindings,
        folding_challenge,
    );
    let folded_eval_outputs = fold_scalars(
        witness.eval_outputs,
        &random_eval_outputs,
        folding_challenge,
    );
    let folded_eval_blindings = fold_scalars(
        witness.eval_blindings,
        &random_eval_blindings,
        folding_challenge,
    );
    let mut folded_eval_output_openings = Vec::new();
    let mut folded_eval_blinding_openings = Vec::new();
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            let (opening, opened) = open_witness_coordinate(
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
            );
            assert_eq!(opened, folded_eval_outputs[index]);
            folded_eval_output_openings.push(opening);
        }
        if let Some(coordinate) = coordinates.blinding {
            let (opening, opened) = open_witness_coordinate(
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
            );
            assert_eq!(opened, folded_eval_blindings[index]);
            folded_eval_blinding_openings.push(opening);
        }
    }
    for opening in &folded_eval_output_openings {
        append_vector_opening(
            transcript,
            b"bf_eval_out_open",
            b"bf_eval_out_blind",
            opening,
        );
    }
    for opening in &folded_eval_blinding_openings {
        append_vector_opening(
            transcript,
            b"bf_eval_blind_open",
            b"bf_eval_blind_bl",
            opening,
        );
    }

    transcript.append(&Label(b"bf_spartan"));
    let outer_num_vars =
        log2(protocol.dimensions.error.row_count) + log2(protocol.dimensions.error.row_len);
    let tau = transcript.challenge_vector(outer_num_vars);
    let outer_trace = prove_slow_sumcheck(
        outer_num_vars,
        3,
        Fr::from_u64(0),
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
        |point| {
            outer_function(
                &protocol.r1cs,
                folded_u,
                &flatten(&folded_witness_rows),
                &folded_error_rows,
                &tau,
                point,
            )
        },
    );

    let (az_rx, bz_rx, cz_rx) = abc_at_point(
        &protocol.r1cs,
        folded_u,
        &flatten(&folded_witness_rows),
        &outer_trace.point,
    );
    let (error_row_point, error_entry_point) = outer_trace
        .point
        .split_at(log2(protocol.dimensions.error.row_count));
    let (error_opening, _) = WrapperBlindFoldVc::open_committed_rows(
        &flatten(&folded_error_rows),
        &folded_error_blindings,
        protocol.dimensions.error.row_len,
        error_row_point,
        error_entry_point,
    )
    .expect("folded error rows open");

    append_values(transcript, b"bf_az_bz_cz", &[az_rx, bz_rx, cz_rx]);
    append_vector_opening(
        transcript,
        b"bf_error_opening",
        b"bf_error_blind",
        &error_opening,
    );

    let ra = transcript.challenge();
    let rb = transcript.challenge();
    let rc = transcript.challenge();
    let inner_num_vars =
        log2(protocol.dimensions.witness.row_count) + log2(protocol.dimensions.witness.row_len);
    let row_weights = EqPolynomial::<Fr>::evals(&outer_trace.point, None);
    let public = protocol
        .r1cs
        .public_column_contributions(&row_weights, 0, folded_u)
        .expect("public column contributions evaluate");
    let inner_claim = ra * (az_rx - public.a) + rb * (bz_rx - public.b) + rc * (cz_rx - public.c);
    let inner_trace = prove_slow_sumcheck(
        inner_num_vars,
        2,
        inner_claim,
        b"inner_sumcheck_poly",
        transcript,
        |point| {
            inner_function(
                &protocol.r1cs,
                &outer_trace.point,
                &folded_witness_rows,
                ra,
                rb,
                rc,
                point,
            )
        },
    );
    let (witness_row_point, witness_entry_point) = inner_trace
        .point
        .split_at(log2(protocol.dimensions.witness.row_count));
    let (witness_opening, _) = WrapperBlindFoldVc::open_committed_rows(
        &flatten(&folded_witness_rows),
        &folded_witness_blindings,
        protocol.dimensions.witness.row_len,
        witness_row_point,
        witness_entry_point,
    )
    .expect("folded witness rows open");

    jolt_blindfold::BlindFoldProof {
        auxiliary_row_commitments,
        random_round_commitments,
        random_output_claim_row_commitments,
        random_auxiliary_row_commitments,
        random_error_row_commitments,
        random_eval_commitments,
        random_u,
        cross_term_error_row_commitments,
        outer_sumcheck: outer_trace.proof,
        az_rx,
        bz_rx,
        cz_rx,
        inner_sumcheck: inner_trace.proof,
        witness_opening,
        error_opening,
        folded_eval_outputs,
        folded_eval_blindings,
        folded_eval_output_openings,
        folded_eval_blinding_openings,
    }
}

#[cfg(feature = "zk")]
fn rng_field(rng: &mut impl RngCore) -> Fr {
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
    Fr::from_bytes_array(&bytes)
}

#[cfg(feature = "zk")]
fn inverse(value: Fr) -> Fr {
    value.inverse().expect("test values are nonzero")
}

#[cfg(feature = "zk")]
fn eval_poly(coefficients: &[Fr], point: Fr) -> Fr {
    let mut result = Fr::from_u64(0);
    let mut power = Fr::from_u64(1);
    for coefficient in coefficients {
        result += *coefficient * power;
        power *= point;
    }
    result
}

#[cfg(feature = "zk")]
fn commit_rows(setup: &PedersenSetup<Bn254G1>, rows: &[Vec<Fr>], blindings: &[Fr]) -> Vec<Bn254G1> {
    rows.iter()
        .zip(blindings)
        .map(|(row, blinding)| WrapperBlindFoldVc::commit(setup, row, blinding))
        .collect()
}

#[cfg(feature = "zk")]
fn open_witness_coordinate(
    witness_rows: &[Vec<Fr>],
    witness_blindings: &[Fr],
    coordinate: jolt_blindfold::WitnessCoordinate,
) -> (VectorCommitmentOpening<Fr>, Fr) {
    let row_vars = log2(witness_rows.len());
    let entry_vars = log2(witness_rows[0].len());
    WrapperBlindFoldVc::open_committed_rows(
        &flatten(witness_rows),
        witness_blindings,
        witness_rows[0].len(),
        &boolean_point(coordinate.row, row_vars),
        &boolean_point(coordinate.column, entry_vars),
    )
    .expect("folded witness coordinate opens")
}

#[cfg(feature = "zk")]
fn boolean_point(index: usize, num_vars: usize) -> Vec<Fr> {
    (0..num_vars)
        .map(|bit| {
            let shift = num_vars - bit - 1;
            Fr::from_u64(((index >> shift) & 1) as u64)
        })
        .collect()
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
struct RelaxedInstanceLabels {
    u: &'static [u8],
    witness: &'static [u8],
    error: &'static [u8],
    eval: &'static [u8],
}

#[cfg(feature = "zk")]
fn append_relaxed_instance(
    transcript: &mut WrapperTranscript,
    labels: RelaxedInstanceLabels,
    u: Fr,
    witness_commitments: &[Bn254G1],
    error_commitments: &[Bn254G1],
    eval_commitments: &[Bn254G1],
) {
    transcript.append(&Label(labels.u));
    u.append_to_transcript(transcript);
    append_values(transcript, labels.witness, witness_commitments);
    append_values(transcript, labels.error, error_commitments);
    append_values(transcript, labels.eval, eval_commitments);
}

#[cfg(feature = "zk")]
fn append_values<A: AppendToTranscript>(
    transcript: &mut WrapperTranscript,
    label: &'static [u8],
    values: &[A],
) {
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}

#[cfg(feature = "zk")]
fn append_vector_opening(
    transcript: &mut WrapperTranscript,
    row_label: &'static [u8],
    blinding_label: &'static [u8],
    opening: &VectorCommitmentOpening<Fr>,
) {
    append_values(transcript, row_label, &opening.combined_vector);
    transcript.append(&Label(blinding_label));
    opening.combined_blinding.append_to_transcript(transcript);
}

#[cfg(feature = "zk")]
fn zero_rows(row_count: usize, row_len: usize) -> Vec<Vec<Fr>> {
    vec![vec![Fr::from_u64(0); row_len]; row_count]
}

#[cfg(feature = "zk")]
fn random_rows(row_count: usize, row_len: usize, rng: &mut impl RngCore) -> Vec<Vec<Fr>> {
    (0..row_count)
        .map(|_| (0..row_len).map(|_| rng_field(rng)).collect())
        .collect()
}

#[cfg(feature = "zk")]
fn fold_rows(real: &[Vec<Fr>], random: &[Vec<Fr>], challenge: Fr) -> Vec<Vec<Fr>> {
    real.iter()
        .zip(random)
        .map(|(real_row, random_row)| {
            real_row
                .iter()
                .zip(random_row)
                .map(|(&real, &random)| real + challenge * random)
                .collect()
        })
        .collect()
}

#[cfg(feature = "zk")]
fn fold_scalars(real: &[Fr], random: &[Fr], challenge: Fr) -> Vec<Fr> {
    real.iter()
        .zip(random)
        .map(|(&real, &random)| real + challenge * random)
        .collect()
}

#[cfg(feature = "zk")]
fn fold_error_rows(
    real: &[Vec<Fr>],
    cross: &[Vec<Fr>],
    random: &[Vec<Fr>],
    challenge: Fr,
) -> Vec<Vec<Fr>> {
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((real_row, cross_row), random_row)| {
            real_row
                .iter()
                .zip(cross_row)
                .zip(random_row)
                .map(|((&real, &cross), &random)| {
                    real + challenge * cross + challenge_squared * random
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "zk")]
fn fold_error_scalars(real: &[Fr], cross: &[Fr], random: &[Fr], challenge: Fr) -> Vec<Fr> {
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((&real, &cross), &random)| real + challenge * cross + challenge_squared * random)
        .collect()
}

#[cfg(feature = "zk")]
fn error_rows_for(
    r1cs: &ConstraintMatrices<Fr>,
    u: Fr,
    witness: &[Fr],
    row_len: usize,
    row_count: usize,
) -> Vec<Vec<Fr>> {
    let z = z_vector(u, witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &z) * dot(&r1cs.b[row_index], &z)
                - u * dot(&r1cs.c[row_index], &z)
        })
        .collect::<Vec<_>>();
    pad_to_row_count(&mut errors, row_len, row_count);
    errors.chunks(row_len).map(<[Fr]>::to_vec).collect()
}

#[cfg(feature = "zk")]
fn cross_term_error_rows_for(
    r1cs: &ConstraintMatrices<Fr>,
    real_u: Fr,
    real_witness: &[Fr],
    random_u: Fr,
    random_witness: &[Fr],
    row_len: usize,
    row_count: usize,
) -> Vec<Vec<Fr>> {
    let real_z = z_vector(real_u, real_witness);
    let random_z = z_vector(random_u, random_witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &real_z) * dot(&r1cs.b[row_index], &random_z)
                + dot(&r1cs.a[row_index], &random_z) * dot(&r1cs.b[row_index], &real_z)
                - real_u * dot(&r1cs.c[row_index], &random_z)
                - random_u * dot(&r1cs.c[row_index], &real_z)
        })
        .collect::<Vec<_>>();
    pad_to_row_count(&mut errors, row_len, row_count);
    errors.chunks(row_len).map(<[Fr]>::to_vec).collect()
}

#[cfg(feature = "zk")]
fn pad_to_row_count(values: &mut Vec<Fr>, row_len: usize, row_count: usize) {
    values.resize(row_len * row_count, Fr::from_u64(0));
}

#[cfg(feature = "zk")]
fn z_vector(u: Fr, witness: &[Fr]) -> Vec<Fr> {
    let mut z = Vec::with_capacity(witness.len() + 1);
    z.push(u);
    z.extend_from_slice(witness);
    z
}

#[cfg(feature = "zk")]
fn flatten(rows: &[Vec<Fr>]) -> Vec<Fr> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

#[cfg(feature = "zk")]
fn abc_at_point(
    r1cs: &ConstraintMatrices<Fr>,
    u: Fr,
    witness: &[Fr],
    point: &[Fr],
) -> (Fr, Fr, Fr) {
    let row_weights = EqPolynomial::<Fr>::evals(point, None);
    let z = z_vector(u, witness);
    let mut az = Fr::from_u64(0);
    let mut bz = Fr::from_u64(0);
    let mut cz = Fr::from_u64(0);
    for (row_index, &row_weight) in row_weights.iter().enumerate().take(r1cs.num_constraints) {
        az += row_weight * dot(&r1cs.a[row_index], &z);
        bz += row_weight * dot(&r1cs.b[row_index], &z);
        cz += row_weight * dot(&r1cs.c[row_index], &z);
    }
    (az, bz, cz)
}

#[cfg(feature = "zk")]
fn outer_function(
    r1cs: &ConstraintMatrices<Fr>,
    u: Fr,
    witness: &[Fr],
    error_rows: &[Vec<Fr>],
    tau: &[Fr],
    point: &[Fr],
) -> Fr {
    let (az, bz, cz) = abc_at_point(r1cs, u, witness, point);
    let error = mle_eval(&flatten(error_rows), point);
    EqPolynomial::<Fr>::mle(tau, point) * (az * bz - u * cz - error)
}

#[cfg(feature = "zk")]
fn inner_function(
    r1cs: &ConstraintMatrices<Fr>,
    outer_point: &[Fr],
    witness_rows: &[Vec<Fr>],
    ra: Fr,
    rb: Fr,
    rc: Fr,
    point: &[Fr],
) -> Fr {
    let row_weights = EqPolynomial::<Fr>::evals(outer_point, None);
    let column_weights = EqPolynomial::<Fr>::evals(point, None);
    let l_w = r1cs
        .linear_form_bilinear_eval(
            &row_weights,
            &column_weights,
            1,
            column_weights.len(),
            [ra, rb, rc],
        )
        .expect("inner linear form dimensions match");
    l_w * mle_eval(&flatten(witness_rows), point)
}

#[cfg(feature = "zk")]
fn mle_eval(values: &[Fr], point: &[Fr]) -> Fr {
    EqPolynomial::<Fr>::evals(point, None)
        .iter()
        .zip(values)
        .map(|(&weight, &value)| weight * value)
        .sum()
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
struct SumcheckTrace {
    proof: CompressedSumcheckProof<Fr>,
    point: Vec<Fr>,
}

#[cfg(feature = "zk")]
fn prove_slow_sumcheck(
    num_vars: usize,
    degree: usize,
    claim: Fr,
    label: &'static [u8],
    transcript: &mut WrapperTranscript,
    eval: impl Fn(&[Fr]) -> Fr,
) -> SumcheckTrace {
    let mut running_sum = claim;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut rounds = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        let remaining = num_vars - round - 1;
        let values = (0..=degree)
            .map(|point| {
                let mut sum = Fr::from_u64(0);
                for suffix in 0..(1usize << remaining) {
                    let mut evaluation_point = prefix.clone();
                    evaluation_point.push(Fr::from_u64(point as u64));
                    for bit in 0..remaining {
                        evaluation_point.push(Fr::from_u64(((suffix >> bit) & 1) as u64));
                    }
                    sum += eval(&evaluation_point);
                }
                sum
            })
            .collect::<Vec<_>>();
        let coefficients = interpolate_zero_to_degree(&values);
        let round_sum = coefficients[0] + coefficients.iter().copied().sum::<Fr>();
        assert_eq!(round_sum, running_sum);
        let mut compressed = Vec::with_capacity(degree);
        compressed.push(coefficients[0]);
        compressed.extend_from_slice(&coefficients[2..]);
        append_values(transcript, label, &compressed);
        let challenge = transcript.challenge();
        running_sum = eval_poly(&coefficients, challenge);
        prefix.push(challenge);
        rounds.push(CompressedPoly::new(compressed));
    }

    SumcheckTrace {
        proof: CompressedSumcheckProof {
            round_polynomials: rounds,
        },
        point: prefix,
    }
}

#[cfg(feature = "zk")]
fn interpolate_zero_to_degree(values: &[Fr]) -> Vec<Fr> {
    let degree = values.len() - 1;
    let mut result = vec![Fr::from_u64(0); degree + 1];
    for (j, &value) in values.iter().enumerate() {
        let x_j = Fr::from_u64(j as u64);
        let mut basis = vec![Fr::from_u64(1)];
        let mut denominator = Fr::from_u64(1);
        for m in 0..=degree {
            if m == j {
                continue;
            }
            let x_m = Fr::from_u64(m as u64);
            basis = multiply_by_linear(&basis, -x_m, Fr::from_u64(1));
            denominator *= x_j - x_m;
        }
        let scale = value * inverse(denominator);
        for (coefficient, basis_coefficient) in result.iter_mut().zip(basis) {
            *coefficient += scale * basis_coefficient;
        }
    }
    result
}

#[cfg(feature = "zk")]
fn multiply_by_linear(poly: &[Fr], constant: Fr, linear: Fr) -> Vec<Fr> {
    let mut result = vec![Fr::from_u64(0); poly.len() + 1];
    for (index, &coefficient) in poly.iter().enumerate() {
        result[index] += coefficient * constant;
        result[index + 1] += coefficient * linear;
    }
    result
}

#[cfg(feature = "zk")]
fn log2(value: usize) -> usize {
    assert!(value.is_power_of_two());
    value.trailing_zeros() as usize
}

struct ProvedSpartan {
    proof: SpartanProof<Fr>,
    witness_point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Fr>,
    witness_opening_claim: Fr,
}

fn prove_spartan(
    relation: &ConstraintMatrices<Fr>,
    witness_poly: &Polynomial<Fr>,
    relation_output: &r1cs_relation::R1csRelationOutput<'_, Fr>,
    transcript: &mut WrapperTranscript,
) -> ProvedSpartan {
    let dimensions = relation_output.statement_facts.spartan;
    let row_polys = row_value_polys(
        relation,
        witness_poly.evaluations(),
        dimensions.num_constraints_padded(),
    );
    transcript.append(&Label(WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL));
    let tau = transcript.challenge_vector(dimensions.num_constraint_rounds());

    transcript.append(&Label(WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL));
    let eq_tau = EqPolynomial::new(tau.as_slice().to_vec()).evaluations();
    let (outer_sumcheck, outer_reduction) = prove_sum_of_products_sumcheck(
        vec![
            SumcheckProductTerm::new(
                Fr::from_u64(1),
                vec![
                    eq_tau.clone(),
                    row_polys.a.evaluations().to_vec(),
                    row_polys.b.evaluations().to_vec(),
                ],
            ),
            SumcheckProductTerm::new(
                -Fr::from_u64(1),
                vec![eq_tau, row_polys.c.evaluations().to_vec()],
            ),
        ],
        WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
        transcript,
    );
    let outer_claims = row_polys.evaluate(&outer_reduction.point);

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL));
    transcript.append(&outer_claims.a);
    transcript.append(&outer_claims.b);
    transcript.append(&outer_claims.c);
    let batching = SpartanInnerBatchingCoefficients {
        a: transcript.challenge_scalar(),
        b: transcript.challenge_scalar(),
        c: transcript.challenge_scalar(),
    };
    let combined_matrix_poly = combined_matrix_column_poly(
        relation,
        &outer_reduction.point,
        batching,
        dimensions.num_vars_padded(),
    );

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL));
    let (inner_sumcheck, inner_reduction) = prove_sum_of_products_sumcheck(
        vec![SumcheckProductTerm::new(
            Fr::from_u64(1),
            vec![
                combined_matrix_poly.evaluations().to_vec(),
                witness_poly.evaluations().to_vec(),
            ],
        )],
        WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
        transcript,
    );
    let witness_opening_claim = witness_poly.evaluate(&inner_reduction.point);

    ProvedSpartan {
        proof: SpartanProof::new(
            outer_sumcheck,
            outer_claims,
            inner_sumcheck,
            witness_opening_claim,
        ),
        witness_point: inner_reduction.point,
        witness_opening_claim,
    }
}

struct SumcheckProductTerm {
    coefficient: Fr,
    factors: Vec<Vec<Fr>>,
}

impl SumcheckProductTerm {
    fn new(coefficient: Fr, factors: Vec<Vec<Fr>>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }
}

fn prove_sum_of_products_sumcheck(
    mut terms: Vec<SumcheckProductTerm>,
    degree: usize,
    transcript: &mut WrapperTranscript,
) -> (
    CompressedSumcheckProof<Fr>,
    jolt_sumcheck::EvaluationClaim<Fr>,
) {
    let table_len = terms[0].factors[0].len();
    assert!(
        table_len.is_power_of_two(),
        "product tables must be over a Boolean hypercube"
    );
    for term in &terms {
        assert!(
            term.factors.len() <= degree,
            "term arity must fit the sumcheck degree"
        );
        for factor in &term.factors {
            assert_eq!(factor.len(), table_len, "product tables must align");
        }
    }

    let num_vars = table_len.trailing_zeros() as usize;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);
    for _round in 0..num_vars {
        let round_evals = sum_of_products_round_evals(&terms, degree);
        let round_poly = UnivariatePoly::from_evals(&round_evals);
        let compressed =
            CompressedLabeledRoundPoly::new(&round_poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, Fr> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        let challenge = transcript.challenge();
        bind_sumcheck_terms(&mut terms, challenge);
        prefix.push(challenge);
        round_polynomials.push(round_poly.compress());
    }
    let value = final_sum_of_products_claim(&terms);
    (
        CompressedSumcheckProof { round_polynomials },
        jolt_sumcheck::EvaluationClaim::new(prefix, value),
    )
}

fn sum_of_products_round_evals(terms: &[SumcheckProductTerm], degree: usize) -> Vec<Fr> {
    let half = terms[0].factors[0].len() / 2;
    let sums = if half < 256 {
        (0..half).fold(vec![Fr::from_u64(0); degree + 1], |mut sums, index| {
            add_sum_of_products_round_terms(&mut sums, terms, index);
            sums
        })
    } else {
        (0..half)
            .into_par_iter()
            .map(|index| {
                let mut sums = vec![Fr::from_u64(0); degree + 1];
                add_sum_of_products_round_terms(&mut sums, terms, index);
                sums
            })
            .reduce(
                || vec![Fr::from_u64(0); degree + 1],
                |mut lhs, rhs| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                        *lhs += rhs;
                    }
                    lhs
                },
            )
    };
    sums
}

fn add_sum_of_products_round_terms(sums: &mut [Fr], terms: &[SumcheckProductTerm], index: usize) {
    let half = terms[0].factors[0].len() / 2;
    for (x, sum) in sums.iter_mut().enumerate() {
        let x = Fr::from_u64(x as u64);
        for term in terms {
            let mut product = term.coefficient;
            for factor in &term.factors {
                let lo = factor[index];
                let hi = factor[index + half];
                product *= lo + x * (hi - lo);
            }
            *sum += product;
        }
    }
}

fn bind_sumcheck_terms(terms: &mut [SumcheckProductTerm], challenge: Fr) {
    for term in terms {
        for factor in &mut term.factors {
            bind_table(factor, challenge);
        }
    }
}

fn bind_table(table: &mut Vec<Fr>, challenge: Fr) {
    let half = table.len() / 2;
    if half < 256 {
        for index in 0..half {
            let lo = table[index];
            let hi = table[index + half];
            table[index] = lo + challenge * (hi - lo);
        }
    } else {
        let (lo, hi) = table.split_at_mut(half);
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(lo, hi)| *lo += challenge * (*hi - *lo));
    }
    table.truncate(half);
}

fn final_sum_of_products_claim(terms: &[SumcheckProductTerm]) -> Fr {
    terms.iter().fold(Fr::from_u64(0), |acc, term| {
        acc + term
            .factors
            .iter()
            .fold(term.coefficient, |product, factor| product * factor[0])
    })
}

struct RowValuePolys {
    a: Polynomial<Fr>,
    b: Polynomial<Fr>,
    c: Polynomial<Fr>,
}

impl RowValuePolys {
    fn evaluate(&self, point: &[Fr]) -> SpartanOuterEvaluationClaims<Fr> {
        SpartanOuterEvaluationClaims::new(
            self.a.evaluate(point),
            self.b.evaluate(point),
            self.c.evaluate(point),
        )
    }
}

fn row_value_polys(
    relation: &ConstraintMatrices<Fr>,
    witness: &[Fr],
    padded_rows: usize,
) -> RowValuePolys {
    let mut a = vec![Fr::from_u64(0); padded_rows];
    let mut b = vec![Fr::from_u64(0); padded_rows];
    let mut c = vec![Fr::from_u64(0); padded_rows];
    for row in 0..relation.num_constraints {
        a[row] = dot(&relation.a[row], witness);
        b[row] = dot(&relation.b[row], witness);
        c[row] = dot(&relation.c[row], witness);
    }

    RowValuePolys {
        a: Polynomial::new(a),
        b: Polynomial::new(b),
        c: Polynomial::new(c),
    }
}

fn combined_matrix_column_poly(
    relation: &ConstraintMatrices<Fr>,
    row_point: &[Fr],
    batching: SpartanInnerBatchingCoefficients<Fr>,
    padded_columns: usize,
) -> Polynomial<Fr> {
    let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
    let mut columns = vec![Fr::from_u64(0); padded_columns];
    for (row, &row_weight) in row_weights
        .iter()
        .enumerate()
        .take(relation.num_constraints)
    {
        add_weighted_row(&mut columns, &relation.a[row], batching.a * row_weight);
        add_weighted_row(&mut columns, &relation.b[row], batching.b * row_weight);
        add_weighted_row(&mut columns, &relation.c[row], batching.c * row_weight);
    }
    Polynomial::new(columns)
}

fn add_weighted_row(columns: &mut [Fr], row: &[(usize, Fr)], row_weight: Fr) {
    for &(column, coefficient) in row {
        columns[column] += row_weight * coefficient;
    }
}

fn dot(row: &[(usize, Fr)], witness: &[Fr]) -> Fr {
    row.iter()
        .map(|&(variable, coefficient)| coefficient * witness[variable])
        .sum()
}

fn witness_polynomial(protocol: &WrapperR1csProtocol<Fr>) -> Polynomial<Fr> {
    let padded = protocol.r1cs.num_vars.max(1).next_power_of_two();
    let mut witness = protocol.witness.clone();
    witness.resize(padded, Fr::from_u64(0));
    Polynomial::new(witness)
}

fn relation_dimensions(protocol: &WrapperR1csProtocol<Fr>) -> WrapperRelationDimensions {
    WrapperRelationDimensions::new(
        protocol.r1cs.num_vars,
        protocol.r1cs.num_constraints,
        protocol.public_inputs.len(),
    )
}

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Pairing>, HyperKZGVerifierSetup<Pairing>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0047_5354_4152);
    let prover = KzgPCS::setup(
        &mut rng,
        max_degree,
        Pairing::g1_generator(),
        Pairing::g2_generator(),
    );
    let verifier = KzgPCS::verifier_setup(&prover);
    (prover, verifier)
}

#[cfg(feature = "zk")]
fn make_zk_setup(
    max_degree: usize,
) -> (
    HyperKZGProverSetup<Pairing>,
    HyperKZGVerifierSetup<Pairing>,
    PedersenSetup<Bn254G1>,
) {
    make_zk_setup_with_hiding_scalars(max_degree, Fr::from_u64(17), Fr::from_u64(17))
}

#[cfg(feature = "zk")]
fn make_zk_setup_with_hiding_scalars(
    max_degree: usize,
    pcs_hiding_scalar: Fr,
    vc_hiding_scalar: Fr,
) -> (
    HyperKZGProverSetup<Pairing>,
    HyperKZGVerifierSetup<Pairing>,
    PedersenSetup<Bn254G1>,
) {
    let g1 = Pairing::g1_generator();
    let pcs_hiding_g1 = g1.scalar_mul(&pcs_hiding_scalar);
    let prover = KzgPCS::setup_zk_from_secret(
        Fr::from_u64(0x5eed),
        max_degree,
        g1,
        pcs_hiding_g1,
        Pairing::g2_generator(),
    );
    let verifier = KzgPCS::verifier_setup(&prover);
    let vc_hiding_g1 = g1.scalar_mul(&vc_hiding_scalar);
    let vc_setup = PedersenSetup::new(
        (0..4)
            .map(|index| g1.scalar_mul(&Fr::from_u64(index as u64 + 1)))
            .collect(),
        vc_hiding_g1,
    );
    (prover, verifier, vc_setup)
}

fn fr_to_fq(value: Fr) -> Fq {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}
