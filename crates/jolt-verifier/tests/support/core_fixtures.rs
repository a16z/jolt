#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when core proof construction or serialization breaks"
)]

use std::sync::Mutex;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
#[cfg(not(feature = "zk"))]
use jolt_dory::DoryCommitment;
use jolt_dory::{DoryScheme, DoryVerifierSetup};
use jolt_field::Fr;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{
    compat::convert::ImportedCoreProof, verify, JoltVerifierPreprocessing, VerifierError,
};

#[cfg(not(feature = "zk"))]
use jolt_verifier::compat::claims::LegacyOpeningClaims;
#[cfg(feature = "zk")]
use jolt_verifier::compat::convert::CoreCurveBridge;
#[cfg(not(feature = "zk"))]
use jolt_verifier::compat::convert::CorePcsBridge;
#[cfg(feature = "zk")]
use jolt_verifier::compat::convert::LegacyBlindFoldProof;

use jolt_core::{
    curve::Bn254Curve,
    host,
    poly::commitment::{
        commitment_scheme::CommitmentScheme as CoreCommitmentScheme, dory::DoryCommitmentScheme,
    },
    zkvm::{
        prover::JoltProverPreprocessing,
        verifier::{
            JoltSharedPreprocessing, JoltVerifierPreprocessing as CoreVerifierPreprocessing,
        },
        RV64IMACProof, RV64IMACProver, RV64IMACVerifier, Serializable,
    },
};

#[cfg(not(feature = "zk"))]
use jolt_core::{
    poly::{
        commitment::dory::{DoryContext, DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
    },
    zkvm::ram::populate_memory_states,
};

static CORE_FIXTURE_LOCK: Mutex<()> = Mutex::new(());

type CoreField = jolt_core::ark_bn254::Fr;
type CoreProof = RV64IMACProof;
type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreOpeningHint = <DoryCommitmentScheme as CoreCommitmentScheme>::OpeningProofHint;
type ConvertedProof = ImportedCoreProof<CoreField, Bn254Curve, DoryCommitmentScheme>;
type ConvertedPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type TrustedAdviceCommitter = fn(
    &JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    &[u8],
) -> (CoreCommitment, CoreOpeningHint);

#[cfg(not(feature = "zk"))]
pub struct CoreVerifierCase {
    pub preprocessing: ConvertedPreprocessing,
    pub public_io: JoltDevice,
    pub proof: ConvertedProof,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

#[cfg(not(feature = "zk"))]
impl CoreVerifierCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, LegacyOpeningClaims<Fr>, ()>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            false,
        )
    }
}

#[cfg(feature = "zk")]
pub struct CoreZkVerifierCase {
    pub preprocessing: ConvertedPreprocessing,
    pub public_io: JoltDevice,
    pub proof: ConvertedProof,
}

#[cfg(feature = "zk")]
impl CoreZkVerifierCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            Blake2bTranscript,
            (),
            LegacyBlindFoldProof<CoreField, Bn254Curve>,
        >(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            None,
            true,
        )
    }
}

#[cfg(not(feature = "zk"))]
pub fn standard_muldiv_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(generate_muldiv())
}

#[cfg(feature = "zk")]
pub fn zk_muldiv_case() -> CoreZkVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    zk_case_from_accepted_fixture(generate_muldiv())
}

#[cfg(not(feature = "zk"))]
pub fn standard_advice_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(generate_unused_advice_commitments())
}

#[cfg(not(feature = "zk"))]
pub fn public_io_memory_layout_mismatch_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    let fixture = generate_muldiv();
    let mut public_io = fixture.public_io.clone();
    public_io.memory_layout.heap_size += 1;
    assert_core_rejects(&fixture, fixture.proof.clone_via_bytes(), public_io.clone());
    case_from_parts(fixture, public_io)
}

#[cfg(not(feature = "zk"))]
pub fn invalid_trace_length_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    let mut fixture = generate_muldiv();
    fixture.proof.trace_length = 3;
    assert_core_rejects(
        &fixture,
        fixture.proof.clone_via_bytes(),
        fixture.public_io.clone(),
    );
    let public_io = fixture.public_io.clone();
    case_from_parts(fixture, public_io)
}

#[cfg(not(feature = "zk"))]
pub fn invalid_ram_k_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    let mut fixture = generate_muldiv();
    fixture.proof.ram_K = 3;
    assert_core_rejects(
        &fixture,
        fixture.proof.clone_via_bytes(),
        fixture.public_io.clone(),
    );
    let public_io = fixture.public_io.clone();
    case_from_parts(fixture, public_io)
}

#[cfg(not(feature = "zk"))]
fn case_from_accepted_fixture(fixture: GeneratedCoreFixture) -> CoreVerifierCase {
    assert_core_accepts(
        &fixture,
        fixture.proof.clone_via_bytes(),
        fixture.public_io.clone(),
    );
    let public_io = fixture.public_io.clone();
    case_from_parts(fixture, public_io)
}

#[cfg(feature = "zk")]
fn zk_case_from_accepted_fixture(fixture: GeneratedCoreFixture) -> CoreZkVerifierCase {
    assert_core_accepts(
        &fixture,
        fixture.proof.clone_via_bytes(),
        fixture.public_io.clone(),
    );
    CoreZkVerifierCase {
        preprocessing: convert_preprocessing(&fixture.core_preprocessing),
        public_io: fixture.public_io,
        proof: fixture.proof.into(),
    }
}

#[cfg(not(feature = "zk"))]
fn case_from_parts(fixture: GeneratedCoreFixture, public_io: JoltDevice) -> CoreVerifierCase {
    CoreVerifierCase {
        preprocessing: convert_preprocessing(&fixture.core_preprocessing),
        public_io,
        proof: fixture.proof.into(),
        trusted_advice_commitment: fixture
            .trusted_advice_commitment
            .map(<DoryCommitmentScheme as CorePcsBridge<CoreField>>::commitment_into_verifier),
    }
}

struct GeneratedCoreFixture {
    core_preprocessing: CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    public_io: JoltDevice,
    proof: CoreProof,
    trusted_advice_commitment: Option<CoreCommitment>,
}

trait CloneCoreProofViaBytes {
    fn clone_via_bytes(&self) -> CoreProof;
}

impl CloneCoreProofViaBytes for CoreProof {
    fn clone_via_bytes(&self) -> CoreProof {
        let proof_bytes = self.serialize_to_bytes().expect("serialize proof");
        CoreProof::deserialize_from_bytes(&proof_bytes).expect("deserialize proof")
    }
}

fn assert_core_accepts(fixture: &GeneratedCoreFixture, proof: CoreProof, public_io: JoltDevice) {
    let result = RV64IMACVerifier::new(
        &fixture.core_preprocessing,
        proof,
        public_io,
        fixture.trusted_advice_commitment,
        None,
    )
    .and_then(RV64IMACVerifier::verify);
    assert!(
        result.is_ok(),
        "core verifier should accept generated fixture proof: {result:?}",
    );
}

#[cfg(not(feature = "zk"))]
fn assert_core_rejects(fixture: &GeneratedCoreFixture, proof: CoreProof, public_io: JoltDevice) {
    let result = RV64IMACVerifier::new(
        &fixture.core_preprocessing,
        proof,
        public_io,
        fixture.trusted_advice_commitment,
        None,
    )
    .and_then(RV64IMACVerifier::verify);
    assert!(result.is_err(), "core verifier accepted tampered fixture");
}

fn generate_muldiv() -> GeneratedCoreFixture {
    let program = host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    generate_core_fixture(program, inputs, Vec::new(), Vec::new(), None)
}

#[cfg(not(feature = "zk"))]
fn generate_unused_advice_commitments() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("fibonacci-guest"),
        postcard::to_stdvec(&5u32).expect("serialize fibonacci input"),
        vec![9u8; 64],
        vec![7u8; 64],
        Some(commit_trusted_advice_preprocessing_only),
    )
}

fn generate_core_fixture(
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    trusted_advice_committer: Option<TrustedAdviceCommitter>,
) -> GeneratedCoreFixture {
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, _, _, public_io) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        public_io.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        entry_address,
    )
    .expect("preprocess core fixture");
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .expect("elf contents should exist");

    let (trusted_advice_commitment, trusted_advice_hint) = trusted_advice_committer
        .map(|commit| commit(&prover_preprocessing, &trusted_advice))
        .map_or((None, None), |(commitment, hint)| {
            (Some(commitment), Some(hint))
        });

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        trusted_advice_commitment,
        trusted_advice_hint,
        None,
    );
    let public_io = prover.program_io.clone();
    let (proof, _) = prover.prove();
    let core_preprocessing = CoreVerifierPreprocessing::from(&prover_preprocessing);

    GeneratedCoreFixture {
        core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
    }
}

#[cfg(not(feature = "zk"))]
fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (CoreCommitment, CoreOpeningHint) {
    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
    populate_memory_states(
        0,
        trusted_advice_bytes,
        Some(&mut trusted_advice_words),
        None,
    );

    let poly = MultilinearPolynomial::<CoreField>::from(trusted_advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
    let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
    DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
}

fn convert_preprocessing(
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> ConvertedPreprocessing {
    JoltVerifierPreprocessing::new(
        JoltProgramPreprocessing {
            bytecode: preprocessing.shared.bytecode.as_ref().clone(),
            ram: preprocessing.shared.ram.clone(),
            memory_layout: preprocessing.shared.memory_layout.clone(),
            max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
        },
        preprocessing.shared.digest(),
        DoryVerifierSetup(preprocessing.generators.clone()),
        convert_vc_setup(preprocessing),
    )
}

#[cfg(not(feature = "zk"))]
fn convert_vc_setup(
    _preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> Option<PedersenSetup<Bn254G1>> {
    None
}

#[cfg(feature = "zk")]
fn convert_vc_setup(
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> Option<PedersenSetup<Bn254G1>> {
    let setup = &preprocessing
        .blindfold_setup
        .as_ref()
        .expect("ZK core preprocessing must carry BlindFold setup")
        .0;
    Some(PedersenSetup::new(
        setup
            .message_generators
            .iter()
            .copied()
            .map(<Bn254Curve as CoreCurveBridge<CoreField>>::g1_into_verifier)
            .collect(),
        <Bn254Curve as CoreCurveBridge<CoreField>>::g1_into_verifier(setup.blinding_generator),
    ))
}
