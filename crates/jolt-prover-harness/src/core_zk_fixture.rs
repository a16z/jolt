use std::sync::Mutex;

use common::jolt_device::JoltDevice;
use jolt_core::{
    curve::Bn254Curve,
    host,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme as CoreCommitmentScheme,
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals},
        },
        multilinear_polynomial::MultilinearPolynomial,
    },
    zkvm::{
        prover::JoltProverPreprocessing,
        ram::populate_memory_states,
        verifier::{
            JoltSharedPreprocessing, JoltVerifierPreprocessing as CoreVerifierPreprocessing,
        },
        RV64IMACProof, RV64IMACProver, RV64IMACVerifier, Serializable,
    },
};
use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
use jolt_dory::{DoryScheme, DoryVerifierSetup};
use jolt_field::Fr;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{
    compat::convert::{CoreCurveBridge, CorePcsBridge, ImportedCoreProof},
    verify, JoltVerifierPreprocessing, VerifierError,
};

use crate::{FeatureMode, FixtureKind, FixtureRequest, HarnessError, HarnessResult};

static CORE_ZK_FIXTURE_LOCK: Mutex<()> = Mutex::new(());

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

#[derive(Clone)]
pub struct CoreZkVerifierFixture {
    pub preprocessing: ConvertedPreprocessing,
    pub public_io: JoltDevice,
    pub proof: ConvertedProof,
    pub trusted_advice_commitment: Option<jolt_dory::DoryCommitment>,
}

impl CoreZkVerifierFixture {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            true,
        )
    }
}

pub fn load_zk_core_verifier_fixture(
    request: &FixtureRequest,
) -> HarnessResult<CoreZkVerifierFixture> {
    if request.feature_mode != FeatureMode::Zk {
        return Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "ZK core verifier fixtures require ZK feature mode",
        });
    }

    let _guard = CORE_ZK_FIXTURE_LOCK
        .lock()
        .map_err(|error| core_zk_fixture_error(request, "lock core ZK fixture generator", error))?;
    let generated = generate_fixture(request)?;
    assert_core_accepts(request, &generated)?;
    let fixture = convert_fixture(request, generated)?;
    fixture.verify().map_err(|error| {
        core_zk_fixture_error(request, "verify converted ZK core fixture", error)
    })?;
    Ok(fixture)
}

fn generate_fixture(request: &FixtureRequest) -> HarnessResult<GeneratedCoreZkFixture> {
    match request.kind {
        FixtureKind::ZkMuldivSmall => generate_muldiv(request),
        FixtureKind::ZkAdviceConsumer => generate_advice_consumer(request),
        _ => Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "no ZK core verifier fixture is registered for this kind",
        }),
    }
}

fn generate_muldiv(request: &FixtureRequest) -> HarnessResult<GeneratedCoreZkFixture> {
    let inputs = postcard::to_stdvec(&[9_u32, 5, 3])
        .map_err(|error| core_zk_fixture_error(request, "serialize muldiv inputs", error))?;
    generate_core_fixture(
        request,
        host::Program::new("muldiv-guest"),
        inputs,
        Vec::new(),
        Vec::new(),
        None,
    )
}

fn generate_advice_consumer(request: &FixtureRequest) -> HarnessResult<GeneratedCoreZkFixture> {
    let inputs = postcard::to_stdvec(&12_u64)
        .map_err(|error| core_zk_fixture_error(request, "serialize advice public input", error))?;
    let untrusted_advice = postcard::to_stdvec(&5_u64)
        .map_err(|error| core_zk_fixture_error(request, "serialize untrusted advice", error))?;
    let trusted_advice = postcard::to_stdvec(&7_u64)
        .map_err(|error| core_zk_fixture_error(request, "serialize trusted advice", error))?;

    generate_core_fixture(
        request,
        host::Program::new("advice-consumer-guest"),
        inputs,
        untrusted_advice,
        trusted_advice,
        Some(commit_trusted_advice_preprocessing_only),
    )
}

fn generate_core_fixture(
    request: &FixtureRequest,
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    trusted_advice_committer: Option<TrustedAdviceCommitter>,
) -> HarnessResult<GeneratedCoreZkFixture> {
    let jolt_program = program
        .jolt_program()
        .map_err(|error| core_zk_fixture_error(request, "build Jolt program", error))?;
    let mut tracer_backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(
            &mut tracer_backend,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        )
        .map_err(|error| core_zk_fixture_error(request, "trace Jolt program", error))?;

    let shared_preprocessing = JoltSharedPreprocessing::new(
        jolt_program.expanded_bytecode,
        trace.device.memory_layout.clone(),
        jolt_program.memory_init,
        1 << 16,
        jolt_program.entry_address,
    )
    .map_err(|error| core_zk_fixture_error(request, "preprocess core ZK fixture", error))?;
    let prover_preprocessing: JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| core_zk_fixture_error(request, "read guest ELF", "missing ELF contents"))?;

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

    Ok(GeneratedCoreZkFixture {
        core_preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
    })
}

fn assert_core_accepts(
    request: &FixtureRequest,
    fixture: &GeneratedCoreZkFixture,
) -> HarnessResult<()> {
    let proof = clone_core_proof(request, &fixture.proof)?;
    RV64IMACVerifier::new(
        &fixture.core_preprocessing,
        proof,
        fixture.public_io.clone(),
        fixture.trusted_advice_commitment,
        None,
    )
    .and_then(RV64IMACVerifier::verify)
    .map_err(|error| core_zk_fixture_error(request, "verify generated core ZK fixture", error))
}

fn clone_core_proof(request: &FixtureRequest, proof: &CoreProof) -> HarnessResult<CoreProof> {
    let bytes = proof
        .serialize_to_bytes()
        .map_err(|error| core_zk_fixture_error(request, "serialize core ZK proof", error))?;
    CoreProof::deserialize_from_bytes(&bytes)
        .map_err(|error| core_zk_fixture_error(request, "deserialize core ZK proof", error))
}

fn convert_fixture(
    request: &FixtureRequest,
    fixture: GeneratedCoreZkFixture,
) -> HarnessResult<CoreZkVerifierFixture> {
    Ok(CoreZkVerifierFixture {
        preprocessing: convert_preprocessing(request, &fixture.core_preprocessing)?,
        public_io: fixture.public_io,
        proof: fixture
            .proof
            .try_into()
            .map_err(|error| core_zk_fixture_error(request, "convert ZK core proof", error))?,
        trusted_advice_commitment: fixture
            .trusted_advice_commitment
            .as_ref()
            .copied()
            .map(<DoryCommitmentScheme as CorePcsBridge<CoreField>>::commitment_into_verifier),
    })
}

fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (CoreCommitment, CoreOpeningHint) {
    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0_u64; (max_trusted_advice_size as usize) / 8];
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
    request: &FixtureRequest,
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> HarnessResult<ConvertedPreprocessing> {
    Ok(JoltVerifierPreprocessing::new(
        JoltProgramPreprocessing {
            bytecode: preprocessing.shared.bytecode.as_ref().clone(),
            ram: preprocessing.shared.ram.clone(),
            memory_layout: preprocessing.shared.memory_layout.clone(),
            max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
        },
        preprocessing.shared.digest(),
        DoryVerifierSetup(preprocessing.generators.clone()),
        Some(convert_vc_setup(request, preprocessing)?),
    ))
}

fn convert_vc_setup(
    request: &FixtureRequest,
    preprocessing: &CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
) -> HarnessResult<PedersenSetup<Bn254G1>> {
    let setup = &preprocessing
        .blindfold_setup
        .as_ref()
        .ok_or_else(|| {
            core_zk_fixture_error(
                request,
                "convert ZK vector commitment setup",
                "missing core BlindFold setup",
            )
        })?
        .0;
    Ok(PedersenSetup::new(
        setup
            .message_generators
            .iter()
            .copied()
            .map(<Bn254Curve as CoreCurveBridge<CoreField>>::g1_into_verifier)
            .collect(),
        <Bn254Curve as CoreCurveBridge<CoreField>>::g1_into_verifier(setup.blinding_generator),
    ))
}

struct GeneratedCoreZkFixture {
    core_preprocessing: CoreVerifierPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme>,
    public_io: JoltDevice,
    proof: CoreProof,
    trusted_advice_commitment: Option<CoreCommitment>,
}

fn core_zk_fixture_error(
    request: &FixtureRequest,
    context: &'static str,
    error: impl ToString,
) -> HarnessError {
    HarnessError::CoreFixture {
        fixture: request.kind,
        context,
        reason: error.to_string(),
    }
}
