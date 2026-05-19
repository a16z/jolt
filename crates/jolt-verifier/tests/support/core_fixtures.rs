#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "fixture generation should fail loudly when core proof construction or serialization breaks"
)]

use std::{
    env, fs,
    io::{Cursor, Read},
    path::PathBuf,
    sync::Mutex,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
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
        commitment::dory::ArkGT,
        commitment::dory::{DoryContext, DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
    },
    zkvm::ram::populate_memory_states,
};

static CORE_FIXTURE_LOCK: Mutex<()> = Mutex::new(());
const ARTIFACT_MAGIC: &[u8; 8] = b"JVCF0001";
const REGENERATE_ARTIFACTS_ENV: &str = "JOLT_VERIFIER_REGENERATE_CORE_FIXTURES";

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
#[derive(Clone)]
pub struct CoreVerifierCase {
    pub preprocessing: ConvertedPreprocessing,
    pub public_io: JoltDevice,
    pub proof: ConvertedProof,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

#[cfg(not(feature = "zk"))]
impl CoreVerifierCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, ()>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            false,
        )
    }
}

#[cfg(feature = "zk")]
#[derive(Clone)]
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
    case_from_accepted_fixture(CoreFixtureKind::MulDivSmall, generate_muldiv)
}

#[cfg(not(feature = "zk"))]
pub fn standard_fibonacci_small_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::FibonacciSmall, generate_fibonacci_small)
}

#[cfg(not(feature = "zk"))]
pub fn standard_fibonacci_medium_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::FibonacciMedium, generate_fibonacci_medium)
}

#[cfg(not(feature = "zk"))]
pub fn standard_memory_ops_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::MemoryOps, generate_memory_ops)
}

#[cfg(not(feature = "zk"))]
pub fn standard_collatz_small_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::CollatzSmall, generate_collatz_small)
}

#[cfg(not(feature = "zk"))]
pub fn standard_sha2_small_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::Sha2Small, generate_sha2_small)
}

#[cfg(feature = "zk")]
pub fn zk_muldiv_case() -> CoreZkVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    zk_case_from_accepted_fixture(generate_muldiv())
}

#[cfg(not(feature = "zk"))]
pub fn standard_advice_consumer_case() -> CoreVerifierCase {
    let _guard = CORE_FIXTURE_LOCK
        .lock()
        .expect("core fixture lock poisoned");
    case_from_accepted_fixture(CoreFixtureKind::AdviceConsumer, generate_advice_consumer)
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
fn case_from_accepted_fixture(
    kind: CoreFixtureKind,
    generate: impl FnOnce() -> GeneratedCoreFixture,
) -> CoreVerifierCase {
    let fixture = load_or_generate_fixture(kind, generate);
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

#[cfg(not(feature = "zk"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoreFixtureKind {
    MulDivSmall,
    FibonacciSmall,
    FibonacciMedium,
    MemoryOps,
    CollatzSmall,
    Sha2Small,
    AdviceConsumer,
}

#[cfg(not(feature = "zk"))]
impl CoreFixtureKind {
    const fn artifact_name(self) -> &'static str {
        match self {
            Self::MulDivSmall => "standard-muldiv-small",
            Self::FibonacciSmall => "standard-fibonacci-small",
            Self::FibonacciMedium => "standard-fibonacci-medium",
            Self::MemoryOps => "standard-memory-ops",
            Self::CollatzSmall => "standard-collatz-small",
            Self::Sha2Small => "standard-sha2-small",
            Self::AdviceConsumer => "standard-advice-consumer",
        }
    }
}

#[cfg(not(feature = "zk"))]
fn load_or_generate_fixture(
    kind: CoreFixtureKind,
    generate: impl FnOnce() -> GeneratedCoreFixture,
) -> GeneratedCoreFixture {
    let path = artifact_path(kind);
    let regenerate = env::var_os(REGENERATE_ARTIFACTS_ENV).is_some();
    if !regenerate && path.exists() {
        return read_fixture_artifact(&path);
    }

    let fixture = generate();
    if regenerate {
        write_fixture_artifact(&path, &fixture);
    }
    fixture
}

#[cfg(not(feature = "zk"))]
fn artifact_path(kind: CoreFixtureKind) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("core")
        .join(format!("{}.jvcf", kind.artifact_name()))
}

#[cfg(not(feature = "zk"))]
fn write_fixture_artifact(path: &PathBuf, fixture: &GeneratedCoreFixture) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create core fixture artifact directory");
    }

    let preprocessing = fixture
        .core_preprocessing
        .serialize_to_bytes()
        .expect("serialize verifier preprocessing");
    let public_io = fixture
        .public_io
        .serialize_to_bytes()
        .expect("serialize public I/O");
    let proof = fixture.proof.serialize_to_bytes().expect("serialize proof");
    let trusted_advice_commitment = fixture
        .trusted_advice_commitment
        .map(serialize_core_commitment);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(ARTIFACT_MAGIC);
    write_section(&mut bytes, &preprocessing);
    write_section(&mut bytes, &public_io);
    write_section(&mut bytes, &proof);
    match trusted_advice_commitment {
        Some(commitment) => {
            bytes.push(1);
            write_section(&mut bytes, &commitment);
        }
        None => bytes.push(0),
    }

    fs::write(path, bytes).expect("write core fixture artifact");
}

#[cfg(not(feature = "zk"))]
fn read_fixture_artifact(path: &PathBuf) -> GeneratedCoreFixture {
    let bytes = fs::read(path).expect("read core fixture artifact");
    let mut cursor = Cursor::new(bytes.as_slice());
    let mut magic = [0; ARTIFACT_MAGIC.len()];
    cursor
        .read_exact(&mut magic)
        .expect("read core fixture artifact magic");
    assert_eq!(
        &magic, ARTIFACT_MAGIC,
        "invalid core fixture artifact magic"
    );

    let preprocessing = read_section(&mut cursor);
    let public_io = read_section(&mut cursor);
    let proof = read_section(&mut cursor);
    let mut has_trusted_advice_commitment = [0];
    cursor
        .read_exact(&mut has_trusted_advice_commitment)
        .expect("read trusted advice commitment marker");
    let trusted_advice_commitment = match has_trusted_advice_commitment[0] {
        0 => None,
        1 => Some(deserialize_core_commitment(&read_section(&mut cursor))),
        marker => panic!("invalid trusted advice commitment marker {marker}"),
    };

    GeneratedCoreFixture {
        core_preprocessing: CoreVerifierPreprocessing::deserialize_from_bytes(&preprocessing)
            .expect("deserialize verifier preprocessing"),
        public_io: JoltDevice::deserialize_from_bytes(&public_io).expect("deserialize public I/O"),
        proof: CoreProof::deserialize_from_bytes(&proof).expect("deserialize proof"),
        trusted_advice_commitment,
    }
}

#[cfg(not(feature = "zk"))]
fn write_section(out: &mut Vec<u8>, section: &[u8]) {
    out.extend_from_slice(&(section.len() as u64).to_le_bytes());
    out.extend_from_slice(section);
}

#[cfg(not(feature = "zk"))]
fn read_section(cursor: &mut Cursor<&[u8]>) -> Vec<u8> {
    let mut len = [0; 8];
    cursor
        .read_exact(&mut len)
        .expect("read core fixture artifact section length");
    let mut section = vec![0; u64::from_le_bytes(len) as usize];
    cursor
        .read_exact(&mut section)
        .expect("read core fixture artifact section");
    section
}

#[cfg(not(feature = "zk"))]
fn serialize_core_commitment(commitment: CoreCommitment) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .0
        .serialize_compressed(&mut bytes)
        .expect("serialize trusted advice commitment");
    bytes
}

#[cfg(not(feature = "zk"))]
fn deserialize_core_commitment(bytes: &[u8]) -> CoreCommitment {
    ArkGT(
        ark_bn254::Fq12::deserialize_compressed(Cursor::new(bytes))
            .expect("deserialize trusted advice commitment"),
    )
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
fn generate_fibonacci_small() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("fibonacci-guest"),
        postcard::to_stdvec(&5u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_fibonacci_medium() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("fibonacci-guest"),
        postcard::to_stdvec(&100u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_memory_ops() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("memory-ops-guest"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_collatz_small() -> GeneratedCoreFixture {
    let mut program = host::Program::new("collatz-guest");
    program.set_func("collatz_convergence");
    generate_core_fixture(
        program,
        postcard::to_stdvec(&19u128).expect("serialize collatz input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_sha2_small() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("sha2-guest"),
        postcard::to_stdvec(&[5u8; 32]).expect("serialize sha2 input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_advice_consumer() -> GeneratedCoreFixture {
    generate_core_fixture(
        host::Program::new("advice-consumer-guest"),
        postcard::to_stdvec(&12u64).expect("serialize advice consumer public input"),
        postcard::to_stdvec(&5u64).expect("serialize untrusted advice"),
        postcard::to_stdvec(&7u64).expect("serialize trusted advice"),
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
