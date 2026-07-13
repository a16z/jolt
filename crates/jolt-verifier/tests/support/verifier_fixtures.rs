#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when verifier object construction or serialization breaks"
)]

use std::{
    env, fs,
    io::{self, Cursor, Read},
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

#[cfg(unix)]
use std::{os::fd::AsRawFd, os::raw::c_int};

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryCommitment;
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};

use jolt_prover_legacy::{
    curve::Bn254Curve,
    host,
    poly::commitment::{
        commitment_scheme::CommitmentScheme as ProverCommitmentScheme, dory::DoryCommitmentScheme,
    },
    zkvm::{
        preprocessing::JoltSharedPreprocessing,
        program::ProgramPreprocessing,
        proof::{verifier_preprocessing_from_prover, ProofCommitmentScheme},
        prover::JoltProverPreprocessing,
        RV64IMACProver,
    },
};

#[cfg(not(feature = "zk"))]
use jolt_prover_legacy::{
    poly::{
        commitment::dory::{DoryContext, DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
    },
    zkvm::ram::populate_memory_states,
};

static VERIFIER_FIXTURE_LOCK: Mutex<()> = Mutex::new(());
// Bumped for the InstructionClaimReductionOutputClaims Option<C> -> C wire flip
// so stale cached fixtures regenerate instead of panicking mid-decode.
const FIXTURE_MAGIC: &[u8; 8] = b"JVCF0003";
const REGENERATE_ARTIFACTS_ENV: &str = "JOLT_VERIFIER_REGENERATE_VERIFIER_FIXTURES";
const VERIFIER_FIXTURE_LOCK_FILE: &str = "jolt-verifier-fixtures.lock";

#[cfg(unix)]
const LOCK_EX: c_int = 2;
#[cfg(unix)]
const LOCK_UN: c_int = 8;

#[cfg(unix)]
unsafe extern "C" {
    fn flock(fd: c_int, operation: c_int) -> c_int;
}

#[cfg(unix)]
struct VerifierFixtureLock {
    _guard: MutexGuard<'static, ()>,
    file: fs::File,
}

#[cfg(not(unix))]
struct VerifierFixtureLock {
    _guard: MutexGuard<'static, ()>,
}

#[cfg(unix)]
impl Drop for VerifierFixtureLock {
    fn drop(&mut self) {
        // SAFETY: `self.file` owns a live file descriptor for the lifetime of
        // the guard. Unlocking in Drop mirrors the successful lock operation.
        let _ = unsafe { flock(self.file.as_raw_fd(), LOCK_UN) };
    }
}

#[cfg(unix)]
fn verifier_fixture_lock() -> VerifierFixtureLock {
    let guard = VERIFIER_FIXTURE_LOCK
        .lock()
        .expect("verifier fixture mutex poisoned");
    let lock_path = env::temp_dir().join(VERIFIER_FIXTURE_LOCK_FILE);
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open verifier fixture file lock");
    lock_exclusive(&file);

    VerifierFixtureLock {
        _guard: guard,
        file,
    }
}

#[cfg(not(unix))]
fn verifier_fixture_lock() -> VerifierFixtureLock {
    VerifierFixtureLock {
        _guard: VERIFIER_FIXTURE_LOCK
            .lock()
            .expect("verifier fixture mutex poisoned"),
    }
}

#[cfg(unix)]
fn lock_exclusive(file: &fs::File) {
    loop {
        // SAFETY: `file` owns a live file descriptor, and `LOCK_EX` is a valid
        // `flock(2)` operation. The call blocks until the process-wide fixture
        // lock is available.
        let result = unsafe { flock(file.as_raw_fd(), LOCK_EX) };
        if result == 0 {
            return;
        }
        let error = io::Error::last_os_error();
        assert_eq!(
            error.kind(),
            io::ErrorKind::Interrupted,
            "lock verifier fixture file: {error}"
        );
    }
}

type ProverField = jolt_prover_legacy::ark_bn254::Fr;
type ProverCommitment = <DoryCommitmentScheme as ProverCommitmentScheme>::Commitment;
type ProverOpeningHint = <DoryCommitmentScheme as ProverCommitmentScheme>::OpeningProofHint;
type VerifierFixtureProof = jolt_verifier::JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type VerifierFixturePreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type TrustedAdviceCommitter = fn(
    &JoltProverPreprocessing<ProverField, Bn254Curve, DoryCommitmentScheme>,
    &[u8],
) -> (ProverCommitment, ProverOpeningHint);

#[cfg(not(feature = "zk"))]
#[derive(Clone)]
pub struct VerifierFixtureCase {
    pub preprocessing: VerifierFixturePreprocessing,
    pub public_io: JoltDevice,
    pub proof: VerifierFixtureProof,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

#[cfg(not(feature = "zk"))]
impl VerifierFixtureCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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
pub struct ZkVerifierFixtureCase {
    pub preprocessing: VerifierFixturePreprocessing,
    pub public_io: JoltDevice,
    pub proof: VerifierFixtureProof,
}

#[cfg(feature = "zk")]
impl ZkVerifierFixtureCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            None,
            true,
        )
    }
}

#[cfg(not(feature = "zk"))]
pub fn standard_muldiv_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(VerifierFixtureKind::MulDivSmall, generate_muldiv)
}

#[cfg(not(feature = "zk"))]
pub fn standard_fibonacci_small_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(
        VerifierFixtureKind::FibonacciSmall,
        generate_fibonacci_small,
    )
}

#[cfg(not(feature = "zk"))]
pub fn standard_fibonacci_medium_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(
        VerifierFixtureKind::FibonacciMedium,
        generate_fibonacci_medium,
    )
}

#[cfg(not(feature = "zk"))]
pub fn standard_memory_ops_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(VerifierFixtureKind::MemoryOps, generate_memory_ops)
}

#[cfg(not(feature = "zk"))]
pub fn standard_collatz_small_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(VerifierFixtureKind::CollatzSmall, generate_collatz_small)
}

#[cfg(not(feature = "zk"))]
pub fn standard_sha2_small_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(VerifierFixtureKind::Sha2Small, generate_sha2_small)
}

#[cfg(feature = "zk")]
pub fn zk_muldiv_case() -> ZkVerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    let fixture = load_or_generate_fixture(VerifierFixtureKind::ZkMulDivSmall, || {
        let fixture = generate_muldiv();
        assert_verifier_accepts(&fixture, fixture.proof.clone(), fixture.public_io.clone());
        fixture
    });
    zk_case_from_parts(fixture)
}

#[cfg(feature = "zk")]
pub fn zk_committed_muldiv_case() -> ZkVerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    let fixture = load_or_generate_fixture(VerifierFixtureKind::ZkCommittedMulDivSmall, || {
        let fixture = generate_committed_muldiv();
        assert_verifier_accepts(&fixture, fixture.proof.clone(), fixture.public_io.clone());
        fixture
    });
    zk_case_from_parts(fixture)
}

#[cfg(feature = "zk")]
pub fn fresh_zk_muldiv_case() -> ZkVerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    zk_case_from_parts(generate_muldiv())
}

#[cfg(not(feature = "zk"))]
pub fn standard_advice_consumer_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(
        VerifierFixtureKind::AdviceConsumer,
        generate_advice_consumer,
    )
}

#[cfg(not(feature = "zk"))]
pub fn standard_committed_muldiv_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    case_from_accepted_fixture(
        VerifierFixtureKind::CommittedMulDivSmall,
        generate_committed_muldiv,
    )
}

#[cfg(not(feature = "zk"))]
fn case_from_accepted_fixture(
    kind: VerifierFixtureKind,
    generate: impl FnOnce() -> GeneratedVerifierFixture,
) -> VerifierFixtureCase {
    let fixture = load_or_generate_fixture(kind, generate);
    assert_verifier_accepts(&fixture, fixture.proof.clone(), fixture.public_io.clone());
    let public_io = fixture.public_io.clone();
    case_from_parts(fixture, public_io)
}

#[cfg(feature = "zk")]
fn zk_case_from_parts(fixture: GeneratedVerifierFixture) -> ZkVerifierFixtureCase {
    ZkVerifierFixtureCase {
        preprocessing: fixture.preprocessing,
        public_io: fixture.public_io,
        proof: fixture.proof,
    }
}

#[cfg(not(feature = "zk"))]
fn case_from_parts(
    fixture: GeneratedVerifierFixture,
    public_io: JoltDevice,
) -> VerifierFixtureCase {
    VerifierFixtureCase {
        preprocessing: fixture.preprocessing,
        public_io,
        proof: fixture.proof,
        trusted_advice_commitment: fixture.trusted_advice_commitment,
    }
}

#[derive(Clone)]
struct GeneratedVerifierFixture {
    preprocessing: VerifierFixturePreprocessing,
    public_io: JoltDevice,
    proof: VerifierFixtureProof,
    trusted_advice_commitment: Option<DoryCommitment>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierFixtureKind {
    #[cfg(not(feature = "zk"))]
    MulDivSmall,
    #[cfg(not(feature = "zk"))]
    FibonacciSmall,
    #[cfg(not(feature = "zk"))]
    FibonacciMedium,
    #[cfg(not(feature = "zk"))]
    MemoryOps,
    #[cfg(not(feature = "zk"))]
    CollatzSmall,
    #[cfg(not(feature = "zk"))]
    Sha2Small,
    #[cfg(not(feature = "zk"))]
    AdviceConsumer,
    #[cfg(not(feature = "zk"))]
    CommittedMulDivSmall,
    #[cfg(feature = "zk")]
    ZkMulDivSmall,
    #[cfg(feature = "zk")]
    ZkCommittedMulDivSmall,
}

impl VerifierFixtureKind {
    const fn fixture_name(self) -> &'static str {
        match self {
            #[cfg(not(feature = "zk"))]
            Self::MulDivSmall => "standard-muldiv-small",
            #[cfg(not(feature = "zk"))]
            Self::FibonacciSmall => "standard-fibonacci-small",
            #[cfg(not(feature = "zk"))]
            Self::FibonacciMedium => "standard-fibonacci-medium",
            #[cfg(not(feature = "zk"))]
            Self::MemoryOps => "standard-memory-ops",
            #[cfg(not(feature = "zk"))]
            Self::CollatzSmall => "standard-collatz-small",
            #[cfg(not(feature = "zk"))]
            Self::Sha2Small => "standard-sha2-small",
            #[cfg(not(feature = "zk"))]
            Self::AdviceConsumer => "standard-advice-consumer",
            #[cfg(not(feature = "zk"))]
            Self::CommittedMulDivSmall => "standard-committed-muldiv-small",
            #[cfg(feature = "zk")]
            Self::ZkMulDivSmall => "zk-muldiv-small-continued-transcript",
            #[cfg(feature = "zk")]
            Self::ZkCommittedMulDivSmall => "zk-committed-muldiv-small",
        }
    }
}

fn load_or_generate_fixture(
    kind: VerifierFixtureKind,
    generate: impl FnOnce() -> GeneratedVerifierFixture,
) -> GeneratedVerifierFixture {
    let path = fixture_path(kind);
    let regenerate = env::var_os(REGENERATE_ARTIFACTS_ENV).is_some();
    if !regenerate && path.exists() {
        if let Some(fixture) = read_fixture_file(&path) {
            return fixture;
        }
    }

    let fixture = generate();
    if regenerate || cfg!(feature = "zk") {
        write_fixture_file(&path, &fixture);
    }
    fixture
}

fn fixture_path(kind: VerifierFixtureKind) -> PathBuf {
    let filename = format!("{}.jvcf", kind.fixture_name());
    if cfg!(feature = "zk") {
        env::temp_dir()
            .join("jolt-verifier-fixtures")
            .join(filename)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("verifier")
            .join(filename)
    }
}

fn write_fixture_file(path: &PathBuf, fixture: &GeneratedVerifierFixture) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create verifier fixture directory");
    }

    let preprocessing = serialize_verifier_object(&fixture.preprocessing);
    let public_io = serialize_verifier_object(&fixture.public_io);
    let proof = serialize_verifier_object(&fixture.proof);
    let trusted_advice_commitment = fixture
        .trusted_advice_commitment
        .as_ref()
        .map(serialize_verifier_object);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(FIXTURE_MAGIC);
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

    fs::write(path, bytes).expect("write verifier fixture file");
}

/// Returns `None` on any decode failure, not just a magic mismatch: cached
/// fixtures embed proof types whose serialized layout can change under a
/// dependency bump, and a stale cache must count as a miss so
/// `load_or_generate_fixture` regenerates it.
fn read_fixture_file(path: &PathBuf) -> Option<GeneratedVerifierFixture> {
    let bytes = fs::read(path).expect("read verifier fixture file");
    let mut cursor = Cursor::new(bytes.as_slice());
    let mut magic = [0; FIXTURE_MAGIC.len()];
    cursor.read_exact(&mut magic).ok()?;
    if &magic != FIXTURE_MAGIC {
        return None;
    }

    let preprocessing = read_section(&mut cursor)?;
    let public_io = read_section(&mut cursor)?;
    let proof = read_section(&mut cursor)?;
    let mut has_trusted_advice_commitment = [0];
    cursor.read_exact(&mut has_trusted_advice_commitment).ok()?;
    let trusted_advice_commitment = match has_trusted_advice_commitment[0] {
        0 => None,
        1 => Some(deserialize_verifier_object(&read_section(&mut cursor)?)?),
        _ => return None,
    };

    Some(GeneratedVerifierFixture {
        preprocessing: deserialize_verifier_object(&preprocessing)?,
        public_io: deserialize_verifier_object(&public_io)?,
        proof: deserialize_verifier_object(&proof)?,
        trusted_advice_commitment,
    })
}

fn write_section(out: &mut Vec<u8>, section: &[u8]) {
    out.extend_from_slice(&(section.len() as u64).to_le_bytes());
    out.extend_from_slice(section);
}

fn read_section(cursor: &mut Cursor<&[u8]>) -> Option<Vec<u8>> {
    let mut len = [0; 8];
    cursor.read_exact(&mut len).ok()?;
    let len = usize::try_from(u64::from_le_bytes(len)).ok()?;
    let remaining = (cursor.get_ref().len() as u64).saturating_sub(cursor.position());
    if len as u64 > remaining {
        return None;
    }
    let mut section = vec![0; len];
    cursor.read_exact(&mut section).ok()?;
    Some(section)
}

fn serialize_verifier_object<T: serde::Serialize>(item: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(item, bincode::config::standard())
        .expect("serialize verifier object")
}

fn deserialize_verifier_object<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    let (value, consumed) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).ok()?;
    (consumed == bytes.len()).then_some(value)
}

fn assert_verifier_accepts(
    fixture: &GeneratedVerifierFixture,
    proof: VerifierFixtureProof,
    public_io: JoltDevice,
) {
    let result = verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        &fixture.preprocessing,
        &public_io,
        &proof,
        fixture.trusted_advice_commitment.as_ref(),
        cfg!(feature = "zk"),
    );
    assert!(
        result.is_ok(),
        "canonical verifier should accept generated fixture proof: {result:?}",
    );
}

fn generate_muldiv() -> GeneratedVerifierFixture {
    let program = host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    generate_verifier_fixture(program, inputs, Vec::new(), Vec::new(), None)
}

#[cfg(not(feature = "zk"))]
fn generate_fibonacci_small() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        host::Program::new("fibonacci-guest"),
        postcard::to_stdvec(&5u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_fibonacci_medium() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        host::Program::new("fibonacci-guest"),
        postcard::to_stdvec(&100u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_memory_ops() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        host::Program::new("memory-ops-guest"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_collatz_small() -> GeneratedVerifierFixture {
    let mut program = host::Program::new("collatz-guest");
    program.set_func("collatz_convergence");
    generate_verifier_fixture(
        program,
        postcard::to_stdvec(&19u128).expect("serialize collatz input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_sha2_small() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        host::Program::new("sha2-guest"),
        postcard::to_stdvec(&[5u8; 32]).expect("serialize sha2 input"),
        Vec::new(),
        Vec::new(),
        None,
    )
}

#[cfg(not(feature = "zk"))]
fn generate_advice_consumer() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        host::Program::new("advice-consumer-guest"),
        postcard::to_stdvec(&12u64).expect("serialize advice consumer public input"),
        postcard::to_stdvec(&5u64).expect("serialize untrusted advice"),
        postcard::to_stdvec(&7u64).expect("serialize trusted advice"),
        Some(commit_trusted_advice_preprocessing_only),
    )
}

fn generate_committed_muldiv() -> GeneratedVerifierFixture {
    const BYTECODE_CHUNK_COUNT: usize = 2;

    let mut program = host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, _, _, public_io) = program.trace(&inputs, &[], &[]);

    let program_preprocessing =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("preprocess committed verifier fixture");
    let (shared_preprocessing, committed_program_prover_data, generators) =
        JoltSharedPreprocessing::new_committed(
            program_preprocessing,
            public_io.memory_layout.clone(),
            1 << 16,
            BYTECODE_CHUNK_COUNT,
        );
    let prover_preprocessing = JoltProverPreprocessing::new_committed(
        shared_preprocessing,
        committed_program_prover_data,
        generators,
    );
    let elf_contents = program
        .get_elf_contents()
        .expect("elf contents should exist");

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let public_io = prover.program_io.clone();
    let (proof, _) = prover.prove().expect("prove verifier object fixture");
    let preprocessing = verifier_preprocessing_from_prover(&prover_preprocessing);

    GeneratedVerifierFixture {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: None,
    }
}

fn generate_verifier_fixture(
    mut program: host::Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    trusted_advice_committer: Option<TrustedAdviceCommitter>,
) -> GeneratedVerifierFixture {
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, _, _, public_io) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let program_preprocessing =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("preprocess verifier fixture");
    let shared_preprocessing = JoltSharedPreprocessing::new(
        program_preprocessing,
        public_io.memory_layout.clone(),
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .expect("elf contents should exist");

    let (trusted_advice_commitment, trusted_advice_hint) = trusted_advice_committer
        .map(|commit| commit(&prover_preprocessing, &trusted_advice))
        .map_or((None, None), |(commitment, hint)| {
            (Some(commitment), Some(hint))
        });
    let verifier_trusted_advice_commitment = trusted_advice_commitment.map(
        <DoryCommitmentScheme as ProofCommitmentScheme<ProverField>>::commitment_into_verifier,
    );

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
    let (proof, _) = prover.prove().expect("prove verifier object fixture");
    let preprocessing = verifier_preprocessing_from_prover(&prover_preprocessing);

    GeneratedVerifierFixture {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: verifier_trusted_advice_commitment,
    }
}

#[cfg(not(feature = "zk"))]
fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<ProverField, Bn254Curve, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (ProverCommitment, ProverOpeningHint) {
    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
    populate_memory_states(
        0,
        trusted_advice_bytes,
        Some(&mut trusted_advice_words),
        None,
    );

    let poly = MultilinearPolynomial::<ProverField>::from(trusted_advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
    let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
    DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
}
