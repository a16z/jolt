#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when verifier object construction or serialization breaks"
)]

use std::{
    env, fs,
    io::{self, Cursor, Read},
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

#[cfg(unix)]
use std::{os::fd::AsRawFd, os::raw::c_int};

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryCommitment;
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_host::Program;
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput};
use jolt_prover::dory::DoryProverPreprocessing;
use jolt_prover::{JoltBackend, JoltSharedPreprocessing, ProverConfig};
use jolt_riscv::JoltTraceRow;
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use super::guest_fixtures::{prepare_guest, PreparedGuest};

static VERIFIER_FIXTURE_LOCK: Mutex<()> = Mutex::new(());
// Modular preprocessing has a different encoded shape from the retired prover.
const FIXTURE_MAGIC: &[u8; 8] = b"JVCF0004";
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

type VerifierFixtureProof = jolt_verifier::JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type VerifierFixturePreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

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
pub fn fresh_standard_muldiv_address_major_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    fresh_case_from_accepted_fixture(|| {
        generate_muldiv_with_order(TracePolynomialOrder::AddressMajor)
    })
}

#[cfg(not(feature = "zk"))]
pub fn fresh_standard_committed_muldiv_address_major_case(
    bytecode_chunk_count: usize,
) -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    fresh_case_from_accepted_fixture(|| {
        generate_committed_muldiv_with_order(
            bytecode_chunk_count,
            TracePolynomialOrder::AddressMajor,
        )
    })
}

#[cfg(not(feature = "zk"))]
pub fn fresh_standard_committed_advice_case() -> VerifierFixtureCase {
    let _guard = verifier_fixture_lock();
    fresh_case_from_accepted_fixture(generate_committed_advice_consumer)
}

#[cfg(not(feature = "zk"))]
fn fresh_case_from_accepted_fixture(
    generate: impl FnOnce() -> GeneratedVerifierFixture,
) -> VerifierFixtureCase {
    let fixture = generate();
    assert_verifier_accepts(&fixture, fixture.proof.clone(), fixture.public_io.clone());
    let public_io = fixture.public_io.clone();
    case_from_parts(fixture, public_io)
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
            // ZK names carry a transcript-scheme suffix: they key the temp-dir
            // cache, so a ZK transcript change must rename them or stale cached
            // proofs fail verification instead of regenerating.
            #[cfg(feature = "zk")]
            Self::ZkMulDivSmall => "zk-muldiv-small-degree-bound",
            #[cfg(feature = "zk")]
            Self::ZkCommittedMulDivSmall => "zk-committed-muldiv-small-degree-bound",
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
    );
    assert!(
        result.is_ok(),
        "canonical verifier should accept generated fixture proof: {result:?}",
    );
}

fn generate_muldiv() -> GeneratedVerifierFixture {
    generate_muldiv_with_order(TracePolynomialOrder::CycleMajor)
}

fn generate_muldiv_with_order(order: TracePolynomialOrder) -> GeneratedVerifierFixture {
    let program = Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    generate_verifier_fixture_with_order(program, inputs, Vec::new(), Vec::new(), order)
}

#[cfg(not(feature = "zk"))]
fn generate_fibonacci_small() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        Program::new("fibonacci-guest"),
        postcard::to_stdvec(&5u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
    )
}

#[cfg(not(feature = "zk"))]
fn generate_fibonacci_medium() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        Program::new("fibonacci-guest"),
        postcard::to_stdvec(&100u32).expect("serialize fibonacci input"),
        Vec::new(),
        Vec::new(),
    )
}

#[cfg(not(feature = "zk"))]
fn generate_memory_ops() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        Program::new("memory-ops-guest"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

#[cfg(not(feature = "zk"))]
fn generate_collatz_small() -> GeneratedVerifierFixture {
    let mut program = Program::new("collatz-guest");
    program.set_func("collatz_convergence");
    generate_verifier_fixture(
        program,
        postcard::to_stdvec(&19u128).expect("serialize collatz input"),
        Vec::new(),
        Vec::new(),
    )
}

#[cfg(not(feature = "zk"))]
fn generate_sha2_small() -> GeneratedVerifierFixture {
    generate_verifier_fixture(
        Program::new("sha2-guest"),
        postcard::to_stdvec(&[5u8; 32]).expect("serialize sha2 input"),
        Vec::new(),
        Vec::new(),
    )
}

#[cfg(not(feature = "zk"))]
fn generate_advice_consumer() -> GeneratedVerifierFixture {
    generate_advice_consumer_with_committed_program(false)
}

#[cfg(not(feature = "zk"))]
fn generate_committed_advice_consumer() -> GeneratedVerifierFixture {
    generate_advice_consumer_with_committed_program(true)
}

#[cfg(not(feature = "zk"))]
fn generate_advice_consumer_with_committed_program(
    committed_program: bool,
) -> GeneratedVerifierFixture {
    let program = Program::new("advice-consumer-guest");
    let inputs = postcard::to_stdvec(&12u64).expect("serialize advice consumer public input");
    let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
    let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
    let run = prepare_guest(program, &inputs, &untrusted_advice, &trusted_advice);
    let config = derive_config(&run);
    let preprocessing = if committed_program {
        jolt_prover::dory::preprocess_committed(run.program_preprocessing, 2)
            .expect("committed preprocessing")
    } else {
        let shared =
            JoltSharedPreprocessing::new(run.program_preprocessing).expect("shared preprocessing");
        jolt_prover::dory::from_shared(shared)
    };
    prove_prepared(
        run.program,
        run.trace,
        config,
        preprocessing,
        &trusted_advice,
    )
}

fn generate_committed_muldiv() -> GeneratedVerifierFixture {
    generate_committed_muldiv_with_order(2, TracePolynomialOrder::CycleMajor)
}

fn generate_committed_muldiv_with_order(
    bytecode_chunk_count: usize,
    order: TracePolynomialOrder,
) -> GeneratedVerifierFixture {
    let program = Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(program, &inputs, &[], &[]);
    let mut config = derive_config(&run);
    config.trace_polynomial_order = order;
    let preprocessing = jolt_prover::dory::preprocess_committed_with_order(
        run.program_preprocessing,
        bytecode_chunk_count,
        order,
    )
    .expect("committed preprocessing");
    prove_prepared(run.program, run.trace, config, preprocessing, &[])
}

fn generate_verifier_fixture(
    program: Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
) -> GeneratedVerifierFixture {
    generate_verifier_fixture_with_order(
        program,
        inputs,
        untrusted_advice,
        trusted_advice,
        TracePolynomialOrder::CycleMajor,
    )
}

fn generate_verifier_fixture_with_order(
    program: Program,
    inputs: Vec<u8>,
    untrusted_advice: Vec<u8>,
    trusted_advice: Vec<u8>,
    order: TracePolynomialOrder,
) -> GeneratedVerifierFixture {
    let run = prepare_guest(program, &inputs, &untrusted_advice, &trusted_advice);
    let mut config = derive_config(&run);
    config.trace_polynomial_order = order;
    let shared =
        JoltSharedPreprocessing::new(run.program_preprocessing).expect("shared preprocessing");
    let preprocessing = jolt_prover::dory::from_shared(shared);
    prove_prepared(
        run.program,
        run.trace,
        config,
        preprocessing,
        &trusted_advice,
    )
}

fn derive_config(run: &PreparedGuest) -> ProverConfig {
    ProverConfig::derive_compact::<Fr>(
        run.trace.trace.as_slice(),
        &run.program_preprocessing.memory_layout,
        run.program_preprocessing.ram.min_bytecode_address,
        run.program_preprocessing.ram.bytecode_words.len(),
        1 << 16,
    )
    .expect("derive config")
}

fn prove_prepared(
    program: Arc<JoltProgram>,
    trace: TraceOutput<Arc<Vec<JoltTraceRow>>>,
    config: ProverConfig,
    preprocessing: DoryProverPreprocessing,
    trusted_advice: &[u8],
) -> GeneratedVerifierFixture {
    let program_preprocessing = preprocessing
        .program_arc()
        .expect("full program retained by prover preprocessing");
    let public_io = trace.device.clone();
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        )
        .include_untrusted_advice(!public_io.untrusted_advice.is_empty())
        .include_trusted_advice(!trusted_advice.is_empty()),
        JoltVmWitnessInputs::new(&program, &program_preprocessing, trace),
    );
    let trusted = (!trusted_advice.is_empty()).then(|| {
        jolt_prover::dory::commit_trusted_advice(&preprocessing, trusted_advice)
            .expect("trusted advice commitment")
    });
    let proof =
        jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
            &JoltBackend::optimized(),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &witness,
            &public_io,
        )
        .expect("prove verifier fixture");
    GeneratedVerifierFixture {
        preprocessing: preprocessing.verifier,
        public_io,
        proof,
        trusted_advice_commitment: trusted.map(|object| object.commitment),
    }
}
