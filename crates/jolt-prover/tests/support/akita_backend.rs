#![expect(clippy::expect_used, clippy::panic)]

use std::ops::Range;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc::{self, Sender},
    Arc,
};
use std::time::Duration;

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_kernels::akita::commitment::{CommitWitness, WitnessCommitRequest, WitnessCommitment};
use jolt_kernels::{KernelError, ProofSession, ReferenceBackend};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::{
    ChunkVisitor, JoltWitnessOracle, JoltWitnessPlane, ProgramSource, RandomAccessRows, RowSource,
    Shape, WitnessError,
};

use jolt_prover::JoltBackend;
use jolt_prover::ProverError;

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_program::execution::OwnedTrace;
use jolt_prover::akita;
use jolt_prover::akita::preprocessing::{AkitaProverPreprocessing, AkitaTranscript, AkitaVc};
use jolt_prover::ProverConfig;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

struct UnreadableWitness;

impl JoltWitnessOracle<AkitaField> for UnreadableWitness {
    fn shape(&self, _id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        panic!("commit dispatch must use the coordinator's layout")
    }
    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        panic!("commit dispatch must use the coordinator's layout")
    }
    fn oracle_table(&self, _id: JoltPolynomialId) -> Result<Vec<AkitaField>, WitnessError> {
        panic!("commit dispatch must not request polynomial tables")
    }
}

impl RowSource for UnreadableWitness {
    fn visit_chunks(
        &self,
        _range: Range<usize>,
        _chunk_size: usize,
        _visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        panic!("commit dispatch must not request trace rows")
    }
    fn random_access(&self) -> Option<RandomAccessRows> {
        panic!("commit dispatch must not request random access")
    }
}

impl ProgramSource for UnreadableWitness {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        panic!("commit dispatch must use the coordinator's preprocessing")
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ExecutionState {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    released: Sender<()>,
    trace_length: usize,
}

impl Drop for ExecutionState {
    fn drop(&mut self) {
        let _ = self.released.send(());
    }
}

struct SessionCommit {
    calls: Arc<AtomicUsize>,
    refuse: bool,
}

impl CommitWitness<AkitaField, AkitaScheme> for SessionCommit {
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<AkitaField>,
        request: WitnessCommitRequest<'_, AkitaScheme>,
    ) -> Result<WitnessCommitment<AkitaScheme>, KernelError<AkitaField>> {
        let execution = session
            .state::<ExecutionState>()
            .expect("full prove must receive the caller's execution state");
        assert_eq!(execution.trace_length, 1usize << request.shape().log_t);
        assert!(request.precommitted_hints().is_empty());
        assert_eq!(
            request.plan().num_vars(),
            request.shape().log_t + request.shape().log_k_chunk,
        );
        let _ = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse {
            Err(KernelError::Unsupported {
                reason: "test commitment refused",
            })
        } else {
            ReferenceBackend.commit_witness(session, _witness, request)
        }
    }
}

#[test]
fn registry_reuses_short_lived_witnesses_and_releases_each_session() {
    let fixture = muldiv_fixture();
    let calls = Arc::new(AtomicUsize::new(0));
    let (release_sender, release_receiver) = mpsc::channel();
    let registry = JoltBackend::reference_kernels(SessionCommit {
        calls: Arc::clone(&calls),
        refuse: true,
    });
    let mut expected_calls = 0;
    // Reuse the registry across a commitment failure, invalid geometry, and
    // another commitment failure, each with a fresh short-lived witness.
    for valid in [true, false, true] {
        let mut config = fixture.config;
        if !valid {
            config.trace_length = 0;
        }
        let witness = UnreadableWitness;
        let mut session = registry.begin_proof();
        session.park(ExecutionState {
            released: release_sender.clone(),
            trace_length: fixture.config.trace_length,
        });
        let backend = registry.with_witness(&witness).with_session(session);
        let result = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            backend,
            &fixture.preprocessing,
            &config,
            None,
            &fixture.public_io,
        );
        if valid {
            assert!(matches!(
                result,
                Err(ProverError::Kernel(KernelError::Unsupported {
                    reason: "test commitment refused"
                }))
            ));
            expected_calls += 1;
        } else {
            assert!(
                result.is_err(),
                "invalid trace length must fail before dispatch"
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        release_receiver
            .try_recv()
            .expect("failed proof must release its execution state before returning");
    }
}

#[test]
fn successful_proof_releases_the_bound_witness_and_session() {
    let fixture = muldiv_fixture();
    let witness = Arc::new(fixture.witness);
    let witness_lifetime = Arc::downgrade(&witness);
    let calls = Arc::new(AtomicUsize::new(0));
    let (release_sender, release_receiver) = mpsc::channel();
    let registry = JoltBackend::reference_kernels(SessionCommit {
        calls: Arc::clone(&calls),
        refuse: false,
    })
    .with_optimized_compute();
    let mut session = registry.begin_proof();
    session.park(ExecutionState {
        released: release_sender,
        trace_length: fixture.config.trace_length,
    });
    let context = registry.with_witness(witness).with_session(session);
    let _proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
        context,
        &fixture.preprocessing,
        &fixture.config,
        None,
        &fixture.public_io,
    )
    .expect("native proof with commitment-produced hints");
    // Successful proofs may queue session destruction on the background pool.
    release_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("successful proof must release its execution state");
    assert!(witness_lifetime.upgrade().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct Fixture {
    config: ProverConfig,
    preprocessing: AkitaProverPreprocessing,
    public_io: JoltDevice,
    witness: TraceBackend<OwnedTrace>,
}

fn muldiv_fixture() -> Fixture {
    use crate::support::GuestCase;
    let mut case = GuestCase::new("muldiv-guest");
    case.inputs = postcard::to_stdvec(&[9u32, 5, 3]).expect("guest inputs");
    let run = crate::support::prepare(&case);
    let config = ProverConfig::derive_from_dimensions::<AkitaField>(
        run.trace.dimensions,
        &run.preprocessing.memory_layout,
        run.preprocessing.ram.min_bytecode_address,
        run.preprocessing.ram.bytecode_words.len(),
        run.preprocessing.max_padded_trace_length,
    )
    .expect("derive config");
    let preprocessing = akita::preprocessing::preprocess_full_with_advice(
        &AkitaScheduleArtifacts::shared_from_default_directory(),
        run.preprocessing,
        &config,
        false,
        false,
    )
    .expect("preprocessing");
    let public_io = run.trace.device.clone();
    let program = preprocessing.program_arc().expect("program preprocessing");
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&run.program, &program, run.trace),
    );
    Fixture {
        config,
        preprocessing,
        public_io,
        witness,
    }
}
