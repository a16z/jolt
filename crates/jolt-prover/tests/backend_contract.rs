#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline"),
    not(feature = "zk")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test failures must identify the violated contract"
)]
mod tests {
    use crate::support::{self, GuestCase};
    use common::jolt_device::JoltDevice;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_kernels::optimized::StreamingFamilies;
    use jolt_kernels::{
        CommitWitness, CommitmentGrid, KernelContext, KernelError, ProofSession, WitnessCommitment,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::{
        dory, JoltBackend, JoltProverPreprocessing, JoltSharedPreprocessing, ProverConfig,
        ProverError,
    };
    use jolt_transcript::{Keccak, ProverState, VerifierState};
    use jolt_witness::{
        ChunkVisitor, JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle,
        JoltWitnessPlane, ProgramSource, RowSource, Shape, TraceBackend, WitnessError,
    };
    use std::{
        ops::Range,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
    struct ExecutionState {
        #[cfg_attr(feature = "allocative", allocative(skip))]
        releases: Arc<AtomicUsize>,
        trace_length: usize,
    }
    impl Drop for ExecutionState {
        fn drop(&mut self) {
            let _ = self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct SessionCommit(Arc<AtomicUsize>);
    impl CommitWitness<Fr, DoryScheme> for SessionCommit {
        fn commit_witness(
            &self,
            session: &mut ProofSession,
            _: &dyn RowSource,
            _: &[JoltCommittedPolynomial],
            grid: CommitmentGrid,
            _: &<DoryScheme as jolt_openings::CommitmentScheme>::ProverSetup,
        ) -> Result<Vec<WitnessCommitment<DoryScheme>>, KernelError<Fr>> {
            let execution = session
                .state::<ExecutionState>()
                .expect("full prove must retain caller-initialized execution state");
            assert_eq!(execution.trace_length, 1usize << grid.log_t);
            let _ = self.0.fetch_add(1, Ordering::SeqCst);
            Err(KernelError::Unsupported {
                reason: "test commitment refused",
            })
        }
        fn commit_advice(
            &self,
            _: &mut ProofSession,
            _: &dyn JoltWitnessOracle<Fr>,
            _: JoltCommittedPolynomial,
            _: CommitmentGrid,
            _: &<DoryScheme as jolt_openings::CommitmentScheme>::ProverSetup,
        ) -> Result<WitnessCommitment<DoryScheme>, KernelError<Fr>> {
            panic!("fixture has no advice")
        }
    }

    struct RejectPolynomial<'a> {
        witness: &'a dyn JoltWitnessPlane<Fr>,
        rejected: JoltPolynomialId,
    }
    impl JoltWitnessOracle<Fr> for RejectPolynomial<'_> {
        fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
            if id == self.rejected {
                return Err(WitnessError::UnavailableView {
                    label: "unsupported test polynomial",
                });
            }
            self.witness.shape(id)
        }
        fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
            panic!("stage-0 commitment order comes from canonical geometry")
        }
        fn oracle_table(&self, _: JoltPolynomialId) -> Result<Vec<Fr>, WitnessError> {
            panic!("unsupported polynomials must fail before reading tables")
        }
    }
    impl RowSource for RejectPolynomial<'_> {
        fn visit_chunks(
            &self,
            _: Range<usize>,
            _: usize,
            _: &mut ChunkVisitor<'_>,
        ) -> Result<(), WitnessError> {
            panic!("unsupported polynomials must fail before reading rows")
        }
    }
    impl ProgramSource for RejectPolynomial<'_> {
        fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
            panic!("stage-0 geometry comes from verifier preprocessing")
        }
    }

    struct Fixture {
        config: ProverConfig,
        preprocessing: JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>,
        public_io: JoltDevice,
        witness: TraceBackend,
    }

    fn fixture() -> Fixture {
        let mut case = GuestCase::new("muldiv-guest");
        case.inputs = postcard::to_stdvec(&[9u32, 5, 3]).expect("guest inputs");
        prepared_fixture(&case, |program| {
            dory::from_shared(JoltSharedPreprocessing::new(program).expect("shared preprocessing"))
                .expect("preprocessing")
        })
    }

    /// The memory-ops guest's `panic_halt` entry asked to panic, under
    /// committed preprocessing: its run ends at `ebreak`.
    fn committed_panic_fixture() -> Fixture {
        let case = GuestCase {
            func: Some("panic_halt"),
            inputs: postcard::to_stdvec(&true).expect("guest inputs"),
            panics: true,
            ..GuestCase::new("memory-ops-guest")
        };
        prepared_fixture(&case, |program| {
            dory::preprocess_committed(program).expect("committed preprocessing")
        })
    }

    fn prepared_fixture(
        case: &GuestCase,
        preprocess: impl FnOnce(
            JoltProgramPreprocessing,
        ) -> JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>,
    ) -> Fixture {
        let run = support::prepare(case);
        let config = ProverConfig::derive_from_dimensions::<Fr>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let preprocessing = preprocess(run.preprocessing);
        let public_io = run.trace.device.clone();
        let program = preprocessing.program_arc().expect("program preprocessing");
        let witness = TraceBackend::new(
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

    #[test]
    fn stage0_validates_the_bound_witness_before_commitment() {
        let Fixture {
            config,
            preprocessing,
            public_io,
            witness,
        } = fixture();
        let registry = JoltBackend::<Fr, DoryScheme>::reference();
        for rejected in [
            JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::InstructionRafFlag),
        ] {
            let mut context = registry.with_witness(RejectPolynomial {
                witness: &witness,
                rejected,
            });
            let result = dory::stages::stage0::prove_stage0::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                ProverState<Keccak>,
            >(&mut context, &preprocessing, &config, None, &public_io);
            assert!(
                matches!(result, Err(ProverError::Witness(WitnessError::InvalidWitnessData { label: "stage-0 validation", reason }))
                if reason.contains(&format!("{rejected:?}")) && reason.contains("unsupported test polynomial"))
            );
        }
    }

    #[test]
    fn full_prove_consumes_initialized_sessions_and_reuses_the_registry() {
        let Fixture {
            config,
            preprocessing,
            public_io,
            witness,
        } = fixture();
        let calls = Arc::new(AtomicUsize::new(0));
        let releases = Arc::new(AtomicUsize::new(0));
        let registry = JoltBackend::reference_kernels(SessionCommit(Arc::clone(&calls)));
        let mut expected_calls = 0;
        for (index, valid) in [true, false, true].into_iter().enumerate() {
            let mut proof_config = config;
            if !valid {
                proof_config.trace_length = 0;
            }
            let mut session = ProofSession::default();
            session.park(ExecutionState {
                releases: Arc::clone(&releases),
                trace_length: config.trace_length,
            });
            let context = KernelContext::new(&registry, &witness, session);
            let result = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
                context,
                &preprocessing,
                &proof_config,
                None,
                &public_io,
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
                    "invalid config must fail before commitment"
                );
            }
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            assert_eq!(releases.load(Ordering::SeqCst), index + 1);
        }
    }

    /// Every streaming family is selected, and the proof must equal the
    /// default kernels' byte for byte. The rounds a kernel streams follow
    /// from its cap and its clamps, so this test pins none of them: the
    /// parity tests in `jolt-kernels` cover every round at which a kernel
    /// can turn dense. What this trace and config add:
    /// - The trace ends inside its padded domain, so every streamed pass,
    ///   walk and fold weighs a constant suffix once, and every table a
    ///   kernel turns dense holds it as its tail.
    /// - The derived config binds the register read-write check cycle-first,
    ///   the order in which its streaming kernel runs. `e2e_matrix`'s
    ///   all-families case binds address-first in the builds that compile
    ///   this file, which keeps the default kernel.
    /// - The guest's bytecode table is small enough for the direct path of
    ///   the bytecode read-RAF address walk; the unit tests in `jolt-kernels`
    ///   cover the routed one.
    /// - The RAM kernels have no crossover: they read the event log, and
    ///   stage 6b its own address column, at every trace size.
    /// - Stage 8's joint opening collects no columns.
    #[test]
    fn streaming_kernels_reproduce_the_default_proof() {
        assert_streaming_reproduces_the_default_proof(fixture());
    }

    /// `streaming_kernels_reproduce_the_default_proof` for a committed
    /// program whose run ends at `ebreak`, so the proof claims
    /// `panic = true`. Stage 8 opens the precommitted bytecode and
    /// program-image tables beside the trace views, and the bytecode table's
    /// `Halt` and `HaltPanic` lanes hold that `ebreak`'s flags.
    #[test]
    fn streaming_kernels_reproduce_the_default_committed_proof() {
        assert_streaming_reproduces_the_default_proof(committed_panic_fixture());
    }

    fn assert_streaming_reproduces_the_default_proof(fixture: Fixture) {
        let Fixture {
            config,
            preprocessing,
            public_io,
            witness,
        } = fixture;
        let rows = witness.random_access().expect("resident rows");
        assert!(
            rows.physical_cycles() < rows.cycles(),
            "the instruction-rows kernels must meet a constant suffix",
        );
        let prove = |backend: JoltBackend<Fr, DoryScheme>| {
            dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
                backend.with_witness(&witness),
                &preprocessing,
                &config,
                None,
                &public_io,
            )
            .expect("Dory proof")
        };
        let streamed = prove(JoltBackend::optimized().with_streaming(StreamingFamilies {
            spartan: true,
            instruction_input: true,
            ram: true,
            registers: true,
            instruction_rows: true,
        }));
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, VerifierState<'_, Keccak>>(
            &preprocessing.verifier,
            &public_io,
            &streamed,
            None,
        )
        .expect("the streamed proof must verify");
        let default = prove(JoltBackend::optimized());
        assert!(
            postcard::to_stdvec(&streamed).expect("encode the streamed proof")
                == postcard::to_stdvec(&default).expect("encode the default proof"),
            "the streaming kernels must reproduce the default proof bytes",
        );
    }
}
