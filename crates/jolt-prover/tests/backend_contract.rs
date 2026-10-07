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
    use jolt_kernels::{
        CommitWitness, CommitmentGrid, KernelError, ProofSession, WitnessCommitment,
    };
    use jolt_program::{execution::OwnedTrace, preprocess::JoltProgramPreprocessing};
    use jolt_prover::{
        dory, JoltBackend, JoltProverPreprocessing, JoltSharedPreprocessing, ProverConfig,
        ProverError,
    };
    use jolt_transcript::LegacyBlake2bTranscript;
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
        witness: TraceBackend<OwnedTrace>,
    }

    fn fixture() -> Fixture {
        let mut case = GuestCase::new("muldiv-guest");
        case.inputs = postcard::to_stdvec(&[9u32, 5, 3]).expect("guest inputs");
        let run = support::prepare(&case);
        let config = ProverConfig::derive_from_dimensions::<Fr>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let preprocessing = dory::from_shared(
            JoltSharedPreprocessing::new(run.preprocessing).expect("shared preprocessing"),
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
            let context = registry.with_witness(RejectPolynomial {
                witness: &witness,
                rejected,
            });
            let mut session = registry.begin_proof();
            let result = dory::stages::stage0::prove_stage0::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                LegacyBlake2bTranscript,
            >(
                &context,
                &mut session,
                &preprocessing,
                &config,
                None,
                &public_io,
            );
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
            let mut session = registry.begin_proof();
            session.park(ExecutionState {
                releases: Arc::clone(&releases),
                trace_length: config.trace_length,
            });
            let context = registry.with_witness(&witness).with_session(session);
            let result = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
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
}
