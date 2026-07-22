//! The per-stage [`StageProver`](crate::driver::StageProver) /
//! [`KernelSource`](crate::driver::KernelSource) impl expansions: one
//! member-list callback invocation per stage batch, each in a module that
//! imports the batch's relation and aggregate names so the derive-emitted
//! tokens resolve. This file is the prove side's complete stage-driver
//! surface — no stage's member list, order, or presence appears anywhere
//! else in this crate.

mod stage1 {
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage1::outputs::{
        Stage1BatchChallenges, Stage1BatchInputClaims, Stage1BatchInputPoints,
        Stage1BatchOutputClaims, Stage1BatchOutputPoints, Stage1BatchSumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage1_batch_sumchecks_members!(impl_stage_prover);
}

mod stage2 {
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
    use jolt_verifier::stages::stage2::outputs::{
        Stage2BatchChallenges, Stage2BatchInputClaims, Stage2BatchInputPoints,
        Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2BatchSumchecks,
    };
    use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
    use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
    use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
    use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage2_batch_sumchecks_members!(impl_stage_prover);
}

mod stage3 {
    use jolt_verifier::stages::stage3::outputs::{
        InstructionInput, RegistersClaimReduction, SpartanShift, Stage3Challenges,
        Stage3InputClaims, Stage3InputPoints, Stage3OutputClaims, Stage3OutputPoints,
        Stage3Sumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage3_sumchecks_members!(impl_stage_prover);
}

mod stage4 {
    use jolt_verifier::stages::stage4::outputs::{
        Stage4Challenges, Stage4InputClaims, Stage4InputPoints, Stage4OutputClaims,
        Stage4OutputPoints, Stage4Sumchecks,
    };
    use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
    use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage4_sumchecks_members!(impl_stage_prover);
}

mod stage5 {
    use jolt_verifier::stages::stage5::outputs::{
        Stage5Challenges, Stage5InputClaims, Stage5InputPoints, Stage5OutputClaims,
        Stage5OutputPoints, Stage5Sumchecks,
    };
    use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
    use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
    use jolt_verifier::stages::stage5::InstructionReadRaf;

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage5_sumchecks_members!(impl_stage_prover);
}

mod stage6a {
    use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
    use jolt_verifier::stages::stage6a::outputs::{
        Stage6aChallenges, Stage6aInputClaims, Stage6aInputPoints, Stage6aOutputClaims,
        Stage6aOutputPoints, Stage6aSumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage6a_sumchecks_members!(impl_stage_prover);
}

mod stage6b {
    use jolt_claims::protocols::jolt::JoltRelationId;
    use jolt_verifier::stages::stage6b::booleanity::Booleanity;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
    use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
        BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
        UntrustedAdviceCyclePhase,
    };
    use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
    use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
    use jolt_verifier::stages::stage6b::outputs::{
        Stage6bChallenges, Stage6bInputClaims, Stage6bInputPoints, Stage6bOutputClaims,
        Stage6bOutputPoints, Stage6bSumchecks,
    };
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
    use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
    use jolt_verifier::stages::stage6b::stage6b_opening_values;
    use jolt_verifier::VerifierError;

    use crate::driver::impl_stage_prover;

    // The stage's `no_opening_values` curation: the promoted verifier
    // helper's canonical order, including the runtime dedup of booleanity's
    // `BytecodeRa` claims against the bytecode read-RAF points.
    jolt_verifier::stage6b_sumchecks_members!(impl_stage_prover
        curate = |_batch, claims, points| {
            let booleanity_opening_point =
                points.booleanity_opening_point().ok_or_else(|| {
                    VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::Booleanity,
                        reason: "stage-6b booleanity produced no opening point".to_string(),
                    }
                })?;
            Ok(stage6b_opening_values(
                claims,
                &points.bytecode_read_raf.bytecode_ra,
                booleanity_opening_point,
            ))
        },
    );
}

mod stage7 {
    use jolt_verifier::stages::stage7::advice_address_phase::{
        TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
    };
    use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
        BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
    };
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
    use jolt_verifier::stages::stage7::outputs::{
        Stage7Challenges, Stage7InputClaims, Stage7InputPoints, Stage7OutputClaims,
        Stage7OutputPoints, Stage7Sumchecks,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::stage7_sumchecks_members!(impl_stage_prover);
}

/// Twin locks for the macro-expanded [`StageProver`](crate::driver::StageProver)
/// driver against a hand-rolled toy stage: three self-consistent dense
/// relations — a plain member, an `Option` member (exercised absent and
/// present), and a session-carried member whose kernel is reclaimed from a
/// [`ProofSession`](jolt_kernels::ProofSession) carry (the uni-skip-remainder
/// / precommitted-span pattern) — driven end to end (head → prepare → round
/// loop → typed extraction → shape validation → final-claim self-check →
/// finish) and byte-compared against the generated `verify_clear` on a twin
/// transcript.
#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod twin_tests {
    use core::marker::PhantomData;

    use jolt_claims::protocols::jolt::{
        JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims::{opening, NoChallenges, OutputClaims as _, SymbolicSumcheck};
    use jolt_field::{Field, Fr, FromPrimitiveInt, RingCore};
    use jolt_kernels::{
        KernelError, KernelSlots, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
        SumcheckKernelError,
    };
    use jolt_poly::UnivariatePoly;
    use jolt_sumcheck::{
        ClearSumcheckRecorder, CommittedSumcheckRecorder, ProveRounds, SumcheckError,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch, SumcheckOutputClaims};
    use jolt_verifier::VerifierError;
    use jolt_witness::protocols::jolt_vm::{
        JoltVmNamespace, JoltVmStage5InstructionReadRafRows, JoltVmStage6Rows, JoltVmWitnessPlane,
        Stage5InstructionReadRafRow,
    };
    use jolt_witness::{
        OracleDescriptor, OracleRef, PolynomialView, ViewRequirement, WitnessError, WitnessProvider,
    };

    use crate::driver::{impl_stage_prover, Proved};
    use crate::{ProverError, StageProver as _};

    /// Declare one toy dense relation: a single produced opening, a single
    /// consumed claim carrying the true table sum, no challenges, degree 1.
    macro_rules! toy_relation {
        (
            $symbolic:ident, $relation:ident, $inputs:ident, $outputs:ident,
            rel = $rel:ident, output = $output:ident, input = $input:ident
        ) => {
            #[derive(Clone, Debug, Default, PartialEq, Eq, jolt_claims::InputClaims)]
            struct $inputs<C> {
                #[opening($input, from = $rel)]
                claimed_sum: C,
            }

            #[derive(
                Clone,
                Debug,
                PartialEq,
                Eq,
                jolt_claims::OutputClaims,
                serde::Serialize,
                serde::Deserialize,
            )]
            #[relation($rel)]
            struct $outputs<C> {
                #[opening($output)]
                value: C,
            }

            #[derive(Clone)]
            struct $symbolic {
                rounds: usize,
            }

            impl SymbolicSumcheck for $symbolic {
                type RelationId = JoltRelationId;
                type OpeningId = JoltOpeningId;
                type DerivedId = jolt_claims::protocols::jolt::JoltDerivedId;
                type ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId;
                type Shape = usize;
                type Challenges<F> = NoChallenges<F>;
                type Inputs<C> = $inputs<C>;
                type Outputs<C> = $outputs<C>;

                fn new(shape: usize) -> Self {
                    Self { rounds: shape }
                }

                fn id() -> JoltRelationId {
                    JoltRelationId::$rel
                }

                fn rounds(&self) -> usize {
                    self.rounds
                }

                fn degree(&self) -> usize {
                    1
                }

                fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$input,
                        JoltRelationId::$rel,
                    ))
                }

                fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$output,
                        JoltRelationId::$rel,
                    ))
                }
            }

            #[derive(Clone)]
            struct $relation<F: Field> {
                symbolic: $symbolic,
                _field: PhantomData<F>,
            }

            impl<F: Field> $relation<F> {
                fn new(rounds: usize) -> Self {
                    Self {
                        symbolic: $symbolic::new(rounds),
                        _field: PhantomData,
                    }
                }
            }

            impl<F: Field> ConcreteSumcheck<F> for $relation<F> {
                type Symbolic = $symbolic;

                fn symbolic(&self) -> &$symbolic {
                    &self.symbolic
                }

                fn derive_opening_points(
                    &self,
                    sumcheck_point: &[F],
                    _input_points: &$inputs<Vec<F>>,
                ) -> Result<$outputs<Vec<F>>, VerifierError> {
                    Ok($outputs {
                        value: sumcheck_point.to_vec(),
                    })
                }
            }
        };
    }

    toy_relation!(
        AlphaSymbolic,
        ToyAlpha,
        ToyAlphaInputs,
        ToyAlphaOutputs,
        rel = RegistersValEvaluation,
        output = LookupOutput,
        input = UnexpandedPC
    );
    toy_relation!(
        BetaSymbolic,
        ToyBeta,
        ToyBetaInputs,
        ToyBetaOutputs,
        rel = RamValCheck,
        output = LeftLookupOperand,
        input = UnexpandedPC
    );
    toy_relation!(
        GammaSymbolic,
        ToyGamma,
        ToyGammaInputs,
        ToyGammaOutputs,
        rel = SpartanShift,
        output = RightLookupOperand,
        input = UnexpandedPC
    );

    #[derive(SumcheckBatch)]
    struct ToyDriverSumchecks<F: Field> {
        alpha: ToyAlpha<F>,
        beta: Option<ToyBeta<F>>,
        gamma: ToyGamma<F>,
    }

    // Unqualified: a macro-expanded `#[macro_export]` macro from the SAME
    // crate is reachable only textually, not by absolute path (#52234).
    toy_driver_sumchecks_members!(impl_stage_prover);

    /// A dense multilinear kernel with a prescribed total sum (HighToLow
    /// binding, degree 1): the single produced opening is the fully bound
    /// table value, which is exactly the relation's `expected_output`.
    struct DenseKernel<R> {
        evals: Vec<Fr>,
        num_rounds: usize,
        _relation: PhantomData<fn() -> R>,
    }

    impl<R> DenseKernel<R> {
        fn with_sum(num_rounds: usize, sum: Fr, seed: u64) -> Self {
            let size = 1u64 << num_rounds;
            let mut evals: Vec<Fr> = (0..size)
                .map(|i| Fr::from_u64(seed + 31 * i + 11))
                .collect();
            let current: Fr = evals.iter().copied().sum();
            evals[0] += sum - current;
            Self {
                evals,
                num_rounds,
                _relation: PhantomData,
            }
        }

        fn bind(&mut self, challenge: Fr) {
            let half = self.evals.len() / 2;
            for i in 0..half {
                self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
            }
            self.evals.truncate(half);
        }
    }

    impl<R> ProveRounds<Fr> for DenseKernel<R> {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn prove_round(
            &mut self,
            bind: Option<Fr>,
            _round: usize,
            previous_claim: Fr,
        ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
            if let Some(challenge) = bind {
                self.bind(challenge);
            }
            let half = self.evals.len() / 2;
            let eval_0: Fr = self.evals[..half].iter().copied().sum();
            let eval_1: Fr = self.evals[half..].iter().copied().sum();
            assert_eq!(eval_0 + eval_1, previous_claim);
            Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
        }

        fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
            self.bind(bind);
            Ok(())
        }
    }

    impl<R> SumcheckKernel<Fr> for DenseKernel<R>
    where
        R: ConcreteSumcheck<Fr>,
        SumcheckOutputClaims<Fr, R>: jolt_claims::OutputClaims<Fr>,
        jolt_verifier::stages::relations::SumcheckInputClaims<Fr, R>: jolt_claims::InputClaims<Fr>,
        jolt_verifier::stages::relations::ConcreteSumcheckChallenges<Fr, R>:
            jolt_claims::SumcheckChallenges<Fr, jolt_claims::protocols::jolt::JoltChallengeId>,
    {
        type Relation = R;

        fn output_claims(
            &mut self,
        ) -> Result<SumcheckOutputClaims<Fr, R>, SumcheckKernelError<Fr>> {
            assert_eq!(self.evals.len(), 1, "kernel extracted before fully bound");
            let value = self.evals[0];
            SumcheckOutputClaims::<Fr, R>::from_opening_values(|_| Some(value))
                .map_err(SumcheckKernelError::from)
        }
    }

    /// The prepare call order, recorded through the proof session (the
    /// universal `prepare` takes `&self`, so the log rides on the session's
    /// backend-private state instead of preparer mutability).
    #[derive(Default)]
    struct PrepareCallLog(Vec<&'static str>);

    fn log_prepare(session: &mut ProofSession, member: &'static str) {
        session
            .state_or_insert_with(PrepareCallLog::default)
            .0
            .push(member);
    }

    /// Mints a fresh dense kernel whose table sums to the member's consumed
    /// claim, read off the `ProverInputs` bundle like a real backend slot.
    struct DensePrepare {
        member: &'static str,
        seed: u64,
    }

    macro_rules! impl_dense_prepare {
        ($($relation:ident),+) => {$(
            impl PrepareKernel<Fr, $relation<Fr>> for DensePrepare {
                fn prepare(
                    &self,
                    session: &mut ProofSession,
                    _witness: &dyn JoltVmWitnessPlane<Fr>,
                    inputs: ProverInputs<'_, Fr, $relation<Fr>>,
                ) -> Result<
                    Box<dyn SumcheckKernel<Fr, Relation = $relation<Fr>>>,
                    KernelError<Fr>,
                > {
                    log_prepare(session, self.member);
                    Ok(Box::new(DenseKernel::<$relation<Fr>>::with_sum(
                        inputs.relation.rounds(),
                        inputs.claims.claimed_sum,
                        self.seed,
                    )))
                }
            }
        )+};
    }

    impl_dense_prepare!(ToyAlpha, ToyBeta);

    /// The gamma kernel is a `ProofSession` carry, parked by the toy front
    /// before `prove` — the uni-skip-remainder / precommitted-span pattern. A
    /// missing carry is a proof-time `KernelError`.
    struct ParkedToyGamma(DenseKernel<ToyGamma<Fr>>);

    struct SessionCarriedToyGamma;

    impl PrepareKernel<Fr, ToyGamma<Fr>> for SessionCarriedToyGamma {
        fn prepare(
            &self,
            session: &mut ProofSession,
            _witness: &dyn JoltVmWitnessPlane<Fr>,
            _inputs: ProverInputs<'_, Fr, ToyGamma<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ToyGamma<Fr>>>, KernelError<Fr>> {
            log_prepare(session, "gamma");
            let ParkedToyGamma(kernel) =
                session
                    .take::<ParkedToyGamma>()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "the toy front parked no gamma kernel for the carried member",
                    })?;
            Ok(Box::new(kernel))
        }
    }

    // The toy registry, resolved exactly like `JoltBackend`: one derived
    // `HasKernel` impl per `Box<dyn PrepareKernel<..>>` slot.
    #[derive(KernelSlots)]
    struct ToyKernels {
        alpha: Box<dyn PrepareKernel<Fr, ToyAlpha<Fr>>>,
        beta: Box<dyn PrepareKernel<Fr, ToyBeta<Fr>>>,
        gamma: Box<dyn PrepareKernel<Fr, ToyGamma<Fr>>>,
    }

    fn toy_kernels() -> ToyKernels {
        ToyKernels {
            alpha: Box::new(DensePrepare {
                member: "alpha",
                seed: 5,
            }),
            beta: Box::new(DensePrepare {
                member: "beta",
                seed: 91,
            }),
            gamma: Box::new(SessionCarriedToyGamma),
        }
    }

    /// A witness plane the toy kernels never read: every access errors.
    struct NoWitness;

    impl WitnessProvider<Fr, JoltVmNamespace> for NoWitness {
        fn describe_oracle(
            &self,
            _oracle: OracleRef<JoltVmNamespace>,
        ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
            Err(WitnessError::UnsupportedView {
                view: "toy driver twin witness",
            })
        }

        fn view_requirements(
            &self,
            _oracle: OracleRef<JoltVmNamespace>,
        ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
            Err(WitnessError::UnsupportedView {
                view: "toy driver twin witness",
            })
        }

        fn oracle_view(
            &self,
            _requirement: ViewRequirement<JoltVmNamespace>,
        ) -> Result<PolynomialView<'_, Fr, JoltVmNamespace>, WitnessError> {
            Err(WitnessError::UnsupportedView {
                view: "toy driver twin witness",
            })
        }
    }

    impl JoltVmStage5InstructionReadRafRows for NoWitness {
        fn stage5_instruction_read_raf_rows(
            &self,
            _log_t: usize,
        ) -> Result<Vec<Stage5InstructionReadRafRow>, WitnessError> {
            Err(WitnessError::UnsupportedView {
                view: "toy driver twin witness",
            })
        }
    }

    impl JoltVmStage6Rows for NoWitness {
        fn stage6_rows(
            &self,
        ) -> Result<Vec<jolt_witness::protocols::jolt_vm::JoltVmStage6Row>, WitnessError> {
            Err(WitnessError::UnsupportedView {
                view: "toy driver twin witness",
            })
        }
    }

    const ALPHA_ROUNDS: usize = 3;
    const BETA_ROUNDS: usize = 2;
    const GAMMA_ROUNDS: usize = 3;
    const GAMMA_SUM: u64 = 4242;

    fn fixture(beta: bool) -> ToyDriverSumchecks<Fr> {
        ToyDriverSumchecks {
            alpha: ToyAlpha::new(ALPHA_ROUNDS),
            beta: beta.then(|| ToyBeta::new(BETA_ROUNDS)),
            gamma: ToyGamma::new(GAMMA_ROUNDS),
        }
    }

    fn inputs(beta: bool) -> ToyDriverInputClaims<Fr> {
        let fr = Fr::from_u64;
        ToyDriverInputClaims {
            alpha: ToyAlphaInputs {
                claimed_sum: fr(1234),
            },
            beta: beta.then(|| ToyBetaInputs {
                claimed_sum: fr(777),
            }),
            gamma: ToyGammaInputs {
                claimed_sum: fr(GAMMA_SUM),
            },
        }
    }

    /// Drive the macro-expanded `prove` and its `verify_clear` twin, assert
    /// byte-identical transcript states, and return the driver's output plus
    /// the recorded prepare call order.
    fn drive(beta: bool) -> (Proved<Fr, ToyDriverSumchecks<Fr>, Fr>, Vec<&'static str>) {
        let sumchecks = fixture(beta);
        let inputs = inputs(beta);
        let kernels = toy_kernels();
        let mut session = ProofSession::default();
        session.park(ParkedToyGamma(DenseKernel::with_sum(
            GAMMA_ROUNDS,
            Fr::from_u64(GAMMA_SUM),
            23,
        )));

        let mut prover_transcript = Blake2bTranscript::new(b"prove-driver-twin");
        let challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        let proved = sumchecks
            .prove(
                &kernels,
                &mut session,
                &NoWitness,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<Fr, Fr>::new(),
                &mut prover_transcript,
            )
            .unwrap();

        // Verifier twin: generated draw + verify_clear + output-claim absorbs.
        let mut verifier_transcript = Blake2bTranscript::new(b"prove-driver-twin");
        let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let verified = sumchecks
            .verify_clear(
                &inputs,
                &verifier_challenges,
                &proved.recorded.proof,
                &mut verifier_transcript,
            )
            .unwrap();
        sumchecks.append_output_claims(&mut verifier_transcript, &proved.output_claims);

        assert_eq!(verified.reduction.value, proved.final_claim);
        assert_eq!(
            sumchecks
                .expected_final_claim(
                    &verified.coefficients,
                    &input_points,
                    &proved.output_claims,
                    &sumchecks
                        .derive_opening_points(&verified.reduction.point, &input_points)
                        .unwrap(),
                    &verifier_challenges,
                )
                .unwrap(),
            proved.final_claim,
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let calls = session.take::<PrepareCallLog>().unwrap().0;
        (proved, calls)
    }

    #[test]
    fn driver_twin_with_present_option_member() {
        let (proved, calls) = drive(true);
        // Prepare ran in declaration order.
        assert_eq!(calls, vec!["alpha", "beta", "gamma"]);
        assert!(proved.output_claims.beta.is_some());
        // Typed extraction filled every slot, the session carry included.
        assert_eq!(proved.output_claims.alpha.opening_values().len(), 1);
        assert_eq!(proved.output_claims.gamma.opening_values().len(), 1);
    }

    #[test]
    fn driver_twin_with_absent_option_member() {
        let (proved, calls) = drive(false);
        assert_eq!(calls, vec!["alpha", "gamma"]);
        assert!(proved.output_claims.beta.is_none());
    }

    /// A session-carried member with no parked carry fails at prepare with a
    /// kernel error — the accepted cost of session carries over v1's
    /// compile-time-visible external members.
    #[test]
    fn missing_session_carry_fails_at_prepare() {
        let sumchecks = fixture(false);
        let inputs = inputs(false);
        let kernels = toy_kernels();
        let mut session = ProofSession::default();

        let mut transcript = Blake2bTranscript::new(b"prove-driver-twin");
        let challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        let result = sumchecks.prove(
            &kernels,
            &mut session,
            &NoWitness,
            &inputs,
            &input_points,
            &challenges,
            ClearSumcheckRecorder::<Fr, Fr>::new(),
            &mut transcript,
        );
        assert!(matches!(
            result,
            Err(ProverError::Kernel(KernelError::InvariantViolation { .. }))
        ));
    }

    /// `prove` is generic over the recorder: this compiles it against the
    /// committed recorder even though nothing wires the ZK path yet.
    #[expect(dead_code, reason = "compile-only recorder-generality witness")]
    #[expect(clippy::too_many_arguments, reason = "the driver's protocol signature")]
    fn prove_type_checks_with_committed_recorder(
        sumchecks: &ToyDriverSumchecks<Fr>,
        kernels: &ToyKernels,
        session: &mut ProofSession,
        inputs: &ToyDriverInputClaims<Fr>,
        input_points: &ToyDriverInputPoints<Fr>,
        challenges: &ToyDriverChallenges<Fr>,
        recorder: CommittedSumcheckRecorder<
            '_,
            Fr,
            jolt_crypto::Pedersen<jolt_crypto::Bn254G1>,
            rand_core::OsRng,
        >,
        transcript: &mut Blake2bTranscript,
    ) -> Result<Proved<Fr, ToyDriverSumchecks<Fr>, jolt_crypto::Bn254G1>, ProverError<Fr>> {
        sumchecks.prove(
            kernels,
            session,
            &NoWitness,
            inputs,
            input_points,
            challenges,
            recorder,
            transcript,
        )
    }
}
