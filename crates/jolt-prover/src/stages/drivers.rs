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
    #[cfg(feature = "field-inline")]
    use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
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
    #[cfg(feature = "field-inline")]
    use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
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
    #[cfg(feature = "field-inline")]
    use jolt_verifier::stages::stage5::field_registers_val_evaluation::FieldRegistersValEvaluation;
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
        BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase,
    };
    #[cfg(not(feature = "akita"))]
    use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
        TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase,
    };
    #[cfg(feature = "field-inline")]
    use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
    #[cfg(not(feature = "akita"))]
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
    #[cfg(not(feature = "akita"))]
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

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod twin_tests {
    use core::marker::PhantomData;

    use jolt_claims::protocols::jolt::{
        JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims::{opening, NoChallenges, OutputClaims as _, SymbolicSumcheck};
    use jolt_field::{Fr, JoltField, Ring};
    use jolt_kernels::{
        KernelError, KernelSlots, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
        SumcheckKernelError,
    };
    use jolt_poly::UnivariatePoly;
    use jolt_sumcheck::{
        ClearSumcheckRecorder, CommittedSumcheckRecorder, ProveRounds, RoundScheduler,
        SequentialRounds, SumcheckError,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch, SumcheckOutputClaims};
    use jolt_verifier::VerifierError;
    use jolt_witness::{
        ChunkVisitor, JoltWitnessOracle, JoltWitnessPlane, ProgramSource, RowSource, Shape,
        WitnessError,
    };

    use crate::driver::{impl_stage_prover, Proved};
    use crate::{ProverError, StageProver as _};

    macro_rules! toy_relation {
        (
            $symbolic:ident, $relation:ident, $inputs:ident, $outputs:ident,
            rel = $rel:ident, output = $output:ident, input = $input:ident
            $(, head_pad = $head_pad:expr)?
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
            #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
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

                fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$input,
                        JoltRelationId::$rel,
                    ))
                }

                fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$output,
                        JoltRelationId::$rel,
                    ))
                }
            }

            #[derive(Clone)]
            struct $relation<F: JoltField> {
                symbolic: $symbolic,
                _field: PhantomData<F>,
            }

            impl<F: JoltField> $relation<F> {
                fn new(rounds: usize) -> Self {
                    Self {
                        symbolic: $symbolic::new(rounds),
                        _field: PhantomData,
                    }
                }
            }

            impl<F: JoltField> ConcreteSumcheck<F> for $relation<F> {
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

                $(
                    fn instance_point_offset(
                        &self,
                        _batch_num_vars: usize,
                    ) -> Result<usize, VerifierError> {
                        Ok(0)
                    }

                    fn expected_output(
                        &self,
                        _input_points: &$inputs<Vec<F>>,
                        output_values: &$outputs<F>,
                        _output_points: &$outputs<Vec<F>>,
                        _challenges: &NoChallenges<F>,
                    ) -> Result<F, VerifierError> {
                        let scale = F::from_u64(1u64 << $head_pad).inverse().unwrap();
                        Ok(output_values.value * scale)
                    }
                )?
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
    toy_relation!(
        DeltaSymbolic,
        ToyDelta,
        ToyDeltaInputs,
        ToyDeltaOutputs,
        rel = RegistersReadWriteChecking,
        output = RegistersVal,
        input = UnexpandedPC,
        head_pad = HEAD_PAD
    );

    #[derive(SumcheckBatch)]
    struct ToyDriverSumchecks<F: JoltField> {
        alpha: ToyAlpha<F>,
        beta: Option<ToyBeta<F>>,
        gamma: ToyGamma<F>,
    }

    #[derive(SumcheckBatch)]
    struct ToyHeadSumchecks<F: JoltField> {
        alpha: ToyAlpha<F>,
        delta: ToyDelta<F>,
    }

    // Unqualified: a macro-expanded `#[macro_export]` macro from the SAME
    // crate is reachable only textually, not by absolute path (#52234).
    toy_driver_sumchecks_members!(impl_stage_prover);
    toy_head_sumchecks_members!(impl_stage_prover);

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
        R::Symbolic: SymbolicSumcheck<RelationId = JoltRelationId>,
    {
        type Relation = R;

        fn output_claims(
            &mut self,
            _inputs: &jolt_verifier::stages::relations::SumcheckInputClaims<Fr, R>,
        ) -> Result<SumcheckOutputClaims<Fr, R>, SumcheckKernelError<Fr>> {
            assert_eq!(self.evals.len(), 1, "kernel extracted before fully bound");
            let value = self.evals[0];
            SumcheckOutputClaims::<Fr, R>::from_opening_values(|_| Some(value))
                .map_err(SumcheckKernelError::from)
        }

        fn park_residue(self: Box<Self>, session: &mut ProofSession) {
            assert_eq!(self.evals.len(), 1, "kernel parked before fully bound");
            log_residue(session, <R::Symbolic as SymbolicSumcheck>::id());
        }
    }

    #[derive(Default)]
    struct PrepareCallLog(Vec<&'static str>);

    fn log_prepare(session: &mut ProofSession, member: &'static str) {
        session
            .state_or_insert_with(PrepareCallLog::default)
            .0
            .push(member);
    }

    #[derive(Default)]
    struct ResidueCallLog(Vec<JoltRelationId>);

    fn log_residue(session: &mut ProofSession, member: JoltRelationId) {
        session
            .state_or_insert_with(ResidueCallLog::default)
            .0
            .push(member);
    }

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
                    _witness: &dyn JoltWitnessPlane<Fr>,
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

    struct HeadDensePrepare {
        seed: u64,
    }

    impl PrepareKernel<Fr, ToyDelta<Fr>> for HeadDensePrepare {
        fn prepare(
            &self,
            session: &mut ProofSession,
            _witness: &dyn JoltWitnessPlane<Fr>,
            inputs: ProverInputs<'_, Fr, ToyDelta<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ToyDelta<Fr>>>, KernelError<Fr>> {
            log_prepare(session, "delta");
            Ok(Box::new(DenseKernel::<ToyDelta<Fr>>::with_sum(
                inputs.relation.rounds(),
                inputs.claims.claimed_sum.mul_pow_2(HEAD_PAD),
                self.seed,
            )))
        }
    }

    struct ParkedToyGamma(DenseKernel<ToyGamma<Fr>>);

    #[cfg(feature = "allocative")]
    mod carry_visitation {
        use super::*;

        macro_rules! impl_self_sized_allocative {
            ($($ty:ty),+ $(,)?) => {$(
                impl allocative::Allocative for $ty {
                    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                        visitor.enter_self_sized::<Self>().exit();
                    }
                }
            )+};
        }
        impl_self_sized_allocative!(PrepareCallLog, ResidueCallLog, ParkedToyGamma);

        impl<R> allocative::Allocative for DenseKernel<R> {
            fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                let mut visitor = visitor.enter_self_sized::<Self>();
                visitor.visit_simple(
                    allocative::Key::new("evals"),
                    self.evals.capacity() * size_of::<Fr>(),
                );
                visitor.exit();
            }
        }
    }

    struct SessionCarriedToyGamma;

    impl PrepareKernel<Fr, ToyGamma<Fr>> for SessionCarriedToyGamma {
        fn prepare(
            &self,
            session: &mut ProofSession,
            _witness: &dyn JoltWitnessPlane<Fr>,
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

    #[derive(KernelSlots)]
    struct ToyHeadKernels {
        alpha: Box<dyn PrepareKernel<Fr, ToyAlpha<Fr>>>,
        delta: Box<dyn PrepareKernel<Fr, ToyDelta<Fr>>>,
    }

    struct NoWitness;

    impl NoWitness {
        fn unavailable() -> WitnessError {
            WitnessError::UnavailableView {
                label: "toy driver twin witness",
            }
        }
    }

    impl JoltWitnessOracle<Fr> for NoWitness {
        fn shape(
            &self,
            _id: jolt_claims::protocols::jolt::JoltPolynomialId,
        ) -> Result<Shape, WitnessError> {
            Err(Self::unavailable())
        }

        fn oracle_table(
            &self,
            _id: jolt_claims::protocols::jolt::JoltPolynomialId,
        ) -> Result<Vec<Fr>, WitnessError> {
            Err(Self::unavailable())
        }

        fn committed_order(
            &self,
        ) -> Result<Vec<jolt_claims::protocols::jolt::JoltCommittedPolynomial>, WitnessError>
        {
            Err(Self::unavailable())
        }
    }

    impl RowSource for NoWitness {
        fn visit_chunks(
            &self,
            _range: std::ops::Range<usize>,
            _chunk_size: usize,
            _visitor: &mut ChunkVisitor<'_>,
        ) -> Result<(), WitnessError> {
            Err(Self::unavailable())
        }
    }

    impl ProgramSource for NoWitness {
        #[expect(
            clippy::unimplemented,
            reason = "the accessor returns a borrow, so it cannot report the every-access error the other NoWitness accessors do; the toy kernels never read the program"
        )]
        fn program_preprocessing(&self) -> &jolt_program::preprocess::JoltProgramPreprocessing {
            unimplemented!("toy driver twin witness carries no program")
        }
    }

    const ALPHA_ROUNDS: usize = 3;
    const BETA_ROUNDS: usize = 2;
    const GAMMA_ROUNDS: usize = 3;
    const GAMMA_SUM: u64 = 4242;
    const HEAD_ROUNDS: usize = 2;
    const HEAD_PAD: usize = ALPHA_ROUNDS - HEAD_ROUNDS;

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

    #[expect(
        clippy::type_complexity,
        reason = "the twin driver's aggregate return: the proved carrier plus the two recorded call orders"
    )]
    fn drive(
        beta: bool,
    ) -> (
        Proved<Fr, ToyDriverSumchecks<Fr>, Fr>,
        Vec<&'static str>,
        Vec<JoltRelationId>,
    ) {
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
                &mut SequentialRounds,
                &NoWitness,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<Fr, Fr>::new(),
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(b"prove-driver-twin");
        let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let _ = sumchecks
            .verify_clear(
                &inputs,
                &input_points,
                &verifier_challenges,
                &proved.output_claims,
                &proved.recorded.proof,
                &mut verifier_transcript,
                0,
            )
            .unwrap();
        sumchecks.append_output_claims(&mut verifier_transcript, &proved.output_claims);

        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let calls = session.take::<PrepareCallLog>().unwrap().0;
        let residues = session.take::<ResidueCallLog>().unwrap().0;
        (proved, calls, residues)
    }

    #[test]
    fn driver_twin_with_present_option_member() {
        let (proved, calls, residues) = drive(true);
        assert_eq!(calls, vec!["alpha", "beta", "gamma"]);
        assert!(proved.output_claims.beta.is_some());
        assert_eq!(proved.output_claims.alpha.opening_values().len(), 1);
        assert_eq!(proved.output_claims.gamma.opening_values().len(), 1);
        assert_eq!(
            residues,
            vec![
                JoltRelationId::RegistersValEvaluation,
                JoltRelationId::RamValCheck,
                JoltRelationId::SpartanShift,
            ]
        );
    }

    #[test]
    fn driver_twin_with_absent_option_member() {
        let (proved, calls, residues) = drive(false);
        assert_eq!(calls, vec!["alpha", "gamma"]);
        assert!(proved.output_claims.beta.is_none());
        assert_eq!(
            residues,
            vec![
                JoltRelationId::RegistersValEvaluation,
                JoltRelationId::SpartanShift,
            ]
        );
    }

    #[test]
    fn driver_twin_with_head_aligned_member() {
        let sumchecks = ToyHeadSumchecks {
            alpha: ToyAlpha::new(ALPHA_ROUNDS),
            delta: ToyDelta::new(HEAD_ROUNDS),
        };
        let inputs = ToyHeadInputClaims {
            alpha: ToyAlphaInputs {
                claimed_sum: Fr::from_u64(1234),
            },
            delta: ToyDeltaInputs {
                claimed_sum: Fr::from_u64(4321),
            },
        };
        let kernels = ToyHeadKernels {
            alpha: Box::new(DensePrepare {
                member: "alpha",
                seed: 5,
            }),
            delta: Box::new(HeadDensePrepare { seed: 37 }),
        };
        let mut session = ProofSession::default();

        let mut prover_transcript = Blake2bTranscript::new(b"prove-driver-head-twin");
        let challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        let proved = sumchecks
            .prove(
                &kernels,
                &mut session,
                &mut SequentialRounds,
                &NoWitness,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<Fr, Fr>::new(),
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(b"prove-driver-head-twin");
        let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let verified_points = sumchecks
            .verify_clear(
                &inputs,
                &input_points,
                &verifier_challenges,
                &proved.output_claims,
                &proved.recorded.proof,
                &mut verifier_transcript,
                0,
            )
            .unwrap();
        sumchecks.append_output_claims(&mut verifier_transcript, &proved.output_claims);

        assert_eq!(prover_transcript.state(), verifier_transcript.state());
        assert_eq!(verified_points, proved.output_points);
        assert_eq!(
            proved.output_points.delta.value.as_slice(),
            &proved.output_points.alpha.value[..HEAD_ROUNDS]
        );
        assert_eq!(proved.output_claims.alpha.opening_values().len(), 1);
        assert_eq!(proved.output_claims.delta.opening_values().len(), 1);
        let calls = session.take::<PrepareCallLog>().unwrap().0;
        let residues = session.take::<ResidueCallLog>().unwrap().0;
        assert_eq!(calls, vec!["alpha", "delta"]);
        assert_eq!(
            residues,
            vec![
                JoltRelationId::RegistersValEvaluation,
                JoltRelationId::RegistersReadWriteChecking,
            ]
        );
    }

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
            &mut SequentialRounds,
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

    #[test]
    fn populated_cells_for_absent_member_fail_at_prepare() {
        let kernels = toy_kernels();
        let mut session = ProofSession::default();
        let claims = ToyBetaInputs {
            claimed_sum: Fr::from_u64(777),
        };
        let points = ToyBetaInputs {
            claimed_sum: Vec::new(),
        };

        let error = crate::driver::prepare_optional::<Fr, ToyBeta<Fr>, _>(
            &kernels,
            None,
            &mut session,
            &NoWitness,
            Some(&claims),
            Some(&points),
            Some(&NoChallenges::default()),
        )
        .map(|kernel| kernel.map(|_| ()))
        .unwrap_err();
        let ProverError::Verifier(VerifierError::StageClaimSumcheckFailed { stage, .. }) = &error
        else {
            panic!("expected the populated-cell wiring error, got {error:?}");
        };
        assert_eq!(*stage, format!("{:?}", JoltRelationId::RamValCheck));
    }

    #[expect(dead_code, reason = "compile-only recorder-generality witness")]
    #[expect(clippy::too_many_arguments, reason = "the driver's protocol signature")]
    fn prove_type_checks_with_committed_recorder(
        sumchecks: &ToyDriverSumchecks<Fr>,
        kernels: &ToyKernels,
        session: &mut ProofSession,
        scheduler: &mut dyn RoundScheduler<Fr>,
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
            scheduler,
            &NoWitness,
            inputs,
            input_points,
            challenges,
            recorder,
            transcript,
        )
    }
}
