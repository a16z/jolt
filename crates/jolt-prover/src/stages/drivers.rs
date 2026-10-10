//! The per-stage [`StageProver`](crate::driver::StageProver) /
//! [`StageAggregates`](crate::driver::StageAggregates) impl expansions: one
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

    // Stage 4's `no_opening_values` replacement keeps the generated
    // signature (the claims aggregate's hand-ordered `opening_values`, which
    // splices the field-inline openings under `field-inline`), so the driver's default
    // curation serves both feature arms unchanged.
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
    // The Akita batch has no inc member — the fused-inc read-raf stages
    // discharge the reduced inc claims instead.
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

/// Twin locks for the macro-expanded [`StageProver`](crate::driver::StageProver)
/// driver against a hand-rolled toy stage: three self-consistent dense
/// relations — a plain member, an `Option` member (exercised absent and
/// present), and a session-carried member whose kernel is reclaimed from a
/// [`ProofSession`](jolt_kernels::ProofSession) carry (the uni-skip-remainder
/// / precommitted-span pattern) — driven end to end (head → prepare → round
/// loop → typed extraction → per-member `park_residue` → shape validation →
/// final-claim self-check → finish) and byte-compared against the generated
/// `verify_clear` on a twin transcript. A second toy batch pairs a
/// full-window member with a head-aligned shorter member (`offset = 0`,
/// trailing dummy rounds), locking the engine's delayed `finish_rounds`
/// bookkeeping through the generated driver.
#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod twin_tests {
    use core::marker::PhantomData;

    use jolt_claims::protocols::jolt::{
        JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims::{opening, NoChallenges, OutputClaims as _, SymbolicSumcheck};
    use jolt_field::{Fr, JoltField, One, Ring, Zero, F128};
    use jolt_kernels::{
        JoltPlane, KernelError, KernelSlots, PrepareKernel, ProofSession, ProverInputs,
        SumcheckKernel, SumcheckKernelError,
    };
    use jolt_poly::{Polynomial, UnivariatePoly};
    use jolt_sumcheck::{ClearSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckError};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::{
        ConcreteSumcheck, SumcheckBatch, SumcheckInputClaims, SumcheckOutputClaims,
    };
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
            $(, head = $head:expr)?
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
            // The SumcheckBatch derive's aggregates require Allocative of
            // every member's outputs under the expanding crate's
            // `allocative` feature (the profile harness's flamegraphs).
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

                $(toy_relation!(@head $head);)?

                $(
                    fn instance_point_offset(
                        &self,
                        _batch_num_vars: usize,
                    ) -> Result<usize, VerifierError> {
                        Ok(0)
                    }

                    /// The engine halves an inactive member's claim once per
                    /// round, so the head-aligned member's final batch claim
                    /// is the fully bound (padded-scale) table value with the
                    /// trailing dummy rounds halved back out.
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
        (@head $head:expr) => {
            fn instance_point_offset(
                &self,
                _batch_num_vars: usize,
            ) -> Result<usize, VerifierError> {
                Ok(0)
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

    /// The head-aligned twin batch: a full-window member plus a shorter
    /// member active from round 0, whose final bind the engine delivers only
    /// after the trailing dummy rounds (the delayed `finish_rounds` path).
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
        // `log_residue` records a typed `JoltRelationId`, so the toy kernel is
        // pinned to jolt-family relations.
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

    /// The `park_residue` call order — the toy kernels' residue is a log
    /// entry, pinning that the driver consumes every present member into the
    /// session hook after extraction.
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

    /// Mints the head-aligned member's kernel at the dummy-round padding
    /// scale: a head-aligned member is active from round 0 at
    /// `input_claim · 2^(max − rounds)`, so its table must sum to the padded
    /// claim (see `BatchPrelude::new`).
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

    /// The gamma kernel is a `ProofSession` carry, parked by the toy front
    /// before `prove` — the uni-skip-remainder / precommitted-span pattern. A
    /// missing carry is a proof-time `KernelError`.
    struct ParkedToyGamma(DenseKernel<ToyGamma<Fr>>);

    // Session-inserted test state must be `MaybeAllocative`; self-sized
    // visitation is plenty for twin-lock scaffolding.
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

        // The toy kernel is a `SumcheckKernel`, so the mid-stage snapshot's
        // `MaybeAllocative` supertrait reaches it too.
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

    toy_relation!(
        EpsilonSymbolic,
        ToyEpsilon,
        ToyEpsilonInputs,
        ToyEpsilonOutputs,
        rel = RegistersReadWriteChecking,
        output = RegistersVal,
        input = UnexpandedPC,
        head = true
    );

    #[derive(SumcheckBatch)]
    struct ToyBinarySumchecks<F: JoltField> {
        alpha: ToyAlpha<F>,
        beta: Option<ToyBeta<F>>,
        epsilon: ToyEpsilon<F>,
    }

    toy_binary_sumchecks_members!(impl_stage_prover);

    #[cfg_attr(
        feature = "allocative",
        derive(allocative::Allocative),
        allocative(bound = "F: JoltField, R")
    )]
    struct TableKernel<F: JoltField, R> {
        evals: Vec<F>,
        num_rounds: usize,
        _relation: PhantomData<fn() -> R>,
    }

    impl<F: JoltField, R> TableKernel<F, R> {
        fn bind(&mut self, challenge: F) {
            let half = self.evals.len() / 2;
            for i in 0..half {
                let low = self.evals[i];
                let high = self.evals[i + half];
                self.evals[i] = low + challenge * (high - low);
            }
            self.evals.truncate(half);
        }
    }

    impl<F: JoltField, R> ProveRounds<F> for TableKernel<F, R> {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn prove_round(
            &mut self,
            bind: Option<F>,
            round: usize,
            previous_claim: F,
        ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
            if let Some(challenge) = bind {
                self.bind(challenge);
            }
            let half = self.evals.len() / 2;
            let eval_0: F = self.evals[..half].iter().copied().sum();
            let eval_1: F = self.evals[half..].iter().copied().sum();
            let actual = eval_0 + eval_1;
            if actual != previous_claim {
                return Err(SumcheckError::RoundCheckFailed {
                    round,
                    expected: previous_claim,
                    actual,
                });
            }
            Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
        }

        fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
            self.bind(bind);
            Ok(())
        }
    }

    impl<F: JoltField, R: ConcreteSumcheck<F>> SumcheckKernel<F> for TableKernel<F, R> {
        type Relation = R;

        fn output_claims(
            &mut self,
            _inputs: &SumcheckInputClaims<F, R>,
        ) -> Result<SumcheckOutputClaims<F, R>, SumcheckKernelError<F>> {
            assert_eq!(self.evals.len(), 1);
            SumcheckOutputClaims::<F, R>::from_opening_values(|_| Some(self.evals[0]))
                .map_err(SumcheckKernelError::from)
        }
    }

    struct TablePrepare<F> {
        alpha: Vec<F>,
        beta: Vec<F>,
        epsilon: Vec<F>,
    }

    macro_rules! table_prepare {
        ($($relation:ident => $table:ident),+ $(,)?) => {$(
            impl<F: JoltField> PrepareKernel<F, $relation<F>> for TablePrepare<F> {
                fn prepare(
                    &self,
                    _session: &mut ProofSession,
                    _witness: &dyn JoltWitnessPlane<F>,
                    inputs: ProverInputs<'_, F, $relation<F>>,
                ) -> Result<Box<dyn SumcheckKernel<F, Relation = $relation<F>>>, KernelError<F>> {
                    Ok(Box::new(TableKernel {
                        evals: self.$table.clone(),
                        num_rounds: inputs.relation.rounds(),
                        _relation: PhantomData,
                    }))
                }
            }
        )+};
    }

    table_prepare!(ToyAlpha => alpha, ToyBeta => beta, ToyEpsilon => epsilon);

    struct NoWitness;

    impl NoWitness {
        fn unavailable() -> WitnessError {
            WitnessError::UnavailableView {
                label: "toy driver twin witness",
            }
        }
    }

    impl<F: JoltField> JoltWitnessOracle<F> for NoWitness {
        fn shape(
            &self,
            _id: jolt_claims::protocols::jolt::JoltPolynomialId,
        ) -> Result<Shape, WitnessError> {
            Err(Self::unavailable())
        }

        fn oracle_table(
            &self,
            _id: jolt_claims::protocols::jolt::JoltPolynomialId,
        ) -> Result<Vec<F>, WitnessError> {
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

        // Verifier twin: generated draw + composed verify_clear (which runs the
        // derive-opening-points and expected-final-claim checks internally) +
        // output-claim absorbs.
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

    /// The head-aligned driver path: a shorter member active from the batch's
    /// FIRST round alongside a full-window member. Its final bind arrives only
    /// through the engine's delayed `finish_rounds` delivery — after the
    /// trailing dummy rounds — yet typed extraction and `park_residue` see the
    /// kernel fully bound, and the twin `verify_clear` reproduces the
    /// transcript byte for byte.
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

        let error = crate::driver::prepare_optional::<Fr, ToyBeta<Fr>, JoltPlane, _>(
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

    fn drive_binary(beta: bool) {
        const LABEL: &[u8] = b"prove-driver-binary-twin";
        let tables = TablePrepare {
            alpha: [
                0xc763_925a_819b_064e_758c_e429_37f1_d2b0,
                0x1947_bdef_523a_816c_e298_6f03_b571_da42,
                0xa938_712d_04fc_56be_8371_c29a_d564_0fbe,
                0x6bd4_e103_892c_7afe_d912_34cb_56ea_807f,
                0xf37c_98a2_51e6_d0b4_7ab3_0e19_c428_65df,
                0x8291_6fda_c473_0b5e_3d86_e2ac_7059_41bf,
                0x4ab7_2e90_d615_38fc_9a01_7c63_e8b2_f54d,
                0x93e1_a47c_06bd_5f28_c731_9de2_84a0_6bf5,
            ]
            .map(F128::from_raw)
            .to_vec(),
            beta: [
                0xd127_8c6e_935a_0bf4_68e2_a9c7_301f_5db6,
                0x307e_d8a1_b4c6_92f5_7a63_0e19_d52b_84cf,
                0x75ac_9e20_d318_6bf4_0297_fa61_8dce_53b9,
                0xe631_4a8d_7b09_c2f5_946e_13ba_50c7_8df2,
            ]
            .map(F128::from_raw)
            .to_vec(),
            epsilon: [
                0x5f19_c6ae_82db_3407_a698_1de3_7b42_f0c5,
                0x826b_0d94_37fe_a152_c410_8e6d_b975_23af,
                0xb573_29e1_6c0a_8df4_327b_f691_0eac_45d8,
                0x39e4_a2c8_70bd_165f_ca83_9b21_e64d_057a,
            ]
            .map(F128::from_raw)
            .to_vec(),
        };
        let sumchecks = ToyBinarySumchecks {
            alpha: ToyAlpha::new(3),
            beta: beta.then(|| ToyBeta::new(2)),
            epsilon: ToyEpsilon::new(2),
        };
        let inputs = ToyBinaryInputClaims {
            alpha: ToyAlphaInputs {
                claimed_sum: tables.alpha.iter().copied().sum(),
            },
            beta: beta.then(|| ToyBetaInputs {
                claimed_sum: tables.beta.iter().copied().sum(),
            }),
            epsilon: ToyEpsilonInputs {
                claimed_sum: tables.epsilon.iter().copied().sum(),
            },
        };
        let input_points = sumchecks.empty_input_points();
        let mut prover_transcript = Blake2bTranscript::new(LABEL);
        let challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();
        let proved = sumchecks
            .prove(
                &tables,
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &NoWitness,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<F128, F128>::new(),
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(LABEL);
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

        let mut head_transcript = Blake2bTranscript::new(LABEL);
        let head_challenges = sumchecks.draw_challenges(&mut head_transcript).unwrap();
        let (_, coefficients) = sumchecks
            .begin_batch(
                &inputs,
                &head_challenges,
                &mut ClearSumcheckRecorder::<F128, F128>::new(),
                &mut head_transcript,
            )
            .unwrap();
        let r = &proved.output_points.alpha.value;
        assert_eq!(r.len(), 3);
        assert_eq!(proved.output_points.epsilon.value, r[..2]);
        let alpha_value = Polynomial::new(tables.alpha).evaluate(r);
        let epsilon_value = Polynomial::new(tables.epsilon).evaluate(&r[..2]);
        let epsilon_scale = F128::one() - r[2];
        let mut expected =
            coefficients.alpha * alpha_value + coefficients.epsilon * epsilon_scale * epsilon_value;
        assert_eq!(proved.output_claims.alpha.value, alpha_value);
        assert_eq!(proved.output_claims.epsilon.value, epsilon_value);
        if beta {
            let beta_value = Polynomial::new(tables.beta).evaluate(&r[1..]);
            let beta_scale = F128::one() - r[0];
            expected += coefficients.beta.unwrap() * beta_scale * beta_value;
            assert_eq!(
                proved.output_claims.beta.as_ref().unwrap().value,
                beta_value
            );
            assert_eq!(proved.output_points.beta.as_ref().unwrap().value, r[1..]);
        } else {
            assert!(proved.output_claims.beta.is_none());
            assert!(proved.output_points.beta.is_none());
        }
        assert_eq!(proved.final_claim, expected);

        let verify_changed =
            |changed_inputs: &ToyBinaryInputClaims<F128>,
             changed_outputs: &ToyBinaryOutputClaims<F128>| {
                let mut transcript = Blake2bTranscript::new(LABEL);
                let challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
                sumchecks.verify_clear(
                    changed_inputs,
                    &input_points,
                    &challenges,
                    changed_outputs,
                    &proved.recorded.proof,
                    &mut transcript,
                    0,
                )
            };
        let mut changed_inputs = inputs.clone();
        changed_inputs.epsilon.claimed_sum += F128::one();
        assert!(verify_changed(&changed_inputs, &proved.output_claims).is_err());

        assert!(!coefficients.epsilon.is_zero());
        assert!(!epsilon_scale.is_zero());
        let mut changed_outputs = proved.output_claims.clone();
        changed_outputs.epsilon.value += F128::one();
        assert!(verify_changed(&inputs, &changed_outputs).is_err());

        assert!(!coefficients.alpha.is_zero());
        let alpha_scale = F128::one();
        assert!(!alpha_scale.is_zero());
        let mut changed_outputs = proved.output_claims.clone();
        changed_outputs.alpha.value += F128::one();
        assert!(verify_changed(&inputs, &changed_outputs).is_err());
    }

    #[test]
    fn binary_driver_twin_with_present_option_member() {
        drive_binary(true);
    }

    #[test]
    fn binary_driver_twin_with_absent_option_member() {
        drive_binary(false);
    }
}
