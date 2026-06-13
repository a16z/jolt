//! Transpilable Verifier: generic mirror of `JoltVerifier` for symbolic execution.
//!
//! # Temporary Module
//!
//! **This module is intended to be temporary.** The long-term goal is to make
//! `verifier.rs` itself generic over the accumulator type, eliminating the need
//! for this separate file. This would ensure transpilation automatically stays
//! in sync with verifier changes.
//!
//! # Overview
//!
//! A non-ZK verifier for stages 1–7, generic over two seams the transpiler swaps:
//! - `A: AbstractVerifierOpeningAccumulator<F>` — `VerifierOpeningAccumulator<F>` for
//!   real verification, `AstOpeningAccumulator` for symbolic transpilation.
//! - the transcript, taken as `&mut impl VerifierFs<F>` in [`Self::verify`] — a real
//!   spongefish `VerifierState` over the proof's NARG, or a symbolic `VerifierFs`
//!   implementation replaying pre-parsed NARG frames as AST variables.
//!
//! Unlike `JoltVerifier`, this struct holds the proof's *structural* fields directly
//! (no `JoltProof<F, C, PCS, H>`, hence no sponge type parameter): the symbolic path
//! has no real sponge, and the verification logic only needs `trace_length`, `ram_K`,
//! the configs, and the NARG (owned by the caller-built transcript). The non-ZK
//! `opening_claims` are pre-seeded into `A` by the caller, exactly as
//! `JoltVerifier::new` seeds its accumulator.
//!
//! ## Scope
//!
//! Stages 1–7 only (all sumcheck stages), **non-ZK proofs only**. Stage 8 (Dory PCS
//! verification) is NOT transpiled: it requires native elliptic-curve operations that
//! the target proving system performs natively. In non-ZK mode every NARG frame is
//! consumed by the end of stage 7 (stage 8 only absorbs/squeezes), so callers holding
//! a concrete `VerifierState` can — and the parity test does — `check_eof` afterwards.
//!
//! The per-stage methods are `pub` (unlike `JoltVerifier`) so the transpiler driver
//! can set witness-naming context between stages; `verify()` runs them all in order.

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
use crate::poly::opening_proof::AbstractVerifierOpeningAccumulator;
use crate::subprotocols::booleanity::{
    BooleanityAddressSumcheckVerifier, BooleanityCycleSumcheckVerifier, BooleanitySumcheckParams,
};
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcript_msgs::VerifierFs;
use crate::utils::{errors::ProofVerifyError, math::Math};
use crate::zkvm::config::{OneHotConfig, OneHotParams, ReadWriteConfig};
use crate::zkvm::{
    bytecode::read_raf_checking::{
        BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
        BytecodeReadRafSumcheckParams,
    },
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, BytecodeClaimReductionParams,
        BytecodeClaimReductionVerifier, HammingWeightClaimReductionVerifier,
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        PrecommittedClaimReduction, PrecommittedParams, ProgramImageClaimReductionParams,
        ProgramImageClaimReductionVerifier, RamRaClaimReductionSumcheckVerifier,
        RegistersClaimReductionSumcheckVerifier,
    },
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    r1cs::key::UniformSpartanKey,
    ram::{
        compute_max_ram_K, compute_min_ram_K,
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier, val_check::RamValCheckSumcheckVerifier,
        verifier_accumulate_advice, verifier_accumulate_program_image,
    },
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing},
};
use std::marker::PhantomData;
use tracer::JoltDevice;

/// The proof's structural (non-NARG) fields needed by stages 1–7. Mirror of the
/// corresponding `JoltProof` fields; the NARG itself lives in the caller-built
/// transcript, and the non-ZK `opening_claims` are pre-seeded into the accumulator.
#[derive(Clone, Debug)]
pub struct TranspilableProofData {
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

impl TranspilableProofData {
    /// Structural fields extracted from a proof — single source for the replay
    /// entry points.
    pub fn from_proof<
        F: JoltField,
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F>,
        H: jolt_transcript::DuplexSpongeInterface,
    >(
        proof: &crate::zkvm::proof_serialization::JoltProof<F, C, PCS, H>,
    ) -> Self {
        Self {
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: proof.rw_config.clone(),
            one_hot_config: proof.one_hot_config.clone(),
            dory_layout: proof.dory_layout,
        }
    }
}

pub struct TranspilableVerifier<
    'a,
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    A: AbstractVerifierOpeningAccumulator<F>,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof_data: TranspilableProofData,
    /// Witness-polynomial commitments, decoded from the NARG (one `read_slice` frame)
    /// at the start of `verify` — mirrors `JoltVerifier::commitments`.
    pub commitments: Vec<PCS::Commitment>,
    /// Untrusted-advice commitment, decoded from the NARG presence frame (length 0/1).
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
    pub opening_accumulator: A,
    /// Advice claim reduction spans stages 6b and 7; verifier state cached between them.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    /// Bytecode claim reduction spans stages 6b and 7 in committed mode.
    bytecode_reduction_verifier: Option<BytecodeClaimReductionVerifier<F>>,
    /// Program-image claim reduction spans stages 6b and 7 in committed mode.
    program_image_reduction_verifier: Option<ProgramImageClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    _curve: PhantomData<fn() -> C>,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F>,
        A: AbstractVerifierOpeningAccumulator<F>,
    > TranspilableVerifier<'a, F, C, PCS, A>
{
    /// Validates the structural proof data and constructs the verifier — the same
    /// checks as `JoltVerifier::new`. The caller supplies the accumulator pre-seeded
    /// with the proof's `opening_claims` (real path) or symbolic claims (transpiler).
    ///
    /// `untrusted_advice_commitment` starts `None`; it is decoded from the NARG
    /// presence frame during `verify`, mirroring `JoltVerifier`.
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
        proof_data: TranspilableProofData,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        opening_accumulator: A,
    ) -> Result<Self, ProofVerifyError> {
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        if !proof_data.trace_length.is_power_of_two()
            || proof_data.trace_length > preprocessing.shared.max_padded_trace_length
        {
            return Err(ProofVerifyError::InvalidTraceLength(
                proof_data.trace_length,
                preprocessing.shared.max_padded_trace_length,
            ));
        }

        // Truncate trailing zero bytes from outputs, matching `JoltVerifier::new` —
        // the Fiat-Shamir instance must be computed over the truncated outputs.
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let spartan_key = UniformSpartanKey::new(proof_data.trace_length.next_power_of_two());

        proof_data
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        let ram_preprocessing = crate::zkvm::ram::RAMPreprocessing {
            min_bytecode_address: preprocessing.shared.program_meta.min_bytecode_address,
            bytecode_words: vec![0; preprocessing.shared.program_meta.program_image_len_words],
        };
        let min_ram_K = compute_min_ram_K(&ram_preprocessing, &preprocessing.shared.memory_layout);
        let max_ram_K = compute_max_ram_K(&preprocessing.shared.memory_layout);
        if !proof_data.ram_K.is_power_of_two()
            || proof_data.ram_K < min_ram_K
            || proof_data.ram_K > max_ram_K
        {
            return Err(ProofVerifyError::InvalidRamK {
                got: proof_data.ram_K,
                min: min_ram_K,
                max: max_ram_K,
            });
        }

        proof_data
            .rw_config
            .validate(proof_data.trace_length.log_2(), proof_data.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        let bytecode_K = preprocessing.shared.bytecode_size();
        let one_hot_params =
            OneHotParams::from_config(&proof_data.one_hot_config, bytecode_K, proof_data.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof_data,
            commitments: Vec::new(),
            untrusted_advice_commitment: None,
            preprocessing,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            bytecode_reduction_verifier: None,
            program_image_reduction_verifier: None,
            spartan_key,
            one_hot_params,
            _curve: PhantomData,
        })
    }

    #[inline]
    fn main_total_vars(&self) -> usize {
        let trace_log_t = self.proof_data.trace_length.log_2();
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        JoltSharedPreprocessing::<PCS>::max_total_vars_from_candidates(
            trace_log_t + log_k_chunk,
            self.preprocessing.shared.precommitted_candidate_total_vars(
                self.preprocessing.shared.program.is_committed(),
                self.trusted_advice_commitment.is_some(),
                self.untrusted_advice_commitment.is_some(),
            ),
        )
    }

    /// Verify stages 1–7 against `transcript`, which the caller has already seeded with
    /// the Fiat-Shamir instance (for the real path: `verifier_transcript(b"Jolt",
    /// fiat_shamir_instance(..), H::default(), &proof.narg)` computed over the TRUNCATED
    /// `program_io` — use `self.program_io` after construction).
    ///
    /// Mirrors `JoltVerifier::verify_inner` from the Dory-context guard through stage 7;
    /// stage 8 and `check_eof` are the caller's concern (see module docs).
    pub fn verify(&mut self, transcript: &mut impl VerifierFs<F>) -> Result<(), ProofVerifyError> {
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.proof_data.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(self.proof_data.dory_layout),
        );

        self.read_commitment_frames(transcript)?;

        self.verify_stage1(transcript)
            .inspect_err(|e| tracing::error!("Stage 1: {e}"))?;
        self.verify_stage2(transcript)
            .inspect_err(|e| tracing::error!("Stage 2: {e}"))?;
        self.verify_stage3(transcript)
            .inspect_err(|e| tracing::error!("Stage 3: {e}"))?;
        self.verify_stage4(transcript)
            .inspect_err(|e| tracing::error!("Stage 4: {e}"))?;
        self.verify_stage5(transcript)
            .inspect_err(|e| tracing::error!("Stage 5: {e}"))?;
        self.verify_stage6(transcript)
            .inspect_err(|e| tracing::error!("Stage 6: {e}"))?;
        self.verify_stage7(transcript)
            .inspect_err(|e| tracing::error!("Stage 7: {e}"))?;
        // Stage 8 (PCS) is not transpiled — see module docs.

        Ok(())
    }

    /// Pre-stage NARG frames + shared-commitment absorbs, mirroring
    /// `JoltVerifier::verify_inner` between the Dory guard and stage 1.
    pub fn read_commitment_frames(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        // Witness-polynomial commitments: ONE frame (matching the prover's single
        // `write_commitments`), absorbed in the process.
        self.commitments = transcript
            .read_commitments()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        // Untrusted-advice presence frame (length-0/1 vec) → Option. The prover ALWAYS
        // writes this frame, so the read position is the same in both cases.
        let untrusted_advice: Vec<PCS::Commitment> = transcript
            .read_commitments()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        if untrusted_advice.len() > 1 {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        self.untrusted_advice_commitment = untrusted_advice.into_iter().next();

        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            transcript.absorb_commitment(trusted_advice_commitment);
        }
        if let Some(trusted_bytecode) = self.preprocessing.shared.program.bytecode_commitments() {
            for commitment in &trusted_bytecode.commitments {
                transcript.absorb_commitment(commitment);
            }
        }
        if self.preprocessing.shared.program.is_committed() {
            let trusted = self.preprocessing.shared.program.as_committed()?;
            transcript.absorb_commitment(&trusted.program_image_commitment);
        }
        Ok(())
    }

    pub fn verify_stage1(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let (uni_skip_params, _uni_skip_challenge, _zk_readback) =
            verify_stage1_uni_skip::<F, C, _, A>(
                false,
                &self.spartan_key,
                &mut self.opening_accumulator,
                transcript,
            )?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof_data.trace_length,
            &uni_skip_params,
            &self.opening_accumulator,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![&spartan_outer_remaining];

        let _r_stage1 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    pub fn verify_stage2(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let (uni_skip_params, _uni_skip_challenge, _zk_readback) =
            verify_stage2_uni_skip::<F, C, _, A>(false, &mut self.opening_accumulator, transcript)?;

        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            transcript,
            &self.one_hot_params,
            self.proof_data.trace_length,
            &self.proof_data.rw_config,
        );

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof_data.trace_length,
            uni_skip_params.clone(),
            &self.opening_accumulator,
        );

        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof_data.trace_length,
            &self.opening_accumulator,
            transcript,
        );

        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            self.proof_data.trace_length,
            &self.proof_data.rw_config,
            &self.opening_accumulator,
        );

        let ram_output_check = OutputSumcheckVerifier::new(
            self.proof_data.ram_K,
            &self.program_io,
            transcript,
            self.proof_data.trace_length,
            &self.proof_data.rw_config,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![
            &ram_read_write_checking,
            &spartan_product_virtual_remainder,
            &instruction_claim_reduction,
            &ram_raf_evaluation,
            &ram_output_check,
        ];

        let _r_stage2 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    pub fn verify_stage3(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof_data.trace_length.log_2(),
            &self.opening_accumulator,
            transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof_data.trace_length,
            &self.opening_accumulator,
            transcript,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![
            &spartan_shift,
            &spartan_instruction_input,
            &spartan_registers_claim_reduction,
        ];

        let _r_stage3 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    pub fn verify_stage4(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof_data.trace_length,
            &self.opening_accumulator,
            transcript,
            &self.proof_data.rw_config,
        );
        verifier_accumulate_advice::<F, A>(
            self.proof_data.ram_K,
            &self.program_io,
            self.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
        );
        if self.preprocessing.shared.program.is_committed() {
            verifier_accumulate_program_image::<F>(
                self.proof_data.ram_K,
                &mut self.opening_accumulator,
            );
        }
        // Domain-separate the batching challenge.
        let ram_val_check_gamma: F = transcript.challenge_field();
        let initial_ram_state = if self.preprocessing.shared.program.is_full() {
            crate::zkvm::ram::gen_ram_initial_memory_state::<F>(
                self.proof_data.ram_K,
                &self.preprocessing.shared.program.as_full()?.ram,
                &self.program_io,
            )
        } else {
            vec![0u64; self.proof_data.ram_K]
        };
        let ram_preprocessing = if self.preprocessing.shared.program.is_full() {
            self.preprocessing.shared.program.as_full()?.ram.clone()
        } else {
            crate::zkvm::ram::RAMPreprocessing {
                min_bytecode_address: self.preprocessing.shared.program_meta.min_bytecode_address,
                bytecode_words: vec![
                    0;
                    self.preprocessing
                        .shared
                        .program_meta
                        .program_image_len_words
                ],
            }
        };
        let ram_val_check = RamValCheckSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            &ram_preprocessing,
            self.proof_data.trace_length,
            self.proof_data.ram_K,
            &self.proof_data.rw_config,
            ram_val_check_gamma,
            &self.opening_accumulator,
            self.preprocessing.shared.program.is_committed(),
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> =
            vec![&registers_read_write_checking, &ram_val_check];

        let _r_stage4 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    pub fn verify_stage5(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let n_cycle_vars = self.proof_data.trace_length.log_2();

        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof_data.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![
            &lookups_read_raf,
            &ram_ra_reduction,
            &registers_val_evaluation,
        ];

        let _r_stage5 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    pub fn verify_stage6(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let (bytecode_read_raf_params, booleanity_params) = self.verify_stage6a(transcript)?;
        self.verify_stage6b(transcript, bytecode_read_raf_params, booleanity_params)?;
        Ok(())
    }

    /// NOTE: the Dory main-embedding initialization lives here (not in
    /// `verify_stage6`) so drivers that run 6a/6b individually (the transpiler, for
    /// witness-naming context) get identical behavior to `verify_stage6`.
    pub fn verify_stage6a(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<
        (
            BytecodeReadRafSumcheckParams<F>,
            BooleanitySumcheckParams<F>,
        ),
        ProofVerifyError,
    > {
        let _ = DoryGlobals::initialize_main_with_log_embedding(
            self.one_hot_params.k_chunk,
            self.proof_data.trace_length,
            self.main_total_vars(),
            Some(self.proof_data.dory_layout),
        );
        let n_cycle_vars = self.proof_data.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafAddressSumcheckVerifier::new(
            &self.preprocessing.shared.program,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let booleanity = BooleanityAddressSumcheckVerifier::new(BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        ));
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> =
            vec![&bytecode_read_raf, &booleanity];

        let _r_stage6a = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )
        .inspect_err(|err| tracing::error!("Stage 6a: {err}"))?;

        Ok((bytecode_read_raf.into_params(), booleanity.into_params()))
    }

    pub fn verify_stage6b(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<(), ProofVerifyError> {
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let booleanity =
            BooleanityCycleSumcheckVerifier::new(booleanity_params, &self.opening_accumulator);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof_data.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof_data.trace_length,
            &self.opening_accumulator,
            transcript,
        );

        let main_total_vars =
            self.proof_data.trace_length.log_2() + self.one_hot_params.log_k_chunk;
        let precommitted_candidates = self.preprocessing.shared.precommitted_candidate_total_vars(
            self.preprocessing.shared.program.is_committed(),
            self.trusted_advice_commitment.is_some(),
            self.untrusted_advice_commitment.is_some(),
        );
        let precommitted_scheduling_reference =
            PrecommittedClaimReduction::<F>::scheduling_reference(
                main_total_vars,
                &precommitted_candidates,
            );

        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                self.program_io.memory_layout.max_trusted_advice_size as usize,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            ));
        }
        if self.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                self.program_io.memory_layout.max_untrusted_advice_size as usize,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            ));
        }
        if self.preprocessing.shared.program.is_committed() {
            let bytecode_chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            let bytecode_reduction_params = BytecodeClaimReductionParams::new(
                bytecode_read_raf_params.stage_gammas(),
                self.preprocessing.shared.bytecode_size(),
                bytecode_chunk_count,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
                transcript,
            );
            self.bytecode_reduction_verifier = Some(BytecodeClaimReductionVerifier::new(
                bytecode_reduction_params,
            ));

            let padded_len_words = self
                .preprocessing
                .shared
                .program_meta
                .committed_program_image_num_words(&self.program_io.memory_layout);
            let program_image_reduction_params = ProgramImageClaimReductionParams::new(
                &self.program_io,
                self.preprocessing.shared.program_meta.min_bytecode_address,
                padded_len_words,
                self.proof_data.ram_K,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            );
            self.program_image_reduction_verifier = Some(ProgramImageClaimReductionVerifier::new(
                program_image_reduction_params,
            ));
        }

        let bytecode_read_raf = BytecodeReadRafCycleSumcheckVerifier::new(
            bytecode_read_raf_params,
            &self.opening_accumulator,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![
            &bytecode_read_raf,
            &booleanity,
            &ram_hamming_booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }
        if let Some(ref reduction) = self.bytecode_reduction_verifier {
            instances.push(reduction);
        }
        if let Some(ref reduction) = self.program_image_reduction_verifier {
            instances.push(reduction);
        }

        let _r_stage6b = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )
        .inspect_err(|err| tracing::error!("Stage 6b: {err}"))?;

        Ok(())
    }

    pub fn verify_stage7(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(), ProofVerifyError> {
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, A>> = vec![&hw_verifier];
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            if advice_reduction_verifier_trusted
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                advice_reduction_verifier_trusted
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            if advice_reduction_verifier_untrusted
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                advice_reduction_verifier_untrusted
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(advice_reduction_verifier_untrusted);
            }
        }
        if let Some(bytecode_reduction_verifier) = self.bytecode_reduction_verifier.as_mut() {
            if bytecode_reduction_verifier
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                bytecode_reduction_verifier
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(bytecode_reduction_verifier);
            }
        }
        if let Some(program_image_reduction_verifier) =
            self.program_image_reduction_verifier.as_mut()
        {
            if program_image_reduction_verifier
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                program_image_reduction_verifier
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(program_image_reduction_verifier);
            }
        }

        let _r_stage7 = BatchedSumcheck::verify_standard::<F, A>(
            instances,
            &mut self.opening_accumulator,
            transcript,
        )?;

        Ok(())
    }
}
