use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::{
    bytecode::{
        read_raf_checking::ReadRafSumcheckVerifier as BytecodeReadRafSumcheckVerifier,
        BytecodePreprocessing,
    },
    claim_reductions::{
        AdviceClaimReductionPhase1Verifier, AdviceClaimReductionPhase2Verifier,
        HammingWeightClaimReductionVerifier, IncReductionSumcheckVerifier,
        InstructionLookupsClaimReductionSumcheckVerifier, RamRaReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::ReadRafSumcheckVerifier as LookupsReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        self, hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice, RAMPreprocessing,
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
    ProverDebugInfo, Serializable,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{
        DoryOpeningState, OpeningAccumulator, OpeningPoint, SumcheckId, VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// Phase-bridge randomness for two-phase advice claim reduction.
    advice_reduction_gamma_trusted: Option<F>,
    advice_reduction_gamma_untrusted: Option<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>
    JoltVerifier<'a, F, PCS, ProofTranscript>
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &proof.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        #[cfg(test)]
        let mut transcript = ProofTranscript::new(b"Jolt");
        #[cfg(not(test))]
        let transcript = ProofTranscript::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());
        let one_hot_params = OneHotParams::new_with_log_k_chunk(
            proof.log_k_chunk,
            proof.lookups_ra_virtual_log_k_chunk,
            proof.bytecode_K,
            proof.ram_K,
        );

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_gamma_trusted: None,
            advice_reduction_gamma_untrusted: None,
            spartan_key,
            one_hot_params,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify");
        // Parameters are computed from trace length as needed

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript.append_serializable(commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(trusted_advice_commitment);
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        self.verify_stage6()?;
        // Advice claims are now reduced in Stage 6 and verified in Stage 8 batch opening
        self.verify_stage7()?;
        self.verify_stage8()?;

        Ok(())
    }

    fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
        );
        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, &mut self.transcript);
        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &spartan_product_virtual_remainder,
                &ram_raf_evaluation,
                &ram_read_write_checking,
                &ram_output_check,
                &instruction_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            vec![
                &spartan_shift as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            ram::read_write_checking::needs_single_advice_opening(self.proof.trace_length),
        );
        let initial_ram_state = ram::gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.ram,
            &self.program_io,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);
        // Note: RamHammingBooleanity moved to Stage 6 so it shares r_cycle_stage6
        let ram_ra_reduction = RamRaReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = LookupsReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_ra_reduction,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
            &self.preprocessing.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // RamHammingBooleanity - uses r_cycle from prior sumcheck
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);

        // Booleanity: combines instruction, bytecode, and ram booleanity into one
        // (extracts r_address and r_cycle from Stage 5 internally)
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);

        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction = IncReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
        let trusted_advice_phase1 = AdviceClaimReductionPhase1Verifier::new_trusted(
            &self.program_io.memory_layout,
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        if let Some(ref v) = trusted_advice_phase1 {
            self.advice_reduction_gamma_trusted = Some(v.gamma());
        }
        let untrusted_advice_phase1 = AdviceClaimReductionPhase1Verifier::new_untrusted(
            &self.program_io.memory_layout,
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        if let Some(ref v) = untrusted_advice_phase1 {
            self.advice_reduction_gamma_untrusted = Some(v.gamma());
        }

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &bytecode_read_raf,
            &ram_hamming_booleanity,
            &booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref advice) = trusted_advice_phase1 {
            instances.push(advice);
        }
        if let Some(ref advice) = untrusted_advice_phase1 {
            instances.push(advice);
        }

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // 1. Get r_cycle_stage6 from accumulator (extract from any RA claim's opening point)
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // 3. Verify Stage 7 batched sumcheck (address rounds only).
        // Includes HammingWeightClaimReduction plus Phase 2 advice reduction instances (if needed).
        let trusted_advice_phase2 = self.advice_reduction_gamma_trusted.and_then(|gamma| {
            AdviceClaimReductionPhase2Verifier::new_trusted(
                &self.program_io.memory_layout,
                self.proof.trace_length,
                gamma,
                &self.opening_accumulator,
            )
        });
        let untrusted_advice_phase2 = self.advice_reduction_gamma_untrusted.and_then(|gamma| {
            AdviceClaimReductionPhase2Verifier::new_untrusted(
                &self.program_io.memory_layout,
                self.proof.trace_length,
                gamma,
                &self.opening_accumulator,
            )
        });

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];
        if let Some(ref v) = trusted_advice_phase2 {
            instances.push(v);
        }
        if let Some(ref v) = untrusted_advice_phase2 {
            instances.push(v);
        }
        let r_address_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        // 4. Collect all claims for DoryOpeningState
        let mut claims = Vec::new();
        let mut polynomials = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncReduction,
        );
        let (_, rd_inc_claim) = self
            .opening_accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::IncReduction);

        // Apply Lagrange factor for dense polys
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        claims.push(ram_inc_claim * lagrange_factor);
        claims.push(rd_inc_claim * lagrange_factor);
        polynomials.push(CommittedPolynomial::RamInc);
        polynomials.push(CommittedPolynomial::RdInc);

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::RamRa(i));
        }

        // 5. Build unified opening point: (r_address_stage7 || r_cycle_stage6)
        let mut r_address_be = r_address_stage7.clone();
        r_address_be.reverse();

        // Extract r_cycle from Booleanity (same source as HammingWeightClaimReduction uses)
        let (unified_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_cycle_stage6 = &unified_point.r[log_k_chunk..];

        let opening_point = [r_address_be.as_slice(), r_cycle_stage6].concat();

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with Main context dimensions so they can be batched.
        // They have fewer variables than main polynomials, so we apply Lagrange factors.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
        {
            let advice_vars = advice_point.len();
            let mut r_le = opening_point.clone();
            r_le.reverse();
            let total_vars = r_le.len();
            let sigma = total_vars.div_ceil(2);
            let (r_cols, r_rows) = r_le.split_at(sigma);

            let sigma_a = advice_vars.div_ceil(2);
            let nu_a = advice_vars - sigma_a;

            let row_factor: F = r_rows
                .iter()
                .skip(nu_a)
                .map(|r| F::one() - (*r).into())
                .product();
            let col_prefix_factor: F = r_cols
                .iter()
                .skip(sigma_a)
                .map(|r| F::one() - (*r).into())
                .product();

            claims.push(advice_claim * row_factor * col_prefix_factor);
            polynomials.push(CommittedPolynomial::TrustedAdvice);
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
        {
            let advice_vars = advice_point.len();
            let mut r_le = opening_point.clone();
            r_le.reverse();
            let total_vars = r_le.len();
            let sigma = total_vars.div_ceil(2);
            let (r_cols, r_rows) = r_le.split_at(sigma);

            let sigma_a = advice_vars.div_ceil(2);
            let nu_a = advice_vars - sigma_a;

            let row_factor: F = r_rows
                .iter()
                .skip(nu_a)
                .map(|r| F::one() - (*r).into())
                .product();
            let col_prefix_factor: F = r_cols
                .iter()
                .skip(sigma_a)
                .map(|r| F::one() - (*r).into())
                .product();

            claims.push(advice_claim * row_factor * col_prefix_factor);
            polynomials.push(CommittedPolynomial::UntrustedAdvice);
        }

        // 6. Sample gamma and compute powers for RLC
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // 7. Store DoryOpeningState for Stage 8
        self.opening_accumulator.dory_opening_state = Some(DoryOpeningState {
            opening_point,
            gamma_powers,
            claims,
            polynomials,
        });

        Ok(())
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        let state = self
            .opening_accumulator
            .dory_opening_state
            .as_ref()
            .expect("Stage 7 must be called before Stage 8");

        // Build commitments map
        let mut commitments_map = HashMap::new();
        for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
            .into_iter()
            .zip_eq(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        // Add advice commitments if they're part of the batch
        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomials
                .contains(&CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            if state
                .polynomials
                .contains(&CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        // Compute joint commitment: Σ γ_i · C_i
        let joint_commitment =
            self.compute_joint_commitment_new::<PCS>(&mut commitments_map, state);

        // Test assertion
        #[cfg(test)]
        if let Some(ref prover_joint_commitment) = self.proof.joint_commitment_for_test {
            assert_eq!(
                joint_commitment, *prover_joint_commitment,
                "joint commitment mismatch"
            );
        }

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = state
            .gamma_powers
            .iter()
            .zip(state.claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Verify opening
        PCS::verify(
            &self.proof.joint_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &state.opening_point,
            &joint_claim,
            &joint_commitment,
        )
        .context("Stage 8")
    }

    /// Compute joint commitment from DoryOpeningState.
    fn compute_joint_commitment_new<PCS2: CommitmentScheme<Field = F>>(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS2::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> PCS2::Commitment {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = HashMap::new();
        for (gamma, poly) in state.gamma_powers.iter().zip(state.polynomials.iter()) {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS2::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();

        PCS2::combine_commitments(&commitments, &coeffs)
    }

    // Note: verify_trusted_advice_opening_proofs and verify_untrusted_advice_opening_proofs
    // have been removed. Advice claims are now reduced via AdviceClaimReduction in Stage 6
    // and verified as part of the batched Stage 8 opening proof.
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}
