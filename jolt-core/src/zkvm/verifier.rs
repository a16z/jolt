use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use ark_bn254::Fq;
use ark_grumpkin::Projective as GrumpkinProjective;

use crate::poly::commitment::{
    commitment_scheme::{CommitmentScheme, RecursionExt},
    hyrax::Hyrax,
};
use crate::poly::opening_proof::OpeningId;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier,
    claim_reductions::{
        HammingWeightClaimReductionVerifier, IncClaimReductionSumcheckVerifier,
        InstructionLookupsClaimReductionSumcheckVerifier, RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        self, hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice,
    },
    recursion::recursion_verifier::{RecursionVerifier, RecursionVerifierInput},
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    ProverDebugInfo,
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
use tracer::instruction::Instruction;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    PCS: RecursionExt<F>,
    ProofTranscript: Transcript,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, PCS, ProofTranscript>
where
    <PCS as RecursionExt<F>>::Hint: Send + Sync + Clone + 'static,
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
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

        // Validate configs from the proof
        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
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
        self.verify_trusted_advice_opening_proofs()?;
        self.verify_untrusted_advice_opening_proofs()?;
        self.verify_stage7()?;
        self.verify_stage8_with_recursion()?;

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
            &self.proof.rw_config,
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
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2()),
        );
        let initial_ram_state = ram::gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.shared.ram,
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
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
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
            &self.preprocessing.shared.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
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
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            vec![
                &bytecode_read_raf as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_hamming_booleanity,
                &booleanity,
                &ram_ra_virtual,
                &lookups_ra_virtual,
                &inc_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Verify sumcheck (only log_k_chunk rounds)
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![&hw_verifier];
        let _r_address_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }

    /// Stage 8: Dory batch opening verification.
    #[allow(dead_code)]
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Apply Lagrange factor for dense polys
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Build commitments map
        let mut commitments_map = HashMap::new();
        for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
            .into_iter()
            .zip_eq(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        // Compute joint commitment: Σ γ_i · C_i
        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state);

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Verify opening
        PCS::verify(
            &self.proof.stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
        )
        .context("Stage 8")
    }

    /// Stage 8: PCS batch opening verification using a recursion hint (when supported by the PCS).
    fn verify_stage8_with_pcs_hint(
        &mut self,
        stage8_hint: &<PCS as RecursionExt<F>>::Hint,
    ) -> Result<(), anyhow::Error>
    where
        PCS: RecursionExt<F>,
    {
        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Apply Lagrange factor for dense polys
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Compute joint commitment: Σ γ_i · C_i
        // Use precomputed hint if available, otherwise compute directly
        let joint_commitment = if let Some(combine_hint) = &self.proof.stage8_combine_hint {
            // Use the precomputed hint (recursion-offloaded path)
            tracing::info!("[Homomorphic Combine] Verifier using precomputed combine_hint");
            PCS::combine_with_hint_fq12(combine_hint)
        } else {
            // Build commitments map and compute directly
            let mut commitments_map = HashMap::new();
            for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
                .into_iter()
                .zip_eq(&self.proof.commitments)
            {
                commitments_map.insert(polynomial, commitment.clone());
            }
            self.compute_joint_commitment(&mut commitments_map, &state)
        };

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Verify opening using the hint-based PCS verifier
        PCS::verify_with_hint(
            &self.proof.stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
            stage8_hint,
        )
        .context("Stage 8 (hint)")
    }

    /// Compute joint commitment for the batch opening.
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> PCS::Commitment {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();

        PCS::combine_commitments(&commitments, &coeffs)
    }

    /// Verify Stage 8 with recursion proof
    fn verify_stage8_with_recursion(&mut self) -> Result<(), anyhow::Error>
    where
        PCS: RecursionExt<F>,
        <PCS as RecursionExt<F>>::Hint: Clone,
    {
        // 1. Verify Dory proof with hints
        let hint = self
            .proof
            .stage9_pcs_hint
            .clone()
            .ok_or_else(|| anyhow::anyhow!("stage9_pcs_hint is required for recursion verification"))?;

        self.verify_stage8_with_pcs_hint(&hint)?;

        // 2. Extract data for RecursionVerifier
        let recursion_proof = &self.proof.recursion_proof;

        // Build RecursionVerifierInput from proof data
        // Note: This is a simplified version. In practice, this data would come from preprocessing
        // or be computed from the proof structure

        // Extract constraint counts from the opening claims in recursion_proof
        // We can infer the structure from the types of virtual polynomials
        let mut num_gt_exp = 0;
        let mut num_gt_mul = 0;
        let mut num_g1_scalar_mul = 0;

        // Count constraint types based on the virtual polynomial types in opening claims
        for (key, _) in &recursion_proof.opening_claims {
            match key {
                OpeningId::Virtual(poly, _) => {
                    match poly {
                        crate::zkvm::witness::VirtualPolynomial::RecursionBase(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionRhoPrev(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionRhoCurr(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionQuotient(_) => {
                            // These are GT exp constraints
                            num_gt_exp = num_gt_exp.max(match poly {
                                crate::zkvm::witness::VirtualPolynomial::RecursionBase(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionRhoPrev(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionRhoCurr(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionQuotient(i) => {
                                    *i + 1
                                }
                                _ => 0,
                            });
                        }
                        crate::zkvm::witness::VirtualPolynomial::RecursionMulLhs(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionMulRhs(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionMulResult(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionMulQuotient(_) => {
                            // These are GT mul constraints
                            num_gt_mul = num_gt_mul.max(match poly {
                                crate::zkvm::witness::VirtualPolynomial::RecursionMulLhs(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionMulRhs(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionMulResult(i)
                                | crate::zkvm::witness::VirtualPolynomial::RecursionMulQuotient(
                                    i,
                                ) => *i + 1,
                                _ => 0,
                            });
                        }
                        crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXA(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYA(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXT(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYT(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXANext(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYANext(_)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulIndicator(
                            _,
                        ) => {
                            // These are G1 scalar mul constraints
                            num_g1_scalar_mul = num_g1_scalar_mul.max(match poly {
                        crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXA(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYA(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXT(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYT(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulXANext(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulYANext(i)
                        | crate::zkvm::witness::VirtualPolynomial::RecursionG1ScalarMulIndicator(i) => *i + 1,
                        _ => 0,
                    });
                        }
                        _ => {} // Ignore other virtual polynomial types
                    }
                }
                _ => {} // Ignore non-virtual openings
            }
        }

        // Use metadata from proof instead of placeholders
        let metadata = &self.proof.stage10_recursion_metadata;

        let constraint_types = metadata.constraint_types.clone();
        let num_constraints = constraint_types.len();
        let num_constraints_padded = num_constraints.next_power_of_two();

        // The dense polynomial is created by the jagged bijection which compresses
        // the sparse matrix by removing redundant evaluations. We should use the
        // actual dense_num_vars from the metadata rather than trying to recalculate it.

        // Calculate the constraint system parameters for the verifier input
        // These are based on the original constraint matrix structure
        const NUM_POLY_TYPES: usize = 15; // PolyType::NUM_TYPES
        let num_rows_unpadded = NUM_POLY_TYPES * num_constraints_padded;
        let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
        let num_constraint_vars = 8; // All constraints padded to 8 variables
        let num_vars = num_s_vars + num_constraint_vars;

        let jagged_bijection = metadata.jagged_bijection.clone();
        let jagged_mapping = metadata.jagged_mapping.clone();
        let matrix_rows = metadata.matrix_rows.clone();

        let verifier_input = RecursionVerifierInput {
            constraint_types,
            num_vars,
            num_constraint_vars,
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            jagged_bijection,
            jagged_mapping,
            matrix_rows,
        };

        // 3. Verify recursion proof
        let recursion_verifier = RecursionVerifier::<Fq>::new(verifier_input);

        // Sample the same challenges from the main transcript that the prover did
        let _gamma: Fq = self.transcript.challenge_scalar();
        let _delta: Fq = self.transcript.challenge_scalar();

        type HyraxPCS = Hyrax<1, GrumpkinProjective>;

        // Setup Hyrax from proof metadata
        let dense_num_vars = metadata.dense_num_vars;
        let hyrax_prover_setup = <HyraxPCS as CommitmentScheme>::setup_prover(dense_num_vars);
        let hyrax_verifier_setup =
            <HyraxPCS as CommitmentScheme>::setup_verifier(&hyrax_prover_setup);

        // Add dense commitment to transcript (must match prover's order)
        self.transcript.append_serializable(&recursion_proof.dense_commitment);

        let verification_result = recursion_verifier
            .verify::<ProofTranscript, HyraxPCS>(
                recursion_proof,
                &mut self.transcript,
                &recursion_proof.dense_commitment,
                &hyrax_verifier_setup,
            )
            .map_err(|e| anyhow::anyhow!("Recursion verification failed: {:?}", e))?;

        if !verification_result {
            return Err(anyhow::anyhow!("Recursion proof verification failed"));
        }

        Ok(())
    }

    fn verify_trusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        if let Some(ref commitment) = self.trusted_advice_commitment {
            // Verify at RamValEvaluation point
            let Some(ref proof) = self.proof.trusted_advice_val_evaluation_proof else {
                return Err(anyhow::anyhow!(
                    "Trusted advice val evaluation proof not found"
                ));
            };
            let Some((point, eval)) = self
                .opening_accumulator
                .get_trusted_advice_opening(SumcheckId::RamValEvaluation)
            else {
                return Err(anyhow::anyhow!("Trusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}")
            })?;

            // Verify at RamValFinalEvaluation point - only if different from ValEvaluation
            if !self
                .proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2())
            {
                let Some(ref proof_val_final) = self.proof.trusted_advice_val_final_proof else {
                    return Err(anyhow::anyhow!("Trusted advice val final proof not found"));
                };
                let Some((point_val_final, eval_val_final)) = self
                    .opening_accumulator
                    .get_trusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                else {
                    return Err(anyhow::anyhow!(
                        "Trusted advice val final opening not found"
                    ));
                };
                PCS::verify(
                    proof_val_final,
                    &self.preprocessing.generators,
                    &mut self.transcript,
                    &point_val_final.r,
                    &eval_val_final,
                    commitment,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Trusted advice val final opening proof verification failed: {e:?}"
                    )
                })?;
            }
        }

        Ok(())
    }

    fn verify_untrusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        use crate::poly::opening_proof::SumcheckId;
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            // Verify at RamValEvaluation point
            let Some(ref proof) = self.proof.untrusted_advice_val_evaluation_proof else {
                return Err(anyhow::anyhow!(
                    "Untrusted advice val evaluation proof not found"
                ));
            };
            let Some((point, eval)) = self
                .opening_accumulator
                .get_untrusted_advice_opening(SumcheckId::RamValEvaluation)
            else {
                return Err(anyhow::anyhow!("Untrusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
            })?;

            // Verify at RamValFinalEvaluation point - only if different from ValEvaluation
            if !self
                .proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2())
            {
                let Some(ref proof_val_final) = self.proof.untrusted_advice_val_final_proof else {
                    return Err(anyhow::anyhow!(
                        "Untrusted advice val final proof not found"
                    ));
                };
                let Some((point_val_final, eval_val_final)) = self
                    .opening_accumulator
                    .get_untrusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                else {
                    return Err(anyhow::anyhow!(
                        "Untrusted advice val final opening not found"
                    ));
                };
                PCS::verify(
                    proof_val_final,
                    &self.preprocessing.generators,
                    &mut self.transcript,
                    &point_val_final.r,
                    &eval_val_final,
                    commitment,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Untrusted advice val final opening proof verification failed: {e:?}"
                    )
                })?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl CanonicalSerialize for JoltSharedPreprocessing {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        // Serialize the inner BytecodePreprocessing (not the Arc wrapper)
        self.bytecode
            .as_ref()
            .serialize_with_mode(&mut writer, compress)?;
        self.ram.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.bytecode.serialized_size(compress)
            + self.ram.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
    }
}

impl CanonicalDeserialize for JoltSharedPreprocessing {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let bytecode =
            BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram = RAMPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            bytecode: Arc::new(bytecode),
            ram,
            memory_layout,
        })
    }
}

impl ark_serialize::Valid for JoltSharedPreprocessing {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode.check()?;
        self.ram.check()?;
        self.memory_layout.check()
    }
}

impl JoltSharedPreprocessing {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
    ) -> JoltSharedPreprocessing {
        let bytecode = Arc::new(BytecodePreprocessing::preprocess(bytecode));
        let ram = RAMPreprocessing::preprocess(memory_init);
        Self {
            bytecode,
            ram,
            memory_layout,
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
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

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> JoltVerifierPreprocessing<F, PCS> {
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared: shared.clone(),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}
