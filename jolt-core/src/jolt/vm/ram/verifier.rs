use crate::{
    field::JoltField,
    jolt::vm::ram::{RAMPreprocessing, RAMTwistProof, RafEvaluationProof, ReadWriteCheckingProof},
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{
            MultilinearPolynomial, PolynomialEvaluation,
        }
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::Transcript,
    }
};
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::output_check::OutputSumcheck;
use crate::jolt::vm::ram::{remap_address, BooleanitySumcheck, BooleanityVerifierState, HammingWeightSumcheck, HammingWeightVerifierState, ValEvaluationSumcheck, ValEvaluationSumcheckClaims, ValEvaluationVerifierState};
use crate::jolt::vm::ram_read_write_checking::RamReadWriteChecking;
use crate::jolt::witness::CommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::ra_virtual::RASumcheck;
use crate::subprotocols::sumcheck::BatchableSumcheckInstance;

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        K: usize,
        transcript: &mut ProofTranscript,
        memory_layout: &MemoryLayout,
    ) -> Result<Vec<F>, ProofVerifyError> {
        const DEGREE: usize = 2;

        // Verify the sumcheck proof
        let (sumcheck_claim, r_raf_sumcheck) =
            self.sumcheck_proof
                .verify(self.raf_claim, K.log_2(), DEGREE, transcript)?;

        let unmap_eval = UnmapRamAddressPolynomial::new(K.log_2(), memory_layout.input_start)
            .evaluate(&r_raf_sumcheck);

        // Verify sumcheck_claim = unmap(r_raf_sumcheck) * ra(r_raf_sumcheck, r_cycle)
        let expected_product = unmap_eval * self.ra_claim;
        if expected_product != sumcheck_claim {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(r_raf_sumcheck)
    }
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    fn new_verifier(log_K: usize, ra_claims: Vec<F>, z_powers: Vec<F>, d: usize) -> Self {
        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(HammingWeightVerifierState { log_K, d, z_powers }),
            cached_claims: Some(ra_claims),
            d,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    pub fn verify<PCS: CommitmentScheme<ProofTranscript, crate::poly::commitment::commitment_scheme::Field= F>>(
        &self,
        T: usize,
        preprocessing: &RAMPreprocessing,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let log_K = self.K.log_2();
        let log_T = T.log_2();
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) = RamReadWriteChecking::verify(
            &self.read_write_checking_proof,
            program_io,
            self.K,
            &r_prime,
            transcript,
        )?;

        let mut initial_memory_state = vec![0; self.K];
        // Copy bytecode
        let mut index = remap_address(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word as i64;
            index += 1;
        }
        // Copy input bytes
        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate `v_init`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word as i64;
            index += 1;
        }

        // TODO: Verifier currently materializes and evaluates Val_init itself,
        // but this is not tractable for large K
        let val_init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
        let init_eval = val_init.evaluate(&r_address);

        // Create the sumcheck instance for verification
        let sumcheck_instance = ValEvaluationSumcheck {
            claimed_evaluation: self.read_write_checking_proof.claims.val_claim,
            init_eval,
            prover_state: None,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: log_T,
                r_address: r_address.clone(),
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim: self.val_evaluation_proof.inc_claim,
                wa_claim: self.val_evaluation_proof.wa_claim,
            }),
        };

        let mut r_cycle_prime = sumcheck_instance.verify_single(
            &self.val_evaluation_proof.sumcheck_proof,
            transcript,
        )?;

        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_commitment = &commitments.commitments[CommittedPolynomials::RamInc.to_index()];
        opening_accumulator.append(
            &[inc_commitment],
            r_cycle_prime,
            &[self.val_evaluation_proof.inc_claim],
            transcript,
        );

        // TODO: Append Inc claim to opening proof accumulator

        let mut r_address: Vec<F> = transcript.challenge_vector(log_K);
        r_address = r_address.into_iter().rev().collect();

        // Calculate D dynamically
        // let d = (log_K / 8).max(1);
        let d = 1; // @TODO(markosg04) keeping d = 1 for legacy prove

        // Get z challenges for batching
        let z: F = transcript.challenge_scalar();
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z;
        }

        let sumcheck_instance = BooleanitySumcheck {
            K: self.K,
            T,
            d,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState {
                K: self.K,
                T,
                d,
                r_address: r_address.clone(),
                r_prime: r_prime.clone(),
                z_powers: z_powers.clone(),
            }),
            ra_claims: Some(self.booleanity_proof.ra_claims.clone()),
            current_round: 0,
            trace: None,
            memory_layout: None,
        };

        let r_booleanity = sumcheck_instance.verify_single(
            &self.booleanity_proof.sumcheck_proof,
            transcript,
        )?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(log_K);

        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::RamRa(0).to_index()];

        let ra_claim = self.booleanity_proof.ra_claims[0]; // d = 1

        let mut r_cycle_bound = RASumcheck::<F>::verify(
            ra_claim,
            self.ra_proof.ra_i_claims.clone(),
            r_cycle_prime,
            T,
            d,
            &self.ra_proof.sumcheck_proof,
            transcript,
        )?;

        r_cycle_bound.reverse();
        let r_concat = [r_address_prime.as_slice(), r_cycle_bound.as_slice()].concat();

        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &self.ra_proof.ra_i_claims,
            transcript,
        );

        // Get z challenges for hamming weight batching
        let z_hw_challenge: F = transcript.challenge_scalar();
        let mut z_hw_powers = vec![F::one(); d];
        for i in 1..d {
            z_hw_powers[i] = z_hw_powers[i - 1] * z_hw_challenge;
        }

        let sumcheck_instance = HammingWeightSumcheck::new_verifier(
            log_K,
            self.hamming_weight_proof.ra_claims.clone(),
            z_hw_powers,
            d,
        );

        let _r_hamming_weight = sumcheck_instance
            .verify_single(&self.hamming_weight_proof.sumcheck_proof, transcript)?;

        let _r_address_raf =
            self.raf_evaluation_proof
                .verify(self.K, transcript, &program_io.memory_layout)?;

        OutputSumcheck::verify(
            program_io,
            val_init,
            &r_address_prime,
            T,
            &self.output_proof,
            transcript,
        )?;

        // TODO: Append to opening proof accumulator

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Vec<F>, Vec<F>) {
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        let z: F = transcript.challenge_scalar();

        let (sumcheck_claim, r_sumcheck) = self
            .sumcheck_proof
            .verify(
                self.rv_claim + z * self.inc_claim,
                T.log_2() + K.log_2(),
                3,
                transcript,
            )
            .unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[self.sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..self.sumcheck_switch_index].iter().rev());
        // Final log(K) rounds bind address variables
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::new(r).evaluate(&r_address);

        assert_eq!(
            eq_eval_cycle * self.ra_claim * self.val_claim
                + z * eq_eval_address
                * eq_eval_cycle
                * self.ra_claim
                * (self.wv_claim - self.val_claim),
            sumcheck_claim,
            "Read/write-checking sumcheck failed"
        );

        (r_address, r_cycle)
    }
}
