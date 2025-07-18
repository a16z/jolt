use crate::jolt::vm::output_check::OutputSumcheck;
use crate::jolt::vm::ram::{
    remap_address, BooleanitySumcheck, BooleanityVerifierState, HammingWeightSumcheck,
    HammingWeightVerifierState, RafEvaluationSumcheck, RafEvaluationVerifierState,
    RamReadWriteChecking, ValEvaluationSumcheck, ValEvaluationSumcheckClaims,
    ValEvaluationVerifierState,
};
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::CommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::ra_virtual::RASumcheck;
use crate::subprotocols::sumcheck::BatchableSumcheckVerifierInstance;
use crate::{
    field::JoltField,
    jolt::vm::ram::{RAMPreprocessing, RAMTwistProof, RafEvaluationProof},
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::identity_poly::UnmapRamAddressPolynomial;

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for ValEvaluationSumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.inc.len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation - self.init_eval
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let claims = self.claims.as_ref().expect("Claims not cached");

        // r contains r_cycle_prime in low-to-high order
        let r_cycle_prime: Vec<F> = r.iter().rev().copied().collect();

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().zip(verifier_state.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        // Return inc_claim * wa_claim * lt_eval
        claims.inc_claim * claims.wa_claim * lt_eval
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero() // Always zero for booleanity
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_claims = self.ra_claims.as_ref().expect("RA claims not cached");

        let K_log = self.K.log_2();
        let (r_address_prime, r_cycle_prime) = r.split_at(K_log);

        let eq_eval_address = EqPolynomial::mle(&verifier_state.r_address, r_address_prime);
        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&verifier_state.r_prime, &r_cycle_prime);

        // Compute batched booleanity check: sum_{i=0}^{d-1} z^i * (ra_i^2 - ra_i)
        let mut result = F::zero();
        for (i, ra_claim) in ra_claims.iter().enumerate() {
            result += verifier_state.z_powers[i] * (ra_claim.square() - *ra_claim);
        }

        eq_eval_address * eq_eval_cycle * result
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        K: usize,
        transcript: &mut ProofTranscript,
        memory_layout: &MemoryLayout,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let sumcheck_instance = RafEvaluationSumcheck::new_verifier(
            self.raf_claim,
            K.log_2(),
            memory_layout.input_start,
            self.ra_claim,
        );

        let r_raf_sumcheck = sumcheck_instance.verify_single(&self.sumcheck_proof, transcript)?;

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

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for HammingWeightSumcheck<F>
{
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.ra[0].get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.log_K
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_claims = self.cached_claims.as_ref().expect("RA claims not cached");

        // Compute batched claim: sum_{i=0}^{d-1} z^i * ra_i
        ra_claims
            .iter()
            .zip(verifier_state.z_powers.iter())
            .map(|(ra_claim, z_power)| *ra_claim * z_power)
            .sum()
    }
}

impl<F: JoltField> RafEvaluationSumcheck<F> {
    /// Create a new verifier instance
    fn new_verifier(raf_claim: F, log_K: usize, start_address: u64, ra_claim: F) -> Self {
        Self {
            input_claim: raf_claim,
            prover_state: None,
            verifier_state: Some(RafEvaluationVerifierState {
                log_K,
                start_address,
            }),
            cached_claim: Some(ra_claim),
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for RafEvaluationSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.ra.get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.log_K
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Compute unmap evaluation at r
        let unmap_eval =
            UnmapRamAddressPolynomial::new(verifier_state.log_K, verifier_state.start_address)
                .evaluate(r);

        // Return unmap(r) * ra(r)
        let ra_claim = self.cached_claim.expect("ra_claim not cached");
        unmap_eval * ra_claim
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
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

        let mut r_cycle_prime = <ValEvaluationSumcheck<F> as BatchableSumcheckVerifierInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
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

        let r_booleanity = <BooleanitySumcheck<F> as BatchableSumcheckVerifierInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
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
