use crate::jolt::vm::ram::remap_address;
use crate::{
    field::JoltField,
    jolt::vm::ram::{RAMPreprocessing, RAMTwistProof, RafEvaluationProof, ReadWriteCheckingProof},
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;

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

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        K: usize,
        T: usize,
        preprocessing: &RAMPreprocessing,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_K = K.log_2();
        let log_T = T.log_2();
        let r: Vec<F> = transcript.challenge_vector(log_K);
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) =
            self.read_write_checking_proof
                .verify(r, r_prime.clone(), transcript);

        let mut initial_memory_state = vec![0; K];
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

        let init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
        let init_eval = init.evaluate(&r_address);

        let (sumcheck_claim, r_cycle_prime) = self.val_evaluation_proof.sumcheck_proof.verify(
            self.read_write_checking_proof.val_claim - init_eval,
            log_T,
            2,
            transcript,
        )?;

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().rev().zip(r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        assert_eq!(
            sumcheck_claim,
            lt_eval * self.val_evaluation_proof.inc_claim,
            "Val evaluation sumcheck failed"
        );

        // TODO: Append Inc claim to opening proof accumulator

        let mut r_address: Vec<F> = transcript.challenge_vector(log_K);
        r_address = r_address.into_iter().rev().collect();

        let (sumcheck_claim, r_booleanity) =
            self.booleanity_proof
                .sumcheck_proof
                .verify(F::zero(), log_K + log_T, 3, transcript)?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(log_K);

        let eq_eval_address = EqPolynomial::new(r_address).evaluate(r_address_prime);
        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        // let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle_prime);

        assert_eq!(
            eq_eval_address
                * eq_eval_cycle
                * (self.booleanity_proof.ra_claim.square() - self.booleanity_proof.ra_claim),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        let (sumcheck_claim, _r_hamming_weight) =
            self.hamming_weight_proof
                .sumcheck_proof
                .verify(F::one(), log_K, 1, transcript)?;

        assert_eq!(
            self.hamming_weight_proof.ra_claim, sumcheck_claim,
            "Hamming weight sumcheck failed"
        );

        // Verify RAF evaluation proof
        let _r_address_raf =
            self.raf_evaluation_proof
                .verify(K, transcript, &program_io.memory_layout)?;

        // TODO: Add opening proof verification for ra(r_address_raf, r_cycle)

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
