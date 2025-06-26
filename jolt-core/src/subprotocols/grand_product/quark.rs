use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::grand_product::{BatchedGrandProductProof, BatchedGrandProductVerifier};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::subprotocols::QuarkHybridLayerDepth;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::*;
use ark_std::{One, Zero};
use itertools::Itertools;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct QuarkGrandProductProof<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> {
    pub(crate) sumcheck_proof: SumcheckInstanceProof<PCS::Field, ProofTranscript>,
    pub(crate) g_commitment: PCS::Commitment,
    pub(crate) g_r_sumcheck: PCS::Field,
    pub(crate) g_r_prime: (PCS::Field, PCS::Field),
    pub(crate) v_r_prime: (PCS::Field, PCS::Field),
    pub num_vars: usize,
}

pub struct QuarkGrandProduct<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) batch_size: usize,
    pub(crate) quark_poly: Option<Vec<F>>,
    pub(crate) base_layers: Vec<DenseInterleavedPolynomial<F>>,
    pub(crate) _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QuarkGrandProductConfig {
    pub hybrid_layer_depth: QuarkHybridLayerDepth,
}

pub struct QuarkGrandProductBase<F: JoltField, ProofTranscript: Transcript> {
    _marker: PhantomData<(F, ProofTranscript)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum QuarkError {
    /// returned if the sumcheck fails
    #[error("InvalidSumcheck")]
    InvalidQuarkSumcheck,
    /// Returned if a quark opening proof fails
    #[error("InvalidOpeningProof")]
    InvalidOpeningProof,
    /// Returned if eq(tau, r)*(f(1, r) - f(r, 0)*f(r,1)) does not match the result from sumcheck
    #[error("InvalidOpeningProof")]
    InvalidBinding,
}

impl<F, ProofTranscript> QuarkGrandProductBase<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    /// Verifies the given grand product proof.
    #[tracing::instrument(skip_all, name = "QuarkGrandProduct::verify_grand_product")]
    pub fn verify_quark_grand_product<G, PCS>(
        proof: &BatchedGrandProductProof<PCS, ProofTranscript>,
        claimed_outputs: &[F],
        opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>)
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        G: BatchedGrandProductVerifier<F, PCS, ProofTranscript>,
    {
        transcript.append_scalars(claimed_outputs);
        let r_outputs: Vec<F> =
            transcript.challenge_vector(claimed_outputs.len().next_power_of_two().log_2());
        let claim = DensePolynomial::new_padded(claimed_outputs.to_vec()).evaluate(&r_outputs);

        // Here we must also support the case where the number of layers is very small
        let (claim, rand) = match proof.quark_proof.as_ref() {
            Some(quark) => {
                // In this case we verify the quark which fixes the first log(n)-4 vars in the random eval point.
                let v_len = quark.num_vars;
                quark
                    .verify(
                        r_outputs,
                        claim,
                        opening_accumulator.unwrap(),
                        transcript,
                        v_len,
                    )
                    .unwrap_or_else(|e| panic!("quark verify error: {e:?}"))
            }
            None => {
                // Otherwise we must check the actual claims and the preset random will be empty.
                (claim, r_outputs)
            }
        };

        let (grand_product_claim, grand_product_r) =
            G::verify_layers(&proof.gkr_layers, claim, transcript, rand);

        (grand_product_claim, grand_product_r)
    }
}

impl<PCS, ProofTranscript> QuarkGrandProductProof<PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
{
    /// Verifies the given grand product proof.
    #[allow(clippy::type_complexity)]
    pub fn verify(
        &self,
        r_outputs: Vec<PCS::Field>,
        claim: PCS::Field,
        opening_accumulator: &mut VerifierOpeningAccumulator<PCS::Field, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
        n_rounds: usize,
    ) -> Result<(PCS::Field, Vec<PCS::Field>), QuarkError> {
        self.g_commitment.append_to_transcript(transcript);

        // Next sample the tau and construct the evals poly
        let tau: Vec<PCS::Field> = transcript.challenge_vector(n_rounds);

        // To complete the sumcheck proof we have to validate that our polynomial openings match and are right.
        let (expected, r_sumcheck) = self
            .sumcheck_proof
            .verify(claim, n_rounds, 3, transcript)
            .map_err(|_| QuarkError::InvalidQuarkSumcheck)?;

        // Firstly we append g(r_sumcheck)
        opening_accumulator.append(
            &[&self.g_commitment],
            r_sumcheck.clone(),
            &[self.g_r_sumcheck],
            transcript,
        );

        // (r1, r') := r_sumcheck
        let r_1 = r_sumcheck[0];
        let r_prime = r_sumcheck[1..r_sumcheck.len()].to_vec();

        // Next do the line reduction verification of g(r', 0) and g(r', 1)
        let (r_g, claim_g) =
            line_reduce_verify(self.g_r_prime.0, self.g_r_prime.1, &r_prime, transcript);
        opening_accumulator.append(&[&self.g_commitment], r_g, &[claim_g], transcript);

        // Similarly, we can reduce v(r', 0) and v(r', 1) to a single claim about v:
        let (r_v, claim_v) =
            line_reduce_verify(self.v_r_prime.0, self.v_r_prime.1, &r_prime, transcript);

        // Calculate eq(tau, r_sumcheck) in O(log(n))
        let eq_tau_eval: PCS::Field = r_sumcheck
            .iter()
            .zip_eq(tau.iter())
            .map(|(&r_i, &tau_i)| {
                r_i * tau_i + (PCS::Field::one() - r_i) * (PCS::Field::one() - tau_i)
            })
            .product();

        // Calculate eq(11...10 || r_outputs, r_sumcheck) in O(log(n))
        let mut one_padded_r_outputs = vec![PCS::Field::one(); n_rounds];
        let slice_index = one_padded_r_outputs.len() - r_outputs.len();
        one_padded_r_outputs[slice_index..].copy_from_slice(r_outputs.as_slice());
        one_padded_r_outputs[slice_index - 1] = PCS::Field::zero();
        let eq_output_eval: PCS::Field = r_sumcheck
            .iter()
            .zip_eq(one_padded_r_outputs.iter())
            .map(|(&r_i, &r_output)| {
                r_i * r_output + (PCS::Field::one() - r_i) * (PCS::Field::one() - r_output)
            })
            .product();

        // We calculate:
        // - f(1, r_sumcheck) = g(r_sumcheck)
        // - f(r_sumcheck, 0) = r_1 * g(r', 0) + (1 - r_1) * v(r', 0)
        // - f(r_sumcheck, 1) = r_1 * g(r', 1) + (1 - r_1) * v(r', 1)
        let f_1r = self.g_r_sumcheck;
        let f_r0 = self.v_r_prime.0 + r_1 * (self.g_r_prime.0 - self.v_r_prime.0);
        let f_r1 = self.v_r_prime.1 + r_1 * (self.g_r_prime.1 - self.v_r_prime.1);

        // Finally we check that in fact the polynomial bound by the sumcheck is equal to
        // eq(tau, r) * (f(1, r) - f(r, 0) * f(r, 1)) + eq(11...10|| r_outputs, r) * f(1, r)
        let result_from_openings = eq_tau_eval * (f_1r - f_r0 * f_r1) + eq_output_eval * f_1r;

        if result_from_openings != expected {
            return Err(QuarkError::InvalidBinding);
        }

        Ok((claim_v, r_v))
    }
}

// The verifier's dual of `line_reduce`
fn line_reduce_verify<F: JoltField, ProofTranscript: Transcript>(
    claim_0: F,
    claim_1: F,
    r: &[F],
    transcript: &mut ProofTranscript,
) -> (Vec<F>, F) {
    // We add these to the transcript then sample an r which depends on both
    transcript.append_scalar(&claim_0);
    transcript.append_scalar(&claim_1);
    let rand: F = transcript.challenge_scalar();

    // Now calculate r* := r || rand
    let mut r_star = r.to_vec();
    r_star.push(rand);

    let reduced_claim = claim_0 + rand * (claim_1 - claim_0);
    (r_star, reduced_claim)
}

impl<F, PCS, ProofTranscript> BatchedGrandProductVerifier<F, PCS, ProofTranscript>
    for QuarkGrandProduct<F, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// The bottom/input layer of the grand products
    // (leaf values, batch size)
    type Leaves = (Vec<F>, usize);
    type Config = QuarkGrandProductConfig;

    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::verify_grand_product")]
    fn verify_grand_product(
        proof: &BatchedGrandProductProof<PCS, ProofTranscript>,
        claimed_outputs: &[F],
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        QuarkGrandProductBase::verify_quark_grand_product::<Self, PCS>(
            proof,
            claimed_outputs,
            _opening_accumulator,
            transcript,
        )
    }
}

#[cfg(test)]
mod quark_grand_product_tests {
    use super::*;
    use crate::poly::commitment::zeromorph::*;
    use crate::poly::opening_proof::ProverOpeningAccumulator;
    use crate::subprotocols::grand_product::BatchedGrandProductProver;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::{Bn254, Fr};
    use rand_core::SeedableRng;

    fn quark_hybrid_test_with_config(config: QuarkGrandProductConfig) {
        const LAYER_SIZE: usize = 1 << 8;

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9_u64);

        let leaves_1: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect();
        let leaves_2: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect();
        let known_products: Vec<Fr> = vec![leaves_1.iter().product(), leaves_2.iter().product()];

        let v = [leaves_1, leaves_2].concat();
        let mut prover_transcript: KeccakTranscript = KeccakTranscript::new(b"test_transcript");

        let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << 9);
        let setup = srs.trim(1 << 9);

        let mut hybrid_grand_product =
            <QuarkGrandProduct<Fr, KeccakTranscript> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::construct_with_config((v, 2), config);
        let mut prover_accumulator: ProverOpeningAccumulator<Fr, KeccakTranscript> =
            ProverOpeningAccumulator::new();
        let proof: BatchedGrandProductProof<Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript> =
            hybrid_grand_product
                .prove_grand_product(
                    Some(&mut prover_accumulator),
                    &mut prover_transcript,
                    Some(&setup.0),
                )
                .0;
        let batched_proof = prover_accumulator.reduce_and_prove(&setup.0, &mut prover_transcript);

        // Note resetting the transcript is important
        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let mut verifier_accumulator: VerifierOpeningAccumulator<
            Fr,
            Zeromorph<Bn254, KeccakTranscript>,
            KeccakTranscript,
        > = VerifierOpeningAccumulator::new();
        verifier_accumulator.compare_to(prover_accumulator, &setup.0);

        let _ = QuarkGrandProduct::verify_grand_product(
            &proof,
            &known_products,
            Some(&mut verifier_accumulator),
            &mut verifier_transcript,
        );
        assert!(verifier_accumulator
            .reduce_and_verify(&setup.1, &batched_proof, &mut verifier_transcript)
            .is_ok());
    }

    #[test]
    fn quark_hybrid_default_config_e2e() {
        quark_hybrid_test_with_config(QuarkGrandProductConfig::default());
    }

    #[test]
    fn quark_hybrid_custom_config_e2e() {
        let custom_config = QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Custom(20),
        };
        quark_hybrid_test_with_config(custom_config);
    }

    #[test]
    fn quark_hybrid_min_config_e2e() {
        let zero_crossover_config = QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Min,
        };
        quark_hybrid_test_with_config(zero_crossover_config);
    }

    #[test]
    fn quark_hybrid_max_config_e2e() {
        let zero_crossover_config = QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Max,
        };
        quark_hybrid_test_with_config(zero_crossover_config);
    }
}
