use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};
use rayon::prelude::*;

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<F: JoltField, ProofTranscript: Transcript>(
    increments: Vec<(usize, i64)>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute Inc");
    let _guard = span.enter();

    // Compute the Inc polynomial using the above table
    let inc: Vec<F> = increments
        .par_iter()
        .map(|(k, increment)| eq_r_address[*k] * F::from_i64(*increment))
        .collect();
    let mut inc = MultilinearPolynomial::from(inc);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute E");
    let _guard = span.enter();

    let mut E: Vec<Vec<F>> = Vec::with_capacity(r_cycle.len() + 1);
    E.push(vec![F::one()]);
    for (i, r_i) in r_cycle.iter().enumerate() {
        let eq_table: Vec<F> = E[i]
            .par_iter()
            .flat_map(|eq_j_r| {
                let one_term = *eq_j_r * r_i;
                let zero_term = *eq_j_r - one_term;
                [zero_term, one_term]
            })
            .collect();
        E.push(eq_table);
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute D");
    let _guard = span.enter();

    let mut D: Vec<Vec<F>> = Vec::with_capacity(r_cycle.len() + 1);
    D.push(vec![F::zero()]);
    for (i, r_i) in r_cycle.iter().enumerate() {
        let lt_table: Vec<F> = D[i]
            .par_iter()
            .zip(E[i].par_iter())
            .flat_map(|(D_i_x, E_i_x)| {
                let one_term = *D_i_x;
                let zero_term = *D_i_x + *r_i * E_i_x;
                [zero_term, one_term]
            })
            .collect();
        D.push(lt_table);
    }

    drop(_guard);
    drop(span);

    let mut lt = MultilinearPolynomial::from(D.pop().unwrap());

    let num_rounds = T.log_2();
    let mut previous_claim = claimed_evaluation;
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(num_rounds);

    const DEGREE: usize = 2;

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _round in 0..num_rounds {
        #[cfg(test)]
        {
            let expected: F = (0..inc.len())
                .map(|j| inc.get_bound_coeff(j) * lt.get_bound_coeff(j))
                .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 2] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = inc.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = lt.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [inc_evals[0] * lt_evals[0], inc_evals[1] * lt_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::LowToHigh),
            || lt.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let inc_claim = inc.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_cycle_prime,
        inc_claim,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn val_evaluation_sumcheck() {
        const K: usize = 64;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let increments: Vec<(usize, i64)> = (0..T)
            .map(|_| {
                let address = rng.next_u32() as usize % K;
                let increment = rng.next_u32() as i32 as i64;
                (address, increment)
            })
            .collect();

        // Compute the Val polynomial from increments
        let mut values = vec![Fr::zero(); K];
        let mut val: Vec<Fr> = Vec::with_capacity(K * T);
        for (k, increment) in increments.iter() {
            val.extend(values.iter());
            values[*k] += Fr::from_i64(*increment);
        }
        let val = MultilinearPolynomial::from(val);

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_address: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let val_evaluation = val.evaluate(&[r_cycle.clone(), r_address.clone()].concat());
        let (sumcheck_proof, _, _) = prove_val_evaluation(
            increments,
            r_address,
            r_cycle,
            val_evaluation,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_address: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result =
            sumcheck_proof.verify(val_evaluation, T.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
