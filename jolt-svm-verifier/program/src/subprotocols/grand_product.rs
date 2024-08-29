use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use solana_program::msg;
use jolt_types::field::JoltField;
use jolt_types::subprotocols::grand_product::BatchedGrandProductLayerProof;
use jolt_types::utils::transcript::ProofTranscript;

// Defining a solana specific version of this as a trade-off between pulling apart significant parts
// of jolt-core and keeping the changes minimal.
// This implementation follows the solidity implementation closely.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct GrandProductProof<F: JoltField> {
    layers: Vec<BatchedGrandProductLayerProof<F>>
}

fn eval_eq_mle<F: JoltField>(r_grand_product: &[F], r_sumcheck: &[F]) -> F {
    r_grand_product
        .iter()
        .zip(r_sumcheck.iter().rev())
        .map(|(&r_gp, &r_sc)| {
            r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc)
        })
        .product()
}

pub fn verify_sumcheck_claim<F: JoltField>(
    layer_proof: &BatchedGrandProductLayerProof<F>,
    coeffs: &[F],
    sumcheck_claim: F,
    eq_eval: F,
    r_grand_product: &mut Vec<F>,
    transcript: &mut ProofTranscript,
) -> (Vec<F>, Vec<F>) {
    let expected_sumcheck_claim: F = coeffs
        .iter()
        .zip(layer_proof.left_claims.iter().zip(layer_proof.right_claims.iter()))
        .map(|(&coeff, (&left, &right))| coeff * left * right * eq_eval)
        .sum();
    msg!("expected sumcheck calculated");

    assert_eq!(expected_sumcheck_claim, sumcheck_claim, "Sumcheck claim mismatch");

    let r_layer = transcript.challenge_scalar();

    let new_claims: Vec<F> = layer_proof
        .left_claims
        .iter()
        .zip(layer_proof.right_claims.iter())
        .map(|(&left, &right)| left + r_layer * (right - left))
        .collect();
    msg!("new claims done");

    r_grand_product.push(r_layer);

    (new_claims, r_grand_product.clone())
}

pub fn verify_grand_product<F: JoltField>(
    proof: &GrandProductProof<F>,
    claims: &[F],
    transcript: &mut ProofTranscript,
) -> Vec<F> {
    let mut r_grand_product = Vec::new();
    let mut claims_to_verify = claims.to_vec();
    msg!("Proof layers: {:?}", proof.layers.len());

    for layer in &proof.layers {
        let coeffs: Vec<F> = transcript.challenge_vector(claims_to_verify.len());

        let joined_claim: F = claims_to_verify
            .iter()
            .zip(coeffs.iter())
            .map(|(&claim, &coeff)| claim * coeff)
            .sum();

        assert_eq!(claims.len(), layer.left_claims.len());
        assert_eq!(claims.len(), layer.right_claims.len());

        let (sumcheck_claim, r_sumcheck) = layer.verify(
            joined_claim,
            r_grand_product.len(),
            3,
            transcript,
        );

        for (&left, &right) in layer.left_claims.iter().zip(layer.right_claims.iter()) {
            transcript.append_scalar(&left);
            transcript.append_scalar(&right);
        }
        msg!("Transcript appened scalars");

        assert_eq!(r_grand_product.len(), r_sumcheck.len());

        let eq_eval = eval_eq_mle(&r_grand_product, &r_sumcheck);

        r_grand_product = r_sumcheck.into_iter().rev().collect();
        msg!("Going to verify sumcheck");

        let (new_claims, new_r_grand_product) = verify_sumcheck_claim(
            layer,
            &coeffs,
            sumcheck_claim,
            eq_eval,
            &mut r_grand_product,
            transcript,
        );

        claims_to_verify = new_claims;
        r_grand_product = new_r_grand_product;
    }

    r_grand_product
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_serialize::CanonicalDeserialize;
    use ark_bn254::Fr;
    use jolt_types::utils::transcript::ProofTranscript;
    use crate::test_constants::{GRAND_PRODUCT_BATCH_PROOFS, GRAND_PRODUCT_CLAIMS, GRAND_PRODUCT_R_PROVER};

    #[test]
    fn test_grand_product() {
        // let proof_bytes = include_bytes!("../../test_vectors/grand_product_proof.bytes");
        let proof: GrandProductProof<Fr> = CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_BATCH_PROOFS[..]).unwrap();

        // let claims_bytes = include_bytes!("../../test_vectors/grand_product_claims.bytes");
        let claims: Vec<Fr> = CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_CLAIMS[..]).unwrap();

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let r_grand_product = verify_grand_product(&proof, &claims, &mut transcript);

        // let expected_r_grand_product_bytes =
        //     include_bytes!("../../test_vectors/grand_product_r_grand_product.bytes");
        let expected_r_grand_product: Vec<Fr> =
            CanonicalDeserialize::deserialize_uncompressed(&GRAND_PRODUCT_R_PROVER[..]).unwrap();

        assert_eq!(expected_r_grand_product, r_grand_product);
    }
}
