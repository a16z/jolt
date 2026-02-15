use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::univariate_skip::{UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant};
use crate::transcripts::Transcript;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::outer::{OuterUniSkipParams, OuterUniSkipVerifier};
use crate::zkvm::spartan::product::{ProductVirtualUniSkipParams, ProductVirtualUniSkipVerifier};

pub mod instruction_input;
pub mod outer;
pub mod product;
pub mod shift;

pub fn verify_stage1_uni_skip<F: JoltField, C: JoltCurve, T: Transcript>(
    proof: &UniSkipFirstRoundProofVariant<F, C, T>,
    key: &UniformSpartanKey<F>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(OuterUniSkipParams<F>, F::Challenge), anyhow::Error> {
    let verifier = OuterUniSkipVerifier::new(key, transcript);

    let challenge = match proof {
        UniSkipFirstRoundProofVariant::Standard(std_proof) => {
            UniSkipFirstRoundProof::verify::<
                OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
            >(std_proof, &verifier, opening_accumulator, transcript)
            .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?
        }
        UniSkipFirstRoundProofVariant::Zk(zk_proof) => zk_proof
            .verify_transcript(&verifier, opening_accumulator, transcript)
            .map_err(|_| {
                anyhow::anyhow!("UniSkip ZK first-round transcript verification failed")
            })?,
    };

    Ok((verifier.params, challenge))
}

pub fn verify_stage2_uni_skip<F: JoltField, C: JoltCurve, T: Transcript>(
    proof: &UniSkipFirstRoundProofVariant<F, C, T>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(ProductVirtualUniSkipParams<F>, F::Challenge), anyhow::Error> {
    let verifier = ProductVirtualUniSkipVerifier::new(opening_accumulator, transcript);

    let challenge = match proof {
        UniSkipFirstRoundProofVariant::Standard(std_proof) => {
            UniSkipFirstRoundProof::verify::<
                PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
                PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
            >(std_proof, &verifier, opening_accumulator, transcript)
            .map_err(|_| {
                anyhow::anyhow!("ProductVirtual uni-skip first-round verification failed")
            })?
        }
        UniSkipFirstRoundProofVariant::Zk(zk_proof) => zk_proof
            .verify_transcript(&verifier, opening_accumulator, transcript)
            .map_err(|_| {
                anyhow::anyhow!(
                    "ProductVirtual ZK uni-skip first-round transcript verification failed"
                )
            })?,
    };

    Ok((verifier.params, challenge))
}
