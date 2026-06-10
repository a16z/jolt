use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::opening_proof::AbstractVerifierOpeningAccumulator;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
#[cfg(feature = "zk")]
use crate::subprotocols::univariate_skip::zk_uni_skip_first_round;
use crate::subprotocols::univariate_skip::{uni_skip_first_round, ZkUniSkipReadback};
use crate::transcript_msgs::VerifierFs;
use crate::utils::errors::ProofVerifyError;
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

/// `(uni-skip params, first-round challenge, NARG ZK read-back)`. The read-back is `Some`
/// only in ZK mode (the first-round commitment + degree + output-claim commitments threaded
/// to stage 8 / BlindFold); `None` otherwise.
type UniSkipVerifyResult<F, C, P> = (P, <F as JoltField>::Challenge, Option<ZkUniSkipReadback<C>>);

/// Shared zk-vs-clear selector for a uni-skip first round. `zk_mode` (from
/// `JoltProof::zk_mode`) picks the read path; the per-path NARG read order is identical
/// either way. The ZK read-back (commitment + degree + output-claim commitments, threaded
/// to stage 8 / BlindFold) is `Some` only in ZK mode.
fn verify_uni_skip_first_round<
    F: JoltField,
    C: JoltCurve<F = F>,
    const N: usize,
    const FIRST_ROUND_POLY_NUM_COEFFS: usize,
    A: AbstractVerifierOpeningAccumulator<F>,
    V: SumcheckInstanceVerifier<F, A>,
>(
    zk_mode: bool,
    verifier: &V,
    opening_accumulator: &mut A,
    transcript: &mut impl VerifierFs<F>,
) -> Result<(F::Challenge, Option<ZkUniSkipReadback<C>>), ProofVerifyError> {
    if !zk_mode {
        let challenge = uni_skip_first_round::verify::<F, N, FIRST_ROUND_POLY_NUM_COEFFS, A>(
            verifier,
            opening_accumulator,
            transcript,
        )?;
        Ok((challenge, None))
    } else {
        #[cfg(feature = "zk")]
        {
            let (challenge, readback) = zk_uni_skip_first_round::verify_transcript::<F, C, _, _>(
                verifier,
                opening_accumulator,
                transcript,
            )?;
            Ok((challenge, Some(readback)))
        }
        #[cfg(not(feature = "zk"))]
        {
            Err(ProofVerifyError::ZkFeatureRequired)
        }
    }
}

pub fn verify_stage1_uni_skip<
    F: JoltField,
    C: JoltCurve<F = F>,
    T: VerifierFs<F>,
    A: AbstractVerifierOpeningAccumulator<F>,
>(
    zk_mode: bool,
    key: &UniformSpartanKey<F>,
    opening_accumulator: &mut A,
    transcript: &mut T,
) -> Result<UniSkipVerifyResult<F, C, OuterUniSkipParams<F>>, ProofVerifyError> {
    let verifier = OuterUniSkipVerifier::new(key, transcript);
    let (challenge, zk_readback) = verify_uni_skip_first_round::<
        F,
        C,
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        A,
        _,
    >(zk_mode, &verifier, opening_accumulator, transcript)?;

    Ok((verifier.params, challenge, zk_readback))
}

pub fn verify_stage2_uni_skip<
    F: JoltField,
    C: JoltCurve<F = F>,
    T: VerifierFs<F>,
    A: AbstractVerifierOpeningAccumulator<F>,
>(
    zk_mode: bool,
    opening_accumulator: &mut A,
    transcript: &mut T,
) -> Result<UniSkipVerifyResult<F, C, ProductVirtualUniSkipParams<F>>, ProofVerifyError> {
    let verifier = ProductVirtualUniSkipVerifier::new(opening_accumulator, transcript);
    let (challenge, zk_readback) = verify_uni_skip_first_round::<
        F,
        C,
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        A,
        _,
    >(zk_mode, &verifier, opening_accumulator, transcript)?;

    Ok((verifier.params, challenge, zk_readback))
}
