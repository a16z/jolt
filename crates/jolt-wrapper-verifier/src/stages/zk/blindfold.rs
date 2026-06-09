//! BlindFold statement construction for the wrapper Spartan + HyperKZG verifier.
//!
//! In the ZK wrapper path the Spartan sumcheck claims are hidden behind
//! committed output-claim rows. This module gives those hidden scalars typed
//! names and describes exactly how they are related:
//!
//! - the outer Spartan sumcheck starts from zero and ends at
//!   `eq(tau, rx) * (A(rx) * B(rx) - C(rx))`;
//! - the inner Spartan sumcheck starts from the transcript-batched
//!   `alpha_a * A(rx) + alpha_b * B(rx) + alpha_c * C(rx)` and ends at
//!   `M(rx, ry) * Z(ry)`;
//! - the final opening binds the hidden `Z(ry)` scalar to the hidden
//!   HyperKZG evaluation commitment returned by the ZK PCS verifier.
//!
//! No clear claim scalar crosses this boundary. The verifier supplies only the
//! public transcript-derived coefficients and the committed rows already checked
//! by the Spartan and HyperKZG stages.

use jolt_blindfold::{BlindFoldProtocol, BlindFoldProtocolBuilder};
use jolt_claims::{challenge, constant, opening, public, Expr};
use jolt_field::Field;
use jolt_sumcheck::SumcheckDomainSpec;

use crate::{
    stages::{hyperkzg::HyperKzgZkOutput, spartan::SpartanZkOutput, zk::outputs::BlindFoldOutput},
    WrapperError,
};

type Builder<F, C> =
    BlindFoldProtocolBuilder<F, WrapperOpeningId, C, WrapperPublicId, WrapperChallengeId>;
type WrapperExpr<F> = Expr<F, WrapperOpeningId, WrapperPublicId, WrapperChallengeId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperOpeningId {
    OuterA,
    OuterB,
    OuterC,
    WitnessZ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperPublicId {
    EqTauRx,
    CombinedMatrixEval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperChallengeId {
    InnerA,
    InnerB,
    InnerC,
}

pub fn build<F, C>(
    spartan: &SpartanZkOutput<F, C>,
    hyperkzg: &HyperKzgZkOutput<C>,
) -> Result<BlindFoldOutput<F, C>, WrapperError>
where
    F: Field,
    C: Clone,
{
    let builder = BlindFoldProtocol::<F, C>::builder::<
        WrapperOpeningId,
        WrapperPublicId,
        WrapperChallengeId,
    >()
    .public(WrapperPublicId::EqTauRx, spartan.eq_tau_rx)
    .public(
        WrapperPublicId::CombinedMatrixEval,
        spartan.combined_matrix_eval,
    )
    .challenge(WrapperChallengeId::InnerA, spartan.inner_batching.a)
    .challenge(WrapperChallengeId::InnerB, spartan.inner_batching.b)
    .challenge(WrapperChallengeId::InnerC, spartan.inner_batching.c);

    let builder = add_outer_stage(builder, spartan)?;
    let builder = add_inner_stage(builder, spartan)?;
    let protocol = builder
        .final_opening(
            vec![WrapperOpeningId::WitnessZ],
            vec![F::one()],
            hyperkzg.hiding_evaluation_commitment.clone(),
        )
        .build()
        .map_err(blindfold_error)?;

    Ok(BlindFoldOutput { protocol })
}

fn add_outer_stage<F, C>(
    builder: Builder<F, C>,
    spartan: &SpartanZkOutput<F, C>,
) -> Result<Builder<F, C>, WrapperError>
where
    F: Field,
    C: Clone,
{
    builder
        .stage("wrapper-spartan-outer")
        .sumcheck(spartan.outer_statement)
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(spartan.outer_consistency.clone())
        .output_claim_rows(
            vec![
                WrapperOpeningId::OuterA,
                WrapperOpeningId::OuterB,
                WrapperOpeningId::OuterC,
            ],
            spartan.outer_output_claims.shape.row_len,
            spartan.outer_output_claims.commitments.clone(),
        )
        .input_claim(constant(F::zero()))
        .output_claim(outer_output_claim())
        .finish_stage()
        .map_err(blindfold_stage_error)
}

fn add_inner_stage<F, C>(
    builder: Builder<F, C>,
    spartan: &SpartanZkOutput<F, C>,
) -> Result<Builder<F, C>, WrapperError>
where
    F: Field,
    C: Clone,
{
    builder
        .stage("wrapper-spartan-inner")
        .sumcheck(spartan.inner_statement)
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(spartan.inner_consistency.clone())
        .output_claim_rows(
            vec![WrapperOpeningId::WitnessZ],
            spartan.inner_output_claims.shape.row_len,
            spartan.inner_output_claims.commitments.clone(),
        )
        .input_claim(inner_input_claim())
        .output_claim(inner_output_claim())
        .finish_stage()
        .map_err(blindfold_stage_error)
}

fn outer_output_claim<F: Field>() -> WrapperExpr<F> {
    public(WrapperPublicId::EqTauRx)
        * (opening(WrapperOpeningId::OuterA) * opening(WrapperOpeningId::OuterB)
            - opening(WrapperOpeningId::OuterC))
}

fn inner_input_claim<F: Field>() -> WrapperExpr<F> {
    challenge(WrapperChallengeId::InnerA) * opening(WrapperOpeningId::OuterA)
        + challenge(WrapperChallengeId::InnerB) * opening(WrapperOpeningId::OuterB)
        + challenge(WrapperChallengeId::InnerC) * opening(WrapperOpeningId::OuterC)
}

fn inner_output_claim<F: Field>() -> WrapperExpr<F> {
    public(WrapperPublicId::CombinedMatrixEval) * opening(WrapperOpeningId::WitnessZ)
}

fn blindfold_stage_error(error: jolt_blindfold::Error) -> WrapperError {
    WrapperError::BlindFoldConstructionFailed {
        reason: error.to_string(),
    }
}

fn blindfold_error<F: Field>(error: jolt_blindfold::VerificationError<F>) -> WrapperError {
    WrapperError::BlindFoldConstructionFailed {
        reason: error.to_string(),
    }
}
