//! General Inner Product Arguments
//!
//! This is the building block of other Product Arguments like Tipa
use std::marker::PhantomData;

use super::Error;
use crate::{
    msm::Icicle,
    poly::commitment::bmmtv::gipa::GipaProof,
    field::JoltField,
    poly::commitment::bmmtv::{
        afgho::AfghoCommitment, inner_products::MultiexponentiationInnerProduct,
    },
    utils::transcript::Transcript
};
use ark_ec::pairing::Pairing;
use rayon::prelude::*;
use tracing::Level;

impl<P, ProofTranscript> GipaProof<P, ProofTranscript>
where
    P: Pairing,
    P::G1: Icicle,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    /// Returns vector of recursive commitments and transcripts in reverse order
    ///
    /// This is basically bullet-proofs, we have an array with 2^x elements
    ///
    /// [a11, a12, a13, a14, a15, a16, a17, a18]
    ///
    /// We split it in two halves and compute an operation
    ///
    /// [a11 + a15, a12 + a16, a13 + a17, a14 + a18]
    ///
    /// take the commitment and save it
    /// (com_l1,com_r1)  = (com([a11, a12, a13, a14]), com([a15, a16, a17, a18]))
    ///
    /// We repeat this process until we have a single element
    ///
    /// [a21, a22, a23, a24]
    ///
    /// take the commitment and save it
    /// (com_l2, com_r2) = (com([a21, a22]), com([a23, a24]))
    ///
    /// [a21 + a23, a22 +  a24]
    ///
    /// [a31, a32]
    ///
    /// take the commitment and save it
    /// (com_l3, com_r3) = (com([a31]), com([a32]))
    ///
    /// [a31 + a32]
    ///
    /// This final element is our proof + intermediate commitments
    ///
    /// [a41] + [com_1, com2, com3]
    #[tracing::instrument(name = "Gipa::prove", skip_all)]
    pub fn prove(
        mut g1: Vec<P::G1>,
        mut g2: Vec<P::G2>,
        mut scalars: Vec<P::ScalarField>,
        transcript: &mut ProofTranscript,
    ) -> Result<GipaProof<P, ProofTranscript>, Error> {
        // fiat-shamir steps
        let mut commitment_steps = Vec::new();
        // fiat-shamir transcripts
        let mut r_transcript: Vec<P::ScalarField> = Vec::new();
        assert!(g1.len().is_power_of_two());

        // for loop instead of using recursion
        let (final_msg, final_param) = 'recurse: loop {
            let recurse = tracing::span!(Level::TRACE, "New Round", size = g1.len());
            let _ender = recurse.enter();
            match (&g1.as_slice(), &scalars.as_slice(), &g2.as_slice()) {
                ([m_l], [m_r], [param_l]) => {
                    // base case
                    // when we get to zero
                    break 'recurse ((*m_l, *m_r), *param_l);
                }
                (m_l, m_r, p_l) => {
                    // recursive step
                    // Recurse with problem of half size
                    let split = m_l.len() / 2;

                    let m_ll = &m_l[split..];
                    let m_lr = &m_l[..split];
                    let param_ll = &p_l[..split];
                    let param_lr = &p_l[split..];

                    let m_rl = &m_r[..split];
                    let m_rr = &m_r[split..];

                    let cl = tracing::span!(Level::TRACE, "Commit L");
                    let _enter = cl.enter();
                    let com_l = (
                        // commit to first left half
                        AfghoCommitment::commit(param_ll, m_ll)?,
                        // commit to inner pairing from first half of left msg with first half of right message
                        MultiexponentiationInnerProduct::inner_product(m_ll, m_rl)?,
                    );
                    drop(_enter);
                    let cr = tracing::span!(Level::TRACE, "Commit R");
                    let _enter = cr.enter();
                    let com_r = (
                        // commit to second left half
                        AfghoCommitment::commit(param_lr, m_lr)?,
                        // commit to inner pairing from second half of left msg with second half of right message
                        MultiexponentiationInnerProduct::inner_product(m_lr, m_rr)?,
                    );
                    drop(_enter);

                    // Calculate Fiat-Shamir challenge
                    transcript.append_serializable(&com_l.0);
                    transcript.append_point(&com_l.1);
                    transcript.append_serializable(&com_r.0);
                    transcript.append_point(&com_r.1);
                    let c: P::ScalarField = transcript.challenge_scalar();
                    let c_inv = JoltField::inverse(&c).unwrap();

                    // Set up values for next step of recursion
                    let rescale_ml = tracing::span!(Level::TRACE, "Rescale ML");
                    let _enter = rescale_ml.enter();
                    g1 = m_ll
                        .par_iter()
                        .map(|a| *a * c)
                        .zip(m_lr)
                        .map(|(a_1, a_2)| a_1 + a_2)
                        .collect::<Vec<P::G1>>();
                    drop(_enter);

                    let rescale_mr = tracing::span!(Level::TRACE, "Rescale MR");
                    let _enter = rescale_mr.enter();
                    scalars = m_rr
                        .par_iter()
                        .map(|b| *b * c_inv)
                        .zip(m_rl)
                        .map(|(b_1, b_2)| b_1 + b_2)
                        .collect::<Vec<P::ScalarField>>();
                    drop(_enter);

                    let rescale_pl = tracing::span!(Level::TRACE, "Rescale CK1");
                    let _enter = rescale_pl.enter();
                    g2 = param_lr
                        .par_iter()
                        .map(|a| *a * c_inv)
                        .zip(param_ll)
                        .map(|(a_1, a_2)| a_1 + a_2)
                        .collect::<Vec<P::G2>>();
                    drop(_enter);

                    // add commitment steps
                    commitment_steps.push((com_l, com_r));
                    // add scalar used to trancript
                    r_transcript.push(c);
                }
            }
        };
        // reverse them | TODO why?
        r_transcript.reverse();
        commitment_steps.reverse();

        // return the proofs + transcript
        Ok(GipaProof {
            commitment_steps,
            final_message: final_msg,
            scalar_transcript: r_transcript,
            final_commitment_param: final_param,
            _transcript: PhantomData,
        })
    }
}
