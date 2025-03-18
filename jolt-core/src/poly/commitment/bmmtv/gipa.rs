//! General Inner Produc Arguments
//!
//! This is the building block of other Product Arguments like Tipa

use super::{
    commitments::Dhc, inner_products::InnerProduct, mul_helper, Error, InnerProductArgumentError,
};
use crate::field::JoltField;
use crate::poly::commitment::bmmtv::commitments::afgho16::AfghoCommitment;
use crate::poly::commitment::bmmtv::commitments::identity::IdentityOutput;
use crate::poly::commitment::bmmtv::commitments::identity::{DummyParam, IdentityCommitment};
use crate::poly::commitment::bmmtv::inner_products::MultiexponentiationInnerProduct;
use crate::poly::commitment::bmmtv::tipa::structured_scalar_message::SsmDummyCommitment;
use crate::utils::transcript::Transcript;
use anyhow::bail;
use ark_ec::pairing::Pairing;
use ark_ec::pairing::PairingOutput;
use ark_ff::One;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter;
use ark_std::rand::Rng;
use ark_std::{end_timer, start_timer};
use std::marker::PhantomData;
// #[cfg(feature = "rayon")]
// use rayon::prelude::*;

/// General Inner Product Argument
///
/// This is basically how bullet-proofs are built
pub struct Gipa<P, ProofTranscript> {
    _pairing: PhantomData<P>,
    _transcript: PhantomData<ProofTranscript>,
}

/// Proof of [`Gipa`]
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct GipaProof<P: Pairing> {
    pub(crate) commitment_steps: Vec<(
        (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
        (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
    )>,
    pub(crate) final_message: (P::G1, P::ScalarField),
}

/// Transcript in reverse order and commitment bases?
///
// GipaAux<AfghoCommitment<P>, SsmDummyCommitment<P::ScalarField>>,
///
#[allow(unused)]
#[derive(Clone)]
pub struct GipaAux<P: Pairing> {
    pub(crate) scalar_transcript: Vec<P::ScalarField>,
    pub(crate) final_commitment_param: (P::G2, DummyParam),
}

pub struct GipaCommitment<P: Pairing> {
    l_commit: PairingOutput<P>,
    r_commit: P::ScalarField,
    ip_commit: IdentityOutput<P::G1>,
}

#[derive(Clone)]
pub struct GipaParams<P: Pairing> {
    l_params: Vec<P::G2>,
    r_params: Vec<DummyParam>,
    ip_param: DummyParam,
}
impl<P: Pairing> GipaParams<P> {
    pub fn new_aux(l_params: &[P::G2], r_params: &[DummyParam], ip_param: &[DummyParam]) -> Self {
        Self::new(l_params, r_params, &ip_param[0])
    }
    pub fn new(l_params: &[P::G2], r_params: &[DummyParam], ip_param: &DummyParam) -> Self {
        Self {
            l_params: l_params.to_vec(),
            r_params: r_params.to_vec(),
            ip_param: ip_param.clone(),
        }
    }
}

/// Something to keep in mind for Gipas is that Left Message for the inner product is:
///
/// - InnerProduct::LeftMessage = LeftCommitment::Message
/// - InnerProduct::RightMessage = RightCommitment::Message
/// - InnerProduct::Output = InnerProductCommitment::Output
impl<P, ProofTranscript> Gipa<P, ProofTranscript>
where
    P: Pairing,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    /// Generate setup for all commitments and returm them
    ///
    /// For the Inner Product, we only take the first element
    pub fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<GipaParams<P>, Error> {
        Ok(GipaParams {
            l_params: AfghoCommitment::<P>::setup(rng, size)?,
            r_params: SsmDummyCommitment::<P::ScalarField>::setup(rng, size)?,
            ip_param: IdentityCommitment::<P::G1, P::ScalarField>::setup(rng, 1)?
                .pop()
                .unwrap(),
        })
    }

    /// Create a proof for the provided values
    pub fn prove(
        values: (&[P::G1], &[P::ScalarField], &P::G1),
        params: &GipaParams<P>,
        commitment: &GipaCommitment<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<GipaProof<P>, Error> {
        // check if inner_product(left, right) == provided inner pairing
        if MultiexponentiationInnerProduct::inner_product(values.0, values.1)? != *values.2 {
            bail!(InnerProductArgumentError::InnerProductInvalid);
        }
        // check if it's power of 2
        if values.0.len().count_ones() != 1 {
            // Power of 2 length
            bail!(InnerProductArgumentError::MessageLengthInvalid(
                values.0.len(),
                values.1.len(),
            ));
        }
        let valid_left_commitment =
            AfghoCommitment::verify(&params.l_params, values.0, &commitment.l_commit)?;
        let valid_right_commitment =
            SsmDummyCommitment::verify(&params.r_params, values.1, &commitment.r_commit)?;
        let valid_ip_commitment = IdentityCommitment::<P::G1, P::ScalarField>::verify(
            &[params.ip_param.clone()],
            &[*values.2],
            &commitment.ip_commit,
        )?;
        // check if all provided values correspond to the provided commitments
        if !(valid_left_commitment && valid_right_commitment && valid_ip_commitment) {
            bail!(InnerProductArgumentError::InnerProductInvalid);
        }

        // proceed to generate the proof
        let (proof, _) = Self::prove_with_aux((values.0, values.1), params, transcript)?;
        Ok(proof)
    }

    /// Check if the proof with final commitment (size 1) was calculated from the commitment
    pub fn verify(
        params: &GipaParams<P>,
        commitment: GipaCommitment<P>,
        proof: &GipaProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<bool, Error> {
        if params.l_params.len().count_ones() != 1 || params.l_params.len() != params.r_params.len()
        {
            // Power of 2 length
            bail!(InnerProductArgumentError::MessageLengthInvalid(
                params.l_params.len(),
                params.r_params.len(),
            ));
        }
        // Calculate base commitment and transcript
        let (base_com, transcript) = Self::_compute_recursive_challenges(
            (
                commitment.l_commit,
                commitment.r_commit,
                commitment.ip_commit,
            ),
            proof,
            transcript,
        )?;
        // Calculate base commitment keys
        let (ck_a_base, ck_b_base) = Self::_compute_final_commitment_keys(params, &transcript)?;
        // Verify base commitment
        Self::_verify_base_commitment(
            (&ck_a_base, &ck_b_base, &vec![params.ip_param.clone()]),
            base_com,
            proof,
        )
    }

    /// Same as prove, but prepares the slices into vec for you
    pub fn prove_with_aux(
        values: (&[P::G1], &[P::ScalarField]),
        params: &GipaParams<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(GipaProof<P>, GipaAux<P>), Error> {
        let (msg_l, msg_r) = values;
        Self::_prove((msg_l.to_vec(), msg_r.to_vec()), params.clone(), transcript)
    }

    /// Returns vector of recursive commitments and transcripts in reverse order
    ///
    /// This is basically bullet-proofs, we have an array with 2^x elements
    ///
    /// [a11, a12, a13, a14, a15, a16, a17, a18]
    ///
    /// We split it in two halfs and compute an operation
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
    /// This final element is our proof + intermidiate commitments
    ///
    /// [a41] + [com_1, com2, com3]
    fn _prove(
        values: (Vec<P::G1>, Vec<P::ScalarField>),
        GipaParams {
            mut l_params,
            mut r_params,
            ip_param,
        }: GipaParams<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(GipaProof<P>, GipaAux<P>), Error> {
        let (mut msg_l, mut msg_r) = values;
        let ip_param = &[ip_param];
        // fiat-shamir steps
        let mut r_commitment_steps = Vec::new();
        // fiat-shamir transcripts
        let mut r_transcript: Vec<P::ScalarField> = Vec::new();
        assert!(msg_l.len().is_power_of_two());

        // for loop instead of using recursion
        let (final_msg, final_param) = 'recurse: loop {
            let recurse = start_timer!(|| format!("Recurse round size {}", msg_l.len()));
            match (
                &msg_l.as_slice(),
                &msg_r.as_slice(),
                &l_params.as_slice(),
                &r_params.as_slice(),
            ) {
                ([m_l], [m_r], [param_l], &[param_r]) => {
                    // base case
                    // when we get to zero
                    break 'recurse ((*m_l, *m_r), (*param_l, param_r.clone()));
                }
                (m_l, m_r, p_l, p_r) => {
                    // recursive step
                    // Recurse with problem of half size
                    let split = m_l.len() / 2;

                    let m_ll = &m_l[split..];
                    let m_lr = &m_l[..split];
                    let param_ll = &p_l[..split];
                    let param_lr = &p_l[split..];

                    let m_rl = &m_r[..split];
                    let m_rr = &m_r[split..];
                    let param_rl = &p_r[split..];
                    let param_rr = &p_r[..split];

                    let cl = start_timer!(|| "Commit L");
                    let com_l = (
                        // commit to first left half
                        AfghoCommitment::commit(param_ll, m_ll)?,
                        // commit to first right half
                        SsmDummyCommitment::<P::ScalarField>::commit(param_rl, m_rl)?,
                        // commit to inner pairing from first half of left msg with first half of right message
                        IdentityCommitment::<P::G1, P::ScalarField>::commit(
                            ip_param,
                            &[MultiexponentiationInnerProduct::inner_product(m_ll, m_rl)?],
                        )?,
                    );
                    end_timer!(cl);
                    let cr = start_timer!(|| "Commit R");
                    let com_r = (
                        // commit to second left half
                        AfghoCommitment::commit(param_lr, m_lr)?,
                        // commit to second right half
                        SsmDummyCommitment::<P::ScalarField>::commit(param_rr, m_rr)?,
                        // commit to inner pairing from second half of left msg with second half of right message
                        IdentityCommitment::<P::G1, P::ScalarField>::commit(
                            ip_param,
                            &[MultiexponentiationInnerProduct::inner_product(m_lr, m_rr)?],
                        )?,
                    );
                    end_timer!(cr);

                    // Calculate Fiat-Shamir challenge
                    transcript.append_serializable(&com_l.0);
                    transcript.append_serializable(&com_l.1);
                    transcript.append_serializable(&com_l.2);
                    transcript.append_serializable(&com_r.0);
                    transcript.append_serializable(&com_r.1);
                    transcript.append_serializable(&com_r.2);
                    let c: P::ScalarField = transcript.challenge_scalar();
                    let c_inv = JoltField::inverse(&c).unwrap();

                    // Set up values for next step of recursion
                    let rescale_ml = start_timer!(|| "Rescale ML");
                    msg_l = cfg_iter!(m_ll)
                        .map(|a| mul_helper(a, &c))
                        .zip(m_lr)
                        .map(|(a_1, a_2)| a_1 + a_2)
                        .collect::<Vec<P::G1>>();
                    end_timer!(rescale_ml);

                    let rescale_mr = start_timer!(|| "Rescale MR");
                    msg_r = cfg_iter!(m_rr)
                        .map(|b| mul_helper(b, &c_inv))
                        .zip(m_rl)
                        .map(|(b_1, b_2)| b_1 + b_2)
                        .collect::<Vec<P::ScalarField>>();
                    end_timer!(rescale_mr);

                    let rescale_pl = start_timer!(|| "Rescale CK1");
                    l_params = cfg_iter!(param_lr)
                        .map(|a| mul_helper(a, &c_inv))
                        .zip(param_ll)
                        .map(|(a_1, a_2)| a_1 + a_2)
                        .collect::<Vec<P::G2>>();
                    end_timer!(rescale_pl);

                    let rescale_pr = start_timer!(|| "Rescale CK2");
                    r_params = cfg_iter!(param_rl)
                        .map(|b| mul_helper(b, &c))
                        .zip(param_rr)
                        .map(|(b_1, b_2)| b_1 + b_2.clone())
                        .collect::<Vec<DummyParam>>();
                    end_timer!(rescale_pr);

                    // add commitment steps
                    r_commitment_steps.push((com_l, com_r));
                    // add scalar used to trancript
                    r_transcript.push(c);
                    end_timer!(recurse);
                }
            }
        };
        // reverse them | TODO why?
        r_transcript.reverse();
        r_commitment_steps.reverse();

        // return the proofs + transcript
        Ok((
            GipaProof {
                commitment_steps: r_commitment_steps,
                final_message: final_msg,
            },
            GipaAux {
                scalar_transcript: r_transcript,
                final_commitment_param: final_param,
            },
        ))
    }

    // Helper function used to calculate recursive challenges from proof execution (transcript in reverse)
    pub fn verify_recursive_challenge_transcript(
        com: (&PairingOutput<P>, &P::ScalarField, &IdentityOutput<P::G1>),
        proof: &GipaProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<
        (
            (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
            Vec<P::ScalarField>,
        ),
        Error,
    > {
        Self::_compute_recursive_challenges((*com.0, *com.1, com.2.clone()), proof, transcript)
    }

    fn _compute_recursive_challenges(
        com: (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
        proof: &GipaProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<
        (
            (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
            Vec<P::ScalarField>,
        ),
        Error,
    > {
        let (mut com_a, mut com_b, mut com_t) = com;
        let mut r_transcript: Vec<P::ScalarField> = Vec::new();
        for (com_l, com_r) in proof.commitment_steps.iter().rev() {
            // Fiat-Shamir challenge

            transcript.append_serializable(&com_l.0);
            transcript.append_serializable(&com_l.1);
            transcript.append_serializable(&com_l.2);
            transcript.append_serializable(&com_r.0);
            transcript.append_serializable(&com_r.1);
            transcript.append_serializable(&com_r.2);

            let c: P::ScalarField = transcript.challenge_scalar();
            let c_inv = JoltField::inverse(&c).unwrap();

            com_a = mul_helper(&com_l.0, &c) + com_a + mul_helper(&com_r.0, &c_inv);
            com_b = mul_helper(&com_l.1, &c) + com_b + mul_helper(&com_r.1, &c_inv);
            com_t = mul_helper(&com_l.2, &c) + com_t.clone() + mul_helper(&com_r.2, &c_inv);

            r_transcript.push(c);
        }
        r_transcript.reverse();
        Ok(((com_a, com_b, com_t), r_transcript))
    }

    pub(crate) fn _compute_final_commitment_keys(
        GipaParams {
            l_params, r_params, ..
        }: &GipaParams<P>,
        transcript: &[P::ScalarField],
    ) -> Result<(P::G2, DummyParam), Error> {
        // Calculate base commitment keys
        assert!(l_params.len().is_power_of_two());

        let mut ck_a_agg_challenge_exponents = vec![P::ScalarField::one()];
        let mut ck_b_agg_challenge_exponents = vec![P::ScalarField::one()];
        for (i, c) in transcript.iter().enumerate() {
            let c_inv = JoltField::inverse(c).unwrap();
            for j in 0..(2_usize).pow(i as u32) {
                ck_a_agg_challenge_exponents.push(ck_a_agg_challenge_exponents[j] * c_inv);
                ck_b_agg_challenge_exponents.push(ck_b_agg_challenge_exponents[j] * c);
            }
        }
        assert_eq!(ck_a_agg_challenge_exponents.len(), l_params.len());
        //TODO: Optimization: Use VariableMSM multiexponentiation
        let ck_a_base_init = mul_helper(&l_params[0], &ck_a_agg_challenge_exponents[0]);
        let ck_a_base = l_params[1..]
            .iter()
            .zip(&ck_a_agg_challenge_exponents[1..])
            .map(|(g, x)| mul_helper(g, x))
            .fold(ck_a_base_init, |sum, x| sum + x);
        //.reduce(|| ck_a_base_init.clone(), |sum, x| sum + x);
        let ck_b_base_init = mul_helper(&r_params[0], &ck_b_agg_challenge_exponents[0]);
        let ck_b_base = r_params[1..]
            .iter()
            .zip(&ck_b_agg_challenge_exponents[1..])
            .map(|(g, x)| mul_helper(g, x))
            .fold(ck_b_base_init, |sum, x| sum + x);
        //.reduce(|| ck_b_base_init.clone(), |sum, x| sum + x);
        Ok((ck_a_base, ck_b_base))
    }

    pub(crate) fn _verify_base_commitment(
        base_ck: (&P::G2, &DummyParam, &Vec<DummyParam>),
        base_com: (PairingOutput<P>, P::ScalarField, IdentityOutput<P::G1>),
        proof: &GipaProof<P>,
    ) -> Result<bool, Error> {
        let (com_a, com_b, com_t) = base_com;
        let (ck_a_base, ck_b_base, ck_t) = base_ck;
        let a_base = vec![proof.final_message.0];
        let b_base = vec![proof.final_message.1];
        let t_base = vec![MultiexponentiationInnerProduct::inner_product(
            &a_base, &b_base,
        )?];

        Ok(AfghoCommitment::verify(&[*ck_a_base], &a_base, &com_a)?
            && SsmDummyCommitment::<P::ScalarField>::verify(&[ck_b_base.clone()], &b_base, &com_b)?
            && IdentityCommitment::<P::G1, P::ScalarField>::verify(ck_t, &t_base, &com_t)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::commitments::{
            afgho16::AfghoCommitment, identity::IdentityCommitment, random_generators,
        },
        super::inner_products::{InnerProduct, MultiexponentiationInnerProduct},
        *,
    };
    use crate::poly::commitment::bmmtv::tipa::structured_scalar_message::SsmDummyCommitment;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    /// Inner pairing product commitment in G1
    type AfghoBlsG1 = AfghoCommitment<Bn254>;
    /// Pedersen commitment in G1
    type DummySsm = SsmDummyCommitment<<AfghoBlsG1 as Dhc>::Scalar>;
    // IdentityCommitment<<Bn254 as Pairing>::ScalarField, <Bn254 as Pairing>::G2>;
    const TEST_SIZE: usize = 8;

    #[test]
    fn multiexponentiation_inner_product_test() {
        type IP = MultiexponentiationInnerProduct<<Bn254 as Pairing>::G1>;
        type Ipc = IdentityCommitment<<Bn254 as Pairing>::G1, <Bn254 as Pairing>::ScalarField>;
        type MultiExpGIPA = Gipa<Bn254, KeccakTranscript>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let params = MultiExpGIPA::setup(&mut rng, TEST_SIZE).unwrap();
        let m_a = random_generators(&mut rng, TEST_SIZE);
        let mut m_b = Vec::new();
        for _ in 0..TEST_SIZE {
            m_b.push(<Bn254 as Pairing>::ScalarField::rand(&mut rng));
        }
        let l_commit = AfghoBlsG1::commit(&params.l_params, &m_a).unwrap();
        let r_commit = DummySsm::commit(&params.r_params, &m_b).unwrap();
        let t = vec![IP::inner_product(&m_a, &m_b).unwrap()];
        let ip_commit = Ipc::commit(&[params.ip_param.clone()], &t).unwrap();

        let commitment = GipaCommitment {
            l_commit,
            r_commit,
            ip_commit,
        };

        let mut transcript = KeccakTranscript::new(b"test");

        let proof = MultiExpGIPA::prove((&m_a, &m_b, &t[0]), &params, &commitment, &mut transcript)
            .unwrap();

        let mut transcript = KeccakTranscript::new(b"test");

        assert!(MultiExpGIPA::verify(&params, commitment, &proof, &mut transcript).unwrap());
    }
}
