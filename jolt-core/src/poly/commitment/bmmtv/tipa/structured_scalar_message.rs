use super::super::{
    commitments::{identity::DummyParam, Dhc},
    gipa::GipaParams,
    gipa::{Gipa, GipaProof},
    inner_products::InnerProduct,
    tipa::{
        prove_commitment_key_kzg_opening, structured_generators_scalar_power,
        verify_commitment_key_g2_kzg_opening, Srs, VerifierSrs,
    },
    Error,
};
use crate::field::JoltField;
use ark_ec::{pairing::Pairing, Group};
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, rand::Rng};
use ark_std::{end_timer, start_timer};
use digest::Digest;
use std::{marker::PhantomData, ops::MulAssign};
//TODO: Properly generalize the non-committed message approach of SIPP and MIPP to GIPA
//TODO: Structured message is a special case of the non-committed message and does not rely on TIPA
//TODO: Can support structured group element messages as well as structured scalar messages

/// Use placeholder commitment to commit to vector in clear during GIPA execution
#[derive(Clone)]
pub struct SsmDummyCommitment<F> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> Dhc for SsmDummyCommitment<F> {
    type Scalar = F;
    type Message = F;
    type Param = DummyParam;
    type Output = F;

    fn setup<R: Rng>(_rng: &mut R, size: usize) -> Result<Vec<Self::Param>, Error> {
        Ok(vec![DummyParam {}; size])
    }

    //TODO: Doesn't include message which means scalar b not included in generating challenges
    fn commit(_k: &[Self::Param], _m: &[Self::Message]) -> Result<Self::Output, Error> {
        Ok(F::zero())
    }
}
type GipaWithSsmProof<Com, IpCom> = GipaProof<Com, SsmDummyCommitment<<Com as Dhc>::Scalar>, IpCom>;

// /// General Inner Product Argument with Trusted Setup
pub type GipaWithSsm<Ip, Com, IpCom, D> =
    Gipa<Ip, Com, SsmDummyCommitment<<Com as Dhc>::Scalar>, IpCom, D>;

/// Pairing-based instantiation of GIPA with an updatable
/// (trusted) structured reference string (SRS) to achieve
/// logarithmic-time verification
pub struct TipaWithSsm<Ip, LCom, IpCom, P, D> {
    _inner_product: PhantomData<Ip>,
    _left_commitment: PhantomData<LCom>,
    _inner_product_commitment: PhantomData<IpCom>,
    _pair: PhantomData<P>,
    _digest: PhantomData<D>,
}

/// Proof of [`TipaWithSsm`]
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct TipaWithSsmProof<P, LCom, IpCom>
where
    P: Pairing,
    LCom: Dhc,
    IpCom: Dhc,
{
    gipa_proof: GipaWithSsmProof<LCom, IpCom>,
    final_ck: LCom::Param,
    final_ck_proof: P::G2,
}

impl<Ip, LCom, IpCom, P, D> TipaWithSsm<Ip, LCom, IpCom, P, D>
where
    P::ScalarField: JoltField,
    D: Digest,
    P: Pairing,
    Ip: InnerProduct<
        LeftMessage = LCom::Message,
        RightMessage = LCom::Scalar,
        Output = IpCom::Message,
    >,
    LCom: Dhc<Scalar = P::ScalarField, Param = P::G2>,
    IpCom: Dhc<Scalar = LCom::Scalar>,
    IpCom::Message: MulAssign<P::ScalarField>,
    IpCom::Param: MulAssign<P::ScalarField>,
    IpCom::Output: MulAssign<P::ScalarField>,
    LCom::Message: MulAssign<P::ScalarField>,
    LCom::Output: MulAssign<P::ScalarField>,
{
    //TODO: Don't need full TIPA SRS since only using one set of powers
    pub fn setup<R: Rng>(rng: &mut R, size: usize) -> Result<(Srs<P>, IpCom::Param), Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let beta = <P::ScalarField>::rand(rng);
        let g = P::G1::generator();
        let h = P::G2::generator();
        Ok((
            Srs {
                g_alpha_powers: structured_generators_scalar_power(2 * size - 1, &g, &alpha),
                h_beta_powers: structured_generators_scalar_power(2 * size - 1, &h, &beta),
                g_beta: g * beta,
                h_alpha: h * alpha,
            },
            IpCom::setup(rng, 1)?.pop().unwrap(),
        ))
    }

    pub fn prove_with_structured_scalar_message(
        srs: &Srs<P>,
        values: (&[Ip::LeftMessage], &[Ip::RightMessage]),
        ck: (&[LCom::Param], &IpCom::Param),
    ) -> Result<TipaWithSsmProof<P, LCom, IpCom>, Error> {
        // Run GIPA
        let gipa = start_timer!(|| "GIPA");
        let (proof, aux) = GipaWithSsm::<Ip, LCom, IpCom, D>::prove_with_aux(
            values,
            &GipaParams::new_aux(ck.0, &vec![DummyParam {}; values.1.len()], &[ck.1.clone()]),
        )?;
        end_timer!(gipa);

        // Prove final commitment key is wellformed
        let ck_kzg = start_timer!(|| "Prove commitment key");
        let (ck_a_final, _) = aux.final_commitment_param;
        let transcript = aux.scalar_transcript;
        let transcript_inverse = cfg_iter!(transcript)
            .map(|x| JoltField::inverse(x).unwrap())
            .collect::<Vec<_>>();

        // KZG challenge point
        let mut counter_nonce: usize = 0;
        let c = loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            transcript
                .first()
                .unwrap()
                .serialize_uncompressed(&mut hash_input)?;
            ck_a_final.serialize_uncompressed(&mut hash_input)?;
            if let Some(c) = LCom::Scalar::from_random_bytes(&D::digest(&hash_input)) {
                break c;
            };
            counter_nonce += 1;
        };

        // Complete KZG proof
        let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
            &srs.h_beta_powers,
            &transcript_inverse,
            &<P::ScalarField>::one(),
            &c,
        )?;
        end_timer!(ck_kzg);

        Ok(TipaWithSsmProof {
            gipa_proof: proof,
            final_ck: ck_a_final,
            final_ck_proof: ck_a_kzg_opening,
        })
    }

    pub fn verify_with_structured_scalar_message(
        v_srs: &VerifierSrs<P>,
        ck_t: &IpCom::Param,
        com: (&LCom::Output, &IpCom::Output),
        scalar_b: &P::ScalarField,
        proof: &TipaWithSsmProof<P, LCom, IpCom>,
    ) -> Result<bool, Error> {
        let (base_com, transcript) =
            GipaWithSsm::<Ip, LCom, IpCom, D>::verify_recursive_challenge_transcript(
                (com.0, scalar_b, com.1),
                &proof.gipa_proof,
            )?;
        let transcript_inverse = cfg_iter!(transcript)
            .map(|x| JoltField::inverse(x).unwrap())
            .collect::<Vec<_>>();

        let ck_a_final = &proof.final_ck;
        let ck_a_proof = &proof.final_ck_proof;

        // KZG challenge point
        let mut counter_nonce: usize = 0;
        let c = loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            transcript
                .first()
                .unwrap()
                .serialize_uncompressed(&mut hash_input)?;
            ck_a_final.serialize_uncompressed(&mut hash_input)?;
            if let Some(c) = LCom::Scalar::from_random_bytes(&D::digest(&hash_input)) {
                break c;
            };
            counter_nonce += 1;
        };

        // Check commitment key
        let ck_a_valid = verify_commitment_key_g2_kzg_opening(
            v_srs,
            ck_a_final,
            ck_a_proof,
            &transcript_inverse,
            &P::ScalarField::one(),
            &c,
        )?;

        // Compute final scalar
        let mut power_2_b = *scalar_b;
        let mut product_form = Vec::new();
        for x in transcript.iter() {
            product_form.push(<P::ScalarField>::one() + (JoltField::inverse(x).unwrap() * power_2_b));
            power_2_b *= power_2_b.clone();
        }
        let b_base = cfg_iter!(product_form).product::<P::ScalarField>();

        // Verify base inner product commitment
        let (com_a, _, com_t) = base_com;
        let a_base = vec![proof.gipa_proof.final_message.0.clone()];
        let t_base = vec![Ip::inner_product(&a_base, &[b_base])?];
        let base_valid = LCom::verify(&[*ck_a_final], &a_base, &com_a)?
            && IpCom::verify(&[ck_t.clone()], &t_base, &com_t)?;

        Ok(ck_a_valid && base_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::super::{
            commitments::{
                afgho16::AfghoCommitment, identity::IdentityCommitment, random_generators,
            },
            inner_products::{InnerProduct, MultiexponentiationInnerProduct},
        },
        *,
    };
    use ark_bn254::Bn254;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use sha3::Sha3_256;

    type BlsAfghoG1 = AfghoCommitment<Bn254>;
    type BlsScalarField = <Bn254 as Pairing>::ScalarField;
    type BlsG1 = <Bn254 as Pairing>::G1;

    const TEST_SIZE: usize = 8;

    fn structured_scalar_power<F: Field>(num: usize, s: &F) -> Vec<F> {
        let mut powers = vec![F::one()];
        for i in 1..num {
            powers.push(powers[i - 1] * s);
        }
        powers
    }

    #[test]
    fn tipa_ssm_multiexponentiation_inner_product_test() {
        type IP = MultiexponentiationInnerProduct<BlsG1>;
        type Ipc = IdentityCommitment<BlsG1, BlsScalarField>;
        type MultiExpTipa = TipaWithSsm<IP, BlsAfghoG1, Ipc, Bn254, Sha3_256>;

        let mut rng = StdRng::seed_from_u64(0u64);
        let (srs, ck_t) = MultiExpTipa::setup(&mut rng, TEST_SIZE).unwrap();
        let (ck_a, _) = srs.get_commitment_keys();
        let v_srs = srs.get_verifier_key();
        let m_a = random_generators(&mut rng, TEST_SIZE);
        let b = BlsScalarField::rand(&mut rng);
        let m_b = structured_scalar_power(TEST_SIZE, &b);
        let com_a = BlsAfghoG1::commit(&ck_a, &m_a).unwrap();
        let t = vec![IP::inner_product(&m_a, &m_b).unwrap()];
        let com_t = Ipc::commit(&[ck_t.clone()], &t).unwrap();

        let proof =
            MultiExpTipa::prove_with_structured_scalar_message(&srs, (&m_a, &m_b), (&ck_a, &ck_t))
                .unwrap();

        assert!(MultiExpTipa::verify_with_structured_scalar_message(
            &v_srs,
            &ck_t,
            (&com_a, &com_t),
            &b,
            &proof
        )
        .unwrap());
    }
}
