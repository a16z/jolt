use crate::msm::VariableBaseMSM;
use crate::poly::{unipoly::UniPoly, field::JoltField};
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::PrimeField;
use ark_std::UniformRand;
use rand_core::{CryptoRng, RngCore};
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[derive(Clone, Debug)]
pub struct SRS<P: Pairing> {
    pub g1_powers: Vec<P::G1Affine>,
    pub g2_powers: Vec<P::G2Affine>,
}

impl<P: Pairing> SRS<P> {
    pub fn setup<R: RngCore + CryptoRng>(mut rng: &mut R, max_degree: usize) -> Self {
        let beta = P::ScalarField::rand(&mut rng);
        let g1 = P::G1::rand(&mut rng);
        let g2 = P::G2::rand(&mut rng);

        let beta_powers: Vec<P::ScalarField> = (0..=max_degree)
            .scan(beta, |acc, _| {
                let val = *acc;
                *acc *= beta;
                Some(val)
            })
            .collect();

        let window_size = FixedBase::get_mul_window_size(max_degree);
        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;

        let (g1_powers_projective, g2_powers_projective) = rayon::join(
            || {
                let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
                FixedBase::msm(scalar_bits, window_size, &g1_table, &beta_powers)
            },
            || {
                let g2_table = FixedBase::get_window_table(scalar_bits, window_size, g2);
                FixedBase::msm(scalar_bits, window_size, &g2_table, &beta_powers)
            },
        );

        let (g1_powers, g2_powers) = rayon::join(
            || P::G1::normalize_batch(&g1_powers_projective),
            || P::G2::normalize_batch(&g2_powers_projective),
        );

        Self {
            g1_powers,
            g2_powers,
        }
    }

    pub fn trim(params: Arc<Self>, max_degree: usize) -> (KZGProverKey<P>, KZGVerifierKey<P>) {
        assert!(params.g1_powers.len() > 0, "max_degree is 0");
        assert!(max_degree < params.g1_powers.len(), "SRS length is less than size");
        let g1 = params.g1_powers[0];
        let g2 = params.g2_powers[0];
        let beta_g2 = params.g2_powers[1];
        let pk = KZGProverKey::new(params, 0, max_degree + 1);
        let vk = KZGVerifierKey { g1, g2, beta_g2 };
        (pk, vk)
    }
}

#[derive(Clone, Debug)]
pub struct KZGProverKey<P: Pairing> {
    srs: Arc<SRS<P>>,
    // offset to read into SRS
    offset: usize,
    // max size of srs
    supported_size: usize,
}

impl<P: Pairing> KZGProverKey<P> {
    pub fn new(srs: Arc<SRS<P>>, offset: usize, supported_size: usize) -> Self {
        assert!(
            srs.g1_powers.len() >= offset + supported_size,
            "not enough powers (req: {} from offset {}) in the SRS (length: {})",
            supported_size,
            offset,
            srs.g1_powers.len()
        );
        Self {
            srs,
            offset,
            supported_size,
        }
    }

    pub fn g1_powers(&self) -> &[P::G1Affine] {
        &self.srs.g1_powers[self.offset..self.offset + self.supported_size]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KZGVerifierKey<P: Pairing> {
    pub g1: P::G1Affine,
    pub g2: P::G2Affine,
    pub beta_g2: P::G2Affine,
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct UVKZGPCS<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> UVKZGPCS<P>
where
    <P as Pairing>::ScalarField: JoltField,
{
    pub fn commit_offset(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        offset: usize,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if poly.degree() > pk.g1_powers().len() {
            return Err(ProofVerifyError::KeyLengthError(
                poly.degree(),
                pk.g1_powers().len(),
            ));
        }

        let scalars = poly.as_vec();
        let bases = pk.g1_powers();
        let c = <P::G1 as VariableBaseMSM>::msm(
            &bases[offset..scalars.len()],
            &poly.as_vec()[offset..],
        )
        .unwrap();

        Ok(c.into_affine())
    }

    pub fn commit(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if poly.degree() > pk.g1_powers().len() {
            return Err(ProofVerifyError::KeyLengthError(
                poly.degree(),
                pk.g1_powers().len(),
            ));
        }
        let c = <P::G1 as VariableBaseMSM>::msm(
            &pk.g1_powers()[..poly.as_vec().len()],
            &poly.as_vec().as_slice(),
        )
        .unwrap();
        Ok(c.into_affine())
    }

    pub fn open(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Result<(P::G1Affine, P::ScalarField), ProofVerifyError>
    where
        <P as ark_ec::pairing::Pairing>::ScalarField: JoltField,
    {
        let divisor = UniPoly::from_coeff(vec![-*point, P::ScalarField::one()]);
        let (witness_poly, _) = poly.divide_with_q_and_r(&divisor).unwrap();
        let proof = <P::G1 as VariableBaseMSM>::msm(
            &pk.g1_powers()[..witness_poly.as_vec().len()],
            &witness_poly.as_vec().as_slice(),
        )
        .unwrap();
        let evaluation = poly.evaluate(point);
        Ok((proof.into_affine(), evaluation))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::{Bn254, Fr};
    use ark_ec::AffineRepr;
    use ark_std::{rand::Rng, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn kzg_verify<P: Pairing>(
        vk: &KZGVerifierKey<P>,
        commitment: &P::G1Affine,
        point: &P::ScalarField,
        proof: &P::G1Affine,
        evaluation: &P::ScalarField,
    ) -> Result<bool, ProofVerifyError> {
        let lhs = P::pairing(
            commitment.into_group() - vk.g1.into_group() * evaluation,
            vk.g2,
        );
        let rhs = P::pairing(proof, vk.beta_g2.into_group() - (vk.g2 * point));
        Ok(lhs == rhs)
    }

    #[test]
    fn kzg_commit_prove_verify() -> Result<(), ProofVerifyError> {
        let seed = b"11111111111111111111111111111111";
        for _ in 0..100 {
            let mut rng = &mut ChaCha20Rng::from_seed(*seed);
            let degree = rng.gen_range(2..20);

            let pp = Arc::new(SRS::<Bn254>::setup(&mut rng, degree));
            let (ck, vk) = SRS::trim(pp, degree);
            let p = UniPoly::random::<ChaCha20Rng>(degree, rng);
            let comm = UVKZGPCS::<Bn254>::commit(&ck, &p)?;
            let point = Fr::rand(rng);
            let (proof, value) = UVKZGPCS::<Bn254>::open(&ck, &p, &point)?;
            assert!(
                kzg_verify(&vk, &comm, &point, &proof, &value)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }
}