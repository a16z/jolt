use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_std::{One, UniformRand, Zero};
use rand_core::{CryptoRng, RngCore};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SRS<P: Pairing> {
    pub g1_powers: Vec<P::G1Affine>,
    pub g2_powers: Vec<P::G2Affine>,
}

impl<P: Pairing> SRS<P> {
    pub fn setup<R: RngCore + CryptoRng>(
        mut rng: &mut R,
        num_g1_powers: usize,
        num_g2_powers: usize,
    ) -> Self {
        let beta = P::ScalarField::rand(&mut rng);
        let g1 = P::G1::rand(&mut rng);
        let g2 = P::G2::rand(&mut rng);

        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;

        let (g1_powers_projective, g2_powers_projective) = rayon::join(
            || {
                let beta_powers: Vec<P::ScalarField> = (0..=num_g1_powers)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();
                let window_size = FixedBase::get_mul_window_size(num_g1_powers);
                let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
                FixedBase::msm(scalar_bits, window_size, &g1_table, &beta_powers)
            },
            || {
                let beta_powers: Vec<P::ScalarField> = (0..=num_g2_powers)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();

                let window_size = FixedBase::get_mul_window_size(num_g2_powers);
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
        assert!(!params.g1_powers.is_empty(), "max_degree is 0");
        assert!(
            max_degree < params.g1_powers.len(),
            "SRS length is less than size"
        );
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
pub struct UnivariateKZG<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> UnivariateKZG<P>
where
    <P as Pairing>::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all, name = "KZG::commit_offset")]
    pub fn commit_offset(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        offset: usize,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if pk.g1_powers().len() < poly.coeffs.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                poly.coeffs.len(),
            ));
        }

        let bases = pk.g1_powers();
        let c = <P::G1 as VariableBaseMSM>::msm(
            &bases[offset..poly.coeffs.len()],
            &poly.coeffs[offset..],
        )
        .unwrap();

        Ok(c.into_affine())
    }

    #[tracing::instrument(skip_all, name = "KZG::commit")]
    pub fn commit(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if pk.g1_powers().len() < poly.coeffs.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                poly.coeffs.len(),
            ));
        }
        let c = <P::G1 as VariableBaseMSM>::msm(
            &pk.g1_powers()[..poly.coeffs.len()],
            poly.coeffs.as_slice(),
        )
        .unwrap();
        Ok(c.into_affine())
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_slice")]
    pub fn commit_slice(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if pk.g1_powers().len() < coeffs.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                coeffs.len(),
            ));
        }
        let c = <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers()[..coeffs.len()], coeffs).unwrap();
        Ok(c.into_affine())
    }

    #[tracing::instrument(skip_all, name = "KZG::open")]
    pub fn open(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Result<(P::G1Affine, P::ScalarField), ProofVerifyError>
    where
        <P as ark_ec::pairing::Pairing>::ScalarField: JoltField,
    {
        let divisor = UniPoly::from_coeff(vec![-*point, P::ScalarField::one()]);
        let (witness_poly, _) = poly.divide_with_remainder(&divisor).unwrap();
        let proof = <P::G1 as VariableBaseMSM>::msm(
            &pk.g1_powers()[..witness_poly.coeffs.len()],
            witness_poly.coeffs.as_slice(),
        )
        .unwrap();
        let evaluation = poly.evaluate(point);
        Ok((proof.into_affine(), evaluation))
    }

    pub fn verify(
        vk: &KZGVerifierKey<P>,
        commitment: &P::G1Affine,
        point: &P::ScalarField,
        proof: &P::G1Affine,
        evaluation: &P::ScalarField,
    ) -> Result<bool, ProofVerifyError> {
        Ok(P::multi_pairing(
            [
                commitment.into_group() - vk.g1.into_group() * evaluation,
                -proof.into_group(),
            ],
            [vk.g2, (vk.beta_g2.into_group() - (vk.g2 * point)).into()],
        )
        .is_zero())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::{Bn254, Fr};
    use ark_std::{rand::Rng, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn kzg_commit_prove_verify() -> Result<(), ProofVerifyError> {
        let seed = b"11111111111111111111111111111111";
        for _ in 0..100 {
            let mut rng = &mut ChaCha20Rng::from_seed(*seed);
            let degree = rng.gen_range(2..20);

            let pp = Arc::new(SRS::<Bn254>::setup(&mut rng, degree, 2));
            let (ck, vk) = SRS::trim(pp, degree);
            let p = UniPoly::random::<ChaCha20Rng>(degree, rng);
            let comm = UnivariateKZG::<Bn254>::commit(&ck, &p)?;
            let point = Fr::rand(rng);
            let (proof, value) = UnivariateKZG::<Bn254>::open(&ck, &p, &point)?;
            assert!(
                UnivariateKZG::verify(&vk, &comm, &point, &proof, &value)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }
}
