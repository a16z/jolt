use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_std::{One, UniformRand, Zero};
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SRS<P: Pairing> {
    pub g1_powers: Vec<P::G1Affine>,
    pub g2_powers: Vec<P::G2Affine>,
    pub g_products: Vec<P::G1Affine>,
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

        let g1_window_size = FixedBase::get_mul_window_size(num_g1_powers);
        let g2_window_size = FixedBase::get_mul_window_size(num_g2_powers);
        let g1_table = FixedBase::get_window_table(scalar_bits, g1_window_size, g1);
        let g2_table = FixedBase::get_window_table(scalar_bits, g2_window_size, g2);

        let (g1_powers_projective, g2_powers_projective) = rayon::join(
            || {
                let beta_powers: Vec<P::ScalarField> = (0..=num_g1_powers)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();
                FixedBase::msm(scalar_bits, g1_window_size, &g1_table, &beta_powers)
            },
            || {
                let beta_powers: Vec<P::ScalarField> = (0..=num_g2_powers)
                    .scan(beta, |acc, _| {
                        let val = *acc;
                        *acc *= beta;
                        Some(val)
                    })
                    .collect();
                FixedBase::msm(scalar_bits, g2_window_size, &g2_table, &beta_powers)
            },
        );

        let (g1_powers, g2_powers) = rayon::join(
            || P::G1::normalize_batch(&g1_powers_projective),
            || P::G2::normalize_batch(&g2_powers_projective),
        );

        // Precompute a commitment to each power-of-two length vector of ones, which is just the sum of each power-of-two length prefix of the SRS
        let num_powers = (g1_powers.len() as f64).log2().floor() as usize + 1;
        let all_ones_coeffs: Vec<P::ScalarField> = vec![P::ScalarField::one(); num_g1_powers + 1];
        let powers_of_2 = (0..num_powers).into_par_iter().map(|i| 1usize << i);
        let g_products = powers_of_2
            .map(|power| {
                <P::G1 as VariableBaseMSM>::msm(&g1_powers[..power], &all_ones_coeffs[..power])
                    .unwrap()
                    .into_affine()
            })
            .collect();

        Self {
            g1_powers,
            g2_powers,
            g_products,
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

#[derive(Clone, Copy, Debug)]
pub enum CommitMode {
    Default,
    // We noticed that most (93%) of the coefficients arising from lasso grand products are 1.
    // This mode uses a precomputed commitment, G, to save some compute.
    // Where G is the commitment to the all-ones vector of length 2^k```
    GrandProduct,
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
        Self::commit_inner(pk, &poly.coeffs, offset, CommitMode::Default)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit")]
    pub fn commit(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, &poly.coeffs, 0, CommitMode::Default)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_with_mode")]
    pub fn commit_with_mode(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        mode: CommitMode,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, &poly.coeffs, 0, mode)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_slice")]
    pub fn commit_slice(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, coeffs, 0, CommitMode::Default)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_slice_with_mode")]
    pub fn commit_slice_with_mode(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
        mode: CommitMode,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, coeffs, 0, mode)
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "KZG::commit_inner")]
    fn commit_inner(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
        offset: usize,
        mode: CommitMode,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if pk.g1_powers().len() < coeffs.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                coeffs.len(),
            ));
        }

        match mode {
            CommitMode::Default => {
                let c = <P::G1 as VariableBaseMSM>::msm(
                    &pk.g1_powers()[offset..coeffs.len()],
                    &coeffs[offset..],
                )
                .unwrap();
                Ok(c.into_affine())
            }
            CommitMode::GrandProduct => {
                let g1_powers = &pk.g1_powers()[offset..coeffs.len()];
                let coeffs = &coeffs[offset..];
                // Commit to the non-1 coefficients first then combine them with the G commitment (all-1s vector) in the SRS
                let (non_one_coeffs, non_one_bases): (Vec<_>, Vec<_>) = coeffs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, coeff)| {
                        if *coeff != P::ScalarField::one() {
                            // Subtract 1 from the coeff because we already have a commitment to the all the 1s
                            Some((*coeff - P::ScalarField::one(), g1_powers[i]))
                        } else {
                            None
                        }
                    })
                    .unzip();

                // Perform MSM for the non-1 coefficients
                let non_one_commitment = if !non_one_coeffs.is_empty() {
                    <P::G1 as VariableBaseMSM>::msm(&non_one_bases, &non_one_coeffs).unwrap()
                } else {
                    P::G1::zero()
                };

                // find the right precomputed g_product to use
                let num_powers = (coeffs.len() as f64).log2();
                if num_powers.fract() != 0.0 {
                    return Err(ProofVerifyError::InvalidKeyLength(coeffs.len()));
                }
                let num_powers = num_powers.floor() as usize;

                // Combine G * H: Multiply the precomputed G commitment with the non-1 commitment (H)
                let final_commitment = pk.srs.g_products[num_powers] + non_one_commitment;
                Ok(final_commitment.into_affine())
            }
        }
    }

    #[tracing::instrument(skip_all, name = "KZG::open")]
    pub fn open(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Result<(P::G1Affine, P::ScalarField), ProofVerifyError>
    where
        <P as Pairing>::ScalarField: JoltField,
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

    fn run_kzg_test<F>(degree_generator: F, commit_mode: CommitMode) -> Result<(), ProofVerifyError>
    where
        F: Fn(&mut ChaCha20Rng) -> usize,
    {
        for i in 0..100 {
            let seed = [i; 32];
            let mut rng = &mut ChaCha20Rng::from_seed(seed);
            let degree = degree_generator(&mut rng);

            let pp = Arc::new(SRS::<Bn254>::setup(&mut rng, degree, 2));
            let (ck, vk) = SRS::trim(pp, degree);
            let p = UniPoly::random::<ChaCha20Rng>(degree, rng);
            let comm = UnivariateKZG::<Bn254>::commit_with_mode(&ck, &p, commit_mode)?;
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

    #[test]
    fn kzg_commit_prove_verify() -> Result<(), ProofVerifyError> {
        run_kzg_test(|rng| rng.gen_range(2..20), CommitMode::Default)
    }

    #[test]
    fn kzg_commit_prove_verify_mode() -> Result<(), ProofVerifyError> {
        // This test uses the grand product optimization and ensures only powers of 2 are used for degree generation
        run_kzg_test(|rng| 1 << rng.gen_range(1..8), CommitMode::GrandProduct)
    }
}
