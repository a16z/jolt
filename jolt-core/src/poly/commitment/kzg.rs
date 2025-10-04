use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, UniformRand, Zero};
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
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
    ) -> Self
    where
        P::ScalarField: JoltField,
    {
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
        let all_ones_coeffs: Vec<u8> = vec![1; num_g1_powers + 1];
        let powers_of_2 = (0..num_powers).into_par_iter().map(|i| 1usize << i);
        let g_products = powers_of_2
            .map(|power| {
                <P::G1 as VariableBaseMSM>::msm_u8(&g1_powers[..power], &all_ones_coeffs[..power])
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
        let pk = KZGProverKey::new(params, 0, max_degree + 1);
        let vk = KZGVerifierKey::from(&pk);
        (pk, vk)
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct KZGProverKey<P: Pairing> {
    pub(crate) srs: Arc<SRS<P>>,
    // offset to read into SRS
    offset: usize,
    // max size of srs
    pub(crate) supported_size: usize,
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

    pub fn g2_powers(&self) -> &[P::G2Affine] {
        &self.srs.g2_powers
    }

    pub fn len(&self) -> usize {
        self.g1_powers().len()
    }
}

#[derive(Clone, Copy, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct KZGVerifierKey<P: Pairing> {
    pub g1: P::G1Affine,
    pub g2: P::G2Affine,
    pub alpha_g1: P::G1Affine,
    pub beta_g2: P::G2Affine,
}

impl<P: Pairing> From<&KZGProverKey<P>> for KZGVerifierKey<P> {
    fn from(pk: &KZGProverKey<P>) -> Self {
        let g1 = pk.g1_powers()[0];
        let g2 = pk.g2_powers()[0];
        let alpha_g1 = pk.g1_powers()[1];
        let beta_g2 = pk.g2_powers()[1];
        Self {
            g1,
            g2,
            alpha_g1,
            beta_g2,
        }
    }
}

/// Marker trait
pub trait Group<P: Pairing> {
    type Curve: CurveGroup;
}

/// Marker for operations in G1
pub enum G1 {}
impl<P: Pairing> Group<P> for G1 {
    type Curve = P::G1;
}

/// Marker for operations in G2
pub enum G2 {}
impl<P: Pairing> Group<P> for G2 {
    type Curve = P::G2;
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct UnivariateKZG<P: Pairing, G: Group<P> = G1> {
    _phantom: PhantomData<(P, G)>,
}

impl<P: Pairing> UnivariateKZG<P>
where
    P::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all, name = "KZG::commit_batch")]
    pub fn commit_batch<U>(
        pk: &KZGProverKey<P>,
        polys: &[U],
    ) -> Result<Vec<P::G1Affine>, ProofVerifyError>
    where
        U: Borrow<MultilinearPolynomial<P::ScalarField>> + Sync,
    {
        let g1_powers = &pk.g1_powers();

        // batch commit requires all batches to have the same length
        assert!(polys
            .par_iter()
            .all(|s| s.borrow().len() == polys[0].borrow().len()));

        if let Some(invalid) = polys
            .iter()
            .find(|coeffs| (*coeffs).borrow().len() > g1_powers.len())
        {
            return Err(ProofVerifyError::KeyLengthError(
                g1_powers.len(),
                invalid.borrow().len(),
            ));
        }

        let msm_size = polys[0].borrow().len();
        let commitments = <P::G1 as VariableBaseMSM>::batch_msm(&g1_powers[..msm_size], polys);
        Ok(commitments.into_iter().map(|c| c.into_affine()).collect())
    }

    // This API will try to minimize copies to the GPU or just do the batches in parallel on the CPU
    #[tracing::instrument(skip_all, name = "KZG::commit_variable_batch")]
    pub fn commit_variable_batch(
        pk: &KZGProverKey<P>,
        polys: &[MultilinearPolynomial<P::ScalarField>],
    ) -> Result<Vec<P::G1Affine>, ProofVerifyError> {
        let g1_powers = &pk.g1_powers();

        // batch commit requires all batches be less than the bases in size
        if let Some(invalid) = polys.iter().find(|poly| poly.len() > g1_powers.len()) {
            return Err(ProofVerifyError::KeyLengthError(
                g1_powers.len(),
                invalid.len(),
            ));
        }

        let commitments = <P::G1 as VariableBaseMSM>::batch_msm(g1_powers, polys);
        Ok(commitments.into_iter().map(|c| c.into_affine()).collect())
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_variable_batch_univariate")]
    pub fn commit_variable_batch_univariate(
        pk: &KZGProverKey<P>,
        polys: &[UniPoly<P::ScalarField>],
    ) -> Result<Vec<P::G1Affine>, ProofVerifyError> {
        let g1_powers = &pk.g1_powers();

        // batch commit requires all batches be less than the bases in size
        if let Some(invalid) = polys
            .iter()
            .find(|poly| poly.coeffs.len() > g1_powers.len())
        {
            return Err(ProofVerifyError::KeyLengthError(
                g1_powers.len(),
                invalid.coeffs.len(),
            ));
        }

        let commitments = <P::G1 as VariableBaseMSM>::batch_msm_univariate(g1_powers, polys);
        Ok(commitments.into_iter().map(|c| c.into_affine()).collect())
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_offset")]
    pub fn commit_offset(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        offset: usize,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, &poly.coeffs, offset)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit")]
    pub fn commit(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        Self::commit_inner(pk, &poly.coeffs, 0)
    }

    #[tracing::instrument(skip_all, name = "KZG::commit_as_univariate")]
    pub fn commit_as_univariate(
        pk: &KZGProverKey<P>,
        poly: &MultilinearPolynomial<P::ScalarField>,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        if pk.g1_powers().len() < poly.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                poly.len(),
            ));
        }

        let c = <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers()[..poly.original_len()], poly)?;
        Ok(c.into_affine())
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "KZG::commit_inner")]
    fn commit_inner(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
        offset: usize,
    ) -> Result<P::G1Affine, ProofVerifyError> {
        let final_commitment = Self::commit_inner_helper(pk, &coeffs[offset..], offset)?;
        Ok(final_commitment.into_affine())
    }

    #[inline]
    pub(crate) fn commit_inner_helper(
        pk: &KZGProverKey<P>,
        coeffs: &[P::ScalarField],
        offset: usize,
    ) -> Result<P::G1, ProofVerifyError> {
        if pk.g1_powers().len() < offset + coeffs.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pk.g1_powers().len(),
                offset + coeffs.len(),
            ));
        }

        let c = <P::G1 as VariableBaseMSM>::msm_field_elements(
            &pk.g1_powers()[offset..offset + coeffs.len()],
            coeffs,
        )?;

        Ok(c)
    }

    #[tracing::instrument(skip_all, name = "KZG::open")]
    pub fn open(
        pk: &KZGProverKey<P>,
        poly: &UniPoly<P::ScalarField>,
        point: &<P::ScalarField as JoltField>::Challenge,
    ) -> Result<(P::G1Affine, P::ScalarField), ProofVerifyError>
    where
        <P as Pairing>::ScalarField: JoltField,
    {
        let proof = Self::generic_open(pk.g1_powers(), poly, *point)?;
        let evaluation = poly.evaluate(point);
        Ok((proof.into_affine(), evaluation))
    }

    pub fn verify(
        vk: &KZGVerifierKey<P>,
        commitment: &P::G1Affine,
        point: &<P::ScalarField as JoltField>::Challenge,
        proof: &P::G1Affine,
        evaluation: &P::ScalarField,
    ) -> Result<bool, ProofVerifyError> {
        Ok(P::multi_pairing(
            [
                commitment.into_group() - vk.g1.into_group() * evaluation,
                -proof.into_group(),
            ],
            [
                vk.g2,
                (vk.beta_g2.into_group() - (vk.g2 * (*point).into())).into(),
            ],
        )
        .is_zero())
    }
}

impl<P: Pairing, G: Group<P>> UnivariateKZG<P, G>
where
    P::ScalarField: JoltField,
    G::Curve: CurveGroup<ScalarField = P::ScalarField>,
{
    #[tracing::instrument(skip_all, name = "KZG::open")]
    pub fn generic_open(
        powers: &[<G::Curve as CurveGroup>::Affine],
        poly: &UniPoly<P::ScalarField>,
        point: <P::ScalarField as JoltField>::Challenge,
    ) -> Result<G::Curve, ProofVerifyError>
    where
        <P as Pairing>::ScalarField: JoltField,
    {
        let divisor = UniPoly::from_coeff(vec![-point.into(), P::ScalarField::one()]);
        let (witness_poly, _) = poly.divide_with_remainder(&divisor).unwrap();
        let proof = <G::Curve as VariableBaseMSM>::msm_field_elements(
            &powers[..witness_poly.coeffs.len()],
            witness_poly.coeffs.as_slice(),
        )?;
        Ok(proof)
    }
}

impl<P: Pairing> UnivariateKZG<P, G2> {
    pub fn verify_g2(
        v_srs: &KZGVerifierKey<P>,
        commitment: P::G2,
        point: P::ScalarField,
        proof: P::G2,
        evaluation: P::ScalarField,
    ) -> bool {
        P::multi_pairing(
            [
                v_srs.g1.into_group(),
                v_srs.alpha_g1.into_group() - v_srs.g1 * point,
            ],
            [commitment - v_srs.g2.into_group() * evaluation, -proof],
        )
        .is_zero()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Bn254;
    use ark_std::rand::Rng;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn run_kzg_test<F>(degree_generator: F) -> Result<(), ProofVerifyError>
    where
        F: Fn(&mut ChaCha20Rng) -> usize,
    {
        for i in 0..100 {
            let seed = [i; 32];
            let mut rng = &mut ChaCha20Rng::from_seed(seed);
            let degree = degree_generator(rng);

            let pp = Arc::new(SRS::<Bn254>::setup(&mut rng, degree, 2));
            let (ck, vk) = SRS::trim(pp, degree);
            let p = UniPoly::random::<ChaCha20Rng>(degree, rng);
            let comm = UnivariateKZG::<Bn254>::commit(&ck, &p)?;
            let point = <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::random(&mut rng);
            let (proof, value) = UnivariateKZG::<Bn254>::open(&ck, &p, &point)?;
            assert!(
                UnivariateKZG::<_, G1>::verify(&vk, &comm, &point, &proof, &value)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }

    #[test]
    fn kzg_commit_prove_verify() -> Result<(), ProofVerifyError> {
        run_kzg_test(|rng| rng.gen_range(2..20))
    }
}
