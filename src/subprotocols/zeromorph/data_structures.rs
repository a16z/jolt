use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_std::rand::rngs::StdRng;
use ark_ff::UniformRand;
use lazy_static::lazy_static;
use rand_chacha::rand_core::SeedableRng;
use std::sync::{Arc, Mutex};

use crate::utils::math::Math;

//TODO: The SRS is set with a default value of ____ if this is to be changed (extended) use the cli arg and change it manually.
//TODO: add input specifiying monomial or lagrange basis
lazy_static! {
    pub static ref ZEROMORPH_SRS: Arc<Mutex<ZeromorphSRS<20, Bn254>>> =
    Arc::new(Mutex::new(ZeromorphSRS::setup(None)));
}

#[derive(Debug, Clone, Default)]
pub struct ZeromorphSRS<const N_MAX: usize, P: Pairing> {
    g1_powers: Vec<P::G1Affine>,
    g2_powers: Vec<P::G2Affine>,
}

impl<const N_MAX: usize, P: Pairing> ZeromorphSRS<N_MAX, P> {

    fn compute_g_powers<G: CurveGroup>(tau: G::ScalarField) -> Vec<G::Affine> {
        let g_srs = vec![G::zero(); N_MAX - 1];
    
        let g_srs: Vec<G> = std::iter::once(G::generator())
            .chain(g_srs.iter().scan(G::generator(), |state, _| {
                *state *= &tau;
                Some(*state)
            }))
            .collect();
    
        G::normalize_batch(&g_srs)
    }

    pub fn setup(toxic_waste: Option<&[u8]>) -> ZeromorphSRS<N_MAX,P> {
        let tau: &[u8];
        if toxic_waste.is_none() {
            tau = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        } else {
            tau = toxic_waste.unwrap()
        }
        /*
            if ENV_VAR_NOT_PASSED_IN
        */
        let mut bytes = [0u8; 32];
        let len = tau.len();
        bytes[..len].copy_from_slice(&tau[..len]);
        let rng = &mut StdRng::from_seed(bytes);

        let tau = P::ScalarField::rand(rng);
        let g1_powers = Self::compute_g_powers::<P::G1>(tau);
        let g2_powers = Self::compute_g_powers::<P::G2>(tau);
        ZeromorphSRS { g1_powers, g2_powers }
    }

    pub fn get_prover_key(&self) -> ZeromorphProverKey<P> {
       ZeromorphProverKey { g1: self.g1_powers[0], tau_1: self.g1_powers[1], g1_powers: self.g1_powers.clone() } 
    }

    pub fn get_verifier_key(&self) -> ZeromorphVerifierKey<P> {
        let idx = N_MAX - (2_usize.pow(N_MAX.log_2() as u32) - 1);
        ZeromorphVerifierKey { g1: self.g1_powers[0], g2: self.g2_powers[0], tau_2: self.g2_powers[1], tau_N_max_sub_2_N: self.g2_powers[idx] }
    }

}

#[derive(Clone, Debug)]
pub struct ZeromorphProverKey<P: Pairing> {
  // generator
  pub g1: P::G1Affine,
  pub tau_1: P::G1Affine,
  pub g1_powers: Vec<P::G1Affine>,
}

#[derive(Copy, Clone, Debug)]
pub struct ZeromorphVerifierKey<P: Pairing> {
  pub g1: P::G1Affine,
  pub g2: P::G2Affine,
  pub tau_2: P::G2Affine,
  pub tau_N_max_sub_2_N: P::G2Affine,
}

//TODO: can we upgrade the transcript to give not just absorb
#[derive(Clone, Debug)]
pub struct ZeromorphProof<P: Pairing> {
  pub pi: P::G1Affine,
  pub q_hat_com: P::G1Affine,
  pub q_k_com: Vec<P::G1Affine>,
}

#[cfg(test)]
mod test {
    use ark_bn254::Bn254;
    use ark_ec::{pairing::Pairing, AffineRepr};
    use ark_ff::One;
    use std::ops::Mul;
    use super::*;

    fn expected_srs<E: Pairing>(n: usize, seed: &[u8]) -> (Vec<E::G1Affine>, Vec<E::G2Affine>) {

        let mut bytes = [0u8; 32];
        let len = seed.len();
        bytes[..len].copy_from_slice(&seed[..len]);
        let rng = &mut StdRng::from_seed(bytes);

        let tau = E::ScalarField::rand(rng);

        let powers_of_tau: Vec<E::ScalarField> =
            std::iter::successors(Some(E::ScalarField::one()), |p| Some(*p * tau))
                .take(n)
                .collect();

        let g1_gen = E::G1Affine::generator();
        let g2_gen = E::G2Affine::generator();

        let srs_g1: Vec<E::G1Affine> = powers_of_tau
            .iter()
            .map(|tp| g1_gen.mul(tp).into())
            .collect();
        let srs_g2: Vec<E::G2Affine> = powers_of_tau
            .iter()
            .map(|tp| g2_gen.mul(tp).into())
            .collect();

        (srs_g1, srs_g2)
    }

    #[test]
    fn test_srs() {
        const K: i32 = 1;
        const N: usize = 1 << K;
        let seed = b"111111111111111111111111111";

        let (g1_srs_expected, g2_srs_expected) = expected_srs::<Bn254>(N, seed);

        let srs = ZeromorphSRS::<N, Bn254>::setup(Some(seed));
        assert_eq!(g1_srs_expected, srs.g1_powers);
        assert_eq!(g2_srs_expected, srs.g2_powers);
    }
}