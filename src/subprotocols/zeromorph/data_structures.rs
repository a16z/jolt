use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_std::rand::rngs::StdRng;
use ark_ff::UniformRand;
use lazy_static::lazy_static;
use rand_chacha::rand_core::SeedableRng;
use std::sync::{Arc, Mutex};

//TODO: The SRS is set with a default value of ____ if this is to be changed (extended) use the cli arg and change it manually.
//TODO: add input specifiying monomial or lagrange basis
lazy_static! {
    pub static ref ZEROMORPHSRS: Arc<Mutex<ZeromorphSRS<Bn254>>> =
    Arc::new(Mutex::new(ZeromorphSRS::setup(None)));
}

#[derive(Debug, Clone, Default)]
pub struct ZeromorphSRS<P: Pairing> {
    g1: P::G1Affine,
    g2: P::G2Affine,
    g1_powers: Vec<P::G1Affine>,
    g2_powers: Vec<P::G2Affine>,
}

impl<P: Pairing> ZeromorphSRS<P> {

    fn compute_g_powers<G: CurveGroup>(tau: G::ScalarField, n: usize) -> Vec<G::Affine> {
        let mut g_srs = vec![G::zero(); n - 1];
    
        #[cfg(not(feature = "parallel"))]
        let g_srs: Vec<G> = std::iter::once(G::generator())
            .chain(g_srs.iter().scan(G::generator(), |state, _| {
                *state *= &tau;
                Some(*state)
            }))
            .collect();
    
        #[cfg(feature = "parallel")]
        {
            use ark_ff::Field;
            use ark_ff::Zero;
            g_srs.push(G::zero());
            parallelize(&mut g_srs, |g, start| {
                let mut current_g: G = G::generator();
                current_g = current_g.mul(tau.pow(&[start as u64]));
                for g in g.iter_mut() {
                    *g = current_g;
                    current_g *= tau;
                }
            });
        }
    
        G::normalize_batch(&g_srs)
    }

    pub fn setup(tau: Option<&[u8]>) -> ZeromorphSRS<P> {
        let N_MAX = 250;
        /*
        if tau.is_none() {
            return ZeromorphSRS::default();
            todo!()
        }
        */
        /*
        if ENV_VAR_NOT_PASSED_IN
            */
        let mut bytes = [0u8; 32];
        let len = tau.unwrap().len().min(32);
        bytes[..len].copy_from_slice(&tau.unwrap()[..len]);
        let rng = &mut StdRng::from_seed(bytes);

        let tau = P::ScalarField::rand(rng);
        let g1_powers = Self::compute_g_powers::<P::G1>(tau, N_MAX);
        let g2_powers = Self::compute_g_powers::<P::G2>(tau, N_MAX);
        ZeromorphSRS { g1: g1_powers[0], g2: g2_powers[0], g1_powers, g2_powers }
    }

    pub fn get_verifier_key() -> ProverKey<P> {
        todo!()
    }

    pub fn get_prover_key() -> VerifierKey<P> {
        todo!()
    }
}

pub struct ProverKey<P: Pairing> {
  // generator
  pub g1: P::G1Affine,
  pub tau_1: P::G1Affine,
  // random power of tau + g1 used for commitments
  pub g1_tau_powers: Vec<P::G1Affine>,
}

pub struct VerifierKey<P: Pairing> {
  pub g1: P::G1Affine,
  pub g2: P::G2Affine,
  pub tau_2: P::G2Affine,
  pub tau_N_max_sub_2_N: P::G2Affine,
}

//TODO: can we upgrade the transcript to give not just absorb
pub struct ZeromorphProof<P: Pairing> {
  pub pi: P::G1Affine,
  pub q_hat_com: P::G1Affine,
  pub q_k_com: Vec<P::G1Affine>,
}
