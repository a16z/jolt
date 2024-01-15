use crate::subprotocols::zeromorph::kzg::UNIVERSAL_KZG_SRS;
use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::UniformRand;
use ark_std::rand::rngs::StdRng;
use lazy_static::lazy_static;
use rand_chacha::{
  rand_core::{RngCore, SeedableRng},
  ChaCha20Rng,
};
use std::sync::{Arc, Mutex};

use super::kzg::KZGProverKey;

//TODO: The SRS is set with a default value of ____ if this is to be changed (extended) use the cli arg and change it manually.
//TODO: add input specifiying monomial or lagrange basis
const MAX_VARS: usize = 20;
lazy_static! {
  pub static ref ZEROMORPH_SRS: Arc<Mutex<ZeromorphSRS<Bn254>>> =
    Arc::new(Mutex::new(ZeromorphSRS::setup(
      None,
      1 << (MAX_VARS + 1),
      &mut ChaCha20Rng::from_seed(*b"11111111111111111111111111111111")
    )));
}

#[derive(Debug, Clone, Default)]
pub struct ZeromorphSRS<P: Pairing>(UNIVERSAL_KZG_SRS<P>);

impl<P: Pairing> ZeromorphSRS<P> {
  pub fn setup<R: RngCore>(
    toxic_waste: Option<&[u8]>,
    max_degree: usize,
    mut rng: &mut R,
  ) -> ZeromorphSRS<P> {
    ZeromorphSRS(UNIVERSAL_KZG_SRS::<P>::setup(None, max_degree, rng))
  }

  pub fn get_pk_vk(&self, max_degree: usize) -> (ZeromorphProverKey<P>, ZeromorphVerifierKey<P>) {
    let offset = self.0.g1_powers.len() - max_degree;
    let offset_g1_powers = self.0.g1_powers[offset..].to_vec();
    (
      ZeromorphProverKey {
        g1_powers: KZGProverKey {
          g1_powers: self.0.g1_powers.clone(),
        },
        offset_g1_powers: KZGProverKey {
          g1_powers: offset_g1_powers,
        },
      },
      ZeromorphVerifierKey {
        g1: self.0.g1_powers[0],
        g2: self.0.g2_powers[0],
        tau_2: self.0.g2_powers[1],
        tau_N_max_sub_2_N: self.0.g2_powers[offset],
      },
    )
  }
}

#[derive(Clone, Debug)]
pub struct ZeromorphProverKey<P: Pairing> {
  // generator
  pub g1_powers: KZGProverKey<P>,
  pub offset_g1_powers: KZGProverKey<P>,
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
