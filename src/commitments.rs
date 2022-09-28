use super::group::GroupElement;
use super::scalar::Scalar;
use crate::group::group_basepoint_compressed;
use ark_ec::msm::VariableBaseMSM;
use ark_ec::ProjectiveCurve;
use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;
use digest::{ExtendableOutput, Input};
use rand_chacha::ChaCha20Rng;
use sha3::Shake256;
use std::io::Read;

#[derive(Debug)]
pub struct MultiCommitGens {
  pub n: usize,
  pub G: Vec<GroupElement>,
  pub h: GroupElement,
}

impl MultiCommitGens {
  pub fn new(n: usize, label: &[u8]) -> Self {
    let mut shake = Shake256::default();
    shake.input(label);
    shake.input(group_basepoint_compressed().as_slice());

    let mut reader = shake.xof_result();
    let mut seed = [0u8; 32];
    reader.read_exact(&mut seed).unwrap();
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut gens: Vec<GroupElement> = Vec::new();
    for _ in 0..n + 1 {
      gens.push(GroupElement::rand(&mut rng));
    }

    MultiCommitGens {
      n,
      G: gens[..n].to_vec(),
      h: gens[n],
    }
  }

  pub fn clone(&self) -> MultiCommitGens {
    MultiCommitGens {
      n: self.n,
      h: self.h,
      G: self.G.clone(),
    }
  }

  pub fn split_at(&self, mid: usize) -> (MultiCommitGens, MultiCommitGens) {
    let (G1, G2) = self.G.split_at(mid);

    (
      MultiCommitGens {
        n: G1.len(),
        G: G1.to_vec(),
        h: self.h,
      },
      MultiCommitGens {
        n: G2.len(),
        G: G2.to_vec(),
        h: self.h,
      },
    )
  }
}

pub trait Commitments {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement;
}

impl Commitments for Scalar {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, 1);

    gens_n.G[0].mul(self.into_repr()) + gens_n.h.mul(blind.into_repr())
  }
}

impl Commitments for Vec<Scalar> {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, self.len());

    let mut bases = ProjectiveCurve::batch_normalization_into_affine(gens_n.G.as_ref());
    let mut scalars = self.iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    bases.push(gens_n.h.into_affine());
    scalars.push(blind.into_repr());

    VariableBaseMSM::multi_scalar_mul(bases.as_ref(), scalars.as_ref())
  }
}

impl Commitments for [Scalar] {
  fn commit(&self, blind: &Scalar, gens_n: &MultiCommitGens) -> GroupElement {
    assert_eq!(gens_n.n, self.len());

    let mut bases = ProjectiveCurve::batch_normalization_into_affine(gens_n.G.as_ref());
    let mut scalars = self.iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    bases.push(gens_n.h.into_affine());
    scalars.push(blind.into_repr());

    VariableBaseMSM::multi_scalar_mul(bases.as_ref(), scalars.as_ref())
  }
}
