use ark_ec::msm::VariableBaseMSM;
use ark_ec::ProjectiveCurve;
use ark_ff::PrimeField;
use ark_std::rand::SeedableRng;
use digest::{ExtendableOutput, Input};
use rand_chacha::ChaCha20Rng;
use sha3::Shake256;
use std::io::Read;

#[derive(Debug)]
pub struct MultiCommitGens<G> {
  pub n: usize,
  pub G: Vec<G>,
  pub h: G,
}

impl<G: ProjectiveCurve> MultiCommitGens<G> {
  pub fn new(n: usize, label: &[u8]) -> Self {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut buf = vec![];
    G::prime_subgroup_generator().serialize(&mut buf).unwrap();
    shake.input(buf);

    let mut reader = shake.xof_result();
    let mut seed = [0u8; 32];
    reader.read_exact(&mut seed).unwrap();
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut gens: Vec<G> = Vec::new();
    for _ in 0..n + 1 {
      gens.push(G::rand(&mut rng));
    }

    MultiCommitGens {
      n,
      G: gens[..n].to_vec(),
      h: gens[n],
    }
  }

  pub fn clone(&self) -> Self {
    MultiCommitGens {
      n: self.n,
      h: self.h,
      G: self.G.clone(),
    }
  }

  pub fn split_at(&self, mid: usize) -> (Self, Self) {
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

pub trait Commitments<G: ProjectiveCurve>: Sized {
  fn commit(&self, blind: &G::ScalarField, gens_n: &MultiCommitGens<G>) -> G;
  fn batch_commit(inputs: &[Self], blind: &G::ScalarField, gens_n: &MultiCommitGens<G>) -> G;
}

impl<G: ProjectiveCurve> Commitments<G> for G::ScalarField {
  fn commit(&self, blind: &G::ScalarField, gens_n: &MultiCommitGens<G>) -> G {
    assert_eq!(gens_n.n, 1);

    gens_n.G[0].mul(self.into_repr()) + gens_n.h.mul(blind.into_repr())
  }

  fn batch_commit(inputs: &[Self], blind: &G::ScalarField, gens_n: &MultiCommitGens<G>) -> G {
    assert_eq!(gens_n.n, inputs.len());

    let mut bases = ProjectiveCurve::batch_normalization_into_affine(gens_n.G.as_ref());
    let mut scalars = inputs.iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    bases.push(gens_n.h.into_affine());
    scalars.push(blind.into_repr());

    VariableBaseMSM::multi_scalar_mul(bases.as_ref(), scalars.as_ref())
  }
}
