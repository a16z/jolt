use ark_ec::CurveGroup;
use ark_std::rand::SeedableRng;
use digest::{ExtendableOutput, Input};
use rand_chacha::ChaCha20Rng;
use sha3::Shake256;
use std::io::Read;
use crate::utils::math::Math;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

use super::hyrax::matrix_dimensions;

pub struct PedersenInit<G> {
    pub generators: Vec<G>
}

impl<G: CurveGroup> PedersenInit<G> {
    #[tracing::instrument]
    pub fn new(max_num_vars: usize, label: &[u8]) -> Self {
        // TODO(sragss): Realistically this needs to be moved upstream
        // as it's a hyrax implementation detail.
        let max_len = matrix_dimensions(max_num_vars).1.pow2();
        let mut shake = Shake256::default();
        shake.input(label);
        let mut buf = vec![];
        G::generator().serialize_compressed(&mut buf).unwrap();
        shake.input(buf);

        let mut reader = shake.xof_result();
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed).unwrap();
        let mut rng = ChaCha20Rng::from_seed(seed);

        let mut generators: Vec<G> = Vec::new();
        for _ in 0..max_len {
            generators.push(G::rand(&mut rng));
        }

        Self {
            generators,
        }

    } 

    pub fn sample(&self, n: usize) -> PedersenGenerators<G> {
        assert!(self.generators.len() >= n, "Insufficient number of generators for sampling: required {}, available {}", n, self.generators.len());
        let sample = self.generators[0..n].into();
        PedersenGenerators { generators: sample }
    }
}

#[derive(Clone)]
pub struct PedersenGenerators<G> {
    pub generators: Vec<G>,
}

pub trait PedersenCommitment<G: CurveGroup>: Sized {
    fn commit(&self, gens: &PedersenGenerators<G>) -> G;
    fn commit_vector(inputs: &[Self], bases: &[G::Affine]) -> G;
}

impl<G: CurveGroup> PedersenCommitment<G> for G::ScalarField {
    #[tracing::instrument(skip_all, name = "PedersenCommitment::commit")]
    fn commit(&self, gens: &PedersenGenerators<G>) -> G {
        assert_eq!(gens.generators.len(), 1);
        gens.generators[0] * self
    }

    fn commit_vector(inputs: &[Self], bases: &[G::Affine]) -> G {
        assert_eq!(bases.len(), inputs.len());
        VariableBaseMSM::msm_u64(&bases, &inputs).unwrap()
    }
}
