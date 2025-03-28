use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha3::digest::{ExtendableOutput, Update};
use sha3::Shake256;
use std::io::Read;

use crate::field::JoltField;
use crate::msm::{Icicle, VariableBaseMSM};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PedersenGenerators<G: CurveGroup> {
    pub generators: Vec<G>,
}

impl<G: CurveGroup> PedersenGenerators<G> {
    #[tracing::instrument(skip_all, name = "PedersenGenerators::new")]
    pub fn new(len: usize, label: &[u8]) -> Self {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut buf = vec![];
        G::generator().serialize_compressed(&mut buf).unwrap();
        shake.update(&buf);

        let mut reader = shake.finalize_xof();
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed).unwrap();
        let mut rng = ChaCha20Rng::from_seed(seed);

        let mut generators: Vec<G> = Vec::new();
        for _ in 0..len {
            generators.push(G::rand(&mut rng));
        }

        Self { generators }
    }

    pub fn clone_n(&self, n: usize) -> PedersenGenerators<G> {
        assert!(
            self.generators.len() >= n,
            "Insufficient number of generators for clone_n: required {}, available {}",
            n,
            self.generators.len()
        );
        let slice = &self.generators[..n];
        PedersenGenerators {
            generators: slice.into(),
        }
    }
}

pub trait PedersenCommitment<G: CurveGroup>: Sized {
    fn commit(&self, gens: &PedersenGenerators<G>) -> G;
    fn commit_vector(inputs: &[Self], bases: &[G::Affine]) -> G;
}

impl<G: CurveGroup + Icicle> PedersenCommitment<G> for G::ScalarField
where
    G::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all, name = "PedersenCommitment::commit")]
    fn commit(&self, gens: &PedersenGenerators<G>) -> G {
        assert_eq!(gens.generators.len(), 1);
        gens.generators[0] * self
    }

    fn commit_vector(inputs: &[Self], bases: &[G::Affine]) -> G {
        assert_eq!(bases.len(), inputs.len());
        VariableBaseMSM::msm_field_elements(bases, None, inputs, None, false).unwrap()
    }
}
