use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha3::digest::{ExtendableOutput, Update};
use sha3::Shake256;
use std::fs::File;
use std::io::{Read, Write};

use crate::field::JoltField;
use crate::msm::VariableBaseMSM;

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

    /// Load generators from a URS file if it exists, otherwise generate new ones and save to file
    #[tracing::instrument(skip_all, name = "PedersenGenerators::from_urs_file")]
    pub fn from_urs_file(len: usize, label: &[u8], urs_filename: Option<&str>) -> Self {
        match urs_filename {
            Some(filename) => {
                tracing::info!("Attempting to load Hyrax URS from file: {}", filename);

                if let Ok(mut file) = File::open(filename) {
                    let mut buffer = Vec::new();
                    if file.read_to_end(&mut buffer).is_ok() {
                        if let Ok(generators) = Self::deserialize_compressed(&buffer[..]) {
                            if generators.generators.len() == len {
                                tracing::info!("Successfully loaded Hyrax URS from file");
                                return generators;
                            } else {
                                tracing::warn!(
                                    "Loaded generators have wrong length: expected {}, got {}",
                                    len,
                                    generators.generators.len()
                                );
                            }
                        }
                    }
                }

                // If loading failed, generate new and save
                tracing::info!("Generating new Hyrax URS and saving to file");
                let generators = Self::new(len, label);

                // Try to save to file
                if let Ok(mut file) = File::create(filename) {
                    let mut buffer = Vec::new();
                    if generators.serialize_compressed(&mut buffer).is_ok() {
                        let _ = file.write_all(&buffer);
                        tracing::info!("Saved Hyrax URS to file: {}", filename);
                    }
                }

                generators
            }
            None => Self::new(len, label),
        }
    }
}

pub trait PedersenCommitment<G: CurveGroup>: Sized {
    fn commit(&self, gens: &PedersenGenerators<G>) -> G;
    fn commit_vector(inputs: &[Self], bases: &[G::Affine]) -> G;
}

impl<G: CurveGroup> PedersenCommitment<G> for G::ScalarField
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
        VariableBaseMSM::msm_field_elements(bases, inputs).unwrap()
    }
}
