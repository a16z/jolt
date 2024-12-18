use crate::msm::Icicle;
use crate::msm::{GpuBaseType, VariableBaseMSM};
use ark_ec::CurveGroup;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::rand::SeedableRng;
use ark_std::UniformRand;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "icicle")]
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput, Update};
use sha3::Shake256;
use std::io::{Read, Write};

#[derive(Clone)]
pub struct PedersenGenerators<G: CurveGroup + Icicle> {
    pub generators: Vec<G::Affine>,
    pub gpu_generators: Option<Vec<GpuBaseType<G>>>,
}

impl<G: CurveGroup + Icicle> PedersenGenerators<G> {
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

        let mut generators: Vec<G::Affine> = Vec::new();
        for _ in 0..len {
            generators.push(G::Affine::rand(&mut rng));
        }

        #[cfg(feature = "icicle")]
        let gpu_generators = Some(
            generators
                .par_iter()
                .map(<G as Icicle>::from_ark_affine)
                .collect::<Vec<_>>(),
        );
        #[cfg(not(feature = "icicle"))]
        let gpu_generators = None;

        Self {
            generators,
            gpu_generators,
        }
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
            gpu_generators: self
                .gpu_generators
                .as_ref()
                .map(|gpu_slice| gpu_slice[..n].into()),
        }
    }
}

pub trait PedersenCommitment<G: CurveGroup + Icicle>: Sized {
    fn commit(&self, gens: &PedersenGenerators<G>) -> G;
    fn commit_vector(
        inputs: &[Self],
        bases: &[G::Affine],
        gpu_bases: Option<&[GpuBaseType<G>]>,
    ) -> G;
}

impl<G: CurveGroup + Icicle> PedersenCommitment<G> for G::ScalarField {
    #[tracing::instrument(skip_all, name = "PedersenCommitment::commit")]
    fn commit(&self, gens: &PedersenGenerators<G>) -> G {
        assert_eq!(gens.generators.len(), 1);
        gens.generators[0] * self
    }

    #[tracing::instrument(skip_all, name = "PedersenCommitment::commit_vector")]
    fn commit_vector(
        inputs: &[Self],
        bases: &[G::Affine],
        gpu_bases: Option<&[GpuBaseType<G>]>,
    ) -> G {
        assert_eq!(bases.len(), inputs.len());
        VariableBaseMSM::msm(bases, gpu_bases, inputs).unwrap()
    }
}

impl<G: CurveGroup + Icicle> CanonicalSerialize for PedersenGenerators<G> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.generators.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.generators.serialized_size(compress)
    }
}

impl<G: CurveGroup + Icicle> Valid for PedersenGenerators<G> {
    fn check(&self) -> Result<(), SerializationError> {
        self.generators.check()
    }
}

impl<G: CurveGroup + Icicle> CanonicalDeserialize for PedersenGenerators<G> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let generators = Vec::<G::Affine>::deserialize_with_mode(reader, compress, validate)?;
        #[cfg(feature = "icicle")]
        let gpu_generators = Some(
            generators
                .par_iter()
                .map(<G as Icicle>::from_ark_affine)
                .collect::<Vec<_>>(),
        );
        #[cfg(not(feature = "icicle"))]
        let gpu_generators = None;

        Ok(Self {
            generators,
            gpu_generators,
        })
    }
}
