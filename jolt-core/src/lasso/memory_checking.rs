#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::grand_product::{
        BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof,
    },
};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use itertools::interleave;
use rayon::prelude::*;
use std::iter::zip;
use std::marker::PhantomData;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MultisetHashes<F: JoltField> {
    /// Multiset hash of "read" tuples
    pub read_hashes: Vec<F>,
    /// Multiset hash of "write" tuples
    pub write_hashes: Vec<F>,
    /// Multiset hash of "init" tuples
    pub init_hashes: Vec<F>,
    /// Multiset hash of "final" tuples
    pub final_hashes: Vec<F>,
}

impl<F: JoltField> MultisetHashes<F> {
    pub fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
        transcript.append_scalars(&self.read_hashes);
        transcript.append_scalars(&self.write_hashes);
        transcript.append_scalars(&self.init_hashes);
        transcript.append_scalars(&self.final_hashes);
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<F, C, Openings>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
    Openings: StructuredPolynomialData<F> + Sync + Default,
{
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductProof<C>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductProof<C>,
    /// The openings associated with the grand products.
    pub openings: SerializableWrapper<F, Openings>,
}

pub trait StructuredPolynomialData<T> {
    fn read_write_values(&self) -> Vec<&T>;
    fn init_final_values(&self) -> Vec<&T>;
    fn read_write_values_mut(&mut self) -> Vec<&mut T>;
    fn init_final_values_mut(&mut self) -> Vec<&mut T>;
}

pub struct SerializableWrapper<T, U>(pub U, pub PhantomData<T>);

impl<T, U> CanonicalSerialize for SerializableWrapper<T, U>
where
    U: StructuredPolynomialData<T> + Sync,
    T: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        for value in self
            .0
            .read_write_values()
            .iter()
            .chain(self.0.init_final_values().iter())
        {
            value.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        let mut size = 0;
        for value in self
            .0
            .read_write_values()
            .iter()
            .chain(self.0.init_final_values().iter())
        {
            size += value.serialized_size(compress);
        }

        size
    }
}

impl<T, U> CanonicalDeserialize for SerializableWrapper<T, U>
where
    U: StructuredPolynomialData<T> + Sync + Default,
    T: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let mut result = U::default();
        for value in result.read_write_values_mut().into_iter() {
            *value = T::deserialize_with_mode(&mut reader, compress, validate)?;
        }
        for value in result.init_final_values_mut().into_iter() {
            *value = T::deserialize_with_mode(&mut reader, compress, validate)?;
        }

        Ok(Self(result, PhantomData))
    }
}

impl<T, U> Valid for SerializableWrapper<T, U>
where
    U: StructuredPolynomialData<T> + Sync,
    T: Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        for value in self
            .0
            .read_write_values()
            .iter()
            .chain(self.0.init_final_values().iter())
        {
            value.check()?;
        }

        Ok(())
    }
}

// Empty struct to represent that no preprocessing data is used.
pub struct NoPreprocessing;
pub struct NoAdditionalWitness;

pub trait MemoryCheckingProver<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Self: Sync,
{
    type ReadWriteGrandProduct: BatchedGrandProduct<F, PCS> + Send + 'static =
        BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct: BatchedGrandProduct<F, PCS> + Send + 'static =
        BatchedDenseGrandProduct<F>;

    type Polynomials: StructuredPolynomialData<DensePolynomial<F>>;
    type Openings: StructuredPolynomialData<F> + Sync + Default;
    type Commitments: StructuredPolynomialData<PCS::Commitment>;

    type Preprocessing = NoPreprocessing;
    type AdditionalWitnessData = NoAdditionalWitness;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F);

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_memory_checking")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking<'a>(
        pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &'a Self::Polynomials,
        additional_witness: &Self::AdditionalWitnessData,
        opening_accumulator: &mut ProverOpeningAccumulator<'a, F>,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(
            preprocessing,
            polynomials,
            additional_witness,
            transcript,
            pcs_setup,
        );

        let openings = Self::compute_openings(
            preprocessing,
            opening_accumulator,
            polynomials,
            &r_read_write,
            &r_init_final,
        );

        MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings: SerializableWrapper(openings, PhantomData),
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_grand_products")]
    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        additional_witness: &Self::AdditionalWitnessData,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::Setup,
    ) -> (
        BatchedGrandProductProof<PCS>,
        BatchedGrandProductProof<PCS>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, additional_witness, &gamma, &tau);
        let (mut read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (mut init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (read_write_grand_product, r_read_write) =
            read_write_circuit.prove_grand_product(transcript, Some(pcs_setup));
        let (init_final_grand_product, r_init_final) =
            init_final_circuit.prove_grand_product(transcript, Some(pcs_setup));

        drop_in_background_thread(read_write_circuit);
        drop_in_background_thread(init_final_circuit);

        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    fn compute_openings<'a>(
        _preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut ProverOpeningAccumulator<'a, F>,
        polynomials: &'a Self::Polynomials,
        r_read_write: &[F],
        r_init_final: &[F],
    ) -> Self::Openings {
        let mut openings = Self::Openings::default();

        let eq_read_write = EqPolynomial::evals(r_read_write);
        polynomials
            .read_write_values()
            .par_iter()
            .zip(openings.read_write_values_mut().into_par_iter())
            .for_each(|(poly, opening)| {
                let claim = poly.evaluate_at_chi(&eq_read_write);
                *opening = claim;
            });

        for (poly, claim) in polynomials
            .read_write_values()
            .iter()
            .zip(openings.read_write_values().into_iter())
        {
            opening_accumulator.append(poly, r_read_write.to_vec(), *claim);
        }

        let eq_init_final = EqPolynomial::evals(r_init_final);
        polynomials
            .init_final_values()
            .par_iter()
            .zip(openings.init_final_values_mut().into_par_iter())
            .for_each(|(poly, opening)| {
                let claim = poly.evaluate_at_chi(&eq_init_final);
                *opening = claim;
            });

        for (poly, claim) in polynomials
            .init_final_values()
            .iter()
            .zip(openings.init_final_values().into_iter())
        {
            opening_accumulator.append(poly, r_init_final.to_vec(), *claim);
        }

        openings
    }

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        read_write_leaves: <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS>>::Leaves,
    ) -> (Self::ReadWriteGrandProduct, Vec<F>) {
        let batched_circuit = Self::ReadWriteGrandProduct::construct(read_write_leaves);
        let claims = batched_circuit.claims();
        (batched_circuit, claims)
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        init_final_leaves: <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS>>::Leaves,
    ) -> (Self::InitFinalGrandProduct, Vec<F>) {
        let batched_circuit = Self::InitFinalGrandProduct::construct(init_final_leaves);
        let claims = batched_circuit.claims();
        (batched_circuit, claims)
    }

    fn interleave_hashes(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let init_final_hashes = interleave(
            multiset_hashes.init_hashes.clone(),
            multiset_hashes.final_hashes.clone(),
        )
        .collect();

        (read_write_hashes, init_final_hashes)
    }

    fn uninterleave_hashes(
        _preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), num_memories);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[i];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    /// Computes the MLE of the leaves of the read, write, init, and final grand product circuits,
    /// one of each type per memory.
    /// Returns: (interleaved read/write leaves, interleaved init/final leaves)
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        additional_witness: &Self::AdditionalWitnessData,
        gamma: &F,
        tau: &F,
    ) -> (
        <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS>>::Leaves,
        <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS>>::Leaves,
    );

    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingVerifier<F, PCS>: MemoryCheckingProver<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Verifies a memory checking proof, given its associated polynomial `commitment`.
    fn verify_memory_checking<'a>(
        preprocessing: &Self::Preprocessing,
        pcs_setup: &PCS::Setup,
        mut proof: MemoryCheckingProof<F, PCS, Self::Openings>,
        commitments: &'a Self::Commitments,
        opening_accumulator: &mut VerifierOpeningAccumulator<'a, F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        Self::check_multiset_equality(preprocessing, &proof.multiset_hashes);
        proof.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &proof.multiset_hashes);

        let (claims_read_write, r_read_write) = Self::ReadWriteGrandProduct::verify_grand_product(
            &proof.read_write_grand_product,
            &read_write_hashes,
            transcript,
            Some(pcs_setup),
        );
        let (claims_init_final, r_init_final) = Self::InitFinalGrandProduct::verify_grand_product(
            &proof.init_final_grand_product,
            &init_final_hashes,
            transcript,
            Some(pcs_setup),
        );

        proof
            .openings
            .0
            .read_write_values()
            .into_iter()
            .zip(commitments.read_write_values().iter())
            .for_each(|(opening, commitment)| {
                opening_accumulator.append(commitment, r_read_write.to_vec(), *opening);
            });
        proof
            .openings
            .0
            .init_final_values()
            .into_iter()
            .zip(commitments.init_final_values().iter())
            .for_each(|(opening, commitment)| {
                opening_accumulator.append(commitment, r_init_final.to_vec(), *opening);
            });

        // proof.read_write_openings.verify_openings(
        //     generators,
        //     &proof.read_write_opening_proof,
        //     commitments,
        //     &r_read_write,
        //     transcript,
        // )?;
        // proof.init_final_openings.verify_openings(
        //     generators,
        //     &proof.init_final_opening_proof,
        //     commitments,
        //     &r_init_final,
        //     transcript,
        // )?;

        Self::compute_verifier_openings(
            &mut proof.openings.0,
            preprocessing,
            &r_read_write,
            &r_init_final,
        );

        Self::check_fingerprints(
            preprocessing,
            claims_read_write,
            claims_init_final,
            &proof.openings.0,
            &gamma,
            &tau,
        );

        Ok(())
    }

    /// Often some of the openings do not require an opening proof provided by the prover, and
    /// instead can be efficiently computed by the verifier by itself. This function populates
    /// any such fields in `self`.
    fn compute_verifier_openings(
        _openings: &mut Self::Openings,
        _preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        _r_init_final: &[F],
    ) {
    }

    /// Computes "read" memory tuples (one per memory) from the given `openings`.
    fn read_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "write" memory tuples (one per memory) from the given `openings`.
    fn write_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "init" memory tuples (one per memory) from the given `openings`.
    fn init_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "final" memory tuples (one per memory) from the given `openings`.
    fn final_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
    ) -> Vec<Self::MemoryTuple>;

    /// Checks that the claimed multiset hashes (output by grand product) are consistent with the
    /// openings given by `read_write_openings` and `init_final_openings`.
    fn check_fingerprints(
        preprocessing: &Self::Preprocessing,
        claims_read_write: Vec<F>,
        claims_init_final: Vec<F>,
        openings: &Self::Openings,
        gamma: &F,
        tau: &F,
    ) {
        let read_hashes: Vec<_> = Self::read_tuples(preprocessing, openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let write_hashes: Vec<_> = Self::write_tuples(preprocessing, openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let init_hashes: Vec<_> = Self::init_tuples(preprocessing, openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let final_hashes: Vec<_> = Self::final_tuples(preprocessing, openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        assert_eq!(
            read_hashes.len() + write_hashes.len(),
            claims_read_write.len()
        );
        assert_eq!(
            init_hashes.len() + final_hashes.len(),
            claims_init_final.len()
        );

        let multiset_hashes = MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        };
        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &multiset_hashes);

        for (claim, fingerprint) in zip(claims_read_write, read_write_hashes) {
            assert_eq!(claim, fingerprint);
        }
        for (claim, fingerprint) in zip(claims_init_final, init_final_hashes) {
            assert_eq!(claim, fingerprint);
        }
    }
}
