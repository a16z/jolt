#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::grand_product::{BatchedGrandProduct, BatchedGrandProductProof},
};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use rayon::prelude::*;

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
    pub fn append_to_transcript<ProofTranscript: Transcript>(
        &self,
        transcript: &mut ProofTranscript,
    ) {
        transcript.append_scalars(&self.read_hashes);
        transcript.append_scalars(&self.write_hashes);
        transcript.append_scalars(&self.init_hashes);
        transcript.append_scalars(&self.final_hashes);
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Openings: StructuredPolynomialData<F> + Sync + CanonicalSerialize + CanonicalDeserialize,
    OtherOpenings: ExogenousOpenings<F> + Sync,
    ProofTranscript: Transcript,
{
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The openings associated with the grand products.
    pub openings: Openings,
    pub exogenous_openings: OtherOpenings,
}

/// This type, used within a `StructuredPolynomialData` struct, indicates that the
/// field has a corresponding opening but no corresponding polynomial or commitment ––
/// the prover doesn't need to compute a witness polynomial or commitment because
/// the verifier can compute the opening on its own.
pub type VerifierComputedOpening<T> = Option<T>;

/// This trait is used to capture the relationship between polynomials, commitments, and
/// openings in offline memory-checking. For a given offline memory-checking instance,
/// the "shape" of its polynomials, commitments, and openings is the same. We can define a
/// a single struct with this "shape", parametrized by a generic type `T` (see e.g. `BytecodeStuff`).
/// To avoid manually mapping between the respective polynomials/commitments/openings
/// (which introduces footguns), we implement this trait to define a canonical ordering
/// over the generic struct's fields.
pub trait StructuredPolynomialData<T>: CanonicalSerialize + CanonicalDeserialize {
    /// Returns a `Vec` of references to the read/write values of `self`.
    /// Ordering should mirror `read_write_values_mut`.
    fn read_write_values(&self) -> Vec<&T> {
        vec![]
    }

    /// Returns a `Vec` of references to the init/final values of `self`.
    /// Ordering should mirror `init_final_values_mut`.
    fn init_final_values(&self) -> Vec<&T> {
        vec![]
    }

    /// Returns a `Vec` of mutable references to the read/write values of `self`.
    /// Ordering should mirror `read_write_values`.
    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }

    /// Returns a `Vec` of mutable references to the init/final values of `self`.
    /// Ordering should mirror `init_final_values`.
    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
}

/// Sometimes, an offline memory-checking instance "reuses" polynomials/commitments
/// from a different instance. For example, in `read_write_memory.rs` we use some of
/// the polynomials/commitments defined in `bytecode.rs`, specifically the ones corresponding
/// to the RISC-V registers. We need openings from these polynomials, but we shouldn't
/// recompute or recommit to the polynomials.
/// This trait is used to cherry-pick the "exogenous" polynomials/commitments needed
/// by an offline-memory checking instance.
pub trait ExogenousOpenings<F: JoltField>:
    Default + CanonicalSerialize + CanonicalDeserialize
{
    /// Returns a `Vec` of references to the openings contained in `self`.
    /// Ordering should mirror `openings_mut`.
    fn openings(&self) -> Vec<&F>;
    /// Returns a `Vec` of mutable references to the openings contained in `self`.
    /// Ordering should mirror `openings`.
    fn openings_mut(&mut self) -> Vec<&mut F>;
    /// Cherry-picks the "exogenous" polynomials/commitments needed by an offline-memory
    /// checking instance. The ordering of the returned polynoials/commitments should
    /// mirror `openings`/`openings_mut`.
    fn exogenous_data<T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        polys_or_commitments: &JoltStuff<T>,
    ) -> Vec<&T>;
}

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct NoExogenousOpenings;
impl<F: JoltField> ExogenousOpenings<F> for NoExogenousOpenings {
    fn openings(&self) -> Vec<&F> {
        vec![]
    }

    fn openings_mut(&mut self) -> Vec<&mut F> {
        vec![]
    }

    fn exogenous_data<T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        _: &JoltStuff<T>,
    ) -> Vec<&T> {
        vec![]
    }
}

/// This trait (specifically, the `initialize` function) is used in lieu of `Default`
/// to initialize a `StructuredPolynomialData` struct, which may contain `Vec` fields
/// whose lengths depend on some preprocessing.
pub trait Initializable<T, Preprocessing>: StructuredPolynomialData<T> + Default {
    /// This function is used in lieu of `Default::default()` to initialize a
    /// `StructuredPolynomialData` struct, which may contain `Vec` fields
    /// whose lengths depend on some preprocessing.
    ///
    /// Note that the default implementation of initialize, however, does
    /// just return `Default::default()`.
    fn initialize(_preprocessing: &Preprocessing) -> Self {
        Default::default()
    }

    #[cfg(test)]
    fn test_ordering_consistency(preprocessing: &Preprocessing) {
        use itertools::zip_eq;

        let mut data = Self::initialize(preprocessing);
        let read_write_pointers: Vec<_> = data
            .read_write_values()
            .into_iter()
            .map(|ptr| ptr as *const T)
            .collect();
        let read_write_pointers_mut: Vec<_> = data
            .read_write_values_mut()
            .into_iter()
            .map(|ptr| ptr as *const T)
            .collect();
        for (i, (ptr, ptr_mut)) in zip_eq(read_write_pointers, read_write_pointers_mut).enumerate()
        {
            assert!(
                std::ptr::eq(ptr, ptr_mut),
                "Read-write pointer mismatch at index {i}"
            );
        }

        let init_final_pointers: Vec<_> = data
            .init_final_values()
            .into_iter()
            .map(|ptr| ptr as *const T)
            .collect();
        let init_final_pointers_mut: Vec<_> = data
            .init_final_values_mut()
            .into_iter()
            .map(|ptr| ptr as *const T)
            .collect();
        for (i, (ptr, ptr_mut)) in zip_eq(init_final_pointers, init_final_pointers_mut).enumerate()
        {
            assert!(
                std::ptr::eq(ptr, ptr_mut),
                "Init-final pointer mismatch at index {i}"
            );
        }
    }
}

/// Empty struct to represent that no preprocessing data is used.
pub struct NoPreprocessing;

pub trait MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Self: Sync,
{
    type ReadWriteGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static;
    type InitFinalGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static;

    type Polynomials: StructuredPolynomialData<MultilinearPolynomial<F>>;
    type Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Self::Preprocessing>;
    type Commitments: StructuredPolynomialData<PCS::Commitment>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync;

    type Preprocessing;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple: Copy + Clone;

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_memory_checking")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking(
        pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(
            preprocessing,
            polynomials,
            jolt_polynomials,
            opening_accumulator,
            transcript,
            pcs_setup,
        );

        let read_write_batch_size =
            multiset_hashes.read_hashes.len() + multiset_hashes.write_hashes.len();
        let init_final_batch_size =
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len();

        // For a batch size of k, the first log2(k) elements of `r_read_write`/`r_init_final`
        // form the point at which the output layer's MLE is evaluated. The remaining elements
        // then form the point at which the leaf layer's polynomials are evaluated.
        let (_, r_read_write_opening) =
            r_read_write.split_at(read_write_batch_size.next_power_of_two().log_2());
        let (_, r_init_final_opening) =
            r_init_final.split_at(init_final_batch_size.next_power_of_two().log_2());

        let (openings, exogenous_openings) = Self::compute_openings(
            preprocessing,
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            r_read_write_opening,
            r_init_final_opening,
            transcript,
        );

        MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings,
            exogenous_openings,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_grand_products")]
    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::Setup,
    ) -> (
        BatchedGrandProductProof<PCS, ProofTranscript>,
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, jolt_polynomials, &gamma, &tau);
        let (mut read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (mut init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        #[cfg(test)]
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (read_write_grand_product, r_read_write) = read_write_circuit.prove_grand_product(
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );
        let (init_final_grand_product, r_init_final) = init_final_circuit.prove_grand_product(
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );

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

    fn compute_openings(
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        r_read_write: &[F],
        r_init_final: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self::Openings, Self::ExogenousOpenings) {
        let mut openings = Self::Openings::initialize(preprocessing);
        let mut exogenous_openings = Self::ExogenousOpenings::default();

        let read_write_polys: Vec<_> = [
            polynomials.read_write_values(),
            Self::ExogenousOpenings::exogenous_data(jolt_polynomials),
        ]
        .concat();
        let (read_write_evals, eq_read_write) =
            MultilinearPolynomial::batch_evaluate(&read_write_polys, r_read_write);
        let read_write_openings: Vec<&mut F> = openings
            .read_write_values_mut()
            .into_iter()
            .chain(exogenous_openings.openings_mut())
            .collect();

        for (opening, eval) in read_write_openings.into_iter().zip(read_write_evals.iter()) {
            *opening = *eval;
        }

        opening_accumulator.append(
            &read_write_polys,
            DensePolynomial::new(eq_read_write),
            r_read_write.to_vec(),
            &read_write_evals,
            transcript,
        );

        let init_final_polys = polynomials.init_final_values();
        let (init_final_evals, eq_init_final) =
            MultilinearPolynomial::batch_evaluate(&init_final_polys, r_init_final);

        for (opening, eval) in openings
            .init_final_values_mut()
            .into_iter()
            .zip(init_final_evals.iter())
        {
            *opening = *eval;
        }

        opening_accumulator.append(
            &polynomials.init_final_values(),
            DensePolynomial::new(eq_init_final),
            r_init_final.to_vec(),
            &init_final_evals,
            transcript,
        );

        (openings, exogenous_openings)
    }

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        read_write_leaves: <Self::ReadWriteGrandProduct as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::Leaves,
    ) -> (Self::ReadWriteGrandProduct, Vec<F>) {
        let batched_circuit = Self::ReadWriteGrandProduct::construct(read_write_leaves);
        let claims = batched_circuit.claimed_outputs();
        (batched_circuit, claims)
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Polynomials,
        init_final_leaves: <Self::InitFinalGrandProduct as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::Leaves,
    ) -> (Self::InitFinalGrandProduct, Vec<F>) {
        let batched_circuit = Self::InitFinalGrandProduct::construct(init_final_leaves);
        let claims = batched_circuit.claimed_outputs();
        (batched_circuit, claims)
    }

    fn interleave<T: Copy + Clone>(
        _preprocessing: &Self::Preprocessing,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        let read_write_values = interleave(read_values, write_values).cloned().collect();
        let init_final_values = interleave(init_values, final_values).cloned().collect();

        (read_write_values, init_final_values)
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
        exogenous_polynomials: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (
        <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
        <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
    );

    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingVerifier<F, PCS, ProofTranscript>:
    MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// Verifies a memory checking proof, given its associated polynomial `commitment`.
    fn verify_memory_checking(
        preprocessing: &Self::Preprocessing,
        pcs_setup: &PCS::Setup,
        mut proof: MemoryCheckingProof<
            F,
            PCS,
            Self::Openings,
            Self::ExogenousOpenings,
            ProofTranscript,
        >,
        commitments: &Self::Commitments,
        jolt_commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        Self::check_multiset_equality(preprocessing, &proof.multiset_hashes);
        proof.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) = Self::interleave(
            preprocessing,
            &proof.multiset_hashes.read_hashes,
            &proof.multiset_hashes.write_hashes,
            &proof.multiset_hashes.init_hashes,
            &proof.multiset_hashes.final_hashes,
        );

        let read_write_batch_size = read_write_hashes.len();
        let (read_write_claim, r_read_write) = Self::ReadWriteGrandProduct::verify_grand_product(
            &proof.read_write_grand_product,
            &read_write_hashes,
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );
        // For a batch size of k, the first log2(k) elements of `r_read_write`/`r_init_final`
        // form the point at which the output layer's MLE is evaluated. The remaining elements
        // then form the point at which the leaf layer's polynomials are evaluated.
        let (r_read_write_batch_index, r_read_write_opening) =
            r_read_write.split_at(read_write_batch_size.next_power_of_two().log_2());

        let init_final_batch_size = init_final_hashes.len();
        let (init_final_claim, r_init_final) = Self::InitFinalGrandProduct::verify_grand_product(
            &proof.init_final_grand_product,
            &init_final_hashes,
            Some(opening_accumulator),
            transcript,
            Some(pcs_setup),
        );
        let (r_init_final_batch_index, r_init_final_opening) =
            r_init_final.split_at(init_final_batch_size.next_power_of_two().log_2());

        let read_write_commits: Vec<_> = [
            commitments.read_write_values(),
            Self::ExogenousOpenings::exogenous_data(jolt_commitments),
        ]
        .concat();
        let read_write_claims: Vec<_> = [
            proof.openings.read_write_values(),
            proof.exogenous_openings.openings(),
        ]
        .concat();
        opening_accumulator.append(
            &read_write_commits,
            r_read_write_opening.to_vec(),
            &read_write_claims,
            transcript,
        );

        opening_accumulator.append(
            &commitments.init_final_values(),
            r_init_final_opening.to_vec(),
            &proof.openings.init_final_values(),
            transcript,
        );

        Self::compute_verifier_openings(
            &mut proof.openings,
            preprocessing,
            r_read_write_opening,
            r_init_final_opening,
        );

        Self::check_fingerprints(
            preprocessing,
            read_write_claim,
            init_final_claim,
            r_read_write_batch_index,
            r_init_final_batch_index,
            &proof.openings,
            &proof.exogenous_openings,
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
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "write" memory tuples (one per memory) from the given `openings`.
    fn write_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "init" memory tuples (one per memory) from the given `openings`.
    fn init_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "final" memory tuples (one per memory) from the given `openings`.
    fn final_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;

    /// Checks that the claims output by the grand products are consistent with the openings of
    /// the polynomials comprising the input layers.
    fn check_fingerprints(
        preprocessing: &Self::Preprocessing,
        read_write_claim: F,
        init_final_claim: F,
        r_read_write_batch_index: &[F],
        r_init_final_batch_index: &[F],
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
        gamma: &F,
        tau: &F,
    ) {
        let read_hashes: Vec<_> = Self::read_tuples(preprocessing, openings, exogenous_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let write_hashes: Vec<_> = Self::write_tuples(preprocessing, openings, exogenous_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let init_hashes: Vec<_> = Self::init_tuples(preprocessing, openings, exogenous_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let final_hashes: Vec<_> = Self::final_tuples(preprocessing, openings, exogenous_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();

        let (read_write_hashes, init_final_hashes) = Self::interleave(
            preprocessing,
            &read_hashes,
            &write_hashes,
            &init_hashes,
            &final_hashes,
        );

        assert_eq!(
            read_write_hashes.len().next_power_of_two(),
            r_read_write_batch_index.len().pow2(),
        );
        assert_eq!(
            init_final_hashes.len().next_power_of_two(),
            r_init_final_batch_index.len().pow2()
        );

        // `r_read_write_batch_index`/`r_init_final_batch_index` are used to
        // combine the k claims in the batch into a single claim.
        let combined_read_write_hash: F = read_write_hashes
            .iter()
            .zip(EqPolynomial::evals(r_read_write_batch_index).iter())
            .map(|(hash, eq_eval)| *hash * eq_eval)
            .sum();
        assert_eq!(combined_read_write_hash, read_write_claim);

        let combined_init_final_hash: F = init_final_hashes
            .iter()
            .zip(EqPolynomial::evals(r_init_final_batch_index).iter())
            .map(|(hash, eq_eval)| *hash * eq_eval)
            .sum();
        assert_eq!(combined_init_final_hash, init_final_claim);
    }
}
