#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::{
    dense_mlpoly::DensePolynomial,
    structured_poly::{BatchablePolynomials, StructuredOpeningProof},
};
use crate::subprotocols::grand_product::{
    BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
};
use crate::utils::errors::ProofVerifyError;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::marker::PhantomData;

pub struct MultisetHashes<F: PrimeField> {
    /// Multiset hash of "init" tuple(s)
    hash_init: F,
    /// Multiset hash of "final" tuple(s)
    hash_final: F,
    /// Multiset hash of "read" tuple(s)
    hash_read: F,
    /// Multiset hash of "write" tuple(s)
    hash_write: F,
}

impl<F: PrimeField> MultisetHashes<F> {
    pub fn append_to_transcript<G: CurveGroup<ScalarField = F>>(
        &self,
        transcript: &mut Transcript,
    ) {
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_hash_init",
            &self.hash_init,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_hash_read",
            &self.hash_read,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_hash_write",
            &self.hash_write,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_hash_final",
            &self.hash_final,
        );
    }
}

pub struct MemoryCheckingProof<G, Polynomials, ReadWriteOpenings, InitFinalOpenings>
where
    G: CurveGroup,
    Polynomials: BatchablePolynomials + ?Sized,
    ReadWriteOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
    InitFinalOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
{
    _polys: PhantomData<Polynomials>,
    /// Multiset hashes (init, read, write, final) for each memory.
    multiset_hashes: Vec<MultisetHashes<G::ScalarField>>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    read_write_grand_product: BatchedGrandProductArgument<G::ScalarField>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    init_final_grand_product: BatchedGrandProductArgument<G::ScalarField>,
    /// The opening proofs associated with the read/write grand product.
    read_write_openings: ReadWriteOpenings,
    /// The opening proofs associated with the init/final grand product.
    init_final_openings: InitFinalOpenings,
}

pub trait MemoryCheckingProver<F, G, Polynomials>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Polynomials: BatchablePolynomials,
{
    type ReadWriteOpenings: StructuredOpeningProof<F, G, Polynomials>;
    type InitFinalOpenings: StructuredOpeningProof<F, G, Polynomials>;
    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F);

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver.prove")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking(
        &self,
        polynomials: &Polynomials,
        batched_polys: &Polynomials::BatchedPolynomials,
        commitments: &Polynomials::Commitment,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> MemoryCheckingProof<G, Polynomials, Self::ReadWriteOpenings, Self::InitFinalOpenings> {
        // TODO(JOLT-62): Make sure Polynomials::Commitment have been posted to transcript.

        // fka "ProductLayerProof"
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = self.prove_grand_products(polynomials, transcript);

        // fka "HashLayerProof"
        let read_write_openings = Self::ReadWriteOpenings::prove_openings(
            batched_polys,
            commitments,
            &r_read_write,
            Self::ReadWriteOpenings::open(polynomials, &r_read_write),
            transcript,
            random_tape,
        );
        let init_final_openings = Self::InitFinalOpenings::prove_openings(
            batched_polys,
            commitments,
            &r_init_final,
            Self::InitFinalOpenings::open(polynomials, &r_init_final),
            transcript,
            random_tape,
        );

        MemoryCheckingProof {
            _polys: PhantomData,
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            read_write_openings,
            init_final_openings,
        }
    }

    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        &self,
        polynomials: &Polynomials,
        transcript: &mut Transcript,
    ) -> (
        BatchedGrandProductArgument<F>,
        BatchedGrandProductArgument<F>,
        Vec<MultisetHashes<F>>,
        Vec<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // fka "ProductLayerProof"
        let (read_write_circuit, read_hashes, write_hashes) =
            self.read_write_grand_product(polynomials, &gamma, &tau);
        let (init_final_circuit, init_hashes, final_hashes) =
            self.init_final_grand_product(polynomials, &gamma, &tau);
        debug_assert_eq!(read_hashes.len(), init_hashes.len());
        let num_memories = read_hashes.len();

        let mut multiset_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            let hashes = MultisetHashes {
                hash_init: init_hashes[i],
                hash_final: final_hashes[i],
                hash_read: read_hashes[i],
                hash_write: write_hashes[i],
            };
            debug_assert_eq!(
                hashes.hash_init * hashes.hash_write,
                hashes.hash_final * hashes.hash_read,
                "Multiset hashes don't match"
            );
            hashes.append_to_transcript::<G>(transcript);
            multiset_hashes.push(hashes);
        }

        let (read_write_grand_product, r_read_write) =
            BatchedGrandProductArgument::prove::<G>(read_write_circuit, transcript);
        let (init_final_grand_product, r_init_final) =
            BatchedGrandProductArgument::prove::<G>(init_final_circuit, transcript);
        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given `polynomials`. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProof.read_write_grand_product")]
    fn read_write_grand_product(
        &self,
        polynomials: &Polynomials,
        gamma: &F,
        tau: &F,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
        let read_leaves: Vec<DensePolynomial<F>> = self.read_leaves(polynomials, gamma, tau);
        let write_leaves: Vec<DensePolynomial<F>> = self.write_leaves(polynomials, gamma, tau);
        debug_assert_eq!(read_leaves.len(), write_leaves.len());
        let num_memories = read_leaves.len();

        let mut circuits = Vec::with_capacity(2 * num_memories);
        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            let read_circuit = GrandProductCircuit::new(&read_leaves[i]);
            let write_circuit = GrandProductCircuit::new(&write_leaves[i]);
            read_hashes.push(read_circuit.evaluate());
            write_hashes.push(write_circuit.evaluate());
            circuits.push(read_circuit);
            circuits.push(write_circuit);
        }

        (
            BatchedGrandProductCircuit::new_batch(circuits),
            read_hashes,
            write_hashes,
        )
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given `polynomials`. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProof.read_write_grand_product")]
    fn init_final_grand_product(
        &self,
        polynomials: &Polynomials,
        gamma: &F,
        tau: &F,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
        let init_leaves: Vec<DensePolynomial<F>> = self.init_leaves(polynomials, gamma, tau);
        let final_leaves: Vec<DensePolynomial<F>> = self.final_leaves(polynomials, gamma, tau);
        debug_assert_eq!(init_leaves.len(), final_leaves.len());
        let num_memories = init_leaves.len();

        let mut circuits = Vec::with_capacity(2 * num_memories);
        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            let init_circuit = GrandProductCircuit::new(&init_leaves[i]);
            let final_circuit = GrandProductCircuit::new(&final_leaves[i]);
            init_hashes.push(init_circuit.evaluate());
            final_hashes.push(final_circuit.evaluate());
            circuits.push(init_circuit);
            circuits.push(final_circuit);
        }

        (
            BatchedGrandProductCircuit::new_batch(circuits),
            init_hashes,
            final_hashes,
        )
    }

    /// Computes the MLE of the leaves of the "read" grand product circuit; one per memory.
    fn read_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F)
        -> Vec<DensePolynomial<F>>;
    /// Computes the MLE of the leaves of the "write" grand product circuit; one per memory.
    fn write_leaves(
        &self,
        polynomials: &Polynomials,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>>;
    /// Computes the MLE of the leaves of the "init" grand product circuit; one per memory.
    fn init_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F)
        -> Vec<DensePolynomial<F>>;
    /// Computes the MLE of the leaves of the "final" grand product circuit; one per memory.
    fn final_leaves(
        &self,
        polynomials: &Polynomials,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>>;
    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingVerifier<F, G, Polynomials>:
    MemoryCheckingProver<F, G, Polynomials>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Polynomials: BatchablePolynomials,
{
    /// Verifies a memory checking proof, given its associated polynomial `commitment`.
    fn verify_memory_checking(
        mut proof: MemoryCheckingProof<
            G,
            Polynomials,
            Self::ReadWriteOpenings,
            Self::InitFinalOpenings,
        >,
        commitments: &Polynomials::Commitment,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        for hash in &proof.multiset_hashes {
            // Multiset equality check
            assert_eq!(
                hash.hash_init * hash.hash_write,
                hash.hash_read * hash.hash_final
            );
            hash.append_to_transcript::<G>(transcript);
        }

        let interleaved_read_write_hashes = proof
            .multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_read, hash.hash_write])
            .collect();
        let interleaved_init_final_hashes = proof
            .multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_init, hash.hash_final])
            .collect();

        let (claims_read_write, r_read_write) = proof
            .read_write_grand_product
            .verify::<G, Transcript>(&interleaved_read_write_hashes, transcript);
        let (claims_init_final, r_init_final) = proof
            .init_final_grand_product
            .verify::<G, Transcript>(&interleaved_init_final_hashes, transcript);

        proof
            .read_write_openings
            .verify_openings(commitments, &r_read_write, transcript)?;
        proof
            .init_final_openings
            .verify_openings(commitments, &r_init_final, transcript)?;

        Self::compute_verifier_openings(&mut proof.init_final_openings, &r_init_final);

        assert_eq!(claims_read_write.len(), claims_init_final.len());
        assert!(claims_read_write.len() % 2 == 0);
        let num_memories = claims_read_write.len() / 2;
        let grand_product_claims: Vec<MultisetHashes<F>> = (0..num_memories)
            .map(|i| MultisetHashes {
                hash_read: claims_read_write[2 * i],
                hash_write: claims_read_write[2 * i + 1],
                hash_init: claims_init_final[2 * i],
                hash_final: claims_init_final[2 * i + 1],
            })
            .collect();
        Self::check_fingerprints(
            grand_product_claims,
            &proof.read_write_openings,
            &proof.init_final_openings,
            &gamma,
            &tau,
        );

        Ok(())
    }

    /// Often some of the fields in `InitFinalOpenings` do not require an opening proof provided by
    /// the prover, and instead can be efficiently computed by the verifier by itself. This function
    /// populates any such fields in `openings`.
    fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>);
    /// Computes "read" memory tuples (one per memory) from the given `openings`.
    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple>;
    /// Computes "write" memory tuples (one per memory) from the given `openings`.
    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple>;
    /// Computes "init" memory tuples (one per memory) from the given `openings`.
    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple>;
    /// Computes "final" memory tuples (one per memory) from the given `openings`.
    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple>;

    /// Checks that the claimed multiset hashes (output by grand product) are consistent with the
    /// openings given by `read_write_openings` and `init_final_openings`.
    fn check_fingerprints(
        claims: Vec<MultisetHashes<F>>,
        read_write_openings: &Self::ReadWriteOpenings,
        init_final_openings: &Self::InitFinalOpenings,
        gamma: &F,
        tau: &F,
    ) {
        let read_fingerprints: Vec<_> =
            <Self as MemoryCheckingVerifier<_, _, _>>::read_tuples(read_write_openings)
                .iter()
                .map(|tuple| Self::fingerprint(tuple, gamma, tau))
                .collect();
        let write_fingerprints: Vec<_> =
            <Self as MemoryCheckingVerifier<_, _, _>>::write_tuples(read_write_openings)
                .iter()
                .map(|tuple| Self::fingerprint(tuple, gamma, tau))
                .collect();
        let init_fingerprints: Vec<_> =
            <Self as MemoryCheckingVerifier<_, _, _>>::init_tuples(init_final_openings)
                .iter()
                .map(|tuple| Self::fingerprint(tuple, gamma, tau))
                .collect();
        let final_fingerprints: Vec<_> =
            <Self as MemoryCheckingVerifier<_, _, _>>::final_tuples(init_final_openings)
                .iter()
                .map(|tuple| Self::fingerprint(tuple, gamma, tau))
                .collect();
        for (i, claim) in claims.iter().enumerate() {
            assert_eq!(claim.hash_read, read_fingerprints[i]);
            assert_eq!(claim.hash_write, write_fingerprints[i]);
            assert_eq!(claim.hash_init, init_fingerprints[i]);
            assert_eq!(claim.hash_final, final_fingerprints[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use ark_curve25519::{EdwardsProjective, Fr};
    use ark_ff::Field;
    use ark_std::{One, Zero};

    #[test]
    fn product_layer_proof_trivial() {
        struct NormalMems {
            a_ops: DensePolynomial<Fr>,

            v_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_reads: DensePolynomial<Fr>,
            t_finals: DensePolynomial<Fr>,
        }
        struct FakeType();
        struct FakeOpeningProof();
        #[rustfmt::skip]
    impl StructuredOpeningProof<Fr, EdwardsProjective, NormalMems> for FakeOpeningProof {
      type Openings = FakeType;
      fn open(_: &NormalMems, _: &Vec<Fr>) -> Self::Openings { unimplemented!() }
      fn prove_openings(_: &FakeType, _: &FakeType, _: &Vec<Fr>, _: Self::Openings, _: &mut Transcript, _: &mut RandomTape<EdwardsProjective>) -> Self { unimplemented!() }
      fn verify_openings(&self, _: &FakeType, _: &Vec<Fr>, _: &mut Transcript) -> Result<(), ProofVerifyError> { unimplemented!() }
    }

        #[rustfmt::skip]
    impl BatchablePolynomials for NormalMems {
      type Commitment = FakeType;
      type BatchedPolynomials = FakeType;

      fn batch(&self) -> Self::BatchedPolynomials { unimplemented!() }
      fn commit(_batched_polys: &Self::BatchedPolynomials) -> Self::Commitment { unimplemented!() }
    }

        struct TestProver {}
        #[rustfmt::skip] // Keep leaf functions small
    impl MemoryCheckingProver<Fr, EdwardsProjective, NormalMems> for TestProver {
      type ReadWriteOpenings = FakeOpeningProof;
      type InitFinalOpenings = FakeOpeningProof;

      type MemoryTuple = (Fr, Fr, Fr);

      fn read_leaves(
        &self,
        polynomials: &NormalMems,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        vec![DensePolynomial::new((0..polynomials.a_ops.len())
          .map(|i| Self::fingerprint(&(polynomials.a_ops[i], polynomials.v_ops[i], polynomials.t_reads[i]), gamma, tau))
          .collect())]
      }

      fn write_leaves(
        &self,
        polynomials: &NormalMems,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        vec![DensePolynomial::new((0..polynomials.a_ops.len())
          .map(|i| Self::fingerprint(&(polynomials.a_ops[i], polynomials.v_ops[i], polynomials.t_reads[i] + Fr::one()), gamma, tau))
          .collect())]
      }

      fn init_leaves(
        &self,
        polynomials: &NormalMems,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        vec![DensePolynomial::new((0..polynomials.v_mems.len())
          .map(|i| Self::fingerprint(&(Fr::from(i as u64), polynomials.v_mems[i], Fr::zero()), gamma, tau))
          .collect())]
      }

      fn final_leaves(
        &self,
        polynomials: &NormalMems,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        vec![DensePolynomial::new((0..polynomials.v_mems.len())
          .map(|i| Self::fingerprint(&(Fr::from(i as u64), polynomials.v_mems[i], polynomials.t_finals[i]), gamma, tau))
          .collect())]
      }

      fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
        let (a, v, t) = tuple;
        t * &gamma.square() + v * gamma + a - tau
      }

      fn protocol_name() -> &'static [u8] {
        b"protocol_name"
      }
    }
        // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let v_ops = a_ops.clone();

        let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];

        let a_ops = DensePolynomial::new(a_ops);
        let v_ops = DensePolynomial::new(v_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_reads = DensePolynomial::new(t_reads);
        let t_finals = DensePolynomial::new(t_finals);
        let polys = NormalMems {
            a_ops,
            v_ops,
            v_mems,
            t_reads,
            t_finals,
        };

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let prover = TestProver {};
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            prover.prove_grand_products(&polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        for hash in multiset_hashes.iter() {
            hash.append_to_transcript::<EdwardsProjective>(&mut transcript);
        }

        let interleaved_read_write_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_read, hash.hash_write])
            .collect();
        let interleaved_init_final_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_init, hash.hash_final])
            .collect();
        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }

    fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let set1: HashSet<_> = vec1.iter().cloned().collect();
        let set2: HashSet<_> = vec2.iter().cloned().collect();
        set1.difference(&set2).cloned().collect()
    }

    #[test]
    fn product_layer_proof_batched() {
        // Define a GrandProduct circuit that can be batched across 2 memories
        struct Polys {
            a_0_ops: DensePolynomial<Fr>,
            a_1_ops: DensePolynomial<Fr>,

            v_0_ops: DensePolynomial<Fr>,
            v_1_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_0_reads: DensePolynomial<Fr>,
            t_1_reads: DensePolynomial<Fr>,

            t_0_finals: DensePolynomial<Fr>,
            t_1_finals: DensePolynomial<Fr>,
        }

        struct FakeType();
        struct FakeOpeningProof();
        #[rustfmt::skip]
    impl StructuredOpeningProof<Fr, EdwardsProjective, Polys> for FakeOpeningProof {
      type Openings = FakeType;
      fn open(_: &Polys, _: &Vec<Fr>) -> Self::Openings { unimplemented!() }
      fn prove_openings(_: &FakeType, _: &FakeType, _: &Vec<Fr>, _: Self::Openings, _: &mut Transcript, _: &mut RandomTape<EdwardsProjective>) -> Self { unimplemented!() }
      fn verify_openings(&self, _: &FakeType, _: &Vec<Fr>, _: &mut Transcript) -> Result<(), ProofVerifyError> { unimplemented!() }
    }

        #[rustfmt::skip]
    impl BatchablePolynomials for Polys {
      type Commitment = FakeType;
      type BatchedPolynomials = FakeType;

      fn batch(&self) -> Self::BatchedPolynomials { unimplemented!() }
      fn commit(_batched_polys: &Self::BatchedPolynomials) -> Self::Commitment { unimplemented!() }
    }

        struct TestProver {}
        #[rustfmt::skip] // Keep leaf functions small
    impl MemoryCheckingProver<Fr, EdwardsProjective, Polys> for TestProver {
      type ReadWriteOpenings = FakeOpeningProof;
      type InitFinalOpenings = FakeOpeningProof;

      type MemoryTuple = (Fr, Fr, Fr);

      fn read_leaves(
        &self,
        polynomials: &Polys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.a_0_ops.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index]),
                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index]),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn write_leaves(
        &self,
        polynomials: &Polys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.a_0_ops.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index] + Fr::one()),
                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index] + Fr::one()),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn init_leaves(
        &self,
        polynomials: &Polys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.v_mems.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 | 1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], Fr::zero()),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn final_leaves(
        &self,
        polynomials: &Polys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.v_mems.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_0_finals[leaf_index]),
                1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_1_finals[leaf_index]),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
        let (a, v, t) = tuple;
        t * &gamma.square() + v * gamma + a - tau
      }

      fn protocol_name() -> &'static [u8] {
        b"protocol_name"
      }
    }

        // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
        let v_0_ops = a_0_ops.clone();
        let v_1_ops = a_1_ops.clone();

        let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
        let t_0_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];
        let t_1_finals = vec![
            Fr::from(2),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];

        let a_0_ops = DensePolynomial::new(a_0_ops);
        let a_1_ops = DensePolynomial::new(a_1_ops);
        let v_0_ops = DensePolynomial::new(v_0_ops);
        let v_1_ops = DensePolynomial::new(v_1_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_0_reads = DensePolynomial::new(t_0_reads);
        let t_1_reads = DensePolynomial::new(t_1_reads);
        let t_0_finals = DensePolynomial::new(t_0_finals);
        let t_1_finals = DensePolynomial::new(t_1_finals);
        let polys = Polys {
            a_0_ops,
            a_1_ops,
            v_0_ops,
            v_1_ops,
            v_mems,
            t_0_reads,
            t_1_reads,
            t_0_finals,
            t_1_finals,
        };

        let prover = TestProver {};

        // Check leaves match
        let (gamma, tau) = (&Fr::from(100), &Fr::from(35));
        let init_leaves: Vec<DensePolynomial<Fr>> = prover.init_leaves(&polys, gamma, tau);
        let read_leaves: Vec<DensePolynomial<Fr>> = prover.read_leaves(&polys, gamma, tau);
        let write_leaves: Vec<DensePolynomial<Fr>> = prover.write_leaves(&polys, gamma, tau);
        let final_leaves: Vec<DensePolynomial<Fr>> = prover.final_leaves(&polys, gamma, tau);

        [0, 1].into_iter().for_each(|i| {
            let init_leaves = &init_leaves[i];
            let read_leaves = &read_leaves[i];
            let write_leaves = &write_leaves[i];
            let final_leaves = &final_leaves[i];

            let read_final_leaves = vec![read_leaves.evals(), final_leaves.evals()].concat();
            let init_write_leaves = vec![init_leaves.evals(), write_leaves.evals()].concat();
            let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
            assert_eq!(difference.len(), 0);
        });

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            prover.prove_grand_products(&polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        for hash in multiset_hashes.iter() {
            hash.append_to_transcript::<EdwardsProjective>(&mut transcript);
        }

        let interleaved_read_write_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_read, hash.hash_write])
            .collect();
        let interleaved_init_final_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_init, hash.hash_final])
            .collect();
        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }

    #[test]
    fn product_layer_proof_flags_no_reuse() {
        // Define a GrandProduct circuit that can be batched across 2 memories
        struct FlagPolys {
            a_0_ops: DensePolynomial<Fr>,
            a_1_ops: DensePolynomial<Fr>,

            v_0_ops: DensePolynomial<Fr>,
            v_1_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_0_reads: DensePolynomial<Fr>,
            t_1_reads: DensePolynomial<Fr>,

            t_0_finals: DensePolynomial<Fr>,
            t_1_finals: DensePolynomial<Fr>,

            flags_0: DensePolynomial<Fr>,
            flags_1: DensePolynomial<Fr>,
        }

        struct FakeType();
        struct FakeOpeningProof();
        #[rustfmt::skip]
    impl StructuredOpeningProof<Fr, EdwardsProjective, FlagPolys> for FakeOpeningProof {
      type Openings = FakeType;
      fn open(_: &FlagPolys, _: &Vec<Fr>) -> Self::Openings { unimplemented!() }
      fn prove_openings(_: &FakeType, _: &FakeType, _: &Vec<Fr>, _: Self::Openings, _: &mut Transcript, _: &mut RandomTape<EdwardsProjective>) -> Self { unimplemented!() }
      fn verify_openings(&self, _: &FakeType, _: &Vec<Fr>, _: &mut Transcript) -> Result<(), ProofVerifyError> { unimplemented!() }
    }

        #[rustfmt::skip]
    impl BatchablePolynomials for FlagPolys {
      type Commitment = FakeType;
      type BatchedPolynomials = FakeType;

      fn batch(&self) -> Self::BatchedPolynomials { unimplemented!() }
      fn commit(_batched_polys: &Self::BatchedPolynomials) -> Self::Commitment { unimplemented!() }
    }

        struct TestProver {}
        #[rustfmt::skip] // Keep leaf functions small
    impl MemoryCheckingProver<Fr, EdwardsProjective, FlagPolys> for TestProver {
      type ReadWriteOpenings = FakeOpeningProof;
      type InitFinalOpenings = FakeOpeningProof;

      type MemoryTuple = (Fr, Fr, Fr, Option<Fr>);

      fn read_leaves(
        &self,
        polynomials: &FlagPolys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.a_0_ops.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index], None),
                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index], None),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn write_leaves(
        &self,
        polynomials: &FlagPolys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.a_0_ops.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index] + Fr::one(), None),
                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index] + Fr::one(), None),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn init_leaves(
        &self,
        polynomials: &FlagPolys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.v_mems.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 | 1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], Fr::zero(), None),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn final_leaves(
        &self,
        polynomials: &FlagPolys,
        gamma: &Fr,
        tau: &Fr,
      ) -> Vec<DensePolynomial<Fr>> {
        [0,1].iter().map(|memory_index| {
          DensePolynomial::new((0..polynomials.v_mems.len())
            .map(|leaf_index| {
              let tuple = match memory_index {
                0 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_0_finals[leaf_index], None),
                1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_1_finals[leaf_index], None),
                _ => unimplemented!()
              };
              Self::fingerprint(&tuple, gamma, tau)
            })
            .collect())
        }).collect()
      }

      fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
        let (a, v, t, flag) = *tuple;
        match flag {
          Some(val) => val * (t * gamma.square() + v * *gamma + a - tau) + Fr::one() - val,
          None => t * gamma.square() + v * *gamma + a - tau,
        }
      }

      // FLAGS OVERRIDES

      // Override read_write_grand product to call BatchedGrandProductCircuit::new_batch_flags and insert our additional toggling layer.
      fn read_write_grand_product(
          &self,
          polynomials: &FlagPolys,
          gamma: &Fr,
          tau: &Fr,
        ) -> (BatchedGrandProductCircuit<Fr>, Vec<Fr>, Vec<Fr>) {
          // Fingerprint will generate "unflagged" leaves for the final layer
          let read_fingerprints: Vec<DensePolynomial<Fr>> = self.read_leaves(polynomials, gamma, tau);
          let write_fingerprints: Vec<DensePolynomial<Fr>> = self.write_leaves(polynomials, gamma, tau);

          // Generate "flagged" leaves for the second to last layer. Input to normal Grand Products
          let num_memories = 2;
          let mut circuits = Vec::with_capacity(2 * num_memories);
          let mut read_hashes = Vec::with_capacity(num_memories);
          let mut write_hashes = Vec::with_capacity(num_memories);

          for i in 0..num_memories {
            let mut toggled_read_fingerprints = read_fingerprints[i].evals();
            let mut toggled_write_fingerprints = write_fingerprints[i].evals();

            let subtable_index = i;
            for leaf_index in 0..polynomials.a_0_ops.len() {
              let flag = match subtable_index {
                0 => polynomials.flags_0[leaf_index],
                1 => polynomials.flags_1[leaf_index],
                _ => unimplemented!()
              };
              if flag == Fr::zero() {
                toggled_read_fingerprints[leaf_index] = Fr::one();
                toggled_write_fingerprints[leaf_index] = Fr::one();
              }
            }

            let read_circuit = GrandProductCircuit::new(&DensePolynomial::new(toggled_read_fingerprints));
            let write_circuit = GrandProductCircuit::new(&DensePolynomial::new(toggled_write_fingerprints));
            read_hashes.push(read_circuit.evaluate());
            write_hashes.push(write_circuit.evaluate());
            circuits.push(read_circuit);
            circuits.push(write_circuit);
          }

          let expanded_flag_map = vec![0, 0, 1, 1];
          let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
            circuits, 
            vec![polynomials.flags_0.clone(), polynomials.flags_1.clone()], 
            expanded_flag_map, 
            vec![read_fingerprints[0].clone(), write_fingerprints[0].clone(), read_fingerprints[1].clone(), write_fingerprints[1].clone()]
          );

          (batched_circuits, read_hashes, write_hashes)
      }

      fn protocol_name() -> &'static [u8] {
        b"protocol_name"
      }
    }

        // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
        let v_0_ops = a_0_ops.clone();
        let v_1_ops = a_1_ops.clone();

        let flags_0 = vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()];
        let flags_1 = vec![
            Fr::one(),
            Fr::zero(), // Flagged off!
            Fr::one(),
            Fr::one(),
        ];

        let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
        let t_0_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];
        let t_1_finals = vec![
            Fr::from(2),
            Fr::zero(), // Flagged off!
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];

        let a_0_ops = DensePolynomial::new(a_0_ops);
        let a_1_ops = DensePolynomial::new(a_1_ops);
        let v_0_ops = DensePolynomial::new(v_0_ops);
        let v_1_ops = DensePolynomial::new(v_1_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_0_reads = DensePolynomial::new(t_0_reads);
        let t_1_reads = DensePolynomial::new(t_1_reads);
        let t_0_finals = DensePolynomial::new(t_0_finals);
        let t_1_finals = DensePolynomial::new(t_1_finals);
        let flags_0 = DensePolynomial::new(flags_0);
        let flags_1 = DensePolynomial::new(flags_1);
        let polys = FlagPolys {
            a_0_ops,
            a_1_ops,
            v_0_ops,
            v_1_ops,
            v_mems,
            t_0_reads,
            t_1_reads,
            t_0_finals,
            t_1_finals,
            flags_0,
            flags_1,
        };

        let prover = TestProver {};

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            prover.prove_grand_products(&polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        for hash in multiset_hashes.iter() {
            hash.append_to_transcript::<EdwardsProjective>(&mut transcript);
        }

        let interleaved_read_write_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_read, hash.hash_write])
            .collect();
        let interleaved_init_final_hashes = multiset_hashes
            .iter()
            .flat_map(|hash| [hash.hash_init, hash.hash_final])
            .collect();
        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }
}
