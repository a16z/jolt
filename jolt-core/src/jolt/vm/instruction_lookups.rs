use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::BatchedGrandProduct;
use crate::subprotocols::sparse_grand_product::ToggledBatchedGrandProduct;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{interleave, Itertools};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::marker::PhantomData;
use tracing::trace_span;

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstructionSet, SubtableIndices};
use crate::jolt::subtable::JoltSubtableSet;
use crate::lasso::memory_checking::{
    Initializable, MultisetHashes, NoExogenousOpenings, StructuredPolynomialData,
    VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::utils::mul_0_1_optimized;
use crate::utils::transcript::Transcript;
use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, transcript::AppendToTranscript},
};

use super::{JoltCommitments, JoltPolynomials, JoltTraceStep};

#[derive(Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionLookupStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    /// `C`-sized vector of polynomials/commitments/openings corresponding to the
    /// indices at which subtables are queried.
    pub(crate) dim: Vec<T>,
    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the read access counts for each memory.
    read_cts: Vec<T>,
    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the final access counts for each memory.
    pub(crate) final_cts: Vec<T>,
    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the values read from each memory.
    pub(crate) E_polys: Vec<T>,
    /// `NUM_INSTRUCTIONS`-sized vector of polynomials/commitments/openings corresponding
    /// to the indicator bitvectors designating which lookup to perform at each step of
    /// the execution trace.
    pub(crate) instruction_flags: Vec<T>,
    /// The polynomial/commitment/opening corresponding to the lookup output for each
    /// step of the execution trace.
    pub(crate) lookup_outputs: T,

    /// Hack: This is only populated for `InstructionLookupPolynomials`, where
    /// the instruction flags are kept in u64 representation for efficient conversion
    /// to memory flags.
    instruction_flag_bitvectors: Option<Vec<Vec<u64>>>,

    a_init_final: VerifierComputedOpening<T>,
    v_init_final: VerifierComputedOpening<Vec<T>>,
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type InstructionLookupPolynomials<F: JoltField> = InstructionLookupStuff<DensePolynomial<F>>;
/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type InstructionLookupOpenings<F: JoltField> = InstructionLookupStuff<F>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type InstructionLookupCommitments<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> = InstructionLookupStuff<PCS::Commitment>;

impl<const C: usize, F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    Initializable<T, InstructionLookupsPreprocessing<C, F>> for InstructionLookupStuff<T>
{
    fn initialize(preprocessing: &InstructionLookupsPreprocessing<C, F>) -> Self {
        Self {
            dim: std::iter::repeat_with(|| T::default()).take(C).collect(),
            read_cts: std::iter::repeat_with(|| T::default())
                .take(preprocessing.num_memories)
                .collect(),
            final_cts: std::iter::repeat_with(|| T::default())
                .take(preprocessing.num_memories)
                .collect(),
            E_polys: std::iter::repeat_with(|| T::default())
                .take(preprocessing.num_memories)
                .collect(),
            instruction_flags: std::iter::repeat_with(|| T::default())
                .take(preprocessing.instruction_to_memory_indices.len())
                .collect(),
            instruction_flag_bitvectors: None,
            lookup_outputs: T::default(),
            a_init_final: None,
            v_init_final: None,
        }
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T>
    for InstructionLookupStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        self.dim
            .iter()
            .chain(self.read_cts.iter())
            .chain(self.E_polys.iter())
            .chain(self.instruction_flags.iter())
            .chain([&self.lookup_outputs])
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        self.final_cts.iter().collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.dim
            .iter_mut()
            .chain(self.read_cts.iter_mut())
            .chain(self.E_polys.iter_mut())
            .chain(self.instruction_flags.iter_mut())
            .chain([&mut self.lookup_outputs])
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        self.final_cts.iter_mut().collect()
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
/// Polynomial openings associated with the "primary sumcheck" of Jolt instruction lookups.
struct PrimarySumcheckOpenings<F>
where
    F: JoltField,
{
    /// Evaluations of the E_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    E_poly_openings: Vec<F>,
    /// Evaluations of the flag polynomials at the opening point. Vector is of length NUM_INSTRUCTIONS.
    flag_openings: Vec<F>,
    /// Evaluation of the lookup_outputs polynomial at the opening point.
    lookup_outputs_opening: F,
}

impl<const C: usize, const M: usize, F, PCS, InstructionSet, Subtables, ProofTranscript>
    MemoryCheckingProver<F, PCS, ProofTranscript>
    for InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    type ReadWriteGrandProduct = ToggledBatchedGrandProduct<F>;

    type Polynomials = InstructionLookupPolynomials<F>;
    type Openings = InstructionLookupOpenings<F>;
    type Commitments = InstructionLookupCommitments<PCS, ProofTranscript>;

    type Preprocessing = InstructionLookupsPreprocessing<C, F>;

    type MemoryTuple = (F, F, F, Option<F>); // (a, v, t, flag)

    fn fingerprint(inputs: &(F, F, F, Option<F>), gamma: &F, tau: &F) -> F {
        let (a, v, t, _flag) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::compute_leaves")]
    fn compute_leaves(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (
        <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
        <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
    ) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dim[dim_index][i];
                        let v = &polynomials.E_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + *a
                            - *tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();
                [read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_final_leaves: Vec<F> = preprocessing
            .materialized_subtables
            .par_iter()
            .enumerate()
            .flat_map_iter(|(subtable_index, subtable)| {
                let mut leaves: Vec<F> = unsafe_allocate_zero_vec(
                    M * (preprocessing.subtable_to_memory_indices[subtable_index].len() + 1),
                );
                // Init leaves
                (0..M).for_each(|i| {
                    let a = &F::from_u64(i as u64).unwrap();
                    let v = &subtable[i];
                    // let t = F::zero();
                    // Compute h(a,v,t) where t == 0
                    leaves[i] = mul_0_1_optimized(v, gamma) + *a - *tau;
                });
                // Final leaves
                let mut leaf_index = M;
                for memory_index in &preprocessing.subtable_to_memory_indices[subtable_index] {
                    let final_cts = &polynomials.final_cts[*memory_index];
                    (0..M).for_each(|i| {
                        leaves[leaf_index] =
                            leaves[i] + mul_0_1_optimized(&final_cts[i], &gamma_squared);
                        leaf_index += 1;
                    });
                }

                leaves
            })
            .collect();

        let memory_flags = Self::memory_flag_indices(
            preprocessing,
            polynomials.instruction_flag_bitvectors.as_ref().unwrap(),
        );

        (
            (memory_flags, read_write_leaves),
            (
                init_final_leaves,
                // # init = # subtables; # final = # memories
                Self::NUM_SUBTABLES + preprocessing.num_memories,
            ),
        )
    }

    fn interleave<T: Copy + Clone>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        // R W R W R W ...
        let read_write_values = interleave(read_values.clone(), write_values.clone()).collect();

        // I F F F F I F F F F ...
        let mut init_final_values = Vec::with_capacity(init_values.len() + final_values.len());
        for subtable_index in 0..Self::NUM_SUBTABLES {
            init_final_values.push(init_values[subtable_index]);
            let memory_indices = &preprocessing.subtable_to_memory_indices[subtable_index];
            memory_indices
                .iter()
                .for_each(|i| init_final_values.push(final_values[*i]));
        }

        (read_write_values, init_final_values)
    }

    fn uninterleave_hashes(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len(), 2 * preprocessing.num_memories);
        assert_eq!(
            init_final_hashes.len(),
            Self::NUM_SUBTABLES + preprocessing.num_memories
        );

        let mut read_hashes = Vec::with_capacity(preprocessing.num_memories);
        let mut write_hashes = Vec::with_capacity(preprocessing.num_memories);
        for i in 0..preprocessing.num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(Self::NUM_SUBTABLES);
        let mut final_hashes = Vec::with_capacity(preprocessing.num_memories);
        let mut init_final_hashes = init_final_hashes.iter();
        for subtable_index in 0..Self::NUM_SUBTABLES {
            // I
            init_hashes.push(*init_final_hashes.next().unwrap());
            // F F F F
            let memory_indices = &preprocessing.subtable_to_memory_indices[subtable_index];
            for _ in memory_indices {
                final_hashes.push(*init_final_hashes.next().unwrap());
            }
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        assert_eq!(multiset_hashes.init_hashes.len(), Self::NUM_SUBTABLES);
        assert_eq!(
            multiset_hashes.read_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(
            multiset_hashes.write_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(
            multiset_hashes.final_hashes.len(),
            preprocessing.num_memories
        );

        (0..preprocessing.num_memories)
            .into_par_iter()
            .for_each(|i| {
                let read_hash = multiset_hashes.read_hashes[i];
                let write_hash = multiset_hashes.write_hashes[i];
                let init_hash =
                    multiset_hashes.init_hashes[preprocessing.memory_to_subtable_index[i]];
                let final_hash = multiset_hashes.final_hashes[i];
                assert_eq!(
                    init_hash * write_hash,
                    final_hash * read_hash,
                    "Multiset hashes don't match"
                );
            });
    }

    fn protocol_name() -> &'static [u8] {
        b"Instruction lookups check"
    }
}

impl<F, PCS, InstructionSet, Subtables, const C: usize, const M: usize, ProofTranscript>
    MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut Self::Openings,
        _preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));
        openings.v_init_final = Some(
            Subtables::iter()
                .map(|subtable| subtable.evaluate_mle(r_init_final))
                .collect(),
        );
    }

    fn read_tuples(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let memory_flags = Self::memory_flags(preprocessing, &openings.instruction_flags);
        (0..preprocessing.num_memories)
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                (
                    openings.dim[dim_index],
                    openings.E_polys[memory_index],
                    openings.read_cts[memory_index],
                    Some(memory_flags[memory_index]),
                )
            })
            .collect()
    }
    fn write_tuples(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        Self::read_tuples(preprocessing, openings, &NoExogenousOpenings)
            .iter()
            .map(|(a, v, t, flag)| (*a, *v, *t + F::one(), *flag))
            .collect()
    }
    fn init_tuples(
        _preprocessing: &InstructionLookupsPreprocessing<C, F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::NUM_SUBTABLES)
            .map(|subtable_index| (a_init, v_init[subtable_index], F::zero(), None))
            .collect()
    }
    fn final_tuples(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..preprocessing.num_memories)
            .map(|memory_index| {
                (
                    a_init,
                    v_init[preprocessing.memory_to_subtable_index[memory_index]],
                    openings.final_cts[memory_index],
                    None,
                )
            })
            .collect()
    }

    /// Checks that the claims output by the grand products are consistent with the openings of
    /// the polynomials comprising the input layers.
    ///
    /// Differs from the default `check_fingerprints` implementation because the input layer
    /// of the read-write grand product is a `BatchedGrandProductToggleLayer`, so we need to
    /// evaluate a multi-*quadratic* extension of the leaves rather than a multilinear extension.
    /// This means we handle the openings a bit differently.
    fn check_fingerprints(
        preprocessing: &Self::Preprocessing,
        read_write_claim: F,
        init_final_claim: F,
        r_read_write_batch_index: &[F],
        r_init_final_batch_index: &[F],
        openings: &Self::Openings,
        exogenous_openings: &NoExogenousOpenings,
        gamma: &F,
        tau: &F,
    ) {
        let read_tuples: Vec<_> = Self::read_tuples(preprocessing, openings, exogenous_openings);
        let write_tuples: Vec<_> = Self::write_tuples(preprocessing, openings, exogenous_openings);
        let init_tuples: Vec<_> = Self::init_tuples(preprocessing, openings, exogenous_openings);
        let final_tuples: Vec<_> = Self::final_tuples(preprocessing, openings, exogenous_openings);

        let (read_write_tuples, init_final_tuples) = Self::interleave(
            preprocessing,
            &read_tuples,
            &write_tuples,
            &init_tuples,
            &final_tuples,
        );

        assert_eq!(
            read_write_tuples.len().next_power_of_two(),
            r_read_write_batch_index.len().pow2(),
        );
        assert_eq!(
            init_final_tuples.len().next_power_of_two(),
            r_init_final_batch_index.len().pow2()
        );

        let mut read_write_flags: Vec<_> = read_write_tuples
            .iter()
            .map(|tuple| tuple.3.unwrap())
            .collect();
        // For the toggled grand product, the flags in the input layer are padded with 1s,
        // while the fingerprints are padded with 0s, so that all subsequent padding layers
        // are all 0s.
        // To see why this is the case, observe that the input layer's gates will output
        // flag * fingerprint + 1 - flag = 1 * 0 + 1 - 1 = 0.
        // Then all subsequent layers will output gate values 0 * 0 = 0.
        read_write_flags.resize(read_write_flags.len().next_power_of_two(), F::one());

        // Let r' := r_read_write_batch_index
        // and r'':= r_read_write_opening.
        //
        // Let k denote the batch size.
        //
        // The `read_write_flags` vector above contains the evaluations of the k individual
        // flag MLEs at r''.
        //
        // What we want to compute is the evaluation of the MLE of ALL the flags, concatenated together,
        // at (r', r''):
        //
        // flags(r', r'') = \sum_j eq(r', j) * flag_j(r'')
        //
        // where flag_j(r'') is what we already have in `read_write_flags`.
        let combined_flags: F = read_write_flags
            .iter()
            .zip(EqPolynomial::evals(r_read_write_batch_index).iter())
            .map(|(flag, eq_eval)| *flag * eq_eval)
            .sum();
        // Similar thing for the fingerprints:
        //
        // fingerprints(r', r'') = \sum_j eq(r', j) * (t_j(r'') * \gamma^2 + v_j(r'') * \gamma + a_j(r'') - \tau)
        let combined_read_write_fingerprint: F = read_write_tuples
            .iter()
            .zip(EqPolynomial::evals(r_read_write_batch_index).iter())
            .map(|(tuple, eq_eval)| Self::fingerprint(tuple, gamma, tau) * eq_eval)
            .sum();

        // Now we combine flags(r', r'') and fingerprints(r', r'') to obtain the evaluation of the
        // multi-*quadratic* extension W of the input layer at (r', r'')
        //
        // W(r', r'') = flags(r', r'') * fingerprints(r', r'') + 1 - flags(r', r'')
        //
        // and this should equal the claim output by the read-write grand product.
        assert_eq!(
            combined_flags * combined_read_write_fingerprint + F::one() - combined_flags,
            read_write_claim
        );

        // The init-final grand product isn't toggled using flags (it's just a "normal" grand product)
        // so we combine the openings the normal way.
        let combined_init_final_fingerprint: F = init_final_tuples
            .iter()
            .zip(EqPolynomial::evals(r_init_final_batch_index).iter())
            .map(|(tuple, eq_eval)| Self::fingerprint(tuple, gamma, tau) * eq_eval)
            .sum();
        assert_eq!(combined_init_final_fingerprint, init_final_claim);
    }
}

/// Proof of instruction lookups for a single Jolt program execution.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionLookupsProof<
    const C: usize,
    const M: usize,
    F,
    PCS,
    InstructionSet,
    Subtables,
    ProofTranscript,
> where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Subtables: JoltSubtableSet<F>,
    InstructionSet: JoltInstructionSet,
    ProofTranscript: Transcript,
{
    _instructions: PhantomData<InstructionSet>,
    _subtables: PhantomData<Subtables>,
    primary_sumcheck: PrimarySumcheck<F, ProofTranscript>,
    memory_checking: MemoryCheckingProof<
        F,
        PCS,
        InstructionLookupOpenings<F>,
        NoExogenousOpenings,
        ProofTranscript,
    >,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    num_rounds: usize,
    openings: PrimarySumcheckOpenings<F>,
    // opening_proof: PCS::BatchedProof,
    _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone)]
pub struct InstructionLookupsPreprocessing<const C: usize, F: JoltField> {
    subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    instruction_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<Vec<F>>,
    num_memories: usize,
}

impl<const C: usize, F: JoltField> InstructionLookupsPreprocessing<C, F> {
    #[tracing::instrument(skip_all, name = "InstructionLookups::preprocess")]
    pub fn preprocess<const M: usize, InstructionSet, Subtables>() -> Self
    where
        InstructionSet: JoltInstructionSet,
        Subtables: JoltSubtableSet<F>,
    {
        let materialized_subtables = Self::materialize_subtables::<M, Subtables>();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); Subtables::COUNT];
        for instruction in InstructionSet::iter() {
            for (subtable, indices) in instruction.subtables::<F>(C, M) {
                subtable_indices[Subtables::enum_index(subtable)].union_with(&indices);
            }
        }

        let mut subtable_to_memory_indices = Vec::with_capacity(Subtables::COUNT);
        let mut memory_to_subtable_index = vec![];
        let mut memory_to_dimension_index = vec![];

        let mut memory_index = 0;
        for (subtable_index, dimension_indices) in subtable_indices.iter().enumerate() {
            subtable_to_memory_indices
                .push((memory_index..memory_index + dimension_indices.len()).collect_vec());
            memory_to_subtable_index.extend(vec![subtable_index; dimension_indices.len()]);
            memory_to_dimension_index.extend(dimension_indices.iter());
            memory_index += dimension_indices.len();
        }
        let num_memories = memory_index;

        let mut instruction_to_memory_indices = vec![vec![]; InstructionSet::COUNT];
        for instruction in InstructionSet::iter() {
            for (subtable, dimension_indices) in instruction.subtables::<F>(C, M) {
                let memory_indices: Vec<_> = subtable_to_memory_indices
                    [Subtables::enum_index(subtable)]
                .iter()
                .filter(|memory_index| {
                    dimension_indices.contains(memory_to_dimension_index[**memory_index])
                })
                .collect();
                instruction_to_memory_indices[InstructionSet::enum_index(&instruction)]
                    .extend(memory_indices);
            }
        }

        Self {
            num_memories,
            materialized_subtables,
            subtable_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            instruction_to_memory_indices,
        }
    }

    /// Materializes all subtables used by this Jolt instance.
    #[tracing::instrument(skip_all)]
    fn materialize_subtables<const M: usize, Subtables>() -> Vec<Vec<F>>
    where
        Subtables: JoltSubtableSet<F>,
    {
        let mut subtables = Vec::with_capacity(Subtables::COUNT);
        for subtable in Subtables::iter() {
            subtables.push(subtable.materialize(M));
        }
        subtables
    }
}

impl<F, PCS, InstructionSet, Subtables, const C: usize, const M: usize, ProofTranscript>
    InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    const NUM_SUBTABLES: usize = Subtables::COUNT;
    const NUM_INSTRUCTIONS: usize = InstructionSet::COUNT;

    #[tracing::instrument(skip_all, name = "InstructionLookups::prove")]
    pub fn prove<'a>(
        generators: &PCS::Setup,
        polynomials: &'a JoltPolynomials<F>,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let trace_length = polynomials.instruction_lookups.dim[0].len();
        let r_eq = transcript.challenge_vector(trace_length.log_2());

        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq);
        let mut eq_poly = DensePolynomial::new(eq_evals);
        let num_rounds = trace_length.log_2();

        // TODO: compartmentalize all primary sumcheck logic
        let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                &mut eq_poly,
                &polynomials.instruction_lookups.E_polys,
                &polynomials.instruction_lookups.instruction_flags,
                &mut polynomials.instruction_lookups.lookup_outputs.clone(),
                Self::sumcheck_poly_degree(),
                transcript,
            );

        // Create a single opening proof for the flag_evals and memory_evals
        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
            lookup_outputs_opening: outputs_eval,
        };

        let primary_sumcheck_polys = polynomials
            .instruction_lookups
            .E_polys
            .iter()
            .chain(polynomials.instruction_lookups.instruction_flags.iter())
            .chain([&polynomials.instruction_lookups.lookup_outputs].into_iter())
            .collect::<Vec<_>>();

        let mut primary_sumcheck_openings: Vec<F> = [
            sumcheck_openings.E_poly_openings.as_slice(),
            sumcheck_openings.flag_openings.as_slice(),
        ]
        .concat();
        primary_sumcheck_openings.push(outputs_eval);

        opening_accumulator.append(
            &primary_sumcheck_polys,
            DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck)),
            r_primary_sumcheck.clone(),
            &primary_sumcheck_openings.iter().collect::<Vec<_>>(),
            transcript,
        );

        let primary_sumcheck = PrimarySumcheck {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking = Self::prove_memory_checking(
            generators,
            preprocessing,
            &polynomials.instruction_lookups,
            polynomials,
            opening_accumulator,
            transcript,
        );

        InstructionLookupsProof {
            _instructions: PhantomData,
            _subtables: PhantomData,
            primary_sumcheck,
            memory_checking,
        }
    }

    pub fn verify(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        pcs_setup: &PCS::Setup,
        proof: InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let r_eq = transcript.challenge_vector(proof.primary_sumcheck.num_rounds);

        // TODO: compartmentalize all primary sumcheck logic
        let (claim_last, r_primary_sumcheck) = proof.primary_sumcheck.sumcheck_proof.verify(
            F::zero(),
            proof.primary_sumcheck.num_rounds,
            Self::sumcheck_poly_degree(),
            transcript,
        )?;

        // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
        assert_eq!(
            eq_eval
                * (Self::combine_lookups(
                    preprocessing,
                    &proof.primary_sumcheck.openings.E_poly_openings,
                    &proof.primary_sumcheck.openings.flag_openings,
                ) - proof.primary_sumcheck.openings.lookup_outputs_opening),
            claim_last,
            "Primary sumcheck check failed."
        );

        let primary_sumcheck_commitments = commitments
            .instruction_lookups
            .E_polys
            .iter()
            .chain(commitments.instruction_lookups.instruction_flags.iter())
            .chain([&commitments.instruction_lookups.lookup_outputs])
            .collect::<Vec<_>>();

        let primary_sumcheck_openings = proof
            .primary_sumcheck
            .openings
            .E_poly_openings
            .iter()
            .chain(proof.primary_sumcheck.openings.flag_openings.iter())
            .chain([&proof.primary_sumcheck.openings.lookup_outputs_opening])
            .collect::<Vec<_>>();

        opening_accumulator.append(
            &primary_sumcheck_commitments,
            r_primary_sumcheck.clone(),
            &primary_sumcheck_openings,
            transcript,
        );

        Self::verify_memory_checking(
            preprocessing,
            pcs_setup,
            proof.memory_checking,
            &commitments.instruction_lookups,
            commitments,
            opening_accumulator,
            transcript,
        )?;

        Ok(())
    }

    /// Constructs the polynomials used in the primary sumcheck and memory checking.
    #[tracing::instrument(skip_all, name = "InstructionLookups::polynomialize")]
    pub fn generate_witness(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        ops: &Vec<JoltTraceStep<InstructionSet>>,
    ) -> InstructionLookupPolynomials<F> {
        let m: usize = ops.len().next_power_of_two();

        let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(ops);

        let polys: Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> = (0
            ..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![0usize; M];
                let mut read_cts_i = vec![0usize; m];
                let mut subtable_lookups = vec![F::zero(); m];

                for (j, op) in ops.iter().enumerate() {
                    if let Some(instr) = &op.instruction_lookup {
                        let memories_used = &preprocessing.instruction_to_memory_indices
                            [InstructionSet::enum_index(instr)];
                        if memories_used.contains(&memory_index) {
                            let memory_address = access_sequence[j];
                            debug_assert!(memory_address < M);

                            let counter = final_cts_i[memory_address];
                            read_cts_i[j] = counter;
                            final_cts_i[memory_address] = counter + 1;
                            subtable_lookups[j] = preprocessing.materialized_subtables
                                [subtable_index][memory_address];
                        }
                    }
                }

                (
                    DensePolynomial::from_usize(&read_cts_i),
                    DensePolynomial::from_usize(&final_cts_i),
                    DensePolynomial::new(subtable_lookups),
                )
            })
            .collect();

        // Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>)
        let (read_cts, final_cts, E_polys): (
            Vec<DensePolynomial<F>>,
            Vec<DensePolynomial<F>>,
            Vec<DensePolynomial<F>>,
        ) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut E_acc), (read, f, E)| {
                read_acc.push(read);
                final_acc.push(f);
                E_acc.push(E);
                (read_acc, final_acc, E_acc)
            },
        );

        let dim: Vec<DensePolynomial<F>> = (0..C)
            .into_par_iter()
            .map(|i| {
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];
                DensePolynomial::from_usize(access_sequence)
            })
            .collect();

        let mut instruction_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; m]; Self::NUM_INSTRUCTIONS];
        for (j, op) in ops.iter().enumerate() {
            if let Some(instr) = &op.instruction_lookup {
                instruction_flag_bitvectors[InstructionSet::enum_index(instr)][j] = 1;
            }
        }

        let instruction_flag_polys: Vec<DensePolynomial<F>> = instruction_flag_bitvectors
            .par_iter()
            .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
            .collect();

        let mut lookup_outputs = Self::compute_lookup_outputs(ops);
        lookup_outputs.resize(m, F::zero());
        let lookup_outputs = DensePolynomial::new(lookup_outputs);

        InstructionLookupPolynomials {
            dim,
            read_cts,
            final_cts,
            instruction_flags: instruction_flag_polys,
            E_polys,
            lookup_outputs,
            a_init_final: None,
            v_init_final: None,
            instruction_flag_bitvectors: Some(instruction_flag_bitvectors),
        }
    }

    /// Prove Jolt primary sumcheck including instruction collation.
    ///
    /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
    /// via the sumcheck protocol.
    /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
    ///
    /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
    ///
    /// Params:
    /// - `claim`: Claimed sumcheck evaluation.
    /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
    /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
    /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
    /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
    /// - `transcript`: Fiat-shamir transcript.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        num_rounds: usize,
        eq_poly: &mut DensePolynomial<F>,
        memory_polys: &[DensePolynomial<F>],
        flag_polys: &[DensePolynomial<F>],
        lookup_outputs_poly: &mut DensePolynomial<F>,
        degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F>,
        Vec<F>,
        Vec<F>,
        F,
    ) {
        // Check all polys are the same size
        let poly_len = eq_poly.len();
        memory_polys
            .iter()
            .for_each(|E_poly| debug_assert_eq!(E_poly.len(), poly_len));
        flag_polys
            .iter()
            .for_each(|flag_poly| debug_assert_eq!(flag_poly.len(), poly_len));
        debug_assert_eq!(lookup_outputs_poly.len(), poly_len);

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let num_eval_points = degree + 1;

        let round_uni_poly = Self::primary_sumcheck_inner_loop(
            preprocessing,
            eq_poly,
            flag_polys,
            memory_polys,
            lookup_outputs_poly,
            num_eval_points,
        );
        compressed_polys.push(round_uni_poly.compress());
        let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
        random_vars.push(r_j);

        let _bind_span = trace_span!("BindPolys");
        let _bind_enter = _bind_span.enter();
        rayon::join(
            || eq_poly.bound_poly_var_top(&r_j),
            || lookup_outputs_poly.bound_poly_var_top_many_ones(&r_j),
        );
        let mut flag_polys_updated: Vec<DensePolynomial<F>> = flag_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top_flags(&r_j))
            .collect();
        let mut memory_polys_updated: Vec<DensePolynomial<F>> = memory_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top(&r_j))
            .collect();
        drop(_bind_enter);
        drop(_bind_span);

        for _round in 1..num_rounds {
            let round_uni_poly = Self::primary_sumcheck_inner_loop(
                preprocessing,
                eq_poly,
                &flag_polys_updated,
                &memory_polys_updated,
                lookup_outputs_poly,
                num_eval_points,
            );
            compressed_polys.push(round_uni_poly.compress());
            let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
            random_vars.push(r_j);

            // Bind all polys
            let _bind_span = trace_span!("BindPolys");
            let _bind_enter = _bind_span.enter();
            rayon::join(
                || eq_poly.bound_poly_var_top(&r_j),
                || lookup_outputs_poly.bound_poly_var_top_many_ones(&r_j),
            );
            flag_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));
            memory_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));

            drop(_bind_enter);
            drop(_bind_span);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = flag_polys_updated.iter().map(|poly| poly[0]).collect();
        let memory_evals = memory_polys_updated.iter().map(|poly| poly[0]).collect();
        let outputs_eval = lookup_outputs_poly[0];

        drop_in_background_thread(flag_polys_updated);
        drop_in_background_thread(memory_polys_updated);

        (
            SumcheckInstanceProof::new(compressed_polys),
            random_vars,
            flag_evals,
            memory_evals,
            outputs_eval,
        )
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::primary_sumcheck_inner_loop")]
    fn primary_sumcheck_inner_loop(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &DensePolynomial<F>,
        flag_polys: &[DensePolynomial<F>],
        memory_polys: &[DensePolynomial<F>],
        lookup_outputs_poly: &DensePolynomial<F>,
        num_eval_points: usize,
    ) -> UniPoly<F> {
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        // Loop over half MLE size (size of MLE next round)
        //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
        //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
        // After: Sum over MLE elements (with combine)
        let evaluations: Vec<F> = (0..mle_half)
            .into_par_iter()
            .map(|low_index| {
                let high_index = mle_half + low_index;

                let mut eq_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut outputs_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut multi_flag_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); Self::NUM_INSTRUCTIONS]; num_eval_points];
                let mut multi_memory_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); preprocessing.num_memories]; num_eval_points];

                eq_evals[0] = eq_poly[low_index];
                eq_evals[1] = eq_poly[high_index];
                let eq_m = eq_poly[high_index] - eq_poly[low_index];
                for eval_index in 2..num_eval_points {
                    eq_evals[eval_index] = eq_evals[eval_index - 1] + eq_m;
                }

                outputs_evals[0] = lookup_outputs_poly[low_index];
                outputs_evals[1] = lookup_outputs_poly[high_index];
                let outputs_m = lookup_outputs_poly[high_index] - lookup_outputs_poly[low_index];
                for eval_index in 2..num_eval_points {
                    outputs_evals[eval_index] = outputs_evals[eval_index - 1] + outputs_m;
                }

                // TODO: Exactly one flag across NUM_INSTRUCTIONS is non-zero
                for flag_instruction_index in 0..Self::NUM_INSTRUCTIONS {
                    multi_flag_evals[0][flag_instruction_index] =
                        flag_polys[flag_instruction_index][low_index];
                    multi_flag_evals[1][flag_instruction_index] =
                        flag_polys[flag_instruction_index][high_index];
                    let flag_m = flag_polys[flag_instruction_index][high_index]
                        - flag_polys[flag_instruction_index][low_index];
                    for eval_index in 2..num_eval_points {
                        let flag_eval =
                            multi_flag_evals[eval_index - 1][flag_instruction_index] + flag_m;
                        multi_flag_evals[eval_index][flag_instruction_index] = flag_eval;
                    }
                }

                // TODO: Some of these intermediates need not be computed if flags is computed
                for memory_index in 0..preprocessing.num_memories {
                    multi_memory_evals[0][memory_index] = memory_polys[memory_index][low_index];

                    multi_memory_evals[1][memory_index] = memory_polys[memory_index][high_index];
                    let memory_m = memory_polys[memory_index][high_index]
                        - memory_polys[memory_index][low_index];
                    for eval_index in 2..num_eval_points {
                        multi_memory_evals[eval_index][memory_index] =
                            multi_memory_evals[eval_index - 1][memory_index] + memory_m;
                    }
                }

                // Accumulate inner terms.
                // S({0,1,... num_eval_points}) = eq * [ INNER TERMS ]
                //            = eq[000] * [ flags_0[000] * g_0(E_0)[000] + flags_1[000] * g_1(E_1)[000]]
                //            + eq[001] * [ flags_0[001] * g_0(E_0)[001] + flags_1[001] * g_1(E_1)[001]]
                //            + ...
                //            + eq[111] * [ flags_0[111] * g_0(E_0)[111] + flags_1[111] * g_1(E_1)[111]]
                let mut inner_sum = vec![F::zero(); num_eval_points];
                for instruction in InstructionSet::iter() {
                    let instruction_index = InstructionSet::enum_index(&instruction);
                    let memory_indices =
                        &preprocessing.instruction_to_memory_indices[instruction_index];

                    for eval_index in 0..num_eval_points {
                        let flag_eval = multi_flag_evals[eval_index][instruction_index];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let terms: Vec<F> = memory_indices
                            .iter()
                            .map(|memory_index| multi_memory_evals[eval_index][*memory_index])
                            .collect();
                        let instruction_collation_eval = instruction.combine_lookups(&terms, C, M);

                        // TODO(sragss): Could sum all shared inner terms before multiplying by the flag eval
                        inner_sum[eval_index] += flag_eval * instruction_collation_eval;
                    }
                }
                let evaluations: Vec<F> = (0..num_eval_points)
                    .map(|eval_index| {
                        eq_evals[eval_index] * (inner_sum[eval_index] - outputs_evals[eval_index])
                    })
                    .collect();
                evaluations
            })
            .reduce(
                || vec![F::zero(); num_eval_points],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + *n)
                        .collect()
                },
            );

        UniPoly::from_evals(&evaluations)
    }

    fn update_primary_sumcheck_transcript(
        round_uni_poly: UniPoly<F>,
        transcript: &mut ProofTranscript,
    ) -> F {
        round_uni_poly.compress().append_to_transcript(transcript);

        transcript.challenge_scalar::<F>()
    }

    /// Combines the subtable values given by `vals` and the flag values given by `flags`.
    /// This function corresponds to the "primary" sumcheck expression:
    /// \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x))
    /// where `vals` corresponds to E_1, ..., E_\alpha,
    /// and `flags` corresponds to the flag_i's
    fn combine_lookups(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        vals: &[F],
        flags: &[F],
    ) -> F {
        assert_eq!(vals.len(), preprocessing.num_memories);
        assert_eq!(flags.len(), Self::NUM_INSTRUCTIONS);

        let mut sum = F::zero();
        for instruction in InstructionSet::iter() {
            let instruction_index = InstructionSet::enum_index(&instruction);
            let memory_indices = &preprocessing.instruction_to_memory_indices[instruction_index];
            let filtered_operands: Vec<F> = memory_indices.iter().map(|i| vals[*i]).collect();
            sum += flags[instruction_index] * instruction.combine_lookups(&filtered_operands, C, M);
        }

        sum
    }

    /// Converts instruction flag values into memory flag values. A memory flag value
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    fn memory_flags(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        instruction_flags: &[F],
    ) -> Vec<F> {
        debug_assert_eq!(instruction_flags.len(), Self::NUM_INSTRUCTIONS);
        let mut memory_flags = vec![F::zero(); preprocessing.num_memories];
        for instruction_index in 0..Self::NUM_INSTRUCTIONS {
            for memory_index in &preprocessing.instruction_to_memory_indices[instruction_index] {
                memory_flags[*memory_index] += instruction_flags[instruction_index];
            }
        }
        memory_flags
    }

    /// Converts instruction flag bitvectors into a sparse representation of the corresponding memory flags.
    /// A memory flag polynomial can be computed by summing over the instructions that use that memory: if a
    /// given execution step accesses the memory, it must be executing exactly one of those instructions.
    fn memory_flag_indices(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        instruction_flag_bitvectors: &[Vec<u64>],
    ) -> Vec<Vec<usize>> {
        let m = instruction_flag_bitvectors[0].len();

        (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let instruction_indices: Vec<_> = (0..Self::NUM_INSTRUCTIONS)
                    .filter(|instruction_index| {
                        preprocessing.instruction_to_memory_indices[*instruction_index]
                            .contains(&memory_index)
                    })
                    .collect();
                let mut memory_flag_indices = vec![];
                for i in 0..m {
                    for instruction_index in instruction_indices.iter() {
                        if instruction_flag_bitvectors[*instruction_index][i] != 0 {
                            memory_flag_indices.push(i);
                            break;
                        }
                    }
                }
                memory_flag_indices
            })
            .collect()
    }

    /// Returns the sumcheck polynomial degree for the "primary" sumcheck. Since the primary sumcheck expression
    /// is \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x)), the degree is
    /// the max over all the instructions' `g_i` polynomial degrees, plus two (one for \tilde{eq}, one for flag_i)
    fn sumcheck_poly_degree() -> usize {
        InstructionSet::iter()
            .map(|instruction| instruction.g_poly_degree(C))
            .max()
            .unwrap()
            + 2 // eq and flag
    }

    /// Converts each instruction in `ops` into its corresponding subtable lookup indices.
    /// The output is `C` vectors, each of length `m`.
    fn subtable_lookup_indices(ops: &[JoltTraceStep<InstructionSet>]) -> Vec<Vec<usize>> {
        let m = ops.len().next_power_of_two();
        let log_M = M.log_2();
        let chunked_indices: Vec<Vec<usize>> = ops
            .iter()
            .map(|op| {
                if let Some(instr) = &op.instruction_lookup {
                    instr.to_indices(C, log_M)
                } else {
                    vec![0; C]
                }
            })
            .collect();

        let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(C);
        for i in 0..C {
            let mut access_sequence: Vec<usize> =
                chunked_indices.iter().map(|chunks| chunks[i]).collect();
            access_sequence.resize(m, 0);
            subtable_lookup_indices.push(access_sequence);
        }
        subtable_lookup_indices
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(max_trace_length: usize) -> Vec<CommitShape> {
        let max_trace_length = max_trace_length.next_power_of_two();

        let read_write_shape = CommitShape::new(max_trace_length, BatchType::Big);
        let init_final_shape = CommitShape::new(M, BatchType::Small);

        vec![read_write_shape, init_final_shape]
    }

    #[tracing::instrument(skip_all, name = "InstructionLookupsProof::compute_lookup_outputs")]
    fn compute_lookup_outputs(instructions: &Vec<JoltTraceStep<InstructionSet>>) -> Vec<F> {
        instructions
            .par_iter()
            .map(|op| {
                if let Some(instr) = &op.instruction_lookup {
                    F::from_u64(instr.lookup_entry()).unwrap()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    fn protocol_name() -> &'static [u8] {
        b"Jolt instruction lookups"
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::jolt::vm::rv32i_vm::{RV32ISubtables, RV32I};

    use super::*;

    #[test]
    fn instruction_lookup_stuff_ordering() {
        const C: usize = 4;
        const M: usize = 1 << 16;
        let preprocessing =
            InstructionLookupsPreprocessing::<C, Fr>::preprocess::<M, RV32I, RV32ISubtables<Fr>>();
        InstructionLookupOpenings::<Fr>::test_ordering_consistency(&preprocessing);
    }
}
