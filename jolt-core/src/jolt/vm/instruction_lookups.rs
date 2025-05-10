use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProduct};
use crate::subprotocols::sparse_grand_product::ToggledBatchedGrandProduct;
use crate::utils::thread::unsafe_allocate_zero_vec;
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
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
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

    a_init_final: VerifierComputedOpening<T>,
    v_init_final: VerifierComputedOpening<Vec<T>>,
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type InstructionLookupPolynomials<F: JoltField> =
    InstructionLookupStuff<MultilinearPolynomial<F>>;
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
    type InitFinalGrandProduct = BatchedDenseGrandProduct<F>;

    type Polynomials = InstructionLookupPolynomials<F>;
    type Openings = InstructionLookupOpenings<F>;
    type Commitments = InstructionLookupCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = NoExogenousOpenings;

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
        let gamma = *gamma;

        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves: Vec<_> = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let dim: &CompactPolynomial<u16, F> =
                    (&polynomials.dim[dim_index]).try_into().unwrap();
                let E_poly: &CompactPolynomial<u32, F> =
                    (&polynomials.E_polys[memory_index]).try_into().unwrap();
                let read_cts: &CompactPolynomial<u32, F> =
                    (&polynomials.read_cts[memory_index]).try_into().unwrap();

                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = dim[i];
                        let v = E_poly[i];
                        let t = read_cts[i];
                        t.field_mul(gamma_squared) + v.field_mul(gamma) + F::from_u16(a) - *tau
                    })
                    .collect();
                let t_adjustment = 1u64.field_mul(gamma_squared);
                let write_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| read_fingerprints[i] + t_adjustment)
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
                    let a = &F::from_u16(i as u16);
                    let v: u32 = subtable[i];
                    // let t = F::zero();
                    // Compute h(a,v,t) where t == 0
                    leaves[i] = v.field_mul(gamma) + *a - *tau;
                });
                // Final leaves
                let mut leaf_index = M;
                for memory_index in &preprocessing.subtable_to_memory_indices[subtable_index] {
                    let final_cts: &CompactPolynomial<u32, F> =
                        (&polynomials.final_cts[*memory_index]).try_into().unwrap();
                    (0..M).for_each(|i| {
                        leaves[leaf_index] = leaves[i] + final_cts[i].field_mul(gamma_squared);
                        leaf_index += 1;
                    });
                }

                leaves
            })
            .collect();

        let memory_flags = Self::memory_flag_indices(
            preprocessing,
            polynomials
                .instruction_flags
                .iter()
                .map(|poly| poly.try_into().unwrap())
                .collect(),
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionLookupsPreprocessing<const C: usize, F: JoltField> {
    subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    instruction_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<Vec<u32>>,
    num_memories: usize,
    _field: PhantomData<F>,
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
            _field: PhantomData,
        }
    }

    /// Materializes all subtables used by this Jolt instance.
    #[tracing::instrument(skip_all)]
    fn materialize_subtables<const M: usize, Subtables>() -> Vec<Vec<u32>>
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
        polynomials: &'a mut JoltPolynomials<F>,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let trace_length = polynomials.instruction_lookups.dim[0].len();
        let r_eq = transcript.challenge_vector(trace_length.log_2());

        let eq_evals: Vec<F> = EqPolynomial::evals(&r_eq);
        let eq_poly = MultilinearPolynomial::from(eq_evals);
        let num_rounds = trace_length.log_2();

        // TODO: compartmentalize all primary sumcheck logic
        let (primary_sumcheck_proof, mut r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(
                preprocessing,
                num_rounds,
                eq_poly,
                &mut polynomials.instruction_lookups.E_polys,
                &mut polynomials.instruction_lookups.instruction_flags,
                &mut polynomials.instruction_lookups.lookup_outputs.clone(),
                transcript,
            );
        r_primary_sumcheck = r_primary_sumcheck.into_iter().rev().collect();

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

        let eq_primary_sumcheck = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        opening_accumulator.append(
            &primary_sumcheck_polys,
            eq_primary_sumcheck,
            r_primary_sumcheck,
            &primary_sumcheck_openings,
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
        let (claim_last, mut r_primary_sumcheck) = proof.primary_sumcheck.sumcheck_proof.verify(
            F::zero(),
            proof.primary_sumcheck.num_rounds,
            Self::sumcheck_poly_degree(),
            transcript,
        )?;
        r_primary_sumcheck = r_primary_sumcheck.into_iter().rev().collect();

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
    #[tracing::instrument(skip_all, name = "InstructionLookupsProof::generate_witness")]
    pub fn generate_witness(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        ops: &Vec<JoltTraceStep<InstructionSet>>,
    ) -> InstructionLookupPolynomials<F> {
        let m: usize = ops.len().next_power_of_two();

        let subtable_lookup_indices: Vec<Vec<u16>> = Self::subtable_lookup_indices(ops);

        let polys: Vec<(
            MultilinearPolynomial<F>,
            MultilinearPolynomial<F>,
            MultilinearPolynomial<F>,
        )> = (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence: &Vec<u16> = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![0u32; M];
                let mut read_cts_i = vec![0u32; m];
                let mut subtable_lookups = vec![0u32; m];

                for (j, op) in ops.iter().enumerate() {
                    if let Some(instr) = &op.instruction_lookup {
                        let memories_used = &preprocessing.instruction_to_memory_indices
                            [InstructionSet::enum_index(instr)];
                        if memories_used.contains(&memory_index) {
                            let memory_address = access_sequence[j] as usize;
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
                    MultilinearPolynomial::from(read_cts_i),
                    MultilinearPolynomial::from(final_cts_i),
                    MultilinearPolynomial::from(subtable_lookups),
                )
            })
            .collect();

        // Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>, MultilinearPolynomial<F>)> ->
        //   (Vec<MultilinearPolynomial<F>>, Vec<MultilinearPolynomial<F>>, Vec<MultilinearPolynomial<F>>)
        let (read_cts, final_cts, E_polys): (
            Vec<MultilinearPolynomial<F>>,
            Vec<MultilinearPolynomial<F>>,
            Vec<MultilinearPolynomial<F>>,
        ) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut E_acc), (read, f, E)| {
                read_acc.push(read);
                final_acc.push(f);
                E_acc.push(E);
                (read_acc, final_acc, E_acc)
            },
        );

        let dim: Vec<MultilinearPolynomial<F>> = subtable_lookup_indices
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        let mut instruction_flag_bitvectors: Vec<Vec<u8>> =
            vec![vec![0; m]; Self::NUM_INSTRUCTIONS];
        for (j, op) in ops.iter().enumerate() {
            if let Some(instr) = &op.instruction_lookup {
                instruction_flag_bitvectors[InstructionSet::enum_index(instr)][j] = 1;
            }
        }

        let instruction_flag_polys: Vec<MultilinearPolynomial<F>> = instruction_flag_bitvectors
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        let mut lookup_outputs = Self::compute_lookup_outputs(ops);
        lookup_outputs.resize(m, 0);
        let lookup_outputs = MultilinearPolynomial::from(lookup_outputs);

        InstructionLookupPolynomials {
            dim,
            read_cts,
            final_cts,
            instruction_flags: instruction_flag_polys,
            E_polys,
            lookup_outputs,
            a_init_final: None,
            v_init_final: None,
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
        mut eq_poly: MultilinearPolynomial<F>,
        memory_polys: &mut [MultilinearPolynomial<F>],
        flag_polys: &mut [MultilinearPolynomial<F>],
        lookup_outputs_poly: &mut MultilinearPolynomial<F>,
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

        let mut previous_claim = F::zero();
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let univariate_poly = Self::primary_sumcheck_prover_message(
                preprocessing,
                &eq_poly,
                flag_polys,
                memory_polys,
                lookup_outputs_poly,
                previous_claim,
            );

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r.push(r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            // Bind all polys
            let _bind_span = trace_span!("bind");
            let _bind_enter = _bind_span.enter();
            flag_polys
                .par_iter_mut()
                .chain(memory_polys.par_iter_mut())
                .chain([&mut eq_poly, lookup_outputs_poly].into_par_iter())
                .for_each(|poly| poly.bind(r_j, BindingOrder::LowToHigh));
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = flag_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let memory_evals = memory_polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();
        let outputs_eval = lookup_outputs_poly.final_sumcheck_claim();

        (
            SumcheckInstanceProof::new(compressed_polys),
            r,
            flag_evals,
            memory_evals,
            outputs_eval,
        )
    }

    #[tracing::instrument(skip_all)]
    fn primary_sumcheck_prover_message(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        eq_poly: &MultilinearPolynomial<F>,
        flag_polys: &[MultilinearPolynomial<F>],
        subtable_polys: &[MultilinearPolynomial<F>],
        lookup_outputs_poly: &MultilinearPolynomial<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let degree = Self::sumcheck_poly_degree();
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let mut evaluations: Vec<F> = (0..mle_half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let output_evals =
                    lookup_outputs_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let flag_evals: Vec<Vec<F>> = flag_polys
                    .iter()
                    .map(|poly| poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh))
                    .collect();
                // Subtable evals are lazily computed in the for-loop below
                let mut subtable_evals: Vec<Vec<F>> = vec![vec![]; subtable_polys.len()];

                let mut inner_sum = vec![F::zero(); degree];
                for instruction in InstructionSet::iter() {
                    let instruction_index = InstructionSet::enum_index(&instruction);
                    let memory_indices =
                        &preprocessing.instruction_to_memory_indices[instruction_index];

                    for j in 0..degree {
                        let flag_eval = flag_evals[instruction_index][j];
                        if flag_eval.is_zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let subtable_terms: Vec<F> = memory_indices
                            .iter()
                            .map(|memory_index| {
                                if subtable_evals[*memory_index].is_empty() {
                                    subtable_evals[*memory_index] = subtable_polys[*memory_index]
                                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                                }
                                subtable_evals[*memory_index][j]
                            })
                            .collect();

                        let instruction_collation_eval =
                            instruction.combine_lookups(&subtable_terms, C, M);
                        inner_sum[j] += flag_eval * instruction_collation_eval;
                    }
                }

                let evaluations: Vec<F> = (0..degree)
                    .map(|eval_index| {
                        eq_evals[eval_index] * (inner_sum[eval_index] - output_evals[eval_index])
                    })
                    .collect();
                evaluations
            })
            .reduce(
                || vec![F::zero(); degree],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + *n)
                        .collect()
                },
            );

        evaluations.insert(1, previous_claim - evaluations[0]);
        UniPoly::from_evals(&evaluations)
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
        instruction_flag_polys: Vec<&CompactPolynomial<u8, F>>,
    ) -> Vec<Vec<usize>> {
        let m = instruction_flag_polys[0].coeffs.len();

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
                        if instruction_flag_polys[*instruction_index].coeffs[i] != 0 {
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
    fn subtable_lookup_indices(ops: &[JoltTraceStep<InstructionSet>]) -> Vec<Vec<u16>> {
        let m = ops.len().next_power_of_two();
        let log_M = M.log_2();
        let chunked_indices: Vec<Vec<u16>> = ops
            .iter()
            .map(|op| {
                if let Some(instr) = &op.instruction_lookup {
                    instr
                        .to_indices(C, log_M)
                        .iter()
                        .map(|i| *i as u16)
                        .collect()
                } else {
                    vec![0; C]
                }
            })
            .collect();

        let mut subtable_lookup_indices: Vec<Vec<u16>> = Vec::with_capacity(C);
        for i in 0..C {
            let mut access_sequence: Vec<u16> =
                chunked_indices.iter().map(|chunks| chunks[i]).collect();
            access_sequence.resize(m, 0);
            subtable_lookup_indices.push(access_sequence);
        }
        subtable_lookup_indices
    }

    #[tracing::instrument(skip_all, name = "InstructionLookupsProof::compute_lookup_outputs")]
    fn compute_lookup_outputs(instructions: &Vec<JoltTraceStep<InstructionSet>>) -> Vec<u32> {
        instructions
            .par_iter()
            .map(|op| {
                if let Some(instr) = &op.instruction_lookup {
                    instr.lookup_entry() as u32
                } else {
                    0
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
