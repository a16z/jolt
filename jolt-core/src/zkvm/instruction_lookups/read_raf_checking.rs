use std::iter::zip;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::XLEN;
use num_traits::Zero;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::Cycle;

use super::LOG_K;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::{eval_linear_prod_accumulate, finish_mles_product_sum_from_evals},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        config::{self, OneHotParams},
        instruction::{Flags, InstructionLookup, InterleavedBitsMarker, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            LookupTables,
        },
        witness::VirtualPolynomial,
    },
};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

// Instruction lookups: Read + RAF batched sumcheck
//
// Notation:
// - Field F. Let K = 2^{LOG_K}, T = 2^{log_T}.
// - Address index k ∈ {0..K-1}, cycle index j ∈ {0..T-1}.
// - eq(k; r_addr) := multilinear equality polynomial over LOG_K vars.
// - eq(j; r_reduction) := equality polynomials over LOG_T vars.
// - ra(k, j) is the selector arising from prefix/suffix condensation.
//   It is decomposed as the product of virtual sub selectors:
//   ra((k_0, k_1, ..., k_{n-1}), j) := ra_0(k_0, j) * ra_1(k_1, j) * ... * ra_{n-1}(k_{n-1}, j).
//   n is typically 1, 2, 4 or 8.
//   logically ra(k, j) = 1 when the j-th cycle's lookup key equals k, and 0 otherwise.// - Val_j(k) ∈ F is the lookup-table value selected by (j, k); concretely Val_j(k) = table_j(k)
//   if cycle j uses a table and 0 otherwise (materialized via prefix/suffix decomposition).
// - raf_flag(j) ∈ {0,1} is 1 iff the instruction at cycle j is NOT interleaved operands.
// - Let LeftPrefix_j, RightPrefix_j, IdentityPrefix_j ∈ F be the address-only (prefix) factors for
//   the left/right operand and identity polynomials at cycle j (from `PrefixSuffixDecomposition`).
//
// We introduce a batching challenge γ ∈ F. Define
//   RafVal_j(k) := (1 - raf_flag(j)) · (LeftPrefix_j + γ · RightPrefix_j)
//                  + raf_flag(j) · γ · IdentityPrefix_j.
// The overall γ-weights are arranged so that γ multiplies RafVal_j(k) in the final identity.
//
// Claims supplied by the accumulator (LHS), all claimed at `SumcheckId::InstructionClaimReduction`
// and `SumcheckId::ProductVirtualization`:
// - rv         := ⟦LookupOutput⟧
// - left_op    := ⟦LeftLookupOperand⟧
// - right_op   := ⟦RightLookupOperand⟧
//   Combined as: rv + γ·left_op + γ^2·right_op
//
// Statement proved by this sumcheck (RHS), for random challenges
// r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T}:
//
//   rv(r_reduction) + γ·left_op(r_reduction) + γ^2·right_op(r_reduction)
//   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ].
//
// Prover structure:
// - First log(K) rounds bind address vars using prefix/suffix decomposition, accumulating:
//   Σ_k ra(k, j)·Val_j(k)  and  Σ_k ra(k, j)·RafVal_j(k)
//   for each j (via u_evals vectors and suffix polynomials).
// - Last log(T) rounds bind cycle vars producing a degree-3 univariate with the required previous-round claim.
// - The published univariate matches the RHS above; the verifier checks it against the LHS claims.

pub struct ReadRafSumcheckParams<F: JoltField> {
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of cycle variables (last rounds bind cycles).
    pub log_T: usize,
    /// How many address variables each virtual ra polynomial has.
    pub ra_virtual_log_k_chunk: usize,
    /// Number of phases for instruction lookups.
    pub phases: usize,
    pub r_reduction: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ReadRafSumcheckParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let phases = config::instruction_sumcheck_phases(n_cycle_vars);
        let (r_reduction, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );

        Self {
            gamma,
            gamma_sqr,
            log_T: n_cycle_vars,
            ra_virtual_log_k_chunk: one_hot_params.lookups_ra_virtual_log_k_chunk,
            phases,
            r_reduction,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ReadRafSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, rv_claim_branch) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rv_claim, rv_claim_branch);
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
    }

    fn degree(&self) -> usize {
        let n_virtual_ra_polys = LOG_K / self.ra_virtual_log_k_chunk;
        n_virtual_ra_polys + 2
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_cycle_prime) = challenges.split_at(LOG_K);
        let r_cycle_prime = r_cycle_prime.iter().copied().rev().collect::<Vec<_>>();

        OpeningPoint::new([r_address_prime.to_vec(), r_cycle_prime].concat())
    }
}

/// Sumcheck prover for [`ReadRafSumcheckVerifier`].
///
/// Binds address variables first using prefix/suffix decomposition to aggregate, per cycle j,
///   Σ_k ra(k, j)·Val_j(k) and Σ_k ra(k, j)·RafVal_j(k),
#[derive(Allocative)]
pub struct ReadRafSumcheckProver<F: JoltField> {
    /// Materialized `ra_i(k_i, j)` polynomials.
    /// Present only in the last log(T) rounds.
    ra_polys: Option<Vec<MultilinearPolynomial<F>>>,
    /// Running list of sumcheck challenges r_j (address then cycle) in binding order.
    r: Vec<F::Challenge>,

    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Indices of cycles grouped by selected lookup table; used to form per-table flags.
    lookup_indices_by_table: Vec<Vec<usize>>,
    /// Cycle indices with interleaved operands (used for left/right operand prefix-suffix Q).
    lookup_indices_uninterleave: Vec<usize>,
    /// Cycle indices with identity path (non-interleaved) used as the RAF flag source.
    lookup_indices_identity: Vec<usize>,
    /// Per-cycle flag: instruction uses interleaved operands.
    is_interleaved_operands: Vec<bool>,
    #[allocative(skip)]
    /// Per-cycle optional lookup table chosen by the instruction; None if no lookup.
    lookup_tables: Vec<Option<LookupTables<XLEN>>>,

    /// Prefix checkpoints for each registered `Prefix` variant, updated every two rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    /// For each lookup table, dense polynomials holding suffix contributions in the current phase.
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// u_evals for read-checking and RAF: eq(r_reduction,j).
    u_evals: Vec<F>,

    /// Gruen-split equality polynomial over cycle vars.
    eq_r_reduction: GruenSplitEqPolynomial<F>,

    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    /// Materialized Val_j(k) over (address, cycle) after phase transitions.
    combined_val_polynomial: Option<MultilinearPolynomial<F>>,
    /// Materialized RafVal_j(k) (with γ-weights folded into prefixes) over (address, cycle).
    combined_raf_val_polynomial: Option<MultilinearPolynomial<F>>,

    #[allocative(skip)]
    params: ReadRafSumcheckParams<F>,
}

impl<F: JoltField> ReadRafSumcheckProver<F> {
    /// Creates a prover-side instance for the Read+RAF batched sumcheck.
    ///
    /// Builds prover-side working state:
    /// - Precomputes per-cycle lookup index, interleaving flags, and table choices
    /// - Buckets cycles by table and by path (interleaved vs identity)
    /// - Allocates per-table suffix accumulators and u-evals for rv/raf parts
    /// - Instantiates the three RAF decompositions and Gruen EQs over cycles
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::initialize")]
    pub fn initialize(params: ReadRafSumcheckParams<F>, trace: &[Cycle]) -> Self {
        let log_T = trace.len().log_2();

        let log_m = LOG_K / params.phases;
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), log_m, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), log_m, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, LOG_K);
        drop(_guard);
        drop(span);

        let num_tables = LookupTables::<XLEN>::COUNT;

        let span = tracing::span!(tracing::Level::INFO, "Build cycle_data");
        let _guard = span.enter();
        struct CycleData<const XLEN: usize> {
            idx: usize,
            lookup_index: LookupBits,
            is_interleaved: bool,
            table: Option<LookupTables<XLEN>>,
        }

        let cycle_data: Vec<CycleData<XLEN>> = trace
            .par_iter()
            .enumerate()
            .map(|(idx, cycle)| {
                let bits = LookupBits::new(LookupQuery::<XLEN>::to_lookup_index(cycle), LOG_K);
                let is_interleaved = cycle
                    .instruction()
                    .circuit_flags()
                    .is_interleaved_operands();
                let table = cycle.lookup_table();

                CycleData {
                    idx,
                    lookup_index: bits,
                    is_interleaved,
                    table,
                }
            })
            .collect();
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Extract vectors");
        let _guard = span.enter();
        // Extract all vectors in parallel using par_extend
        let mut lookup_indices = Vec::with_capacity(cycle_data.len());
        let mut is_interleaved_operands = Vec::with_capacity(cycle_data.len());
        let mut lookup_tables = Vec::with_capacity(cycle_data.len());

        {
            let span = tracing::span!(tracing::Level::INFO, "par_extend basic vectors");
            let _guard = span.enter();
            lookup_indices.par_extend(cycle_data.par_iter().map(|data| data.lookup_index));
            is_interleaved_operands
                .par_extend(cycle_data.par_iter().map(|data| data.is_interleaved));
            lookup_tables.par_extend(cycle_data.par_iter().map(|data| data.table));
        }

        // Collect interleaved and identity indices
        let (lookup_indices_uninterleave, lookup_indices_identity): (Vec<_>, Vec<_>) = {
            let span = tracing::span!(tracing::Level::INFO, "partition_map interleaved/identity");
            let _guard = span.enter();
            cycle_data.par_iter().partition_map(|data| {
                if data.is_interleaved {
                    rayon::iter::Either::Left(data.idx)
                } else {
                    rayon::iter::Either::Right(data.idx)
                }
            })
        };

        // Build lookup_indices_by_table fully in parallel
        // Create a vector for each table in parallel
        let lookup_indices_by_table: Vec<Vec<usize>> = (0..num_tables)
            .into_par_iter()
            .map(|t_idx| {
                // Each table gets its own parallel collection
                cycle_data
                    .par_iter()
                    .filter_map(|data| {
                        data.table.and_then(|t| {
                            if LookupTables::<XLEN>::enum_index(&t) == t_idx {
                                Some(data.idx)
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        drop_in_background_thread(cycle_data);
        drop(_guard);
        drop(span);

        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::default()) // Will be properly initialized in `init_phase`
                    .collect()
            })
            .collect();

        // Build split-eq polynomials and u_evals.
        let span = tracing::span!(tracing::Level::INFO, "Compute u_evals");
        let _guard = span.enter();
        let eq_poly_r_reduction =
            GruenSplitEqPolynomial::<F>::new(&params.r_reduction.r, BindingOrder::LowToHigh);
        let u_evals = EqPolynomial::evals(&params.r_reduction.r);
        drop(_guard);
        drop(span);

        let mut res = Self {
            r: Vec::with_capacity(log_T + LOG_K),
            lookup_tables,
            lookup_indices,

            // Prefix-suffix state (first log(K) rounds)
            lookup_indices_by_table,
            lookup_indices_uninterleave,
            lookup_indices_identity,
            is_interleaved_operands,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            suffix_polys,
            v: (0..params.phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            u_evals,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

            // State for last log(T) rounds
            ra_polys: None,
            eq_r_reduction: eq_poly_r_reduction,
            prefix_registry: PrefixRegistry::new(),
            combined_val_polynomial: None,
            combined_raf_val_polynomial: None,
            params,
        };
        res.init_phase(0);
        res
    }

    /// To be called in the beginning of each phase, before any binding
    /// Phase initialization for address-binding:
    /// - Condenses prior-phase u-evals through the expanding-table v[phase-1]
    /// - Builds Q for RAF (Left/Right dual and Identity) from cycle buckets
    /// - Refreshes per-table read-checking suffix polynomials for this phase
    /// - Initializes/caches P via the shared `PrefixRegistry`
    /// - Resets the current expanding table accumulator for this phase
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_phase")]
    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .zip(&mut self.u_evals)
                .for_each(|(k, u_eval)| {
                    let (prefix, _) = k.split((self.params.phases - phase) * log_m);
                    let k_bound = prefix & m_mask;
                    *u_eval *= self.v[phase - 1][k_bound];
                });
        }

        rayon::scope(|s| {
            // Single pass over lookup_indices_uninterleave for both operands
            s.spawn(|_| {
                PrefixSuffixDecomposition::init_Q_dual(
                    &mut self.left_operand_ps,
                    &mut self.right_operand_ps,
                    &self.u_evals,
                    &self.lookup_indices_uninterleave,
                    &self.lookup_indices,
                )
            });
            s.spawn(|_| {
                self.identity_ps.init_Q(
                    &self.u_evals,
                    &self.lookup_indices_identity,
                    &self.lookup_indices,
                )
            });
        });

        self.init_suffix_polys(phase);

        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v[phase].reset(F::one());
    }

    /// Recomputes per-table suffix accumulators used by read-checking for the
    /// current phase. For each table's suffix family, bucket cycles by the
    /// current chunk value and aggregate weighted contributions into Dense MLEs
    /// of size M = 2^{log_m}.
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (self.lookup_indices.len() / num_chunks).max(1);

        let new_suffix_polys: Vec<_> = {
            LookupTables::<XLEN>::iter()
                .collect::<Vec<_>>()
                .par_iter()
                .zip(self.lookup_indices_by_table.par_iter())
                .map(|(table, lookup_indices)| {
                    let suffixes = table.suffixes();
                    let unreduced_polys = lookup_indices
                        .par_chunks(chunk_size)
                        .map(|chunk| {
                            let mut chunk_result: Vec<Vec<F::Unreduced<6>>> =
                                vec![unsafe_allocate_zero_vec(m); suffixes.len()];

                            for j in chunk {
                                let k = self.lookup_indices[*j];
                                let (prefix_bits, suffix_bits) =
                                    k.split((self.params.phases - 1 - phase) * log_m);
                                for (suffix, result) in suffixes.iter().zip(chunk_result.iter_mut())
                                {
                                    let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                                    if t != 0 {
                                        let u = self.u_evals[*j];
                                        result[prefix_bits & m_mask] += u.mul_u64_unreduced(t);
                                    }
                                }
                            }

                            chunk_result
                        })
                        .reduce(
                            || vec![unsafe_allocate_zero_vec(m); suffixes.len()],
                            |mut acc, new| {
                                for (acc_i, new_i) in acc.iter_mut().zip(new.iter()) {
                                    for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter())
                                    {
                                        *acc_coeff += new_coeff;
                                    }
                                }
                                acc
                            },
                        );

                    // Reduce the unreduced values to field elements
                    unreduced_polys
                        .into_iter()
                        .map(|unreduced_coeffs| {
                            unreduced_coeffs
                                .into_iter()
                                .map(F::from_barrett_reduce)
                                .collect::<Vec<F>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(old, new)| {
                old.iter_mut()
                    .zip(new.into_iter())
                    .for_each(|(poly, mut coeffs)| {
                        *poly = DensePolynomial::new(std::mem::take(&mut coeffs));
                    });
            });
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes all virtual ra_i(k_i,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_sqr: F) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Drop stuff that's no longer needed
        drop_in_background_thread((
            std::mem::take(&mut self.u_evals),
            std::mem::take(&mut self.lookup_indices_uninterleave),
        ));

        let ra_polys: Vec<MultilinearPolynomial<F>> = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomials");
            let _guard = span.enter();
            assert!(self.v.len().is_power_of_two());
            let n = LOG_K / self.params.ra_virtual_log_k_chunk;
            let chunk_size = self.v.len() / n;
            self.v
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_i, v_chunk)| {
                    let phase_offset = chunk_i * chunk_size;
                    let res = self
                        .lookup_indices
                        .par_iter()
                        .with_min_len(1024)
                        .map(|i| {
                            let mut acc = F::one();

                            for (phase, table) in zip(phase_offset.., v_chunk) {
                                let v: u128 = i.into();
                                let i_segment = ((v >> ((self.params.phases - 1 - phase) * log_m))
                                    as usize)
                                    & m_mask;
                                acc *= table[i_segment];
                            }

                            acc
                        })
                        .collect::<Vec<F>>();
                    res.into()
                })
                .collect()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));

        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.lookup_indices.len());
        {
            let span = tracing::span!(tracing::Level::INFO, "Materialize combined_val_poly");
            let _guard = span.enter();
            combined_val_poly
                .par_iter_mut()
                .zip(std::mem::take(&mut self.lookup_tables))
                .for_each(|(val, table)| {
                    if let Some(table) = table {
                        let suffixes: Vec<_> = table
                            .suffixes()
                            .iter()
                            .map(|suffix| {
                                F::from_u64(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0)))
                            })
                            .collect();
                        *val += table.combine(&prefixes, &suffixes);
                    }
                });
        }

        let mut combined_raf_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.lookup_indices.len());
        {
            let span = tracing::span!(tracing::Level::INFO, "Materialize combined_raf_val_poly");
            let _guard = span.enter();
            combined_raf_val_poly
                .par_iter_mut()
                .zip(std::mem::take(&mut self.is_interleaved_operands))
                .for_each(|(val, is_interleaved_operands)| {
                    if is_interleaved_operands {
                        *val += gamma
                            * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                            + gamma_sqr
                                * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
                    } else {
                        *val +=
                            gamma_sqr * self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
                    }
                });
        }

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
        self.combined_raf_val_polynomial = Some(MultilinearPolynomial::from(combined_raf_val_poly));
        self.ra_polys = Some(ra_polys);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReadRafSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            let ra_polys = self.ra_polys.as_ref().unwrap();
            let val = self.combined_val_polynomial.as_ref().unwrap();
            let raf_val = self.combined_raf_val_polynomial.as_ref().unwrap();
            let n_evals = ra_polys.len() + 1;

            let mut sum_evals = self
                .eq_r_reduction
                .E_out_current()
                .par_iter()
                .enumerate()
                .map(|(j_out, e_out)| {
                    // Each pair is a linear polynomial.
                    let mut pairs = vec![(F::zero(), F::zero()); n_evals];
                    let mut evals_acc = vec![F::Unreduced::<9>::zero(); n_evals];

                    for (j_in, e_in) in self.eq_r_reduction.E_in_current().iter().enumerate() {
                        let j = self.eq_r_reduction.group_index(j_out, j_in);

                        let Some((val_pair, ra_pairs)) = pairs.split_first_mut() else {
                            unreachable!()
                        };

                        let val_at_j_0 = val.get_bound_coeff(2 * j);
                        let val_at_j_1 = val.get_bound_coeff(2 * j + 1);
                        let raf_val_at_j_0 = raf_val.get_bound_coeff(2 * j);
                        let raf_val_at_j_1 = raf_val.get_bound_coeff(2 * j + 1);
                        // v = val + raf_val
                        let v_at_0 = val_at_j_0 + raf_val_at_j_0;
                        let v_at_1 = val_at_j_1 + raf_val_at_j_1;
                        // Load linear poly: eq * (val + raf_val).
                        *val_pair = (*e_in * v_at_0, *e_in * v_at_1);
                        // Load ra polys.
                        zip(ra_pairs, ra_polys).for_each(|(pair, ra_poly)| {
                            let eval_at_0 = ra_poly.get_bound_coeff(2 * j);
                            let eval_at_1 = ra_poly.get_bound_coeff(2 * j + 1);
                            *pair = (eval_at_0, eval_at_1);
                        });

                        // TODO: Use unreduced arithmetic in eval_linear_prod_assign.
                        eval_linear_prod_accumulate(&pairs, &mut evals_acc);
                    }

                    evals_acc
                        .into_iter()
                        .map(|v| F::from_montgomery_reduce(v) * e_out)
                        .collect::<Vec<F>>()
                })
                .reduce(
                    || vec![F::zero(); n_evals],
                    |a, b| zip(a, b).map(|(a, b)| a + b).collect(),
                );

            let current_scalar = self.eq_r_reduction.get_current_scalar();
            sum_evals.iter_mut().for_each(|v| *v *= current_scalar);
            finish_mles_product_sum_from_evals(&sum_evals, previous_claim, &self.eq_r_reduction)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::ingest_challenge")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let log_m = LOG_K / self.params.phases;
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / log_m;
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys.par_iter_mut().for_each(|polys| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                    });
                });
                s.spawn(|_| self.identity_ps.bind(r_j));
                s.spawn(|_| self.right_operand_ps.bind(r_j));
                s.spawn(|_| self.left_operand_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });
            {
                if self.r.len().is_multiple_of(2) {
                    // Calculate suffix_len based on phases, using the same formula as original current_suffix_len
                    let suffix_len = LOG_K - (round / log_m + 1) * log_m;
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut self.prefix_checkpoints,
                        self.r[self.r.len() - 2],
                        self.r[self.r.len() - 1],
                        round,
                        suffix_len,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                if phase != self.params.phases - 1 {
                    // if not last phase, init next phase
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                self.init_log_t_rounds(self.params.gamma, self.params.gamma_sqr);
            }
        } else {
            // log(T) rounds

            self.eq_r_reduction.bind(r_j);
            [
                self.combined_val_polynomial.as_mut().unwrap(),
                self.combined_raf_val_polynomial.as_mut().unwrap(),
            ]
            .iter_mut()
            .for_each(|poly| {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            });

            self.ra_polys
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Prover publishes new virtual openings derived by this sumcheck:
        // - Per-table LookupTableFlag(i) at r_cycle
        // - InstructionRa at r_sumcheck (ra MLE's final claim)
        // - InstructionRafFlag at r_cycle
        let (r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);
        let eq_r_cycle_prime = EqPolynomial::<F>::evals(&r_cycle.r);

        let flag_claims = self
            .lookup_indices_by_table
            .par_iter()
            .map(|table_lookups| {
                table_lookups
                    .par_iter()
                    .map(|j| eq_r_cycle_prime[*j])
                    .sum::<F>()
            })
            .collect::<Vec<F>>();
        flag_claims.into_iter().enumerate().for_each(|(i, claim)| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claim,
            );
        });

        let ra_polys = self.ra_polys.as_ref().unwrap();
        let mut r_address_chunks = r_address.r.chunks(LOG_K / ra_polys.len());
        for (i, ra_poly) in self.ra_polys.as_ref().unwrap().iter().enumerate() {
            let r_address = r_address_chunks.next().unwrap();
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
                ra_poly.final_sumcheck_claim(),
            );
        }
        let raf_flag_claim = self
            .lookup_indices_identity
            .par_iter()
            .map(|j| eq_r_cycle_prime[*j])
            .sum::<F>();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
            raf_flag_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> ReadRafSumcheckProver<F> {
    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || {
                read_checking = self.prover_msg_read_checking(round);
            },
            || {
                raf = self.prover_msg_raf();
            },
        );

        let eval_at_0 = read_checking[0] + raf[0];
        let eval_at_2 = read_checking[1] + raf[1];

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    /// RAF part for address rounds.
    ///
    /// Builds two evaluations at X∈{0,2} for the batched
    /// (Left + γ·Right) vs Identity path, folding γ-weights into the result.
    fn prover_msg_raf(&self) -> [F; 2] {
        let len = self.identity_ps.Q_len();
        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (i0, i2) = self.identity_ps.sumcheck_evals(b);
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *(i0 + r0).as_unreduced_ref(),
                    *(i2 + r2).as_unreduced_ref(),
                ]
            })
            .fold_with([F::Unreduced::<5>::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 4],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                    ]
                },
            );
        [
            F::from_montgomery_reduce(
                left_0.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_0.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
            F::from_montgomery_reduce(
                left_2.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_2.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
        ]
    }

    /// Read-checking part for address rounds.
    ///
    /// For each lookup table, evaluates Σ P(0)·Q^L, Σ P(2)·Q^L, Σ P(2)·Q^R via
    /// table-specific suffix families, then returns [g(0), g(2)] by the standard
    /// quadratic interpolation trick.
    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let lookup_tables: Vec<_> = LookupTables::<XLEN>::iter().collect();

        let len = self.suffix_polys[0][0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };

        let [eval_0, eval_2_left, eval_2_right] = (0..len / 2)
            .into_par_iter()
            .flat_map_iter(|b| {
                let b = LookupBits::new(b as u128, log_len - 1);
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect();
                lookup_tables
                    .iter()
                    .zip(self.suffix_polys.iter())
                    .map(move |(table, suffixes)| {
                        let suffixes_left: Vec<_> =
                            suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                        let suffixes_right: Vec<_> = suffixes
                            .iter()
                            .map(|suffix| suffix[usize::from(b) + len / 2])
                            .collect();
                        [
                            table.combine(&prefixes_c0, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_right),
                        ]
                    })
            })
            .fold_with([F::Unreduced::<5>::zero(); 3], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                    running[2] + new[2].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .map(F::from_barrett_reduce);
        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }
}

/// Instruction lookups: batched Read + RAF sumcheck.
///
/// Let K = 2^{LOG_K}, T = 2^{log_T}. For random r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T},
/// this sumcheck proves that the accumulator claims
///   rv + γ·left_op + γ^2·right_op
/// equal the double sum over (j, k):
///   Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ·RafVal_j(k)) ].
/// It is implemented as: first log(K) address-binding rounds (prefix/suffix condensation), then
/// last log(T) cycle-binding rounds driven by [`GruenSplitEqPolynomial`].
pub struct ReadRafSumcheckVerifier<F: JoltField> {
    params: ReadRafSumcheckParams<F>,
}

impl<F: JoltField> ReadRafSumcheckVerifier<F> {
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ReadRafSumcheckParams::new(
            n_cycle_vars,
            one_hot_params,
            opening_accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReadRafSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Verifier's RHS reconstruction from virtual claims at r:
        //
        // Computes Val and RafVal contributions at r_address, forms EQ(r_cycle)
        // for InstructionClaimReduction sumcheck, multiplies by ra claim at r_sumcheck,
        // and returns the batched identity RHS to be matched against the LHS input claim.
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(&r_address_prime.r))
            .collect();

        let r_reduction = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
            )
            .0
            .r;
        let eq_eval_r_reduction = EqPolynomial::<F>::mle(&r_reduction, &r_cycle_prime.r);

        let n_virtual_ra_polys = LOG_K / self.params.ra_virtual_log_k_chunk;
        let ra_claim = (0..n_virtual_ra_polys)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::InstructionRa(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .product::<F>();

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();

        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let val_claim = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(claim, val)| claim * val)
            .sum::<F>();

        let raf_claim = (F::one() - raf_flag_claim)
            * (left_operand_eval + self.params.gamma * right_operand_eval)
            + raf_flag_claim * self.params.gamma * identity_poly_eval;

        eq_eval_r_reduction * ra_claim * (val_claim + self.params.gamma * raf_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Verifier requests the virtual openings that the prover must provide
        // for this sumcheck (same set as published by the prover-side cache).
        let (r_address, r_cycle) = r_sumcheck.split_at(LOG_K);

        (0..LookupTables::<XLEN>::COUNT).for_each(|i| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        });

        for (i, r_address_chunk) in r_address
            .r
            .chunks(self.params.ra_virtual_log_k_chunk)
            .enumerate()
        {
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address_chunk, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_std::Zero;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn random_instruction(rng: &mut StdRng, instruction: &Option<Cycle>) -> Cycle {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % Cycle::COUNT;
            Cycle::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });

        match instruction {
            Cycle::ADD(cycle) => cycle.random(rng).into(),
            Cycle::ADDI(cycle) => cycle.random(rng).into(),
            Cycle::AND(cycle) => cycle.random(rng).into(),
            Cycle::ANDN(cycle) => cycle.random(rng).into(),
            Cycle::ANDI(cycle) => cycle.random(rng).into(),
            Cycle::AUIPC(cycle) => cycle.random(rng).into(),
            Cycle::BEQ(cycle) => cycle.random(rng).into(),
            Cycle::BGE(cycle) => cycle.random(rng).into(),
            Cycle::BGEU(cycle) => cycle.random(rng).into(),
            Cycle::BLT(cycle) => cycle.random(rng).into(),
            Cycle::BLTU(cycle) => cycle.random(rng).into(),
            Cycle::BNE(cycle) => cycle.random(rng).into(),
            Cycle::FENCE(cycle) => cycle.random(rng).into(),
            Cycle::JAL(cycle) => cycle.random(rng).into(),
            Cycle::JALR(cycle) => cycle.random(rng).into(),
            Cycle::LUI(cycle) => cycle.random(rng).into(),
            Cycle::LD(cycle) => cycle.random(rng).into(),
            Cycle::MUL(cycle) => cycle.random(rng).into(),
            Cycle::MULHU(cycle) => cycle.random(rng).into(),
            Cycle::OR(cycle) => cycle.random(rng).into(),
            Cycle::ORI(cycle) => cycle.random(rng).into(),
            Cycle::SLT(cycle) => cycle.random(rng).into(),
            Cycle::SLTI(cycle) => cycle.random(rng).into(),
            Cycle::SLTIU(cycle) => cycle.random(rng).into(),
            Cycle::SLTU(cycle) => cycle.random(rng).into(),
            Cycle::SUB(cycle) => cycle.random(rng).into(),
            Cycle::SD(cycle) => cycle.random(rng).into(),
            Cycle::XOR(cycle) => cycle.random(rng).into(),
            Cycle::XORI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAdvice(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertEQ(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertHalfwordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertWordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertLTE(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidDiv0(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidUnsignedRemainder(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMovsign(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMULI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2I(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2IW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmask(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmaskI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRA(cycle) => cycle.random(rng).into(),
            Cycle::VirtualRev8W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRAI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRL(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRLI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualZeroExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSignExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRIW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisor(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisorW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertMulUNoOverflow(cycle) => cycle.random(rng).into(),
            _ => Cycle::NoOp,
        }
    }

    fn test_read_raf_sumcheck(instruction: Option<Cycle>) {
        let mut rng = StdRng::seed_from_u64(12345);

        let trace: Vec<_> = (0..T)
            .map(|_| random_instruction(&mut rng, &instruction))
            .collect();

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator = VerifierOpeningAccumulator::new(trace.len().log_2());

        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(LOG_T);
        let _r_cycle: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(LOG_T);
        let eq_r_cycle = EqPolynomial::<Fr>::evals(&r_cycle);

        let mut rv_claim = Fr::zero();
        let mut left_operand_claim = Fr::zero();
        let mut right_operand_claim = Fr::zero();

        for (i, cycle) in trace.iter().enumerate() {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            let table: Option<LookupTables<XLEN>> = cycle.lookup_table();
            if let Some(table) = table {
                rv_claim +=
                    JoltField::mul_u64(&eq_r_cycle[i], table.materialize_entry(lookup_index));
            }

            // Compute left and right operand claims
            let (lo, ro) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
            left_operand_claim += JoltField::mul_u64(&eq_r_cycle[i], lo);
            right_operand_claim += JoltField::mul_u128(&eq_r_cycle[i], ro);
        }

        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            left_operand_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            right_operand_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );

        let one_hot_params = OneHotParams::new(trace.len().log_2(), 100, 100);

        let params = ReadRafSumcheckParams::new(
            trace.len().log_2(),
            &one_hot_params,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let mut prover_sumcheck = ReadRafSumcheckProver::initialize(params, &trace);

        let (proof, r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut prover_sumcheck],
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
            OpeningPoint::new(r_cycle.clone()),
        );

        let mut verifier_sumcheck = ReadRafSumcheckVerifier::new(
            trace.len().log_2(),
            &one_hot_params,
            &verifier_opening_accumulator,
            verifier_transcript,
        );

        let r_sumcheck_verif = BatchedSumcheck::verify(
            &proof,
            vec![&mut verifier_sumcheck],
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }

    #[test]
    fn test_random_instructions() {
        test_read_raf_sumcheck(None);
    }

    #[test]
    fn test_add() {
        test_read_raf_sumcheck(Some(Cycle::ADD(Default::default())));
    }

    #[test]
    fn test_addi() {
        test_read_raf_sumcheck(Some(Cycle::ADDI(Default::default())));
    }

    #[test]
    fn test_and() {
        test_read_raf_sumcheck(Some(Cycle::AND(Default::default())));
    }

    #[test]
    fn test_andn() {
        test_read_raf_sumcheck(Some(Cycle::ANDN(Default::default())));
    }

    #[test]
    fn test_andi() {
        test_read_raf_sumcheck(Some(Cycle::ANDI(Default::default())));
    }

    #[test]
    fn test_auipc() {
        test_read_raf_sumcheck(Some(Cycle::AUIPC(Default::default())));
    }

    #[test]
    fn test_beq() {
        test_read_raf_sumcheck(Some(Cycle::BEQ(Default::default())));
    }

    #[test]
    fn test_bge() {
        test_read_raf_sumcheck(Some(Cycle::BGE(Default::default())));
    }

    #[test]
    fn test_bgeu() {
        test_read_raf_sumcheck(Some(Cycle::BGEU(Default::default())));
    }

    #[test]
    fn test_blt() {
        test_read_raf_sumcheck(Some(Cycle::BLT(Default::default())));
    }

    #[test]
    fn test_bltu() {
        test_read_raf_sumcheck(Some(Cycle::BLTU(Default::default())));
    }

    #[test]
    fn test_bne() {
        test_read_raf_sumcheck(Some(Cycle::BNE(Default::default())));
    }

    #[test]
    fn test_fence() {
        test_read_raf_sumcheck(Some(Cycle::FENCE(Default::default())));
    }

    #[test]
    fn test_jal() {
        test_read_raf_sumcheck(Some(Cycle::JAL(Default::default())));
    }

    #[test]
    fn test_jalr() {
        test_read_raf_sumcheck(Some(Cycle::JALR(Default::default())));
    }

    #[test]
    fn test_lui() {
        test_read_raf_sumcheck(Some(Cycle::LUI(Default::default())));
    }

    #[test]
    fn test_ld() {
        test_read_raf_sumcheck(Some(Cycle::LD(Default::default())));
    }

    #[test]
    fn test_mul() {
        test_read_raf_sumcheck(Some(Cycle::MUL(Default::default())));
    }

    #[test]
    fn test_mulhu() {
        test_read_raf_sumcheck(Some(Cycle::MULHU(Default::default())));
    }

    #[test]
    fn test_or() {
        test_read_raf_sumcheck(Some(Cycle::OR(Default::default())));
    }

    #[test]
    fn test_ori() {
        test_read_raf_sumcheck(Some(Cycle::ORI(Default::default())));
    }

    #[test]
    fn test_slt() {
        test_read_raf_sumcheck(Some(Cycle::SLT(Default::default())));
    }

    #[test]
    fn test_slti() {
        test_read_raf_sumcheck(Some(Cycle::SLTI(Default::default())));
    }

    #[test]
    fn test_sltiu() {
        test_read_raf_sumcheck(Some(Cycle::SLTIU(Default::default())));
    }

    #[test]
    fn test_sltu() {
        test_read_raf_sumcheck(Some(Cycle::SLTU(Default::default())));
    }

    #[test]
    fn test_sub() {
        test_read_raf_sumcheck(Some(Cycle::SUB(Default::default())));
    }

    #[test]
    fn test_sd() {
        test_read_raf_sumcheck(Some(Cycle::SD(Default::default())));
    }

    #[test]
    fn test_xor() {
        test_read_raf_sumcheck(Some(Cycle::XOR(Default::default())));
    }

    #[test]
    fn test_xori() {
        test_read_raf_sumcheck(Some(Cycle::XORI(Default::default())));
    }

    #[test]
    fn test_advice() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAdvice(Default::default())));
    }

    #[test]
    fn test_asserteq() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertEQ(Default::default())));
    }

    #[test]
    fn test_asserthalfwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertHalfwordAlignment(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertWordAlignment(Default::default())));
    }

    #[test]
    fn test_assertlte() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertLTE(Default::default())));
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidDiv0(Default::default())));
    }

    #[test]
    fn test_assertvalidunsignedremainder() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidUnsignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_movsign() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMovsign(Default::default())));
    }

    #[test]
    fn test_muli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMULI(Default::default())));
    }

    #[test]
    fn test_pow2() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2(Default::default())));
    }

    #[test]
    fn test_pow2i() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2I(Default::default())));
    }

    #[test]
    fn test_pow2w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2W(Default::default())));
    }

    #[test]
    fn test_pow2iw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2IW(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmask() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmask(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmaski() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmaskI(Default::default())));
    }

    #[test]
    fn test_virtualrotri() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRI(Default::default())));
    }

    #[test]
    fn test_virtualrotriw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRIW(Default::default())));
    }

    #[test]
    fn test_virtualsra() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRA(Default::default())));
    }

    #[test]
    fn test_virtualsrai() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRAI(Default::default())));
    }

    #[test]
    fn test_virtualrev8w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualRev8W(Default::default())));
    }

    #[test]
    fn test_virtualsrl() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRL(Default::default())));
    }

    #[test]
    fn test_virtualsrli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRLI(Default::default())));
    }

    #[test]
    fn test_virtualextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualZeroExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualsignextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSignExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisor() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisor(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisorw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisorW(Default::default())));
    }

    #[test]
    fn test_virtualassertmulnooverflow() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertMulUNoOverflow(Default::default())));
    }
}
