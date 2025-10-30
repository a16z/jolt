use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::XLEN;
use num_traits::Zero;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};

use super::{LOG_K, LOG_M, M, PHASES};

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
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
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        dag::state_manager::StateManager,
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
// - eq_addr(k; r_addr) := multilinear equality polynomial over LOG_K vars.
// - eq_sp(j; r_sp) and eq_br(j; r_br) := equality polynomials over LOG_T vars.
// - ra(k, j) ∈ F is the selector arising from prefix/suffix condensation; logically ra(k, j) = 1
//   when the j-th cycle’s lookup key equals k, and 0 otherwise (implemented via ExpandingTable).
// - Val_j(k) ∈ F is the lookup-table value selected by (j, k); concretely Val_j(k) = table_j(k)
//   if cycle j uses a table and 0 otherwise (materialized via prefix/suffix decomposition).
// - raf_flag(j) ∈ {0,1} is 1 iff the instruction at cycle j is NOT interleaved operands.
// - Let LeftPrefix_j, RightPrefix_j, IdentityPrefix_j ∈ F be the address-only (prefix) factors for
//   the left/right operand and identity polynomials at cycle j (from `PrefixSuffixDecomposition`).
//
// We introduce a batching challenge γ ∈ F. Define
//   RafVal_j(k) := (1 - raf_flag(j)) · (LeftPrefix_j + γ · RightPrefix_j)
//                  + raf_flag(j) · γ · IdentityPrefix_j.
// The overall γ-weights are arranged so that γ^2 multiplies RafVal_j(k) in the final identity.
//
// Claims supplied by the accumulator (LHS):
// - rv_spartan := ⟦LookupOutput⟧ at SumcheckId::SpartanOuter
// - rv_branch  := ⟦LookupOutput⟧ at SumcheckId::ProductVirtualization
// - left_op    := ⟦LeftLookupOperand⟧ at SumcheckId::SpartanOuter
// - right_op   := ⟦RightLookupOperand⟧ at SumcheckId::SpartanOuter
//   Combined as: rv_spartan(r_sp) + γ·rv_branch(r_br) + γ^2·(left_op + γ·right_op)
//
// Statement proved by this sumcheck (RHS), for random challenges
// r_addr ∈ F^{LOG_K}, r_sp, r_br ∈ F^{log_T}:
//
//   rv_spartan(r_sp) + γ·rv_branch(r_br) + γ^2·(left_op + γ·right_op)
//   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ (eq_sp(j; r_sp) + γ·eq_br(j; r_br)) · ra(k, j) · Val_j(k)
//                                   + γ^2 · eq_sp(j; r_sp) · ra(k, j) · RafVal_j(k) ].
//
// Equivalent split (for GruenSplitEqPolynomial in the last log(T) rounds):
//   (i)  rv_spartan(r_sp) + γ^2·raf(r_sp)
//        = Σ_j eq_sp(j; r_sp) · Σ_k ra(k, j) · (Val_j(k) + γ^2·RafVal_j(k))
//   (ii) rv_branch(r_br)
//        = Σ_j eq_br(j; r_br) · Σ_k ra(k, j) · Val_j(k).
//
// Prover structure:
// - First log(K) rounds bind address vars using prefix/suffix decomposition, accumulating:
//   Σ_k ra(k, j)·Val_j(k)  and  Σ_k ra(k, j)·RafVal_j(k)
//   for each j (via u_evals vectors and suffix polynomials).
// - Last log(T) rounds bind cycle vars using two GruenSplitEqPolynomial instances (for r_sp, r_br),
//   producing degree-3 univariates with the required previous-round claims.
// - The published univariate matches the RHS above; the verifier checks it against the LHS claims.

/// Degree bound of the sumcheck round polynomials in [`ReadRafSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

/// Sumcheck prover for [`ReadRafSumcheckVerifier`].
///
/// Binds address variables first using prefix/suffix decomposition to aggregate, per cycle j,
///   Σ_k ra(k, j)·Val_j(k) and Σ_k ra(k, j)·RafVal_j(k),
/// then binds cycle variables using two `GruenSplitEqPolynomial` instances (Spartan and Branch),
/// producing degree-3 univariates with previous-round claims to support the Gruen evaluation.
#[derive(Allocative)]
pub struct ReadRafSumcheckProver<F: JoltField> {
    /// Materialized `ra(k, j)` MLE over (address, cycle) after the first log(K) rounds.
    /// Present only in the last log(T) rounds.
    ra: Option<MultilinearPolynomial<F>>,
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
    /// Expanding tables accumulating address-prefix products per phase (see `u_evals_*`).
    v: [ExpandingTable<F>; PHASES],
    /// u_evals for read-checking part: eq(r_spartan,j) + gamma·eq(r_branch,j).
    u_evals_rv: Vec<F>,
    /// u_evals for RAF part: eq(r_spartan,j).
    u_evals_raf: Vec<F>,

    // State related to Gruen EQ optimization
    /// Gruen-split equality polynomial over cycle vars for Spartan part (high-to-low binding).
    eq_r_spartan: GruenSplitEqPolynomial<F>,
    /// Gruen-split equality polynomial over cycle vars for Branch/ProductVirtualization part.
    eq_r_branch: GruenSplitEqPolynomial<F>,
    /// Previous-round sumcheck claim s_spartan(0)+s_spartan(1) for degree-3 univariate recovery.
    prev_claim_spartan: Option<F>,
    /// Previous-round sumcheck claim s_branch(0)+s_branch(1) for degree-3 univariate recovery.
    prev_claim_branch: Option<F>,
    /// Previous round polynomial for Spartan part, used to derive next claim at r_j.
    prev_round_poly_spartan: Option<UniPoly<F>>,
    /// Previous round polynomial for Branch part, used to derive next claim at r_j.
    prev_round_poly_branch: Option<UniPoly<F>>,

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

impl<'a, F: JoltField> ReadRafSumcheckProver<F> {
    /// Creates a prover-side instance for the Read+RAF batched sumcheck.
    ///
    /// Builds prover-side working state:
    /// - Precomputes per-cycle lookup index, interleaving flags, and table choices
    /// - Buckets cycles by table and by path (interleaved vs identity)
    /// - Allocates per-table suffix accumulators and u-evals for rv/raf parts
    /// - Instantiates the three RAF decompositions and Gruen EQs over cycles
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::gen")]
    pub fn gen(
        sm: &'a mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let params = ReadRafSumcheckParams::new(sm);
        let trace = sm.get_prover_data().2;
        let (r_branch, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
        );
        let (r_spartan, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let log_T = trace.len().log_2();
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), LOG_M, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), LOG_M, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), LOG_M, LOG_K);
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

        lookup_indices.par_extend(cycle_data.par_iter().map(|data| data.lookup_index));
        is_interleaved_operands.par_extend(cycle_data.par_iter().map(|data| data.is_interleaved));
        lookup_tables.par_extend(cycle_data.par_iter().map(|data| data.table));

        // Collect interleaved and identity indices
        let (lookup_indices_uninterleave, lookup_indices_identity): (Vec<_>, Vec<_>) =
            cycle_data.par_iter().partition_map(|data| {
                if data.is_interleaved {
                    rayon::iter::Either::Left(data.idx)
                } else {
                    rayon::iter::Either::Right(data.idx)
                }
            });

        // Build lookup_indices_by_table fully in parallel
        // Create a vector for each table in parallel
        let lookup_indices_by_table: Vec<Vec<usize>> = (0..num_tables)
            .into_par_iter()
            .map(|t_idx| {
                // Each table gets its own parallel collection
                let mut table_vec = Vec::new();
                table_vec.par_extend(cycle_data.par_iter().filter_map(|data| {
                    data.table.and_then(|t| {
                        if LookupTables::<XLEN>::enum_index(&t) == t_idx {
                            Some(data.idx)
                        } else {
                            None
                        }
                    })
                }));
                table_vec
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

        let eq_r_spartan = EqPolynomial::evals(&r_spartan.r);
        let eq_r_branch = EqPolynomial::evals(&r_branch.r);
        let u_evals_rv = eq_r_spartan
            .par_iter()
            .zip(eq_r_branch.into_par_iter())
            .map(|(a, b)| b * params.gamma + a)
            .collect::<Vec<_>>();

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
            v: std::array::from_fn(|_| ExpandingTable::new(M)),
            u_evals_rv,
            u_evals_raf: eq_r_spartan,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

            // State for last log(T) rounds
            ra: None,
            eq_r_spartan: GruenSplitEqPolynomial::new(&r_spartan.r, BindingOrder::HighToLow),
            eq_r_branch: GruenSplitEqPolynomial::new(&r_branch.r, BindingOrder::HighToLow),
            prev_claim_spartan: None,
            prev_claim_branch: None,
            prev_round_poly_spartan: None,
            prev_round_poly_branch: None,
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
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .zip(self.u_evals_rv.par_iter_mut())
                .zip(self.u_evals_raf.par_iter_mut())
                .for_each(|((k, u), u_raf)| {
                    let (prefix, _) = k.split((PHASES - phase) * LOG_M);
                    let k_bound: usize = prefix % M;
                    *u *= self.v[phase - 1][k_bound];
                    *u_raf *= self.v[phase - 1][k_bound];
                });
        }

        rayon::scope(|s| {
            // Single pass over lookup_indices_uninterleave for both operands
            s.spawn(|_| {
                PrefixSuffixDecomposition::init_Q_dual(
                    &mut self.left_operand_ps,
                    &mut self.right_operand_ps,
                    &self.u_evals_raf,
                    &self.lookup_indices_uninterleave,
                    &self.lookup_indices,
                )
            });
            s.spawn(|_| {
                self.identity_ps.init_Q(
                    &self.u_evals_raf,
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
    /// current phase. For each table’s suffix family, bucket cycles by the
    /// current chunk value and aggregate weighted contributions into Dense MLEs
    /// of size M = 2^{LOG_M}.
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (self.lookup_indices.len() / num_chunks).max(1);

        let new_suffix_polys: Vec<_> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .zip(self.lookup_indices_by_table.par_iter())
            .map(|(table, lookup_indices)| {
                let suffixes = table.suffixes();
                let unreduced_polys = lookup_indices
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut chunk_result: Vec<Vec<F::Unreduced<6>>> =
                            vec![unsafe_allocate_zero_vec(M); suffixes.len()];

                        for j in chunk {
                            let k = self.lookup_indices[*j];
                            let (prefix_bits, suffix_bits) = k.split((PHASES - 1 - phase) * LOG_M);
                            for (suffix, result) in suffixes.iter().zip(chunk_result.iter_mut()) {
                                let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                                if t != 0 {
                                    let u = self.u_evals_rv[*j];
                                    result[prefix_bits % M] += u.mul_u64_unreduced(t);
                                }
                            }
                        }

                        chunk_result
                    })
                    .reduce(
                        || vec![unsafe_allocate_zero_vec(M); suffixes.len()],
                        |mut acc, new| {
                            for (acc_i, new_i) in acc.iter_mut().zip(new.iter()) {
                                for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
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
            .collect();

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
    /// - Materializes ra(k,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Computes previous-claim hints for Gruen (Spartan and Branch)
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_sqr: F) {
        // Drop stuff that's no longer needed
        drop_in_background_thread((
            std::mem::take(&mut self.u_evals_raf),
            std::mem::take(&mut self.u_evals_rv),
            std::mem::take(&mut self.lookup_indices_uninterleave),
        ));

        // Materialize ra polynomial
        let ra: Vec<_> = self
            .lookup_indices
            .par_iter()
            .map(|k| {
                (0..PHASES)
                    .map(|phase| {
                        let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                        let k_bound: usize = prefix % M;
                        self.v[phase][k_bound]
                    })
                    .product::<F>()
            })
            .collect();

        drop_in_background_thread(std::mem::take(&mut self.v));

        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.lookup_indices.len());
        combined_val_poly
            .par_iter_mut()
            .zip(std::mem::take(&mut self.lookup_tables))
            .for_each(|(val, table)| {
                if let Some(table) = table {
                    let suffixes: Vec<_> = table
                        .suffixes()
                        .iter()
                        .map(|suffix| F::from_u64(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0))))
                        .collect();
                    *val += table.combine(&prefixes, &suffixes);
                }
            });
        let gamma_cub = gamma * gamma_sqr;

        let mut combined_raf_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.lookup_indices.len());
        combined_raf_val_poly
            .par_iter_mut()
            .zip(std::mem::take(&mut self.is_interleaved_operands))
            .for_each(|(val, is_interleaved_operands)| {
                if is_interleaved_operands {
                    *val += gamma_sqr
                        * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                        + gamma_cub
                            * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
                } else {
                    *val += gamma_cub * self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
                }
            });

        // The first log(K) rounds of this sumcheck effectively batches the following two sumchecks together:
        // (simplified for exposition)
        //
        // 1. rv(r_spartan) + gamma * rv(r_branch) = \sum (eq(r_spartan, j) + gamma * eq(r_branch, j)) * ra(k, j) * Val(k, j)
        // 2. raf(r_spartan) = \sum eq(r_spartan, j) * ra(k, j) * rafVal(k, j)
        //
        // Batched:
        //   rv(r_spartan) + gamma * rv(r_branch) + gamma^2 * raf(r_spartan)
        //     = \sum (eq(r_spartan, j) + gamma * eq(r_branch, j)) * ra(k, j) * Val(k, j) + gamma^2 * eq(r_spartan, j) * ra(k, j) * rafVal(k, j)
        //
        // Where the (eq(r_spartan, j) + gamma * eq(r_branch, j)) term appearing in sumcheck 1 can be represented by a single
        // vector `u_evals_rv`, since it doesn't involve any k variables.
        //
        // But in order to use GruenSplitEqPolynomial to represent eq(r_spartan, j) and eq(r_branch, j) in the last log(T)
        // rounds, we need to split the batched sumcheck by EQ term, rather splitting by rv vs. raf.
        //
        // i. rv(r_spartan) + gamma^2 * raf(r_spartan) = \sum eq(r_spartan, j) * ra(k, j) * (Val(k, j) + gamma^2 rafVal(k, j))
        // ii. rv(r_branch) = \sum \sum eq(r_branch, j) * ra(k, j) * Val(k, j)
        //
        // Note that the batched sumcheck expression can be equivalently represented by (1) + gamma^2 * (2) and
        // (i) + gamma * (ii), so this alternative decomposition works.
        //
        // In order to derive the univariate polynomial in round log(K), GruenSplitEqPolynomial needs the sumcheck claim from round log(K)-1.
        // For sumchecks (i) and (ii), these claims correspond to:
        // \sum_j eq(r_spartan, j) * ra(r_address, j) * (Val(r_address, j) + gamma^2 rafVal(r_address, j)) and
        // \sum_j eq(r_branch, j) * ra(r_address, j) * Val(r_address, j)
        //
        // We compute these two claims below.
        let prev_claim_spartan: F = self
            .eq_r_spartan
            .par_iter_high_to_low()
            .map(|(j, eq)| eq * ra[j] * (combined_val_poly[j] + combined_raf_val_poly[j]))
            .sum();
        let prev_claim_branch: F = self
            .eq_r_branch
            .par_iter_high_to_low()
            .map(|(j, eq)| eq * ra[j] * combined_val_poly[j])
            .sum();

        self.prev_claim_spartan = Some(prev_claim_spartan);
        self.prev_claim_branch = Some(prev_claim_branch);

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
        self.combined_raf_val_polynomial = Some(MultilinearPolynomial::from(combined_raf_val_poly));
        self.ra = Some(ra.into());
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReadRafSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(
        skip_all,
        name = "InstructionReadRafSumcheckProver::compute_prover_message"
    )]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQs to form the Spartan
    ///   and Branch univariates and returns their γ-weighted sum.
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round).to_vec()
        } else {
            let ra = self.ra.as_ref().unwrap();
            let val = self.combined_val_polynomial.as_ref().unwrap();
            let raf_val = self.combined_raf_val_polynomial.as_ref().unwrap();

            debug_assert_eq!(
                self.eq_r_spartan.current_index,
                self.eq_r_branch.current_index
            );
            let out_evals_spartan = self.eq_r_spartan.E_out_current();
            let in_evals_spartan = self.eq_r_spartan.E_in_current();
            let out_evals_branch = self.eq_r_branch.E_out_current();
            let in_evals_branch = self.eq_r_branch.E_in_current();

            let out_len = out_evals_spartan.len();
            let in_len = in_evals_spartan.len();
            let out_n_vars = out_len.ilog2();
            let half_n = self.ra.as_ref().unwrap().len() / 2;

            let [eval_at_0_spartan, eval_at_inf_spartan, eval_at_0_branch, eval_at_inf_branch] = (0
                ..in_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut eval_at_0_spartan = F::zero();
                    let mut eval_at_inf_spartan = F::zero();
                    let mut eval_at_0_branch = F::zero();
                    let mut eval_at_inf_branch = F::zero();

                    for j_lo in 0..out_len {
                        let j = j_lo + (j_hi << out_n_vars);

                        let ra_at_0_j = ra.get_bound_coeff(j);
                        let ra_at_inf_j = ra.get_bound_coeff(j + half_n) - ra_at_0_j;

                        let val_at_0_j = val.get_bound_coeff(j);
                        let val_at_inf_j = val.get_bound_coeff(j + half_n) - val_at_0_j;

                        let raf_val_at_0_j = raf_val.get_bound_coeff(j);
                        let raf_val_at_inf_j = raf_val.get_bound_coeff(j + half_n) - raf_val_at_0_j;

                        eval_at_0_spartan +=
                            out_evals_spartan[j_lo] * ra_at_0_j * (val_at_0_j + raf_val_at_0_j);
                        eval_at_inf_spartan += out_evals_spartan[j_lo]
                            * ra_at_inf_j
                            * (val_at_inf_j + raf_val_at_inf_j);
                        eval_at_0_branch += out_evals_branch[j_lo] * ra_at_0_j * val_at_0_j;
                        eval_at_inf_branch += out_evals_branch[j_lo] * ra_at_inf_j * val_at_inf_j;
                    }

                    [
                        in_evals_spartan[j_hi].mul_unreduced::<9>(eval_at_0_spartan),
                        in_evals_spartan[j_hi].mul_unreduced::<9>(eval_at_inf_spartan),
                        in_evals_branch[j_hi].mul_unreduced::<9>(eval_at_0_branch),
                        in_evals_branch[j_hi].mul_unreduced::<9>(eval_at_inf_branch),
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); 4],
                    |a, b| std::array::from_fn(|i| a[i] + b[i]),
                );

            let univariate_evals_spartan = self.eq_r_spartan.gruen_evals_deg_3(
                F::from_montgomery_reduce(eval_at_0_spartan),
                F::from_montgomery_reduce(eval_at_inf_spartan),
                self.prev_claim_spartan.unwrap(),
            );
            let univariate_evals_branch = self.eq_r_branch.gruen_evals_deg_3(
                F::from_montgomery_reduce(eval_at_0_branch),
                F::from_montgomery_reduce(eval_at_inf_branch),
                self.prev_claim_branch.unwrap(),
            );

            self.prev_round_poly_spartan = Some(UniPoly::from_evals_and_hint(
                self.prev_claim_spartan.unwrap(),
                &univariate_evals_spartan,
            ));
            self.prev_round_poly_branch = Some(UniPoly::from_evals_and_hint(
                self.prev_claim_branch.unwrap(),
                &univariate_evals_branch,
            ));

            univariate_evals_spartan
                .iter()
                .zip(univariate_evals_branch.iter())
                .map(|(eval_spartan, eval_branch)| *eval_spartan + self.params.gamma * eval_branch)
                .collect()
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::bind")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQs; update previous-claim hints via last round's
    /// univariate.
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / LOG_M;
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
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut self.prefix_checkpoints,
                        self.r[self.r.len() - 2],
                        self.r[self.r.len() - 1],
                        round,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(LOG_M) {
                self.prefix_registry.update_checkpoints();
                if phase != PHASES - 1 {
                    // if not last phase, init next phase
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                self.init_log_t_rounds(self.params.gamma, self.params.gamma_sqr);
            }
        } else {
            // log(T) rounds

            self.eq_r_spartan.bind(r_j);
            self.eq_r_branch.bind(r_j);
            [
                self.ra.as_mut().unwrap(),
                self.combined_val_polynomial.as_mut().unwrap(),
                self.combined_raf_val_polynomial.as_mut().unwrap(),
            ]
            .par_iter_mut()
            .for_each(|poly| {
                poly.bind_parallel(r_j, BindingOrder::HighToLow);
            });

            self.prev_claim_spartan =
                Some(self.prev_round_poly_spartan.take().unwrap().evaluate(&r_j));
            self.prev_claim_branch =
                Some(self.prev_round_poly_branch.take().unwrap().evaluate(&r_j));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = get_opening_point::<F>(sumcheck_challenges);
        // Prover publishes new virtual openings derived by this sumcheck:
        // - Per-table LookupTableFlag(i) at r_cycle
        // - InstructionRa at r_sumcheck (ra MLE’s final claim)
        // - InstructionRafFlag at r_cycle
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);
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

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
            self.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
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
    fn compute_prefix_suffix_prover_message(&self, round: usize) -> [F; 2] {
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

        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
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
        let gamma_cub = self.params.gamma * self.params.gamma_sqr;
        [
            F::from_montgomery_reduce(
                left_0.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref())
                    + right_0.mul_trunc::<4, 9>(gamma_cub.as_unreduced_ref()),
            ),
            F::from_montgomery_reduce(
                left_2.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref())
                    + right_2.mul_trunc::<4, 9>(gamma_cub.as_unreduced_ref()),
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

/// Computes the bit-length of the suffix, for the current (`j`th) round
/// of sumcheck.
pub fn current_suffix_len(j: usize) -> usize {
    LOG_K - (j / LOG_M + 1) * LOG_M
}

/// Instruction lookups: batched Read + RAF sumcheck.
///
/// Let K = 2^{LOG_K}, T = 2^{log_T}. For random r_addr ∈ F^{LOG_K}, r_sp, r_br ∈ F^{log_T},
/// this sumcheck proves that the accumulator claims
///   rv_spartan(r_sp) + γ·rv_branch(r_br) + γ^2·(left_op + γ·right_op)
/// equal the double sum over (j, k):
///   Σ_j Σ_k [ (eq_sp(j; r_sp) + γ·eq_br(j; r_br)) · ra(k, j) · Val_j(k)
///             + γ^2 · eq_sp(j; r_sp) · ra(k, j) · RafVal_j(k) ].
/// It is implemented as: first log(K) address-binding rounds (prefix/suffix condensation), then
/// last log(T) cycle-binding rounds driven by two `GruenSplitEqPolynomial`s (Spartan/Branch).
pub struct ReadRafSumcheckVerifier<F: JoltField> {
    params: ReadRafSumcheckParams<F>,
}

impl<F: JoltField> ReadRafSumcheckVerifier<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let params = ReadRafSumcheckParams::new(state_manager);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReadRafSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Verifier’s RHS reconstruction from virtual claims at r:
        //
        // Computes Val and RafVal contributions at r_address, forms EQ(r_cycle)
        // for Spartan/Branch, multiplies by ra claim at r_sumcheck, and returns
        // the batched identity RHS to be matched against the LHS input claim.
        let (r_address_prime, r_cycle_prime) = sumcheck_challenges.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(r_address_prime);
        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(r_address_prime))
            .collect();

        let r_spartan = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let r_branch = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::ProductVirtualization,
            )
            .0
            .r;
        let eq_eval_spartan = EqPolynomial::<F>::mle(&r_spartan, r_cycle_prime);
        let eq_eval_branch = EqPolynomial::<F>::mle(&r_branch, r_cycle_prime);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            )
            .1;

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

        ra_claim
            * (val_claim * (eq_eval_spartan + self.params.gamma * eq_eval_branch)
                + self.params.gamma_sqr * raf_claim * eq_eval_spartan)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = get_opening_point::<F>(sumcheck_challenges);
        // Verifier requests the virtual openings that the prover must provide
        // for this sumcheck (same set as published by the prover-side cache).
        let (_r_address, r_cycle) = r_sumcheck.split_at(LOG_K);

        (0..LookupTables::<XLEN>::COUNT).for_each(|i| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        });

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
        );
    }
}

struct ReadRafSumcheckParams<F: JoltField> {
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    gamma: F,
    gamma_sqr: F,
    /// log2(T): number of cycle variables (last rounds bind cycles).
    log_T: usize,
}

impl<F: JoltField> ReadRafSumcheckParams<F> {
    fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let gamma = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let log_T = state_manager.get_trace_len().log_2();
        Self {
            gamma,
            gamma_sqr,
            log_T,
        }
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim_spartan) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, rv_claim_branch) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
        );
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );
        rv_claim_spartan
            + self.gamma * rv_claim_branch
            + self.gamma_sqr * (left_operand_claim + self.gamma * right_operand_claim)
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::new(sumcheck_challenges.to_vec())
}

#[cfg(test)]
mod tests {
    use std::ops::DerefMut;

    use super::*;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use crate::{
        poly::commitment::mock::MockCommitScheme,
        zkvm::{
            bytecode::BytecodePreprocessing, ram::RAMPreprocessing, JoltProverPreprocessing,
            JoltSharedPreprocessing, JoltVerifierPreprocessing,
        },
    };
    use ark_bn254::Fr;
    use ark_std::Zero;
    use common::jolt_device::MemoryLayout;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::emulator::memory::Memory;
    use tracer::instruction::Cycle;
    use tracer::{JoltDevice, LazyTraceIterator};

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
        let bytecode = vec![];
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let memory_layout = MemoryLayout::default();
        let shared_preprocessing = JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            ram: RAMPreprocessing::preprocess(vec![]),
            memory_layout: memory_layout.clone(),
        };
        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_preprocessing.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_preprocessing,
            };
        let program_io = JoltDevice {
            memory_layout,
            untrusted_advice: vec![],
            trusted_advice: vec![],
            inputs: vec![],
            outputs: vec![],
            panic: false,
        };
        let final_memory_state = Memory::default();

        let lazy_trace = LazyTraceIterator::new_for_test();
        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            lazy_trace,
            trace.clone(),
            program_io.clone(),
            None,
            final_memory_state,
        );
        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            1 << 8,
            prover_sm.twist_sumcheck_switch_index,
        );

        let r_cycle: Vec<<Fr as JoltField>::Challenge> = prover_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let _r_cycle: Vec<<Fr as JoltField>::Challenge> = verifier_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let eq_r_cycle = EqPolynomial::<Fr>::evals(&r_cycle);

        let r_cycle_branch: Vec<<Fr as JoltField>::Challenge> = prover_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let _r_cycle_branch: Vec<<Fr as JoltField>::Challenge> = verifier_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let eq_r_cycle_branch = EqPolynomial::<Fr>::evals(&r_cycle_branch);

        let mut rv_claim = Fr::zero();
        let mut left_operand_claim = Fr::zero();
        let mut right_operand_claim = Fr::zero();
        let mut rv_claim_branch = Fr::zero();

        for (i, cycle) in trace.iter().enumerate() {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            let table: Option<LookupTables<XLEN>> = cycle.lookup_table();
            if let Some(table) = table {
                rv_claim +=
                    JoltField::mul_u64(&eq_r_cycle[i], table.materialize_entry(lookup_index));

                rv_claim_branch += JoltField::mul_u64(
                    &eq_r_cycle_branch[i],
                    table.materialize_entry(lookup_index),
                );
            }

            // Compute left and right operand claims
            let (lo, ro) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
            left_operand_claim += JoltField::mul_u64(&eq_r_cycle[i], lo);
            right_operand_claim += JoltField::mul_u128(&eq_r_cycle[i], ro);
        }

        let prover_accumulator = prover_sm.get_prover_accumulator();
        prover_accumulator.borrow_mut().append_virtual(
            prover_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );
        prover_accumulator.borrow_mut().append_virtual(
            prover_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            left_operand_claim,
        );
        prover_accumulator.borrow_mut().append_virtual(
            prover_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            right_operand_claim,
        );
        prover_accumulator.borrow_mut().append_virtual(
            prover_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
            OpeningPoint::new(r_cycle_branch.clone()),
            rv_claim_branch,
        );

        let mut prover_sumcheck = ReadRafSumcheckProver::gen(&mut prover_sm);

        let mut prover_transcript_ref = prover_sm.transcript.borrow_mut();

        let (proof, r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut prover_sumcheck],
            &mut *prover_accumulator.borrow_mut(),
            &mut *prover_transcript_ref,
        );
        drop(prover_transcript_ref);

        // Take claims
        let prover_acc_borrow = prover_accumulator.borrow();
        let verifier_accumulator = verifier_sm.get_verifier_accumulator();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_acc_borrow
                .openings_mut()
                .insert(*key, (empty_point, *value));
        }
        drop(prover_acc_borrow);
        drop(verifier_acc_borrow);

        verifier_accumulator.borrow_mut().append_virtual(
            verifier_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_accumulator.borrow_mut().append_virtual(
            verifier_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_accumulator.borrow_mut().append_virtual(
            verifier_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_accumulator.borrow_mut().append_virtual(
            verifier_sm.transcript.borrow_mut().deref_mut(),
            VirtualPolynomial::LookupOutput,
            SumcheckId::ProductVirtualization,
            OpeningPoint::new(r_cycle_branch.clone()),
        );

        let mut verifier_sumcheck = ReadRafSumcheckVerifier::new(&mut verifier_sm);

        let r_sumcheck_verif = BatchedSumcheck::verify(
            &proof,
            vec![&mut verifier_sumcheck],
            &mut *verifier_accumulator.borrow_mut(),
            &mut *verifier_sm.transcript.borrow_mut(),
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
