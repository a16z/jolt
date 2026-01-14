use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::accumulation::Acc6S;
use crate::utils::math::{s64_from_diff_u64s, Math};
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::remap_address;
use crate::zkvm::{bytecode::BytecodePreprocessing, witness::CommittedPolynomial};
use allocative::Allocative;
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tracer::ChunksIterator;
use tracer::{instruction::Cycle, LazyTraceIterator};
use tracing::trace_span;

#[derive(Clone, Debug)]
pub struct RLCStreamingData {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub memory_layout: MemoryLayout,
}

/// Source of trace data for streaming VMV computation.
#[derive(Clone, Debug)]
pub enum TraceSource {
    /// Pre-materialized trace in memory (default, efficient single pass)
    Materialized(Arc<Vec<Cycle>>),
    /// Lazy trace iterator (experimental, re-runs tracer)
    /// Boxed to avoid large enum size difference (LazyTraceIterator is ~34KB)
    Lazy(Box<LazyTraceIterator>),
}

impl TraceSource {
    pub fn len(&self) -> usize {
        match self {
            TraceSource::Materialized(trace) => trace.len(),
            // Lazy trace length is not known upfront (would require full iteration)
            TraceSource::Lazy(_) => panic!("Cannot get length of lazy trace"),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TraceSource::Materialized(trace) => trace.is_empty(),
            TraceSource::Lazy(_) => panic!("Cannot check emptiness of lazy trace"),
        }
    }
}

/// Streaming context for RLC evaluation
#[derive(Clone, Debug)]
pub struct StreamingRLCContext<F: JoltField> {
    pub dense_polys: Vec<(CommittedPolynomial, F)>,
    pub onehot_polys: Vec<(CommittedPolynomial, F)>,
    /// Advice polynomials with their RLC coefficients.
    /// These are NOT streamed from trace - they're passed in directly.
    pub advice_polys: Vec<(F, MultilinearPolynomial<F>)>,
    pub trace_source: TraceSource,
    pub preprocessing: Arc<RLCStreamingData>,
    pub one_hot_params: OneHotParams,
}

/// `RLCPolynomial` represents a multilinear polynomial comprised of a
/// random linear combination of multiple polynomials, potentially with
/// different sizes.
#[derive(Default, Clone, Debug, Allocative)]
pub struct RLCPolynomial<F: JoltField> {
    /// Random linear combination of dense (i.e. length T) polynomials.
    /// Empty if using streaming mode.
    pub dense_rlc: Vec<F>,
    /// Random linear combination of one-hot polynomials (length T x K
    /// for some K). Instead of pre-emptively combining these polynomials,
    /// as we do for `dense_rlc`, we store a vector of (coefficient, polynomial)
    /// pairs and lazily handle the linear combination in `commit_rows`
    /// and `vector_matrix_product`.
    pub one_hot_rlc: Vec<(F, Arc<MultilinearPolynomial<F>>)>,
    /// When present, dense_rlc and one_hot_rlc are not materialized.
    #[allocative(skip)]
    pub streaming_context: Option<Arc<StreamingRLCContext<F>>>,
}

impl<F: JoltField> PartialEq for RLCPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        // Compare materialized data only (streaming context is ephemeral)
        self.dense_rlc == other.dense_rlc && self.one_hot_rlc == other.one_hot_rlc
    }
}

impl<F: JoltField> RLCPolynomial<F> {
    pub fn new() -> Self {
        Self {
            dense_rlc: unsafe_allocate_zero_vec(DoryGlobals::get_T()),
            one_hot_rlc: vec![],
            streaming_context: None,
        }
    }

    /// Constructs an `RLCPolynomial` as a linear combination of `polynomials` with the provided
    /// `coefficients`.
    ///
    /// This is a legacy helper (used by some commitment backends) that eagerly combines dense
    /// polynomials into `dense_rlc` and stores one-hot polynomials lazily in `one_hot_rlc`.
    pub fn linear_combination(
        poly_ids: Vec<CommittedPolynomial>,
        polynomials: Vec<Arc<MultilinearPolynomial<F>>>,
        coefficients: &[F],
        streaming_context: Option<Arc<StreamingRLCContext<F>>>,
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());
        debug_assert_eq!(polynomials.len(), poly_ids.len());

        // Partition into dense and one-hot polynomials
        let (dense, one_hot): (Vec<_>, Vec<_>) = polynomials
            .iter()
            .zip(coefficients.iter())
            .partition(|(p, _)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)));

        // Eagerly materialize the dense linear combination (if any).
        let dense_rlc = if dense.is_empty() {
            vec![]
        } else {
            let max_len = dense
                .iter()
                .map(|(p, _)| p.as_ref().original_len())
                .max()
                .unwrap();

            (0..max_len)
                .into_par_iter()
                .map(|idx| {
                    let mut acc = F::zero();
                    for (poly, coeff) in &dense {
                        if idx < poly.as_ref().original_len() {
                            acc += poly.as_ref().get_scaled_coeff(idx, **coeff);
                        }
                    }
                    acc
                })
                .collect()
        };

        // Store one-hot polynomials lazily.
        let one_hot_rlc: Vec<_> = one_hot
            .into_iter()
            .map(|(poly, coeff)| (*coeff, poly.clone()))
            .collect();

        Self {
            dense_rlc,
            one_hot_rlc,
            streaming_context,
        }
    }

    /// Creates a streaming RLC polynomial from polynomial IDs and coefficients.
    /// O(sqrt(T)) space - streams directly from trace without materializing polynomials.
    ///
    /// # Arguments
    /// * `one_hot_params` - Parameters for one-hot polynomial chunking
    /// * `preprocessing` - Bytecode and memory layout for address computation
    /// * `trace_source` - Either materialized trace (default) or lazy trace (experimental)
    /// * `poly_ids` - List of polynomial identifiers
    /// * `coefficients` - RLC coefficients for each polynomial
    /// * `advice_poly_map` - Map of advice polynomial IDs to their actual polynomials
    #[tracing::instrument(skip_all)]
    pub fn new_streaming(
        one_hot_params: OneHotParams,
        preprocessing: Arc<RLCStreamingData>,
        trace_source: TraceSource,
        poly_ids: Vec<CommittedPolynomial>,
        coefficients: &[F],
        mut advice_poly_map: HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
    ) -> Self {
        debug_assert_eq!(poly_ids.len(), coefficients.len());

        let mut dense_polys = Vec::new();
        let mut onehot_polys = Vec::new();
        let mut advice_polys = Vec::new();

        for (poly_id, coeff) in poly_ids.iter().zip(coefficients.iter()) {
            match poly_id {
                CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                    dense_polys.push((*poly_id, *coeff));
                }
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    onehot_polys.push((*poly_id, *coeff));
                }
                CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                    // Advice polynomials are passed in directly (not streamed from trace)
                    if advice_poly_map.contains_key(poly_id) {
                        advice_polys.push((*coeff, advice_poly_map.remove(poly_id).unwrap()));
                    }
                }
            }
        }

        Self {
            dense_rlc: vec![],   // Not materialized in streaming mode
            one_hot_rlc: vec![], // Not materialized in streaming mode
            streaming_context: Some(Arc::new(StreamingRLCContext {
                dense_polys,
                onehot_polys,
                advice_polys,
                trace_source,
                preprocessing,
                one_hot_params,
            })),
        }
    }

    /// Materializes a streaming RLC polynomial for testing purposes.
    #[cfg(test)]
    pub fn materialize(
        &self,
        _poly_ids: &[CommittedPolynomial],
        polynomials: &[Arc<MultilinearPolynomial<F>>],
        coefficients: &[F],
    ) -> Self {
        if self.streaming_context.is_none() {
            return self.clone();
        }

        let mut result = RLCPolynomial::<F>::new();
        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)))
            .map(|(i, _)| i)
            .collect();

        if !dense_indices.is_empty() {
            let dense_len = result.dense_rlc.len();

            result.dense_rlc = (0..dense_len)
                .into_par_iter()
                .map(|i| {
                    let mut acc = F::zero();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        if i < poly.original_len() {
                            acc += poly.get_scaled_coeff(i, coeff);
                        }
                    }
                    acc
                })
                .collect();
        }

        for (i, poly) in polynomials.iter().enumerate() {
            if matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)) {
                result.one_hot_rlc.push((coefficients[i], poly.clone()));
            }
        }

        result
    }

    /// Commits to the rows of `RLCPolynomial`, viewing its coefficients
    /// as a matrix (used in Dory).
    /// We do so by computing the row commitments for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting commitments.
    // TODO(moodlezoup): we should be able to cache the row commitments
    // for each underlying polynomial and take a linear combination of those
    #[tracing::instrument(skip_all, name = "RLCPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<G> {
        let num_rows = DoryGlobals::get_max_num_rows();
        tracing::debug!("Committing to RLC polynomial with {num_rows} rows");
        let row_len = DoryGlobals::get_num_columns();

        let mut row_commitments = vec![G::zero(); num_rows];

        // Compute the row commitments for dense submatrix
        self.dense_rlc
            .par_chunks(row_len)
            .zip(row_commitments.par_iter_mut())
            .for_each(|(dense_row, commitment)| {
                let msm_result: G =
                    VariableBaseMSM::msm_field_elements(&bases[..dense_row.len()], dense_row)
                        .unwrap();
                *commitment += msm_result
            });

        // Compute the row commitments for one-hot polynomials
        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut new_row_commitments: Vec<G> = match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => one_hot.commit_rows(bases),
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            };

            // TODO(moodlezoup): Avoid resize
            new_row_commitments.resize(num_rows, G::zero());

            let updated_row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    new_row_commitments.as_mut_ptr() as *mut G1Projective,
                    new_row_commitments.len(),
                )
            };

            let current_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(
                    row_commitments.as_ptr() as *const G1Projective,
                    row_commitments.len(),
                )
            };

            let coeff_fr = unsafe { *(&raw const *coeff as *const Fr) };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                updated_row_commitments,
                coeff_fr,
                current_row_commitments,
            );

            let _ = std::mem::replace(&mut row_commitments, new_row_commitments);
        }

        row_commitments
    }

    /// Computes a vector-matrix product, viewing the coefficients of the
    /// polynomial as a matrix (used in Dory).
    /// We do so by computing the vector-matrix product for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting products.
    #[tracing::instrument(skip_all, name = "RLCPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(&self, left_vec: &[F]) -> Vec<F> {
        let num_columns = DoryGlobals::get_num_columns();

        // Compute the vector-matrix product for dense submatrix
        let mut result: Vec<F> = if let Some(ctx) = &self.streaming_context {
            // Streaming mode: generate rows on-demand from trace
            self.streaming_vector_matrix_product(left_vec, num_columns, Arc::clone(ctx))
        } else {
            // Linear space mode: use pre-computed dense_rlc
            (0..num_columns)
                .into_par_iter()
                .map(|col_index| {
                    self.dense_rlc
                        .iter()
                        .skip(col_index)
                        .step_by(num_columns)
                        .zip(left_vec.iter())
                        .map(|(&a, &b)| -> F { a * b })
                        .sum::<F>()
                })
                .collect()
        };

        // Compute the vector-matrix product for one-hot polynomials (linear space)
        for (coeff, poly) in self.one_hot_rlc.iter() {
            match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => {
                    one_hot.vector_matrix_product(left_vec, *coeff, &mut result);
                }
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            }
        }

        result
    }

    /// Adds the advice polynomial contribution to the vector-matrix-vector product result.
    ///
    /// In Dory's batch opening, advice polynomials are embedded as the top-left block of the
    /// main matrix. This function computes their contribution to the VMV product:
    /// ```text
    /// result[col] += left_vec[row] * (coeff * advice[row, col])
    /// ```
    /// for rows and columns within the advice block.
    ///
    /// The advice block occupies:
    /// - `sigma_a = ceil(advice_vars/2)`, `nu_a = advice_vars - sigma_a`
    /// - `advice` occupies rows `[0 .. 2^{nu_a})` and cols `[0 .. 2^{sigma_a})`
    ///
    /// # Complexity
    /// It uses O(m + a) space where m is the number of rows
    /// and a is the advice size. However, this is small enough in practice (advice is typically
    /// much smaller than the trace). This function is used in both streaming and
    /// non-streaming contexts, and mutates `result` in place.
    fn vmp_advice_contribution(
        result: &mut [F],
        left_vec: &[F],
        num_columns: usize,
        ctx: &StreamingRLCContext<F>,
    ) {
        // For each advice polynomial, compute its contribution to the result
        ctx.advice_polys
            .iter()
            .filter(|(_, advice_poly)| advice_poly.original_len() > 0)
            .for_each(|(coeff, advice_poly)| {
                let advice_len = advice_poly.original_len();
                let advice_vars = advice_len.log_2();
                let (sigma_a, nu_a) = DoryGlobals::balanced_sigma_nu(advice_vars);
                let advice_cols = 1usize << sigma_a;
                let advice_rows = 1usize << nu_a;

                debug_assert!(
                    advice_cols <= num_columns,
                    "Advice columns (2^{{sigma_a}}={advice_cols}) must fit in main num_columns={num_columns}; \
guardrail in gen_from_trace should ensure sigma_main >= sigma_a."
                );

                // Only the top-left block contributes: rows [0..advice_rows), cols [0..advice_cols)
                let effective_rows = advice_rows.min(left_vec.len());

                // Compute column contributions: for each column, sum contributions from all rows
                // Note: advice_len is always advice_cols * advice_rows (advice size must be power of 2)
                let column_contributions: Vec<F> = (0..advice_cols)
                    .into_par_iter()
                    .map(|col_idx| {
                        // For this column, sum contributions from all non-zero rows
                        left_vec[..effective_rows]
                            .iter()
                            .enumerate()
                            .filter(|(_, &left)| !left.is_zero())
                            .map(|(row_idx, &left)| {
                                let coeff_idx = row_idx * advice_cols + col_idx;
                                let advice_val = advice_poly.get_coeff(coeff_idx);
                                left * *coeff * advice_val
                            })
                            .sum()
                    })
                    .collect();

                // Add column contributions to result in parallel
                result[..advice_cols]
                    .par_iter_mut()
                    .zip(column_contributions.par_iter())
                    .for_each(|(res, &contrib)| {
                        *res += contrib;
                    });
            });
    }

    /// Streaming VMP implementation that generates rows on-demand from trace.
    /// Achieves O(sqrt(n)) space complexity by lazily generating the witness.
    /// Single pass through trace for both dense and one-hot polynomials.
    fn streaming_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        ctx: Arc<StreamingRLCContext<F>>,
    ) -> Vec<F> {
        let T = DoryGlobals::get_T();

        match &ctx.trace_source {
            TraceSource::Materialized(trace) => {
                self.materialized_vector_matrix_product(left_vec, num_columns, trace, &ctx, T)
            }
            TraceSource::Lazy(lazy_trace) => self.lazy_vector_matrix_product(
                left_vec,
                num_columns,
                (**lazy_trace).clone(),
                &ctx,
                T,
            ),
        }
    }

    /// Single-pass VMV over materialized trace. Parallelizes by dividing rows evenly across threads.
    #[tracing::instrument(skip_all)]
    fn materialized_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        trace: &[Cycle],
        ctx: &StreamingRLCContext<F>,
        T: usize,
    ) -> Vec<F> {
        let num_rows = T / num_columns;
        let trace_len = trace.len();

        // Setup: precompute coefficients, row factors, and folded one-hot tables.
        let setup = VmvSetup::new(ctx, left_vec, num_rows);

        // Divide rows evenly among threads using par_chunks on left_vec
        // Only use first num_rows elements (left_vec may be longer due to padding)
        let num_threads = rayon::current_num_threads();
        let rows_per_thread = num_rows.div_ceil(num_threads);

        let (dense_accs, onehot_accs) = left_vec[..num_rows]
            .par_chunks(rows_per_thread)
            .enumerate()
            .map(|(chunk_idx, row_weights)| {
                let (mut dense_accs, mut onehot_accs) =
                    VmvSetup::<F>::create_accumulators(num_columns);

                let row_start = chunk_idx * rows_per_thread;
                for (local_idx, &row_weight) in row_weights.iter().enumerate() {
                    let row_idx = row_start + local_idx;
                    let chunk_start = row_idx * num_columns;

                    // Row-scaled dense coefficients.
                    let scaled_rd_inc = row_weight * setup.rd_inc_coeff;
                    let scaled_ram_inc = row_weight * setup.ram_inc_coeff;
                    let row_factor = setup.row_factors[row_idx];

                    // Split into valid trace range vs padding range.
                    let valid_end = std::cmp::min(chunk_start + num_columns, trace_len);
                    let row_cycles = if chunk_start < valid_end {
                        &trace[chunk_start..valid_end]
                    } else {
                        &trace[0..0] // Fully padded row
                    };

                    // Process valid trace elements.
                    for (col_idx, cycle) in row_cycles.iter().enumerate() {
                        setup.process_cycle(
                            cycle,
                            scaled_rd_inc,
                            scaled_ram_inc,
                            row_factor,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }
                }

                (dense_accs, onehot_accs)
            })
            .reduce(
                || VmvSetup::<F>::create_accumulators(num_columns),
                VmvSetup::<F>::merge_accumulators,
            );

        let mut result = VmvSetup::<F>::finalize(dense_accs, onehot_accs, num_columns);

        // Advice contribution is small and independent of the trace; add it after the streamed pass.
        Self::vmp_advice_contribution(&mut result, left_vec, num_columns, ctx);
        result
    }

    /// Lazy VMV over lazy trace iterator (experimental, re-runs tracer).
    #[tracing::instrument(skip_all)]
    fn lazy_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        lazy_trace: LazyTraceIterator,
        ctx: &StreamingRLCContext<F>,
        T: usize,
    ) -> Vec<F> {
        let num_rows = T / num_columns;

        // Setup: precompute coefficients, row factors, and folded one-hot tables.
        let setup = VmvSetup::new(ctx, left_vec, num_rows);

        let (dense_accs, onehot_accs) = lazy_trace
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(num_columns)
            .enumerate()
            .par_bridge()
            .fold(
                || VmvSetup::<F>::create_accumulators(num_columns),
                |(mut dense_accs, mut onehot_accs), (row_idx, chunk)| {
                    let row_weight = left_vec[row_idx];
                    let scaled_rd_inc = row_weight * setup.rd_inc_coeff;
                    let scaled_ram_inc = row_weight * setup.ram_inc_coeff;
                    let row_factor = setup.row_factors[row_idx];

                    // Process columns within chunk sequentially.
                    for (col_idx, cycle) in chunk.iter().enumerate() {
                        setup.process_cycle(
                            cycle,
                            scaled_rd_inc,
                            scaled_ram_inc,
                            row_factor,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }

                    (dense_accs, onehot_accs)
                },
            )
            .reduce(
                || VmvSetup::<F>::create_accumulators(num_columns),
                VmvSetup::<F>::merge_accumulators,
            );
        let mut result = VmvSetup::<F>::finalize(dense_accs, onehot_accs, num_columns);

        // Advice contribution is small and independent of the trace; add it after the streamed pass.
        Self::vmp_advice_contribution(&mut result, left_vec, num_columns, ctx);
        result
    }
}

/// Precomputed tables for the one-hot VMV fast path.
/// Each polynomial type has its own Vec<F> of length k_chunk.
struct FoldedOneHotTables<F: JoltField> {
    /// Tables for InstructionRa polynomials, indexed by [poly_idx][k]
    instruction: Vec<Vec<F>>,
    /// Tables for BytecodeRa polynomials, indexed by [poly_idx][k]
    bytecode: Vec<Vec<F>>,
    /// Tables for RamRa polynomials, indexed by [poly_idx][k]
    ram: Vec<Vec<F>>,
}

/// Precomputed VMV setup shared between materialized and lazy paths.
struct VmvSetup<'a, F: JoltField> {
    /// Coefficient for RdInc dense polynomial
    rd_inc_coeff: F,
    /// Coefficient for RamInc dense polynomial
    ram_inc_coeff: F,
    /// Row factors from left vector decomposition
    row_factors: Vec<F>,
    /// Folded one-hot tables (coeff * eq_k pre-multiplied)
    folded_tables: FoldedOneHotTables<F>,
    /// Reference to preprocessing data
    bytecode: &'a BytecodePreprocessing,
    memory_layout: &'a MemoryLayout,
    /// Reference to one-hot parameters
    one_hot_params: &'a OneHotParams,
}

impl<'a, F: JoltField> VmvSetup<'a, F> {
    fn new(ctx: &'a StreamingRLCContext<F>, left_vec: &[F], num_rows: usize) -> Self {
        let one_hot_params = &ctx.one_hot_params;
        let k_chunk = one_hot_params.k_chunk;

        debug_assert!(
            left_vec.len() >= k_chunk * num_rows,
            "left_vec too short for one-hot VMV: len={} need_at_least={}",
            left_vec.len(),
            k_chunk * num_rows
        );

        // Compute row_factors and eq_k from left vector
        let (row_factors, eq_k) = Self::compute_row_factors_and_eq_k(left_vec, num_rows, k_chunk);

        // Extract dense coefficients
        let mut rd_inc_coeff = F::zero();
        let mut ram_inc_coeff = F::zero();
        for (poly_id, coeff) in ctx.dense_polys.iter() {
            match poly_id {
                CommittedPolynomial::RdInc => rd_inc_coeff = *coeff,
                CommittedPolynomial::RamInc => ram_inc_coeff = *coeff,
                _ => unreachable!("one-hot polynomial found in dense_polys"),
            }
        }

        // Build folded one-hot tables (non-flattened)
        let folded_tables =
            Self::build_folded_tables(&ctx.onehot_polys, one_hot_params, &eq_k, k_chunk);

        Self {
            rd_inc_coeff,
            ram_inc_coeff,
            row_factors,
            folded_tables,
            bytecode: &ctx.preprocessing.bytecode,
            memory_layout: &ctx.preprocessing.memory_layout,
            one_hot_params,
        }
    }

    /// Compute row_factors and eq_k from the Dory left vector.
    #[inline]
    fn compute_row_factors_and_eq_k(
        left_vec: &[F],
        rows_per_k: usize,
        k_chunk: usize,
    ) -> (Vec<F>, Vec<F>) {
        let mut row_factors: Vec<F> = unsafe_allocate_zero_vec(rows_per_k);
        let mut eq_k: Vec<F> = unsafe_allocate_zero_vec(k_chunk);

        for k in 0..k_chunk {
            let base = k * rows_per_k;
            let mut sum_k = F::zero();
            for row in 0..rows_per_k {
                let v = left_vec[base + row];
                sum_k += v;
                row_factors[row] += v;
            }
            eq_k[k] = sum_k;
        }

        (row_factors, eq_k)
    }

    /// Build per-polynomial folded one-hot tables (non-flattened).
    fn build_folded_tables(
        onehot_polys: &[(CommittedPolynomial, F)],
        one_hot_params: &OneHotParams,
        eq_k: &[F],
        k_chunk: usize,
    ) -> FoldedOneHotTables<F> {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;

        // Initialize tables with zeros
        let mut instruction: Vec<Vec<F>> = (0..instruction_d)
            .map(|_| unsafe_allocate_zero_vec(k_chunk))
            .collect();
        let mut bytecode: Vec<Vec<F>> = (0..bytecode_d)
            .map(|_| unsafe_allocate_zero_vec(k_chunk))
            .collect();
        let mut ram: Vec<Vec<F>> = (0..ram_d)
            .map(|_| unsafe_allocate_zero_vec(k_chunk))
            .collect();

        // Fill tables with coeff * eq_k[k]
        for (poly_id, coeff) in onehot_polys.iter() {
            if coeff.is_zero() {
                continue;
            }
            match poly_id {
                CommittedPolynomial::InstructionRa(idx) => {
                    for k in 0..k_chunk {
                        instruction[*idx][k] = *coeff * eq_k[k];
                    }
                }
                CommittedPolynomial::BytecodeRa(idx) => {
                    for k in 0..k_chunk {
                        bytecode[*idx][k] = *coeff * eq_k[k];
                    }
                }
                CommittedPolynomial::RamRa(idx) => {
                    for k in 0..k_chunk {
                        ram[*idx][k] = *coeff * eq_k[k];
                    }
                }
                _ => unreachable!("dense polynomial found in onehot_polys"),
            }
        }

        FoldedOneHotTables {
            instruction,
            bytecode,
            ram,
        }
    }

    /// Process a single cycle.
    #[inline(always)]
    fn process_cycle(
        &self,
        cycle: &Cycle,
        scaled_rd_inc: F,
        scaled_ram_inc: F,
        row_factor: F,
        dense_acc: &mut Acc6S<F>,
        onehot_acc: &mut F::Unreduced<9>,
    ) {
        // Dense polynomials: accumulate scaled_coeff * (post - pre)
        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
        let diff = s64_from_diff_u64s(post_value, pre_value);
        dense_acc.fmadd(&scaled_rd_inc, &diff);

        if let tracer::instruction::RAMAccess::Write(write) = cycle.ram_access() {
            let diff = s64_from_diff_u64s(write.post_value, write.pre_value);
            dense_acc.fmadd(&scaled_ram_inc, &diff);
        }

        // One-hot polynomials: accumulate using pre-folded K tables (unreduced)
        let mut inner_sum = F::Unreduced::<5>::default();

        // Instruction RA chunks
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        for (i, table) in self.folded_tables.instruction.iter().enumerate() {
            let k = self.one_hot_params.lookup_index_chunk(lookup_index, i) as usize;
            inner_sum += *table[k].as_unreduced_ref();
        }

        // Bytecode RA chunks
        let pc = self.bytecode.get_pc(cycle);
        for (i, table) in self.folded_tables.bytecode.iter().enumerate() {
            let k = self.one_hot_params.bytecode_pc_chunk(pc, i) as usize;
            inner_sum += *table[k].as_unreduced_ref();
        }

        // RAM RA chunks
        let address = cycle.ram_access().address() as u64;
        if let Some(remapped) = remap_address(address, self.memory_layout) {
            for (i, table) in self.folded_tables.ram.iter().enumerate() {
                let k = self.one_hot_params.ram_address_chunk(remapped, i) as usize;
                inner_sum += *table[k].as_unreduced_ref();
            }
        }

        // Reduce inner_sum before multiplying with row_factor
        let inner_sum_reduced = F::from_barrett_reduce::<5>(inner_sum);
        *onehot_acc += row_factor.mul_unreduced::<9>(inner_sum_reduced);
    }

    #[inline]
    fn create_accumulators(num_columns: usize) -> (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>) {
        (
            unsafe_allocate_zero_vec(num_columns),
            unsafe_allocate_zero_vec(num_columns),
        )
    }

    #[inline]
    fn merge_accumulators(
        (mut dense_a, mut onehot_a): (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>),
        (dense_b, onehot_b): (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>),
    ) -> (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>) {
        for (a, b) in dense_a.iter_mut().zip(dense_b.iter()) {
            *a = *a + *b;
        }
        for (a, b) in onehot_a.iter_mut().zip(onehot_b.iter()) {
            *a += *b;
        }
        (dense_a, onehot_a)
    }

    fn finalize(
        dense_accs: Vec<Acc6S<F>>,
        onehot_accs: Vec<F::Unreduced<9>>,
        num_columns: usize,
    ) -> Vec<F> {
        (0..num_columns)
            .into_par_iter()
            .map(|col_idx| {
                dense_accs[col_idx].barrett_reduce()
                    + F::from_montgomery_reduce::<9>(onehot_accs[col_idx])
            })
            .collect()
    }
}
