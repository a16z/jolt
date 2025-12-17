use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::shared_ra_polys::{MAX_BYTECODE_D, MAX_INSTRUCTION_D, MAX_RAM_D};
use crate::utils::accumulation::Acc6S;
use crate::utils::math::s64_from_diff_u64s;
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
use std::sync::Arc;
use tracer::ChunksIterator;
use tracer::{instruction::Cycle, LazyTraceIterator};
use tracing::trace_span;

#[derive(Clone, Debug)]
pub struct RLCStreamingData {
    pub bytecode: BytecodePreprocessing,
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
    #[allow(unused_variables)]
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
    #[tracing::instrument(skip_all)]
    pub fn new_streaming(
        one_hot_params: OneHotParams,
        preprocessing: Arc<RLCStreamingData>,
        trace_source: TraceSource,
        poly_ids: Vec<CommittedPolynomial>,
        coefficients: &[F],
    ) -> Self {
        debug_assert_eq!(poly_ids.len(), coefficients.len());

        let mut dense_polys = Vec::new();
        let mut onehot_polys = Vec::new();

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
            }
        }

        Self {
            dense_rlc: vec![],   // Not materialized in streaming mode
            one_hot_rlc: vec![], // Not materialized in streaming mode
            streaming_context: Some(Arc::new(StreamingRLCContext {
                dense_polys,
                onehot_polys,
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

        // Divide rows evenly among threads.
        let num_threads = rayon::current_num_threads();
        let rows_per_thread = num_rows.div_ceil(num_threads);
        let chunk_ranges: Vec<(usize, usize)> = (0..num_threads)
            .map(|t| {
                let start = t * rows_per_thread;
                let end = std::cmp::min(start + rows_per_thread, num_rows);
                (start, end)
            })
            .filter(|(start, end)| start < end)
            .collect();

        let (dense_accs, onehot_accs) = chunk_ranges
            .into_par_iter()
            .map(|(row_start, row_end)| {
                let (mut dense_accs, mut onehot_accs) =
                    VmvCoeffs::<F>::create_accumulators(num_columns);

                for row_idx in row_start..row_end {
                    let chunk_start = row_idx * num_columns;
                    let row_weight = left_vec[row_idx];

                    // Row-scaled dense coefficients.
                    let scaled_rd_inc = row_weight * setup.coeffs.rd_inc_coeff;
                    let scaled_ram_inc = row_weight * setup.coeffs.ram_inc_coeff;
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
                        setup.coeffs.process_cycle(
                            cycle,
                            scaled_rd_inc,
                            scaled_ram_inc,
                            row_factor,
                            &setup.folded_onehot,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }
                }

                (dense_accs, onehot_accs)
            })
            .reduce(
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                VmvCoeffs::<F>::merge_accumulators,
            );

        VmvCoeffs::<F>::finalize(dense_accs, onehot_accs, num_columns)
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
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                |(mut dense_accs, mut onehot_accs), (row_idx, chunk)| {
                    let row_weight = left_vec[row_idx];
                    let scaled_rd_inc = row_weight * setup.coeffs.rd_inc_coeff;
                    let scaled_ram_inc = row_weight * setup.coeffs.ram_inc_coeff;
                    let row_factor = setup.row_factors[row_idx];

                    // Process columns within chunk sequentially.
                    for (col_idx, cycle) in chunk.iter().enumerate() {
                        setup.coeffs.process_cycle(
                            cycle,
                            scaled_rd_inc,
                            scaled_ram_inc,
                            row_factor,
                            &setup.folded_onehot,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }

                    (dense_accs, onehot_accs)
                },
            )
            .reduce(
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                VmvCoeffs::<F>::merge_accumulators,
            );

        VmvCoeffs::<F>::finalize(dense_accs, onehot_accs, num_columns)
    }
}

// ============================================================================
// VMV Helper Types - Preprocessed coefficients for streaming vector-matrix product
// ============================================================================

/// Precomputed tables for the one-hot VMV fast path that **folds per-polynomial coefficients
/// (γ-powers) into the K-sized eq table**.
///
/// This moves γ-multiplications out of the hot per-cycle loop (O(T * #polys) multiplications)
/// into a tiny precompute (O(K * #polys) multiplications).
struct VmvFoldedOneHotTables<F: JoltField> {
    k_chunk: usize,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    /// Flattened tables: `[instruction_0..d, bytecode_0..d, ram_0..d]`, each length `k_chunk`.
    tables: Vec<F>,
}

/// Preprocessed coefficients for VMV computation.
struct VmvCoeffs<'a, F: JoltField> {
    rd_inc_coeff: F,
    ram_inc_coeff: F,
    instruction_coeffs: [F; MAX_INSTRUCTION_D],
    bytecode_coeffs: [F; MAX_BYTECODE_D],
    ram_coeffs: [F; MAX_RAM_D],
    instruction_shifts: [usize; MAX_INSTRUCTION_D],
    bytecode_shifts: [usize; MAX_BYTECODE_D],
    ram_shifts: [usize; MAX_RAM_D],
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    k_chunk_mask_usize: usize,
    k_chunk_mask_u64: u64,
    k_chunk_mask_u128: u128,
    bytecode: &'a BytecodePreprocessing,
    memory_layout: &'a MemoryLayout,
}

impl<'a, F: JoltField> VmvCoeffs<'a, F> {
    /// Compute `(row_factors, eq_k)` from the Dory left vector.
    #[inline]
    fn compute_row_factors_and_eq_k(
        left_vec: &[F],
        rows_per_k: usize,
        k_chunk: usize,
    ) -> (Vec<F>, Vec<F>) {
        debug_assert!(
            left_vec.len() >= k_chunk * rows_per_k,
            "left_vec too short: len={} need_at_least={}",
            left_vec.len(),
            k_chunk * rows_per_k
        );

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

    /// Build per-polynomial folded one-hot tables.
    #[inline]
    fn build_folded_onehot_tables(&self, eq_k: &[F]) -> VmvFoldedOneHotTables<F> {
        let k_chunk = eq_k.len();
        let num_polys = self.instruction_d + self.bytecode_d + self.ram_d;
        let mut tables: Vec<F> = unsafe_allocate_zero_vec(num_polys * k_chunk);

        for i in 0..self.instruction_d {
            let coeff = self.instruction_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = i * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        let bytecode_base_poly = self.instruction_d;
        for i in 0..self.bytecode_d {
            let coeff = self.bytecode_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = (bytecode_base_poly + i) * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        let ram_base_poly = self.instruction_d + self.bytecode_d;
        for i in 0..self.ram_d {
            let coeff = self.ram_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = (ram_base_poly + i) * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        VmvFoldedOneHotTables {
            k_chunk,
            instruction_d: self.instruction_d,
            bytecode_d: self.bytecode_d,
            ram_d: self.ram_d,
            tables,
        }
    }

    fn new(ctx: &'a StreamingRLCContext<F>) -> Self {
        let one_hot_params = &ctx.one_hot_params;
        let log_k_chunk = one_hot_params.log_k_chunk;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;

        let mut rd_inc_coeff = F::zero();
        let mut ram_inc_coeff = F::zero();
        let mut instruction_coeffs = [F::zero(); MAX_INSTRUCTION_D];
        let mut bytecode_coeffs = [F::zero(); MAX_BYTECODE_D];
        let mut ram_coeffs = [F::zero(); MAX_RAM_D];

        for (poly_id, coeff) in ctx.dense_polys.iter() {
            match poly_id {
                CommittedPolynomial::RdInc => rd_inc_coeff += *coeff,
                CommittedPolynomial::RamInc => ram_inc_coeff += *coeff,
                _ => unreachable!("one-hot polynomial found in dense_polys"),
            }
        }

        for (poly_id, coeff) in ctx.onehot_polys.iter() {
            match poly_id {
                CommittedPolynomial::InstructionRa(idx) => instruction_coeffs[*idx] += *coeff,
                CommittedPolynomial::BytecodeRa(idx) => bytecode_coeffs[*idx] += *coeff,
                CommittedPolynomial::RamRa(idx) => ram_coeffs[*idx] += *coeff,
                _ => unreachable!("dense polynomial found in onehot_polys"),
            }
        }

        let mut instruction_shifts = [0usize; MAX_INSTRUCTION_D];
        let mut bytecode_shifts = [0usize; MAX_BYTECODE_D];
        let mut ram_shifts = [0usize; MAX_RAM_D];

        for i in 0..instruction_d {
            instruction_shifts[i] = log_k_chunk * (instruction_d - 1 - i);
        }
        for i in 0..bytecode_d {
            bytecode_shifts[i] = log_k_chunk * (bytecode_d - 1 - i);
        }
        for i in 0..ram_d {
            ram_shifts[i] = log_k_chunk * (ram_d - 1 - i);
        }

        let k_chunk_mask_usize = one_hot_params.k_chunk - 1;

        Self {
            rd_inc_coeff,
            ram_inc_coeff,
            instruction_coeffs,
            bytecode_coeffs,
            ram_coeffs,
            instruction_shifts,
            bytecode_shifts,
            ram_shifts,
            instruction_d,
            bytecode_d,
            ram_d,
            k_chunk_mask_usize,
            k_chunk_mask_u64: k_chunk_mask_usize as u64,
            k_chunk_mask_u128: k_chunk_mask_usize as u128,
            bytecode: &ctx.preprocessing.bytecode,
            memory_layout: &ctx.preprocessing.memory_layout,
        }
    }

    /// Process a single cycle using delayed reduction for performance.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn process_cycle(
        &self,
        cycle: &Cycle,
        scaled_rd_inc: F,
        scaled_ram_inc: F,
        row_factor: F,
        folded_tables: &VmvFoldedOneHotTables<F>,
        dense_acc: &mut Acc6S<F>,
        onehot_acc: &mut F::Unreduced<9>,
    ) {
        // Dense polynomials: accumulate scaled_coeff * (post - pre)
        let (_, pre_value, post_value) = cycle.rd_write();
        let diff = s64_from_diff_u64s(post_value, pre_value);
        dense_acc.fmadd(&scaled_rd_inc, &diff);

        if let tracer::instruction::RAMAccess::Write(write) = cycle.ram_access() {
            let diff = s64_from_diff_u64s(write.post_value, write.pre_value);
            dense_acc.fmadd(&scaled_ram_inc, &diff);
        }

        // One-hot polynomials: accumulate using pre-folded K tables
        let k_chunk = folded_tables.k_chunk;
        let tables = &folded_tables.tables;
        let mut inner_sum = F::zero();

        // Instruction RA chunks
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        for i in 0..folded_tables.instruction_d {
            let k =
                ((lookup_index >> self.instruction_shifts[i]) & self.k_chunk_mask_u128) as usize;
            inner_sum += tables[i * k_chunk + k];
        }

        // Bytecode RA chunks
        let pc = self.bytecode.get_pc(cycle);
        let bytecode_base = folded_tables.instruction_d;
        for i in 0..folded_tables.bytecode_d {
            let k = (pc >> self.bytecode_shifts[i]) & self.k_chunk_mask_usize;
            inner_sum += tables[(bytecode_base + i) * k_chunk + k];
        }

        // RAM RA chunks
        let address = cycle.ram_access().address() as u64;
        if let Some(remapped) = remap_address(address, self.memory_layout) {
            let ram_base = folded_tables.instruction_d + folded_tables.bytecode_d;
            for i in 0..folded_tables.ram_d {
                let k = ((remapped >> self.ram_shifts[i]) & self.k_chunk_mask_u64) as usize;
                inner_sum += tables[(ram_base + i) * k_chunk + k];
            }
        }

        *onehot_acc += row_factor.mul_unreduced::<9>(inner_sum);
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

/// Precomputed VMV setup shared between materialized and lazy paths.
struct VmvSetup<'a, F: JoltField> {
    coeffs: VmvCoeffs<'a, F>,
    row_factors: Vec<F>,
    folded_onehot: VmvFoldedOneHotTables<F>,
}

impl<'a, F: JoltField> VmvSetup<'a, F> {
    fn new(ctx: &'a StreamingRLCContext<F>, left_vec: &[F], num_rows: usize) -> Self {
        let coeffs = VmvCoeffs::new(ctx);
        let k_chunk = ctx.one_hot_params.k_chunk;

        let (row_factors, eq_k) =
            VmvCoeffs::<F>::compute_row_factors_and_eq_k(left_vec, num_rows, k_chunk);
        let folded_onehot = coeffs.build_folded_onehot_tables(&eq_k);

        debug_assert!(
            left_vec.len() >= k_chunk * num_rows,
            "left_vec too short for one-hot VMV: len={} need_at_least={}",
            left_vec.len(),
            k_chunk * num_rows
        );

        Self {
            coeffs,
            row_factors,
            folded_onehot,
        }
    }
}
