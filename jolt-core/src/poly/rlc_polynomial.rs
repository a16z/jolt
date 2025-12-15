use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
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
    #[allow(unused_variables)]
    pub fn linear_combination(
        poly_ids: Vec<CommittedPolynomial>,
        polynomials: Vec<Arc<MultilinearPolynomial<F>>>,
        coefficients: &[F],
        streaming_context: Option<Arc<StreamingRLCContext<F>>>,
    ) -> Self {
        use crate::utils::small_scalar::SmallScalar;

        debug_assert_eq!(polynomials.len(), coefficients.len());
        debug_assert_eq!(polynomials.len(), poly_ids.len());

        // Collect indices of dense (non-one-hot) polynomials.
        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)))
            .map(|(i, _)| i)
            .collect();

        // Eagerly materialize the dense linear combination (if any).
        let dense_rlc = if dense_indices.is_empty() {
            vec![]
        } else {
            let max_len = dense_indices
                .iter()
                .map(|&i| polynomials[i].as_ref().original_len())
                .max()
                .unwrap();

            (0..max_len)
                .into_par_iter()
                .map(|idx| {
                    let mut acc = F::zero();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        if idx < poly.original_len() {
                            match poly {
                                MultilinearPolynomial::LargeScalars(p) => {
                                    acc += p.Z[idx] * coeff;
                                }
                                MultilinearPolynomial::U8Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U16Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U32Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U64Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::I64Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::I128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::S128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                _ => unreachable!(
                                    "unexpected polynomial variant in RLC linear_combination"
                                ),
                            }
                        }
                    }
                    acc
                })
                .collect()
        };

        // Store one-hot polynomials lazily.
        let mut one_hot_rlc = Vec::new();
        for (i, poly) in polynomials.iter().enumerate() {
            if matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)) {
                one_hot_rlc.push((coefficients[i], poly.clone()));
            }
        }

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
        advice_poly_map: std::collections::HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
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
                    if let Some(poly) = advice_poly_map.get(poly_id) {
                        advice_polys.push((*coeff, poly.clone()));
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
        use crate::utils::small_scalar::SmallScalar;

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
                            match poly {
                                MultilinearPolynomial::U8Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U16Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U32Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::S128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::LargeScalars(p) => {
                                    acc += p.Z[i] * coeff;
                                }
                                _ => unreachable!(),
                            }
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

        // Compute the vector-matrix product for advice polynomials (streaming context)
        // Advice polynomials are small (fewer variables) and stored directly
        if let Some(ctx) = &self.streaming_context {
            for (coeff, advice_poly) in ctx.advice_polys.iter() {
                // Advice polynomial has fewer coefficients than main polynomials.
                // Treat it as occupying the "bottom-left" corner of the matrix.
                // With main matrix dimensions, advice fits in the first few rows.
                let advice_len = advice_poly.original_len();
                let advice_rows = advice_len.div_ceil(num_columns);

                // Only the first `advice_rows` left_vec elements contribute
                for row_idx in 0..advice_rows.min(left_vec.len()) {
                    let row_start = row_idx * num_columns;
                    let row_end = ((row_idx + 1) * num_columns).min(advice_len);
                    let row_slice_len = row_end - row_start;

                    // Get coefficient slice for this row
                    for col_idx in 0..row_slice_len {
                        let coeff_idx = row_start + col_idx;
                        let advice_val = advice_poly.get_coeff(coeff_idx);
                        result[col_idx] += left_vec[row_idx] * *coeff * advice_val;
                    }
                }
            }
        }

        result
    }

    /// Extract dense polynomial value from a cycle
    #[inline]
    fn extract_dense_value(poly_id: &CommittedPolynomial, cycle: &Cycle) -> F {
        match poly_id {
            CommittedPolynomial::RdInc => {
                let (_, pre_value, post_value) = cycle.rd_write();
                F::from_i128(post_value as i128 - pre_value as i128)
            }
            CommittedPolynomial::RamInc => match cycle.ram_access() {
                tracer::instruction::RAMAccess::Write(write) => {
                    F::from_i128(write.post_value as i128 - write.pre_value as i128)
                }
                tracer::instruction::RAMAccess::Read(_) | tracer::instruction::RAMAccess::NoOp => {
                    F::zero()
                }
            },
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => {
                panic!("One-hot polynomials should not be passed to extract_dense_value")
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not be passed to extract_dense_value")
            }
        }
    }

    /// Extract one-hot index k from a cycle for a given polynomial
    #[inline]
    fn extract_onehot_k(
        poly_id: &CommittedPolynomial,
        cycle: &Cycle,
        preprocessing: &RLCStreamingData,
        one_hot_params: &OneHotParams,
    ) -> Option<usize> {
        match poly_id {
            CommittedPolynomial::InstructionRa(idx) => {
                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let pc = preprocessing.bytecode.get_pc(cycle);
                Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
            }
            CommittedPolynomial::RamRa(idx) => remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.memory_layout,
            )
            .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize),
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                panic!("Dense polynomials should not be passed to extract_onehot_k")
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not be passed to extract_onehot_k")
            }
        }
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

        // Divide rows evenly among threads - one allocation per thread
        let num_threads = rayon::current_num_threads();
        let rows_per_thread = num_rows.div_ceil(num_threads);

        // Pre-extract dense polynomial coefficients
        let dense_coeffs: Vec<_> = ctx.dense_polys.iter().map(|(id, c)| (*id, *c)).collect();
        let onehot_coeffs: Vec<_> = ctx.onehot_polys.iter().map(|(id, c)| (*id, *c)).collect();

        (0..num_rows)
            .collect::<Vec<_>>()
            .par_chunks(rows_per_thread)
            .map(|row_chunk| {
                // One allocation per thread
                let mut acc: Vec<F> = unsafe_allocate_zero_vec(num_columns);

                for &row_idx in row_chunk {
                    let chunk_start = row_idx * num_columns;

                    // Precompute scaled dense coefficients for this row
                    let scaled_dense: Vec<_> = dense_coeffs
                        .iter()
                        .map(|(_, coeff)| left_vec[row_idx] * *coeff)
                        .collect();

                    // Split into valid trace range vs padding range (avoid branch in hot loop)
                    let valid_end = std::cmp::min(chunk_start + num_columns, trace_len);
                    let valid_cols = valid_end.saturating_sub(chunk_start);

                    // Process valid trace elements (no branch needed)
                    for col_idx in 0..valid_cols {
                        let cycle = &trace[chunk_start + col_idx];
                        let mut val = F::zero();

                        // Dense polynomials with precomputed scaling
                        for (i, (poly_id, _)) in dense_coeffs.iter().enumerate() {
                            let dense_val = Self::extract_dense_value(poly_id, cycle);
                            val += scaled_dense[i] * dense_val;
                        }

                        // One-hot polynomials
                        for (poly_id, coeff) in &onehot_coeffs {
                            if let Some(k) = Self::extract_onehot_k(
                                poly_id,
                                cycle,
                                &ctx.preprocessing,
                                &ctx.one_hot_params,
                            ) {
                                let onehot_row = k * num_rows + row_idx;
                                val += left_vec[onehot_row] * *coeff;
                            }
                        }

                        acc[col_idx] += val;
                    }

                    // Process padding (NoOp cycles) - typically rare or none
                    // For NoOp: dense polys (RdInc, RamInc) are always zero,
                    // and one-hot polys contribute consistently:
                    // - InstructionRa/BytecodeRa: Some(0) (fetch NoOp at index/PC 0)
                    // - RamRa: None (no RAM access)
                    // Since these are constant per-row, we skip the per-column loop entirely.
                    #[cfg(test)]
                    if valid_cols < num_columns {
                        // Verify dense polynomials are zero for NoOp
                        for (poly_id, _) in &dense_coeffs {
                            debug_assert_eq!(
                                Self::extract_dense_value(poly_id, &Cycle::NoOp),
                                F::zero(),
                                "Expected zero dense value for NoOp, got non-zero for {:?}",
                                poly_id
                            );
                        }

                        // Verify one-hot polynomials have expected values for NoOp
                        for (poly_id, _) in &onehot_coeffs {
                            let k = Self::extract_onehot_k(
                                poly_id,
                                &Cycle::NoOp,
                                &ctx.preprocessing,
                                &ctx.one_hot_params,
                            );
                            match poly_id {
                                CommittedPolynomial::InstructionRa(_)
                                | CommittedPolynomial::BytecodeRa(_) => {
                                    debug_assert_eq!(
                                        k,
                                        Some(0),
                                        "Expected Some(0) for {:?} on NoOp, got {:?}",
                                        poly_id,
                                        k
                                    );
                                }
                                CommittedPolynomial::RamRa(_) => {
                                    debug_assert_eq!(
                                        k, None,
                                        "Expected None for RamRa on NoOp, got {:?}",
                                        k
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }

                acc
            })
            .reduce(
                || unsafe_allocate_zero_vec(num_columns),
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += *y;
                    }
                    a
                },
            )
    }

    /// Lazy VMV over lazy trace iterator (experimental, re-runs tracer).
    fn lazy_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        lazy_trace: LazyTraceIterator,
        ctx: &StreamingRLCContext<F>,
        T: usize,
    ) -> Vec<F> {
        let num_rows = T / num_columns;

        // Pre-extract coefficients
        let dense_coeffs: Vec<_> = ctx.dense_polys.iter().map(|(id, c)| (*id, *c)).collect();
        let onehot_coeffs: Vec<_> = ctx.onehot_polys.iter().map(|(id, c)| (*id, *c)).collect();

        lazy_trace
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(num_columns)
            .enumerate()
            .par_bridge()
            .map(|(row_idx, chunk)| {
                // Precompute scaled dense coefficients for this row
                let scaled_dense: Vec<_> = dense_coeffs
                    .iter()
                    .map(|(_, coeff)| left_vec[row_idx] * *coeff)
                    .collect();

                // Process columns within chunk
                let chunk_result: Vec<F> = chunk
                    .par_iter()
                    .map(|cycle| {
                        let mut val = F::zero();

                        // Dense polynomials with precomputed scaling
                        for (i, (poly_id, _)) in dense_coeffs.iter().enumerate() {
                            let dense_val = Self::extract_dense_value(poly_id, cycle);
                            val += scaled_dense[i] * dense_val;
                        }

                        // One-hot polynomials
                        for (poly_id, coeff) in &onehot_coeffs {
                            if let Some(k) = Self::extract_onehot_k(
                                poly_id,
                                cycle,
                                &ctx.preprocessing,
                                &ctx.one_hot_params,
                            ) {
                                let onehot_row = k * num_rows + row_idx;
                                val += left_vec[onehot_row] * *coeff;
                            }
                        }

                        val
                    })
                    .collect();

                chunk_result
            })
            .reduce(
                || unsafe_allocate_zero_vec(num_columns),
                |mut acc, chunk_result| {
                    acc.par_iter_mut().zip(chunk_result.par_iter()).for_each(
                        |(acc_val, &chunk_val)| {
                            *acc_val += chunk_val;
                        },
                    );
                    acc
                },
            )
    }
}
