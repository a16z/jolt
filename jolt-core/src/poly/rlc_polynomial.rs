use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::read_write_checking::RamReadWriteCheckingProver;
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

/// Streaming context for lazy RLC evaluation
#[derive(Clone, Debug)]
pub struct StreamingRLCContext<F: JoltField> {
    pub dense_polys: Vec<(CommittedPolynomial, F)>,
    pub onehot_polys: Vec<(CommittedPolynomial, F)>,
    pub lazy_trace: LazyTraceIterator,
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
    pub trusted_advice_poly_coeffs: Vec<F>,
    pub trusted_advice_poly_rows: usize,
    pub trusted_advice_poly_columns: usize,
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
            trusted_advice_poly_coeffs: vec![],
            trusted_advice_poly_rows: 0,
            trusted_advice_poly_columns: 0,
        }
    }

    /// O(sqrt(T)) space Vector matrix product constructor for all polys
    /// Allows for just a single pass over the trace
    pub fn new_streaming(
        dense_polys: Vec<(CommittedPolynomial, F)>,
        onehot_polys: Vec<(CommittedPolynomial, F)>,
        lazy_trace: LazyTraceIterator,
        preprocessing: Arc<RLCStreamingData>,
        one_hot_params: &OneHotParams,
        trusted_advice_poly_coeffs: Vec<F>,
        trusted_advice_poly_rows: usize,
        trusted_advice_poly_columns: usize,
    ) -> Self {
        Self {
            dense_rlc: vec![],   // Not materialized in streaming mode
            one_hot_rlc: vec![], // Not materialized in streaming mode
            streaming_context: Some(Arc::new(StreamingRLCContext {
                dense_polys,
                onehot_polys,
                lazy_trace,
                preprocessing,
                one_hot_params: one_hot_params.clone(),
            })),
            trusted_advice_poly_coeffs: trusted_advice_poly_coeffs,
            trusted_advice_poly_rows: trusted_advice_poly_rows,
            trusted_advice_poly_columns: trusted_advice_poly_columns,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn linear_combination(
        poly_ids: Vec<CommittedPolynomial>,
        polynomials: Vec<Arc<MultilinearPolynomial<F>>>,
        coefficients: &[F],
        trusted_advice_poly_coeffs: Vec<F>,
        trusted_advice_poly_rows: usize,
        trusted_advice_poly_columns: usize,
        trusted_advice_gamma: F,
        streaming_context: Option<(LazyTraceIterator, Arc<RLCStreamingData>, OneHotParams)>,
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());
        debug_assert_eq!(poly_ids.len(), coefficients.len());

        let (lazy_trace, preprocessing, one_hot_params) =
            streaming_context.expect("Streaming context must be provided");

        let mut dense_polys = Vec::new();
        let mut onehot_polys = Vec::new();

        for (poly_id, coeff) in poly_ids.iter().zip(coefficients.iter()) {
            match poly_id {
                CommittedPolynomial::RdInc | CommittedPolynomial::RamInc 
                | CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice 
                => {
                    dense_polys.push((*poly_id, *coeff));
                }
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    onehot_polys.push((*poly_id, *coeff));
                }
            }
        }

        let coeffs = trusted_advice_poly_coeffs.iter().map(|c| *c * trusted_advice_gamma).collect();

        Self::new_streaming(
            dense_polys,
            onehot_polys,
            lazy_trace,
            preprocessing,
            &one_hot_params,
            coeffs,
            trusted_advice_poly_rows,
            trusted_advice_poly_columns,
        )
    }

    /// Materializes a streaming RLC polynomial for testing purposes.
    // #[cfg(test)]
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

        tracing::info!("Hereeeeeeeeeee trusted_advice_info={:?}", self.trusted_advice_poly_coeffs);

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
        result.trusted_advice_poly_coeffs = self.trusted_advice_poly_coeffs.clone();
        result.trusted_advice_poly_rows = self.trusted_advice_poly_rows;
        result.trusted_advice_poly_columns = self.trusted_advice_poly_columns;

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
        tracing::info!("Computing vector-matrix product in RLCPolynomial::vector_matrix_product");
        let num_columns = DoryGlobals::get_num_columns();

        // Compute the vector-matrix product for dense submatrix
        let mut result: Vec<F> = if let Some(ctx) = &self.streaming_context {
            // Streaming mode: generate rows on-demand from trace
            self.streaming_vector_matrix_product(left_vec, num_columns, Arc::clone(ctx))
        } else {
            // Linear space mode: use pre-computed dense_rlc
            tracing::info!("Computing vector-matrix product in linear space mode");
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

        let trusted_advice_poly_coeffs = self.trusted_advice_poly_coeffs.clone();
        let ta_columns = self.trusted_advice_poly_columns;
        let ta_rows = self.trusted_advice_poly_rows;
        tracing::info!("Hereeeeee ta_columns={}, ta_rows={}", ta_columns, ta_rows);
        
        // Compute trusted advice contribution for each column
        // Vector-matrix product: result[col] = Σ_row left_vec[row] * TA[row, col]
        // TA[row, col] is stored at flat_index = row * ta_columns + col

        let mut ta_contribution = vec![F::zero(); result.len()];
        for col_index in 0..ta_columns {
            for row_index in 0..ta_rows {
                let flat_index = row_index * ta_columns + col_index;
                ta_contribution[col_index] += trusted_advice_poly_coeffs[flat_index] * left_vec[row_index];
                tracing::info!("flat_index={}, row_index={}, col_index={}, adding {} * {} = {}", flat_index, row_index, col_index, trusted_advice_poly_coeffs[flat_index], left_vec[row_index], trusted_advice_poly_coeffs[flat_index] * left_vec[row_index]);
            }
        }
        // let ta_contribution: Vec<F> = (0..result.len())
        //     .par_iter_mut()
        //     .map(|row_index| {
        //         if row_index >= ta_rows {
        //             F::zero()
        //         } else {
        //             (0..ta_columns)
        //                 .map(|col_index| {
        //                     let flat_index = row_index * ta_columns + col_index;
        //                     tracing::info!("flat_index={}, row_index={}, col_index={}", flat_index, row_index, col_index);
        //                     tracing::info!("addding {} * {} = {}", trusted_advice_poly_coeffs[flat_index], left_vec[row_index], trusted_advice_poly_coeffs[flat_index] * left_vec[row_index]);
        //                     trusted_advice_poly_coeffs[flat_index] * left_vec[row_index]
        //                 })
        //                 .sum::<F>()
        //         }
        //     })
        //     .collect();
        
        // Add trusted advice contribution to result
        for (r, ta) in result.iter_mut().zip(ta_contribution.iter()) {
            *r += *ta;
        }
        let x = result[0];

        // Compute the vector-matrix product for one-hot polynomials (linear space)
        for (coeff, poly) in self.one_hot_rlc.iter() {
            match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => {
                    one_hot.vector_matrix_product(left_vec, *coeff, &mut result);
                }
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            }
        }

        tracing::info!("Result: {:?}, one_hot contribution: {:?}", result, result[0] - x);

        result
    }

    /// Extract dense polynomial value from a cycle
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
                panic!("Trusted or untrusted advice polynomials should not be passed to extract_dense_value")
            }
        }
    }

    /// Extract one-hot index k from a cycle for a given polynomial
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
                panic!("Trusted or untrusted advice polynomials should not be passed to extract_onehot_k")
            }
        }
    }

    /// Streaming VMP implementation that generates rows on-demand from trace.
    /// Achieves O(sqrt(n)) space complexity by lazily generating the witness
    /// Single pass through lazy trace iterator for both dense and one-hot polynomials.
    fn streaming_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        ctx: Arc<StreamingRLCContext<F>>,
    ) -> Vec<F> {
        let T = DoryGlobals::get_T();
        let trace_len = ctx.lazy_trace.clone().count();
        tracing::info!("Hereeee trace_len={}, num_columns={}", trace_len, num_columns);

        let result = ctx
            .lazy_trace
            .clone()
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(num_columns)
            .enumerate()
            .par_bridge()
            .map(|(row_idx, chunk)| {
                // Nested parallelism: process columns within chunk in parallel
                let chunk_result: Vec<F> = chunk
                    .par_iter()
                    .enumerate()
                    .map(|(col_idx, cycle)| {
                        let mut val = F::zero();

                        // Process DENSE POLYNOMIALS (RdInc, RamInc, TrustedAdvice)
                        for (poly_id, coeff) in &ctx.dense_polys {
                            let dense_val = match poly_id {
                                CommittedPolynomial::TrustedAdvice => {
                                    F::zero()
                                }
                                _ => Self::extract_dense_value(poly_id, cycle),
                            };
                            val += left_vec[row_idx] * *coeff * dense_val;
                        }

                        // Process ONE-HOT POLYNOMIALS (InstructionRa, BytecodeRa, RamRa)
                        for (poly_id, coeff) in &ctx.onehot_polys {
                            if let Some(k) = Self::extract_onehot_k(
                                poly_id,
                                cycle,
                                &ctx.preprocessing,
                                &ctx.one_hot_params,
                            ) {
                                let rows_per_k = T / num_columns;
                                let onehot_row = k * rows_per_k + row_idx;
                                val += left_vec[onehot_row] * *coeff;
                            }
                        }

                        val
                    })
                    .collect();

                chunk_result
            })
            .reduce(
                || vec![F::zero(); num_columns],
                |mut acc, chunk_result| {
                    acc.par_iter_mut().zip(chunk_result.par_iter()).for_each(
                        |(acc_val, &chunk_val)| {
                            *acc_val += chunk_val;
                        },
                    );
                    acc
                },
            );

        drop_in_background_thread(ctx);
        result
    }
}
