use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::PrecommittedClaimReduction;
use jolt_field::{Field, Fr};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};

use super::context::{CudaKernelContext, BLOCK};
use super::dense_product::DeviceDenseProduct;
use super::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use crate::precommitted_reduction::{lsb_permutation, permute_challenges};
use crate::{KernelError, SumcheckKernelError};

pub struct DeviceRowPlan<'a> {
    pub source: &'a DeviceFrVec,
    pub source_row: usize,
    pub permute: bool,
}

pub struct DevicePrecommittedTables<F: Field> {
    context: &'static CudaKernelContext,
    packed: DeviceFrVec,
    len: usize,
    aux: usize,
    scale: F,
    scale_inv: F,
    two_inv: F,
}

impl<F: Field> DevicePrecommittedTables<F> {
    pub fn from_rows(
        context: &'static CudaKernelContext,
        reduction: &PrecommittedClaimReduction,
        len: usize,
        rows: &[DeviceRowPlan<'_>],
    ) -> Result<Self, KernelError<F>> {
        let expected = 1usize << reduction.poly_opening_round_permutation_be().len();
        if len != expected {
            return Err(KernelError::TableSizeMismatch {
                table: "cuda precommitted reduction".to_owned(),
                expected,
                got: len,
            });
        }
        if rows.len() < 2 {
            return Err(KernelError::InvariantViolation {
                reason: "a precommitted reduction needs a value row and an eq row",
            });
        }
        let num_vars = reduction.poly_opening_round_permutation_be().len();
        let map = lsb_permutation(reduction.poly_opening_round_permutation_be());
        let new_to_old = map.as_ref().map(|old_to_new| {
            let mut new_to_old = vec![0u32; num_vars];
            for (old_lsb, &new_lsb) in old_to_new.iter().enumerate() {
                new_to_old[new_lsb] = old_lsb as u32;
            }
            new_to_old
        });

        let packed = context.alloc(rows.len() * len)?;
        let mut tables = Self {
            context,
            packed,
            len,
            aux: rows.len() - 2,
            scale: F::one(),
            scale_inv: F::one(),
            two_inv: two_inverse::<F>()?,
        };
        for (destination, row) in rows.iter().enumerate() {
            tables.place_row(row, destination, num_vars, new_to_old.as_deref())?;
        }
        Ok(tables)
    }

    fn place_row(
        &mut self,
        row: &DeviceRowPlan,
        destination: usize,
        num_vars: usize,
        new_to_old: Option<&[u32]>,
    ) -> Result<(), CudaError> {
        let permute = u32::from(row.permute && new_to_old.is_some());
        let identity = [0u32; 1];
        let map = self
            .context
            .upload_u32_slice(new_to_old.unwrap_or(&identity))?;
        let count = CudaKernelContext::count_of(self.len)?;
        let vars = CudaKernelContext::count_of(num_vars)?;
        let source_row = CudaKernelContext::count_of(row.source_row)?;
        let destination_row = CudaKernelContext::count_of(destination)?;
        let mut builder = self
            .context
            .stream()
            .launch_builder(self.context.pcr_place_row());
        let _ = builder.arg(row.source.limbs());
        let _ = builder.arg(&map);
        let _ = builder.arg(&vars);
        let _ = builder.arg(&permute);
        let _ = builder.arg(&source_row);
        let _ = builder.arg(&count);
        let _ = builder.arg(&destination_row);
        let _ = builder.arg(self.packed.limbs_mut());
        // SAFETY: thread `i < table_len` writes only
        // `out[dst_row * table_len + i]`, inside `packed`'s
        // `rows * table_len * LIMBS` u64s because `dst_row < rows`. It reads
        // `src[src_row * table_len + source]` where `source` is either `i` or a
        // bit permutation of `i` built from `new_lsb_to_old_lsb`, whose entries
        // are a permutation of `0..num_vars` (`lsb_permutation` returns a
        // bijection), so `source < table_len` and the read is inside `src`'s
        // `(src_row + 1) * table_len` elements. `map` holds `num_vars` entries
        // when `permute` is set, and is not read otherwise.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        Ok(())
    }

    fn round_evals(&self) -> Result<(F, F), CudaError> {
        let half = self.len / 2;
        let count = CudaKernelContext::count_of(half)?;
        let table_len = CudaKernelContext::count_of(self.len)?;
        let blocks = count.div_ceil(BLOCK).max(1);
        let mut partials = self.context.alloc(2 * blocks as usize)?;
        let mut builder = self
            .context
            .stream()
            .launch_builder(self.context.pcr_round());
        let _ = builder.arg(self.packed.limbs());
        let _ = builder.arg(&table_len);
        let _ = builder.arg(&count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `j < half` reads `value[2j]`, `value[2j+1]`,
        // `eq[2j]`, `eq[2j+1]` where `value` is `packed`'s row 0 and `eq` its
        // row 1, both `table_len` elements long, so every index is below
        // `2 * table_len` and inside `packed`. Writes go only to
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, matching the
        // `2 * blocks` allocation. Shared memory is `BLOCK * LIMBS` u64s as
        // `shared_mem_bytes` declares, and `lane_block_reduce` sits outside the
        // `j < half` guard so every thread reaches each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        let totals = DeviceDenseProduct::reduce_lanes(self.context, partials, 2, blocks)?;
        let host = totals.to_host()?;
        let convert = |value: Fr| {
            fr_into(value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        Ok((convert(host[0])?, convert(host[1])?))
    }

    fn round_message(
        &self,
        active: bool,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        if !active {
            return Ok(UnivariatePoly::new(vec![previous_claim * self.two_inv]));
        }
        let (eval_0, eval_2) = self.round_evals()?;
        let eval_1 = previous_claim * self.scale_inv - eval_0;
        let c2 = (eval_0 - eval_1 - eval_1 + eval_2) * self.two_inv;
        let c1 = eval_1 - eval_0 - c2;
        Ok(UnivariatePoly::new(vec![
            eval_0 * self.scale,
            c1 * self.scale,
            c2 * self.scale,
        ]))
    }

    fn bind_round(&mut self, active: bool, challenge: F) -> Result<(), CudaError> {
        if !active {
            self.scale *= self.two_inv;
            self.scale_inv += self.scale_inv;
            return Ok(());
        }
        let bound = self
            .context
            .bind_rows(&self.packed, self.len, require_fr(challenge)?)?;
        self.packed = bound;
        self.len /= 2;
        Ok(())
    }

    pub fn prove_round(
        &mut self,
        active_rounds: &[usize],
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        if let Some(challenge) = bind {
            self.bind_round(is_active(active_rounds, round - 1), challenge)?;
        }
        self.round_message(is_active(active_rounds, round), previous_claim)
    }

    pub fn finish_rounds(
        &mut self,
        active_rounds: &[usize],
        total_rounds: usize,
        bind: F,
    ) -> Result<(), CudaError> {
        self.bind_round(is_active(active_rounds, total_rounds - 1), bind)
    }

    fn rows_to_host(&self, rows: usize) -> Result<Vec<Vec<F>>, SumcheckKernelError<F>> {
        let host = self.packed.to_host().map_err(device_readback)?;
        (0..rows)
            .map(|row| {
                host[row * self.len..(row + 1) * self.len]
                    .iter()
                    .map(|value| {
                        fr_into(*value).ok_or(SumcheckKernelError::InvariantViolation {
                            reason: "CUDA kernels support only the BN254 scalar field",
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub fn intermediate_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        let rows = self.rows_to_host(2)?;
        let product: F = rows[0]
            .iter()
            .zip(&rows[1])
            .map(|(value, eq)| *value * *eq)
            .sum();
        Ok(product * self.scale)
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.len != 1 {
            return Err(SumcheckKernelError::InvariantViolation {
                reason:
                    "precommitted reduction final claim requested before the polynomial is fully bound",
            });
        }
        Ok(())
    }

    pub fn final_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        Ok(self.rows_to_host(1)?[0][0])
    }

    pub fn final_aux_claims(&self) -> Result<Vec<F>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let rows = self.rows_to_host(2 + self.aux)?;
        Ok(rows[2..].iter().map(|row| row[0]).collect())
    }
}

fn device_readback<F: Field>(_error: CudaError) -> SumcheckKernelError<F> {
    SumcheckKernelError::InvariantViolation {
        reason: "CUDA precommitted reduction claim readback failed",
    }
}

fn two_inverse<F: Field>() -> Result<F, KernelError<F>> {
    F::from_u64(2)
        .inverse()
        .ok_or(KernelError::InvariantViolation {
            reason: "2 is invertible in any Jolt field (large-prime characteristic)",
        })
}

fn is_active(active_rounds: &[usize], round: usize) -> bool {
    active_rounds.binary_search(&round).is_ok()
}

pub struct DevicePrecommittedReductionCarry<F: Field, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: std::marker::PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: Field, R> allocative::Allocative for DevicePrecommittedReductionCarry<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

pub struct DeviceCycleReductionKernel<F: Field, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: std::marker::PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: Field, R> allocative::Allocative for DeviceCycleReductionKernel<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field, R> DeviceCycleReductionKernel<F, R> {
    pub fn new(reduction: PrecommittedClaimReduction, tables: DevicePrecommittedTables<F>) -> Self {
        Self {
            reduction,
            tables,
            _relation: std::marker::PhantomData,
        }
    }

    pub fn has_address_phase(&self) -> bool {
        self.reduction.num_address_phase_rounds() > 0
    }

    pub fn scalar_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        if self.has_address_phase() {
            self.tables.intermediate_claim()
        } else {
            self.tables.final_claim()
        }
    }

    pub fn tables(&self) -> &DevicePrecommittedTables<F> {
        &self.tables
    }

    pub fn park_carry<RA: 'static>(self, session: &mut crate::ProofSession)
    where
        F: 'static,
        R: 'static,
    {
        if !self.has_address_phase() {
            return;
        }
        session.park(DevicePrecommittedReductionCarry::<F, RA> {
            reduction: self.reduction,
            tables: self.tables,
            _relation: std::marker::PhantomData,
        });
    }
}

impl<F: Field, R> ProveRounds<F> for DeviceCycleReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.reduction.cycle_phase_total_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.tables
            .prove_round(
                self.reduction.cycle_phase_rounds(),
                bind,
                round,
                previous_claim,
            )
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda precommitted cycle round",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        let total_rounds = self.num_rounds();
        self.tables
            .finish_rounds(self.reduction.cycle_phase_rounds(), total_rounds, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda precommitted cycle bind",
            })
    }
}

pub struct DeviceAddressReductionKernel<F: Field, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: std::marker::PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: Field, R> allocative::Allocative for DeviceAddressReductionKernel<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field, R> DeviceAddressReductionKernel<F, R> {
    pub fn new(carry: DevicePrecommittedReductionCarry<F, R>) -> Self {
        Self {
            reduction: carry.reduction,
            tables: carry.tables,
            _relation: std::marker::PhantomData,
        }
    }

    pub fn final_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        self.tables.final_claim()
    }

    pub fn final_aux_claims(&self) -> Result<Vec<F>, SumcheckKernelError<F>> {
        self.tables.final_aux_claims()
    }
}

impl<F: Field, R> ProveRounds<F> for DeviceAddressReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.reduction.address_phase_total_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.tables
            .prove_round(
                self.reduction.address_phase_rounds(),
                bind,
                round,
                previous_claim,
            )
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda precommitted address round",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        let total_rounds = self.num_rounds();
        self.tables
            .finish_rounds(self.reduction.address_phase_rounds(), total_rounds, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda precommitted address bind",
            })
    }
}

pub fn permuted_eq_point<F: Field>(
    reduction: &PrecommittedClaimReduction,
    point: &[F],
) -> Result<Vec<Fr>, CudaError> {
    let permuted = match lsb_permutation(reduction.poly_opening_round_permutation_be()) {
        Some(map) => permute_challenges(point, &map),
        None => point.to_vec(),
    };
    Ok(require_fr_slice(&permuted)?.to_vec())
}

pub fn scatter_sparse_row(
    context: &'static CudaKernelContext,
    indices: &[u32],
    values: &[Fr],
    table_len: usize,
    row: usize,
    out: &mut DeviceFrVec,
) -> Result<(), CudaError> {
    if indices.is_empty() {
        return Ok(());
    }
    if indices.len() != values.len() {
        return Err(CudaError::LengthMismatch {
            expected: indices.len(),
            got: values.len(),
        });
    }
    let device_indices = context.upload_u32_slice(indices)?;
    let device_values = context.upload(values)?;
    let count = CudaKernelContext::count_of(indices.len())?;
    let len = CudaKernelContext::count_of(table_len)?;
    let row = CudaKernelContext::count_of(row)?;
    let mut builder = context.stream().launch_builder(context.pcr_scatter());
    let _ = builder.arg(&device_indices);
    let _ = builder.arg(device_values.limbs());
    let _ = builder.arg(&count);
    let _ = builder.arg(&len);
    let _ = builder.arg(&row);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: thread `i < count` reads `values[i]` of `device_values`'s `count`
    // elements and writes `out[row * table_len + indices[i]]`. Every index was
    // produced by `address_cycle_to_index(lane, chunk_cycle, lane_capacity,
    // chunk_cycle_len)` with `lane < lane_capacity` and `chunk_cycle <
    // chunk_cycle_len`, so it is below `table_len = lane_capacity *
    // chunk_cycle_len`, and `row` indexes an allocated row of `out`. Indices are
    // pairwise distinct — `for_each_active_lane_value` visits each lane at most
    // once per instruction and each instruction owns a distinct `chunk_cycle` —
    // so no two threads write the same element.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    Ok(())
}

pub fn fold_chunk_weights(
    context: &'static CudaKernelContext,
    chunks: &DeviceFrVec,
    weights: &DeviceFrVec,
    chunk_count: usize,
    table_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(table_len)?;
    let count = CudaKernelContext::count_of(table_len)?;
    let chunks_count = CudaKernelContext::count_of(chunk_count)?;
    let mut builder = context.stream().launch_builder(context.pcr_value_fold());
    let _ = builder.arg(chunks.limbs());
    let _ = builder.arg(weights.limbs());
    let _ = builder.arg(&chunks_count);
    let _ = builder.arg(&count);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: thread `i < table_len` reads `chunks[c * table_len + i]` for
    // `c < chunk_count`, inside `chunks`'s `chunk_count * table_len` elements,
    // and `weights[c]` of its `chunk_count` elements; it writes only `out[i]` of
    // the `table_len` just allocated.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    Ok(out)
}

pub fn lane_cycle_eq(
    context: &'static CudaKernelContext,
    lane_weights: &DeviceFrVec,
    eq_cycle: &DeviceFrVec,
    chunk_cycle_len: usize,
    lane_capacity: usize,
    lane_outer: bool,
    table_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(table_len)?;
    let count = CudaKernelContext::count_of(table_len)?;
    let cycles = CudaKernelContext::count_of(chunk_cycle_len)?;
    let lanes = CudaKernelContext::count_of(lane_capacity)?;
    let outer = u32::from(lane_outer);
    let mut builder = context.stream().launch_builder(context.pcr_lane_eq());
    let _ = builder.arg(lane_weights.limbs());
    let _ = builder.arg(eq_cycle.limbs());
    let _ = builder.arg(&cycles);
    let _ = builder.arg(&lanes);
    let _ = builder.arg(&outer);
    let _ = builder.arg(&count);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: thread `i < table_len` derives `lane < lane_capacity` and
    // `cycle < chunk_cycle_len` from `i` by division and remainder against the
    // interleave the caller declared, and `table_len = lane_capacity *
    // chunk_cycle_len`, so `lane_weights[lane]` and `eq_cycle[cycle]` are inside
    // their `lane_capacity` and `chunk_cycle_len` elements. It writes only
    // `out[i]`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    Ok(out)
}

pub fn shifted_block_eq(
    context: &'static CudaKernelContext,
    challenges_be: &[Fr],
    start_index: usize,
    domain_mask: usize,
    table_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    let challenges = context.upload(challenges_be)?;
    let mut out = context.alloc(table_len)?;
    let count = CudaKernelContext::count_of(table_len)?;
    let num_vars = CudaKernelContext::count_of(challenges_be.len())?;
    let start = CudaKernelContext::count_of(start_index)?;
    let mask = CudaKernelContext::count_of(domain_mask)?;
    let mut builder = context.stream().launch_builder(context.pcr_shift_eq());
    let _ = builder.arg(challenges.limbs());
    let _ = builder.arg(&num_vars);
    let _ = builder.arg(&start);
    let _ = builder.arg(&mask);
    let _ = builder.arg(&count);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: thread `i < table_len` reads `challenges_be[bit]` for
    // `bit < num_vars`, inside the uploaded `num_vars` elements, and writes only
    // `out[i]` of the `table_len` just allocated. `index` is masked to the RAM
    // domain, so the per-bit product visits exactly `num_vars` factors.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    Ok(out)
}

impl<F: Field> DevicePrecommittedTables<F> {
    pub fn from_host_tables(
        context: &'static CudaKernelContext,
        tables: &crate::precommitted_reduction::PrecommittedTables<F>,
    ) -> Result<Self, KernelError<F>> {
        let len = tables.value.len();
        if tables.eq.len() != len || tables.aux.iter().any(|row| row.len() != len) {
            return Err(KernelError::InvariantViolation {
                reason: "a host precommitted carry has rows of differing widths",
            });
        }
        let mut host = Vec::with_capacity((2 + tables.aux.len()) * len);
        for row in std::iter::once(&tables.value)
            .chain(std::iter::once(&tables.eq))
            .chain(tables.aux.iter())
        {
            host.extend_from_slice(require_fr_slice(row.evals())?);
        }
        Ok(Self {
            context,
            packed: context.upload(&host)?,
            len,
            aux: tables.aux.len(),
            scale: tables.scale,
            scale_inv: tables.scale_inv,
            two_inv: two_inverse::<F>()?,
        })
    }
}

pub fn reclaim_carry<F: Field, R: 'static>(
    session: &mut crate::ProofSession,
    missing: &'static str,
) -> Result<DeviceAddressReductionKernel<F, R>, KernelError<F>> {
    if let Some(carry) = session.take::<DevicePrecommittedReductionCarry<F, R>>() {
        return Ok(DeviceAddressReductionKernel::new(carry));
    }
    let host = session
        .take::<crate::precommitted_reduction::PrecommittedReductionCarry<F, R>>()
        .ok_or(KernelError::InvariantViolation { reason: missing })?;
    let context = crate::cuda::require_context::<F>()?;
    let tables = DevicePrecommittedTables::from_host_tables(context, &host.tables)?;
    Ok(DeviceAddressReductionKernel::new(
        DevicePrecommittedReductionCarry {
            reduction: host.reduction,
            tables,
            _relation: std::marker::PhantomData,
        },
    ))
}
