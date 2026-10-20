//! Device cycle/address reductions for advice, bytecode, and program images.
//!
//! `PrecommittedClaimReduction` owns the round permutation and active rounds.
//! CUDA packs the value and equality rows into one allocation, permutes them
//! with that geometry, and binds only active variables. Inactive rounds adjust
//! the scalar multiplicity rather than duplicating coefficients.
//!
//! Unlike the shared host tables used by reference and optimized, the carry
//! parked between stages 6b and 7 owns device buffers. The corresponding cycle
//! and address slots must therefore both use the CUDA carry type.

use crate::ProofSession;
#[cfg(feature = "allocative")]
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::Visitor;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::PrecommittedClaimReduction;
use jolt_field::{Fr, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::marker::PhantomData;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::primitives::reduce_lanes;
use crate::precommitted_reduction::{lsb_permutation, permute_challenges};
use crate::{KernelError, SumcheckKernelError};

pub struct DeviceRowPlan<'a> {
    pub source: &'a DeviceFrVec,
    pub source_row: usize,
    pub permute: bool,
}

pub struct DevicePrecommittedTables<F: JoltField> {
    context: &'static CudaKernelContext,
    packed: DeviceFrVec,
    len: usize,
    scale: F,
    scale_inv: F,
    two_inv: F,
}

impl<F: JoltField> DevicePrecommittedTables<F> {
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
        let totals = reduce_lanes(self.context, partials, 2, blocks)?;
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
}

fn device_readback<F: JoltField>(_error: CudaError) -> SumcheckKernelError<F> {
    SumcheckKernelError::InvariantViolation {
        reason: "CUDA precommitted reduction claim readback failed",
    }
}

fn two_inverse<F: JoltField>() -> Result<F, KernelError<F>> {
    F::from_u64(2)
        .inverse()
        .ok_or(KernelError::InvariantViolation {
            reason: "2 is invertible in any Jolt field (large-prime characteristic)",
        })
}

fn is_active(active_rounds: &[usize], round: usize) -> bool {
    active_rounds.binary_search(&round).is_ok()
}

pub struct DevicePrecommittedReductionCarry<F: JoltField, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: JoltField, R> Allocative for DevicePrecommittedReductionCarry<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

pub struct DeviceCycleReductionKernel<F: JoltField, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: JoltField, R> Allocative for DeviceCycleReductionKernel<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: JoltField, R> DeviceCycleReductionKernel<F, R> {
    pub fn new(reduction: PrecommittedClaimReduction, tables: DevicePrecommittedTables<F>) -> Self {
        Self {
            reduction,
            tables,
            _relation: PhantomData,
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

    pub fn park_carry<RA: 'static>(self, session: &mut ProofSession)
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
            _relation: PhantomData,
        });
    }
}

impl<F: JoltField, R> ProveRounds<F> for DeviceCycleReductionKernel<F, R> {
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

pub struct DeviceAddressReductionKernel<F: JoltField, R> {
    reduction: PrecommittedClaimReduction,
    tables: DevicePrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

#[cfg(feature = "allocative")]
impl<F: JoltField, R> Allocative for DeviceAddressReductionKernel<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: JoltField, R> DeviceAddressReductionKernel<F, R> {
    pub fn new(carry: DevicePrecommittedReductionCarry<F, R>) -> Self {
        Self {
            reduction: carry.reduction,
            tables: carry.tables,
            _relation: PhantomData,
        }
    }

    pub fn final_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        self.tables.final_claim()
    }
}

impl<F: JoltField, R> ProveRounds<F> for DeviceAddressReductionKernel<F, R> {
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

pub fn permuted_eq_point<F: JoltField>(
    reduction: &PrecommittedClaimReduction,
    point: &[F],
) -> Result<Vec<Fr>, CudaError> {
    let permuted = match lsb_permutation(reduction.poly_opening_round_permutation_be()) {
        Some(map) => permute_challenges(point, &map),
        None => point.to_vec(),
    };
    Ok(require_fr_slice(&permuted)?.to_vec())
}

pub fn reclaim_carry<F: JoltField, R: 'static>(
    session: &mut ProofSession,
    missing: &'static str,
) -> Result<DeviceAddressReductionKernel<F, R>, KernelError<F>> {
    session
        .take::<DevicePrecommittedReductionCarry<F, R>>()
        .map(DeviceAddressReductionKernel::new)
        .ok_or(KernelError::InvariantViolation { reason: missing })
}
