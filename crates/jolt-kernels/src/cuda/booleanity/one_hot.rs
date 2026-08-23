use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};

use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::device_columns::DeviceTraceColumns;
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::split_eq::DeviceSplitEq;

pub const COLLAPSE_AFTER_ROUNDS: usize = 5;

pub const PACKED_BITS: usize = 32;

const LANES: usize = 2;

pub struct DeviceBooleanityRa {
    lookup: Arc<CudaSlice<u64>>,
    pc: Arc<CudaSlice<u32>>,
    ram: Arc<CudaSlice<u32>>,
    tables: DeviceFrVec,
    rho: DeviceFrVec,
    dense: Option<DeviceFrVec>,
    families: [usize; 3],
    addresses: usize,
    chunk_bits: usize,
    cycles: usize,
    rounds_bound: usize,
}

impl DeviceBooleanityRa {
    pub fn from_device<F: Field>(
        context: &CudaKernelContext,
        columns: DeviceTraceColumns,
        cycles: usize,
        chunk_bits: usize,
        families: [usize; 3],
        address_point: &[F],
        gamma: F,
    ) -> Result<Self, CudaError> {
        let polys = families.iter().sum::<usize>();
        if chunk_bits == 0 || address_point.len() != chunk_bits {
            return Err(CudaError::InvariantViolation {
                reason: "a booleanity one-hot family needs one address coordinate per chunk bit",
            });
        }
        if !cycles.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a booleanity one-hot family needs a power-of-two cycle count",
            });
        }
        if polys == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "a booleanity one-hot family needs at least one checked polynomial",
            });
        }
        if chunk_bits * families[1] > PACKED_BITS || chunk_bits * families[2] > PACKED_BITS {
            return Err(CudaError::NotImplemented {
                kernel: "the CUDA booleanity kernels pack the bytecode PC and the remapped RAM \
                         word address into one 32-bit word each",
            });
        }
        if columns.pc.len() < cycles
            || columns.ram.len() < cycles
            || columns.lookup.len() < 2 * cycles
        {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: columns.pc.len(),
            });
        }

        let addresses = 1usize << chunk_bits;
        let base = Self::eq_table(context, chunk_bits, address_point)?;

        let mut powers = Vec::with_capacity(polys);
        let mut power = F::one();
        for _ in 0..polys {
            powers.push(power);
            power *= gamma;
        }
        let rho = context.upload(require_fr_slice(&powers)?)?;

        let mut tables = context.alloc(polys * addresses)?;
        let count = CudaKernelContext::count_of(polys * addresses)?;
        let poly_count = CudaKernelContext::count_of(polys)?;
        let address_count = CudaKernelContext::count_of(addresses)?;
        let mut builder = context.stream().launch_builder(context.brc_tables_init());
        let _ = builder.arg(base.limbs());
        let _ = builder.arg(rho.limbs());
        let _ = builder.arg(tables.limbs_mut());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&address_count);
        // SAFETY: thread `idx < polys * addresses` reads `base[idx % addresses]` and
        // `rho[idx / addresses]` — inside `base`'s `addresses` and `rho`'s `polys`
        // elements — and writes only `out[idx]`, of `polys * addresses`. Index sets
        // are disjoint across threads and `out` is a fresh allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self {
            lookup: columns.lookup,
            pc: columns.pc,
            ram: columns.ram,
            tables,
            rho,
            dense: None,
            families,
            addresses,
            chunk_bits,
            cycles,
            rounds_bound: 0,
        })
    }

    fn eq_table<F: Field>(
        context: &CudaKernelContext,
        chunk_bits: usize,
        address_point: &[F],
    ) -> Result<DeviceFrVec, CudaError> {
        let point = context.upload(require_fr_slice(address_point)?)?;
        let mut table = context.upload(&[Fr::from(1u64)])?;
        for level in 0..chunk_bits {
            let prev_len = 1usize << level;
            let mut next = context.alloc(prev_len * 2)?;
            let count = CudaKernelContext::count_of(prev_len)?;
            let polys = CudaKernelContext::count_of(1)?;
            let prev = CudaKernelContext::count_of(prev_len)?;
            let level_arg = CudaKernelContext::count_of(level)?;
            let bits = CudaKernelContext::count_of(chunk_bits)?;
            let mut builder = context.stream().launch_builder(context.irv_eq_double());
            let _ = builder.arg(table.limbs());
            let _ = builder.arg(point.limbs());
            let _ = builder.arg(next.limbs_mut());
            let _ = builder.arg(&polys);
            let _ = builder.arg(&prev);
            let _ = builder.arg(&level_arg);
            let _ = builder.arg(&bits);
            // SAFETY: thread `idx < prev_len` reads `in[idx]` and `point[level]` —
            // inside `point`'s `chunk_bits` because `level < chunk_bits` — and writes
            // `out[2 * idx]` and `out[2 * idx + 1]`, both inside `out`'s `2 * prev_len`.
            // Index sets are disjoint across threads and `out` is a fresh allocation.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            table = next;
        }
        Ok(table)
    }

    fn from_dense(
        context: &CudaKernelContext,
        columns: &[Vec<Fr>],
        rho: &DeviceFrVec,
        families: [usize; 3],
        chunk_bits: usize,
        tail_rounds: usize,
    ) -> Result<Self, CudaError> {
        let cycles = 1usize << tail_rounds;
        let polys = families.iter().sum::<usize>();
        if columns.len() != polys || columns.iter().any(|column| column.len() != cycles) {
            return Err(CudaError::LengthMismatch {
                expected: polys,
                got: columns.len(),
            });
        }
        let mut flat = Vec::with_capacity(polys * cycles);
        for column in columns {
            flat.extend_from_slice(column);
        }
        Ok(Self {
            lookup: Arc::new(context.alloc_u64(0)?),
            pc: Arc::new(context.alloc_u32(0)?),
            ram: Arc::new(context.alloc_u32(0)?),
            tables: context.alloc(0)?,
            rho: rho.try_clone()?,
            dense: Some(context.upload(&flat)?),
            families,
            addresses: 1usize << chunk_bits,
            chunk_bits,
            cycles,
            rounds_bound: 0,
        })
    }

    fn window_scalars(&self, context: &CudaKernelContext) -> Result<Vec<Fr>, CudaError> {
        if self.len() != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len(),
            });
        }
        match &self.dense {
            Some(dense) => dense.to_host(),
            None => self.gather(context)?.to_host(),
        }
    }

    fn rho(&self) -> &DeviceFrVec {
        &self.rho
    }

    pub fn polys(&self) -> usize {
        self.families.iter().sum()
    }

    pub const fn len(&self) -> usize {
        self.cycles >> self.rounds_bound
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the collapse assertion is the round-for-round test's guard against a \
                      config that never leaves the sparse form"
        )
    )]
    pub const fn is_collapsed(&self) -> bool {
        self.dense.is_some()
    }

    const fn slots(&self) -> usize {
        1usize << self.rounds_bound
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the per-round coefficient view is the round-for-round test's entry point"
        )
    )]
    pub fn coefficients(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        match &self.dense {
            Some(dense) => dense.try_clone(),
            None => self.gather(context),
        }
    }

    fn gather(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let polys = self.polys();
        let len = self.len();
        let mut out = context.alloc(polys * len)?;
        if len == 0 {
            return Ok(out);
        }

        let instruction = CudaKernelContext::count_of(self.families[0])?;
        let bytecode = CudaKernelContext::count_of(self.families[1])?;
        let ram = CudaKernelContext::count_of(self.families[2])?;
        let addresses = CudaKernelContext::count_of(self.addresses)?;
        let slots = CudaKernelContext::count_of(self.slots())?;
        let bits = CudaKernelContext::count_of(self.chunk_bits)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.brc_gather());
        let _ = builder.arg(self.lookup.as_ref());
        let _ = builder.arg(self.pc.as_ref());
        let _ = builder.arg(self.ram.as_ref());
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(&instruction);
        let _ = builder.arg(&bytecode);
        let _ = builder.arg(&ram);
        let _ = builder.arg(&addresses);
        let _ = builder.arg(&slots);
        let _ = builder.arg(&bits);
        let _ = builder.arg(out.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `(p = blockIdx.y < polys, j < len)` reads the source column
        // its family selects at cycle `j * slots + s` for `s < slots` — every such
        // index is below `len * slots == cycles`, so inside `lookup`'s `2 * cycles`
        // u64s and `pc`/`ram`'s `cycles` u32s — and, only for a cycle whose word is
        // not the cold sentinel, `tables[p * slots * addresses + s * addresses + a]`
        // with `a` masked below `addresses`, inside `tables`'s
        // `polys * slots * addresses`. It writes only `out[p * len + j]`, one slot per
        // (poly, cycle) pair, inside `out`'s `polys * len`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    count.div_ceil(BLOCK).max(1),
                    CudaKernelContext::count_of(polys)?,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(out)
    }

    fn split_tables(
        &mut self,
        context: &CudaKernelContext,
        challenge: Fr,
    ) -> Result<(), CudaError> {
        let polys = self.polys();
        let eq_zero = context.upload(&[Fr::from(1u64) - challenge])?;
        let eq_one = context.upload(&[challenge])?;
        let len = self.slots() * self.addresses;
        let mut next = context.alloc(polys * len * 2)?;
        let count = CudaKernelContext::count_of(polys * len)?;
        let poly_count = CudaKernelContext::count_of(polys)?;
        let width = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.irv_tables_split());
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(eq_zero.limbs());
        let _ = builder.arg(eq_one.limbs());
        let _ = builder.arg(next.limbs_mut());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&width);
        // SAFETY: thread `idx < polys * len` reads `in[idx]` plus the two
        // single-element challenges, and writes `out[p * 2 * len + i]` and
        // `out[p * 2 * len + len + i]` for `p = idx / len`, `i = idx % len` —
        // disjoint across threads and inside `out`'s `2 * polys * len`. `out` is a
        // fresh allocation, so no thread reads another's write.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        self.tables = next;
        Ok(())
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if let Some(dense) = &self.dense {
            let bound = context.bind_rows(dense, self.len(), challenge)?;
            self.dense = Some(bound);
            self.rounds_bound += 1;
            return Ok(());
        }
        if self.len() < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        self.split_tables(context, challenge)?;
        self.rounds_bound += 1;
        if self.rounds_bound >= COLLAPSE_AFTER_ROUNDS && self.len() > 1 {
            self.dense = Some(self.gather(context)?);
            self.tables = context.alloc(0)?;
            self.lookup = Arc::new(context.alloc_u64(0)?);
            self.pc = Arc::new(context.alloc_u32(0)?);
            self.ram = Arc::new(context.alloc_u32(0)?);
        }
        Ok(())
    }

    pub fn final_claims<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        if self.len() != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len(),
            });
        }
        let values = match &self.dense {
            Some(dense) => dense.to_host()?,
            None => self.gather(context)?.to_host()?,
        };
        values
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub fn round_coefficients<F: Field>(
        &self,
        context: &CudaKernelContext,
        eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F), CudaError> {
        let half = self.len() / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        let e_in_len = eq.e_in_len();
        if eq.e_out_current().len() * e_in_len != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: eq.e_out_current().len() * e_in_len,
            });
        }

        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let polys = CudaKernelContext::count_of(self.polys())?;
        let e_in_arg = CudaKernelContext::count_of(e_in_len)?;
        let num_x_in_bits = e_in_len.max(1).ilog2();
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        };

        if let Some(dense) = &self.dense {
            let mut builder = context.stream().launch_builder(context.brc_message_dense());
            let _ = builder.arg(dense.limbs());
            let _ = builder.arg(self.rho.limbs());
            let _ = builder.arg(&polys);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(&e_in_arg);
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&num_x_in_bits);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread `g < half` reads `dense[p * 2 * half + 2g]` and
            // `dense[p * 2 * half + 2g + 1]` for every `p < polys` — inside `dense`'s
            // `polys * 2 * half` elements — plus `rho[p]`, and `e_in[g & mask]` and
            // `e_out[g >> num_x_in_bits]`, both bounded because
            // `e_out.len() * e_in.len() == half` is checked above. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of
            // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
            // `shared_mem_bytes`, and the block reduction sits outside the `g < half`
            // guard so every thread reaches each `__syncthreads()`.
            let _ = unsafe { builder.launch(config) }?;
        } else {
            let instruction = CudaKernelContext::count_of(self.families[0])?;
            let bytecode = CudaKernelContext::count_of(self.families[1])?;
            let ram = CudaKernelContext::count_of(self.families[2])?;
            let addresses = CudaKernelContext::count_of(self.addresses)?;
            let slots = CudaKernelContext::count_of(self.slots())?;
            let bits = CudaKernelContext::count_of(self.chunk_bits)?;
            let mut builder = context
                .stream()
                .launch_builder(context.brc_message_sparse());
            let _ = builder.arg(self.lookup.as_ref());
            let _ = builder.arg(self.pc.as_ref());
            let _ = builder.arg(self.ram.as_ref());
            let _ = builder.arg(self.tables.limbs());
            let _ = builder.arg(self.rho.limbs());
            let _ = builder.arg(&instruction);
            let _ = builder.arg(&bytecode);
            let _ = builder.arg(&ram);
            let _ = builder.arg(&addresses);
            let _ = builder.arg(&slots);
            let _ = builder.arg(&bits);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(&e_in_arg);
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&num_x_in_bits);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread `g < half` reads each family's source column at cycle
            // indices `2 * g * slots + t` for `t < 2 * slots`, all below
            // `2 * half * slots == cycles`, so inside `lookup`'s `2 * cycles` u64s and
            // `pc`/`ram`'s `cycles` u32s; and `tables` at
            // `p * slots * addresses + s * addresses + a` with `a` masked below
            // `addresses` and `p < polys`, inside `tables`'s
            // `polys * slots * addresses`. `rho`, `e_in` and `e_out` are indexed as in
            // the dense arm, bounded by the same check. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of
            // `2 * blocks`. Shared memory matches `shared_mem_bytes` and the block
            // reduction sits outside the `g < half` guard.
            let _ = unsafe { builder.launch(config) }?;
        }

        let totals = crate::cuda::common::primitives::reduce_lanes(
            context,
            partials,
            CudaKernelContext::count_of(LANES)?,
            blocks,
        )?;
        let host = totals.to_host()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        let constant = fr_into(host[0]).ok_or_else(unsupported)?;
        let quadratic = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((constant, quadratic))
    }
}

pub(crate) struct BooleanityShard<F: Field> {
    pub(crate) ordinal: usize,
    pub(crate) one_hot: DeviceBooleanityRa,
    pub(crate) eq: DeviceSplitEq<F>,
}

pub(crate) struct ShardedBooleanityRa<F: Field> {
    shards: Vec<BooleanityShard<F>>,
    collapsed: Option<DeviceBooleanityRa>,
    families: [usize; 3],
    chunk_bits: usize,
    local_rounds: usize,
    tail_rounds: usize,
}

impl<F: Field> ShardedBooleanityRa<F> {
    pub(crate) fn new(shards: Vec<BooleanityShard<F>>, log_t: usize) -> Result<Self, CudaError> {
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded booleanity family needs a power-of-two shard count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        let (families, chunk_bits) = shards
            .first()
            .map(|shard| (shard.one_hot.families, shard.one_hot.chunk_bits))
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded booleanity family needs at least one shard",
            })?;
        if tail_rounds > log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded booleanity family cannot split more windows than cycle rounds",
            });
        }
        if count == 1 {
            let shard = shards
                .into_iter()
                .next()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a single-shard booleanity family lost its state",
                })?;
            return Ok(Self {
                shards: Vec::new(),
                collapsed: Some(shard.one_hot),
                families,
                chunk_bits,
                local_rounds: log_t,
                tail_rounds: 0,
            });
        }
        Ok(Self {
            shards,
            collapsed: None,
            families,
            chunk_bits,
            local_rounds: log_t - tail_rounds,
            tail_rounds,
        })
    }

    pub(crate) fn round_coefficients(
        &self,
        whole_eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F), CudaError> {
        if let Some(collapsed) = &self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return collapsed.round_coefficients(context, whole_eq);
        }
        let tasks: Vec<DeviceTask<'_, (F, F), CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, (F, F), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.one_hot.round_coefficients(context, &shard.eq)
                });
                task
            })
            .collect();
        let mut total = (F::zero(), F::zero());
        for part in fan_out(tasks)? {
            total.0 += part.0;
            total.1 += part.1;
        }
        Ok(total)
    }

    pub(crate) fn bind(&mut self, challenge: F, bound: usize) -> Result<(), CudaError> {
        if let Some(collapsed) = &mut self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return collapsed.bind(context, challenge);
        }
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.one_hot.bind(context, challenge)?;
                    shard.eq.bind(challenge);
                    Ok(())
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if bound + 1 == self.local_rounds {
            self.collapse()?;
        }
        Ok(())
    }

    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let shards = std::mem::take(&mut self.shards);
        let tasks: Vec<DeviceTask<'_, Vec<Fr>, CudaError>> = shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, Vec<Fr>, CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.one_hot.window_scalars(context)
                });
                task
            })
            .collect();
        let mut columns: Vec<Vec<Fr>> = Vec::new();
        for scalars in fan_out(tasks)? {
            if columns.is_empty() {
                columns = scalars.iter().map(|_| Vec::new()).collect();
            }
            if scalars.len() != columns.len() {
                return Err(CudaError::LengthMismatch {
                    expected: columns.len(),
                    got: scalars.len(),
                });
            }
            for (column, value) in columns.iter_mut().zip(&scalars) {
                column.push(*value);
            }
        }
        let rho = shards.first().map(|shard| shard.one_hot.rho()).ok_or(
            CudaError::InvariantViolation {
                reason: "a sharded booleanity family lost its shards before collapsing",
            },
        )?;
        self.collapsed = Some(DeviceBooleanityRa::from_dense(
            context,
            &columns,
            rho,
            self.families,
            self.chunk_bits,
            self.tail_rounds,
        )?);
        Ok(())
    }

    pub(crate) fn final_claims(&self) -> Result<Vec<F>, CudaError> {
        let context = context_for(0).ok_or(absent())?;
        self.collapsed
            .as_ref()
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded booleanity family was asked for claims before its tail rounds",
            })?
            .final_claims(context)
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a sharded booleanity window names an absent device",
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
    use proptest::prelude::*;

    use std::sync::Arc;

    use super::{DeviceBooleanityRa, COLLAPSE_AFTER_ROUNDS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::device_columns::DeviceTraceColumns;
    use crate::cuda::common::one_hot_witness::PackedColumns;
    use crate::cuda::common::pack::COLD;
    use crate::cuda::common::split_eq::DeviceSplitEq;
    use crate::cuda::common::testing::{arb_point, fr};

    #[test]
    fn sharded_booleanity_ra_matches_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        const LOG_T: usize = 8;
        const CHUNK_BITS: usize = 3;
        const FAMILIES: [usize; 3] = [2, 1, 1];
        let cycles = 1usize << LOG_T;
        let gamma = fr(37);
        let all = columns(0xB001, cycles, 5, 6, 4);
        let address_point: Vec<Fr> = (0..CHUNK_BITS).map(|i| fr(13 + 5 * i as u64)).collect();
        let cycle_point: Vec<Fr> = (0..LOG_T).map(|i| fr(47 + 9 * i as u64)).collect();
        let scaling = fr(101);

        let build = |base: usize, len: usize| {
            let window = DeviceTraceColumns {
                lookup: Arc::new(
                    context
                        .upload_u64_slice(&all.lookup[base * 2..(base + len) * 2])
                        .expect("upload lookup window"),
                ),
                pc: Arc::new(
                    context
                        .upload_u32_slice(&all.pc[base..base + len])
                        .expect("upload pc window"),
                ),
                ram: Arc::new(
                    context
                        .upload_u32_slice(&all.ram[base..base + len])
                        .expect("upload ram window"),
                ),
            };
            DeviceBooleanityRa::from_device(
                context,
                window,
                len,
                CHUNK_BITS,
                FAMILIES,
                &address_point,
                gamma,
            )
            .expect("window booleanity family")
        };

        for shards in [2usize, 4] {
            let mut expected = build(0, cycles);
            let mut expected_eq = DeviceSplitEq::<Fr>::new_with_scaling(
                context,
                &cycle_point,
                BindingOrder::LowToHigh,
                scaling,
            )
            .expect("whole split-eq");
            let len = cycles / shards;
            let windows: Vec<super::BooleanityShard<Fr>> = (0..shards)
                .map(|shard| super::BooleanityShard {
                    ordinal: 0,
                    one_hot: build(shard * len, len),
                    eq: DeviceSplitEq::<Fr>::new_window_with_scaling(
                        context,
                        &cycle_point,
                        BindingOrder::LowToHigh,
                        scaling,
                        shard,
                        shards,
                    )
                    .expect("window split-eq"),
                })
                .collect();
            let mut got =
                super::ShardedBooleanityRa::new(windows, LOG_T).expect("sharded booleanity");
            let mut got_eq = DeviceSplitEq::<Fr>::new_with_scaling(
                context,
                &cycle_point,
                BindingOrder::LowToHigh,
                scaling,
            )
            .expect("tail split-eq");

            for round in 0..LOG_T {
                let want: (Fr, Fr) = expected
                    .round_coefficients(context, &expected_eq)
                    .expect("whole round coefficients");
                let have: (Fr, Fr) = got
                    .round_coefficients(&got_eq)
                    .expect("sharded round coefficients");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: a booleanity window's eq-weighted mass sums \
                     must add to the whole domain's",
                );
                let challenge = fr(700 + 13 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                expected_eq.bind(challenge);
                got.bind(challenge, round).expect("sharded bind");
                got_eq.bind(challenge);
            }

            let want: Vec<Fr> = expected.final_claims(context).expect("whole final claims");
            let have: Vec<Fr> = got.final_claims().expect("sharded final claims");
            assert_eq!(have, want, "shards={shards}: the final claims diverged");
            assert_eq!(want.len(), FAMILIES.iter().sum::<usize>());
            assert_ne!(
                want.first().copied(),
                Some(Fr::from_u64(0)),
                "a degenerate fixture would hide a divergence",
            );
        }
    }

    fn mix(seed: u64, cycle: usize, salt: u64) -> u64 {
        let value = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + salt);
        value.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (value >> 29)
    }

    fn columns(
        seed: u64,
        cycles: usize,
        bytecode_span: u32,
        ram_span: u32,
        cold_every: usize,
    ) -> PackedColumns {
        let mut lookup = Vec::with_capacity(2 * cycles);
        let mut pc = Vec::with_capacity(cycles);
        let mut ram = Vec::with_capacity(cycles);
        for cycle in 0..cycles {
            lookup.push(mix(seed, cycle, 1));
            lookup.push(mix(seed, cycle, 2));
            pc.push(if cycle % cold_every == 1 {
                COLD
            } else {
                (mix(seed, cycle, 3) % u64::from(bytecode_span)) as u32
            });
            ram.push(if cycle.is_multiple_of(cold_every) {
                COLD
            } else {
                (mix(seed, cycle, 4) % u64::from(ram_span)) as u32
            });
        }
        PackedColumns { lookup, pc, ram }
    }

    fn expected_tables(
        columns: &PackedColumns,
        cycles: usize,
        chunk_bits: usize,
        families: [usize; 3],
        address_point: &[Fr],
        gamma: Fr,
    ) -> Vec<Polynomial<Fr>> {
        let table = EqPolynomial::<Fr>::evals(address_point, None);
        let addresses = 1usize << chunk_bits;
        let mut polys = Vec::new();
        let mut rho = Fr::from_u64(1);
        for (family, count) in families.into_iter().enumerate() {
            for local in 0..count {
                let shift = chunk_bits * (count - 1 - local);
                let evals = (0..cycles)
                    .map(|cycle| {
                        let index = match family {
                            0 => {
                                let wide = u128::from(columns.lookup[2 * cycle])
                                    | (u128::from(columns.lookup[2 * cycle + 1]) << 64);
                                Some(((wide >> shift) as usize) & (addresses - 1))
                            }
                            1 => (columns.pc[cycle] != COLD)
                                .then(|| ((columns.pc[cycle] >> shift) as usize) & (addresses - 1)),
                            _ => (columns.ram[cycle] != COLD).then(|| {
                                ((columns.ram[cycle] >> shift) as usize) & (addresses - 1)
                            }),
                        };
                        index.map_or_else(|| Fr::from_u64(0), |index| rho * table[index])
                    })
                    .collect();
                polys.push(Polynomial::new(evals));
                rho *= gamma;
            }
        }
        polys
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn device_booleanity_ra_matches_cpu_round_for_round(
            log_t in (COLLAPSE_AFTER_ROUNDS + 1)..9usize,
            chunk_bits in prop::sample::select(vec![4usize, 8]),
            bytecode_polys in 1usize..3,
            ram_polys in 1usize..3,
            cold_every in 2usize..5,
            seed in any::<u64>(),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(8),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            if chunk_bits * bytecode_polys > 32 || chunk_bits * ram_polys > 32 {
                return Ok(());
            }
            let cycles = 1usize << log_t;
            let families = [128 / chunk_bits, bytecode_polys, ram_polys];
            let point: Vec<Fr> = (0..chunk_bits)
                .map(|i| fr(seed ^ (i as u64 * 37 + 11)))
                .collect();
            let bytecode_span = 1u32 << (chunk_bits * bytecode_polys - 1);
            let ram_span = 1u32 << (chunk_bits * ram_polys - 1);
            let packed = columns(seed, cycles, bytecode_span, ram_span, cold_every);

            let mut expected =
                expected_tables(&packed, cycles, chunk_bits, families, &point, gamma);
            let uploaded = DeviceTraceColumns {
                lookup: Arc::new(
                    context.upload_u64_slice(&packed.lookup).expect("upload lookup"),
                ),
                pc: Arc::new(context.upload_u32_slice(&packed.pc).expect("upload pc")),
                ram: Arc::new(context.upload_u32_slice(&packed.ram).expect("upload ram")),
            };
            let mut got = DeviceBooleanityRa::from_device(
                context, uploaded, cycles, chunk_bits, families, &point, gamma,
            )
            .expect("device booleanity one-hot family");

            for (round, &challenge) in challenges.iter().take(log_t).enumerate() {
                let coefficients = got
                    .coefficients(context)
                    .expect("gather coefficients")
                    .to_host()
                    .expect("download coefficients");
                let len = got.len();
                for (p, poly) in expected.iter().enumerate() {
                    prop_assert_eq!(
                        &coefficients[p * len..(p + 1) * len],
                        &poly.evals()[..len],
                        "poly {} diverged at round {}", p, round
                    );
                }
                for poly in &mut expected {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                got.bind(context, challenge).expect("device bind");
            }

            prop_assert_eq!(
                got.final_claims::<Fr>(context).expect("device final claims"),
                expected
                    .iter()
                    .map(|poly| poly.evals()[0])
                    .collect::<Vec<Fr>>(),
                "final claims diverged"
            );
            prop_assert!(got.is_collapsed(), "the family never collapsed to dense arrays");
        }
    }
}
