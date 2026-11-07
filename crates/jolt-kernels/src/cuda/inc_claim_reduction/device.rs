use std::sync::Arc;

#[cfg(feature = "allocative")]
use allocative::{Allocative, Key, Visitor};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Fr, JoltField};
use jolt_witness::backend::cuda::DeviceMemoryColumns;

use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_limbs, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::optimized::inc_claim_reduction::PairedEq;

const COLUMNS: usize = 2;
const TERMS: usize = 4;
const CONSOLIDATE_AT: usize = 1 << 12;

enum Weights {
    Split { lo: DeviceFrVec, hi: DeviceFrVec },
    Dense(DeviceFrVec),
}

impl Weights {
    fn new<F: JoltField>(
        context: &CudaKernelContext,
        window: &CycleWindow,
        points: [&[F]; TERMS],
        gamma: F,
    ) -> Result<Self, CudaError> {
        let gamma_squared = gamma * gamma;
        let pairs = [
            PairedEq::new(points[0], F::one(), points[1], gamma),
            PairedEq::new(points[2], gamma_squared, points[3], gamma_squared * gamma),
        ];
        let mut low = Vec::new();
        let mut high = Vec::new();
        for pair in pairs {
            let PairedEq::Split { lo1, hi1, lo2, hi2 } = pair else {
                return Err(invariant("increment weights require a nonempty split"));
            };
            for (lo, hi) in [(lo1, hi1), (lo2, hi2)] {
                if !window.start.is_multiple_of(lo.len()) || !window.len.is_multiple_of(lo.len()) {
                    return Err(invariant("increment window splits a low equality factor"));
                }
                let start = window.start / lo.len();
                let len = window.len / lo.len();
                let part = hi.get(start..start + len).ok_or(invariant(
                    "increment window exceeds the high equality factor",
                ))?;
                low.extend(lo);
                high.extend_from_slice(part);
            }
        }
        Ok(Self::Split {
            lo: context.upload(require_fr_slice(&low)?)?,
            hi: context.upload(require_fr_slice(&high)?)?,
        })
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        match self {
            Self::Split { lo, hi } => {
                let lo_len = lo.len() / TERMS;
                *lo = context.bind_rows(lo, lo_len, challenge)?;
                if lo_len == 2 {
                    let hi_len = CudaKernelContext::count_of(hi.len() / TERMS)?;
                    let mut values = context.alloc(COLUMNS * hi_len as usize)?;
                    let mut builder = context.stream().launch_builder(&context.inc_paired_weights);
                    let _ = builder
                        .arg(lo.limbs())
                        .arg(hi.limbs())
                        .arg(&hi_len)
                        .arg(values.limbs_mut());
                    // SAFETY: the four low rows each have one bound scalar; hi
                    // has four hi_len rows. Every output in two hi_len rows is
                    // written once, with the launch tail guarded in the kernel.
                    let _ = unsafe { builder.launch(LaunchConfig::for_num_elems(hi_len)) }?;
                    *self = Self::Dense(values);
                }
            }
            Self::Dense(values) => {
                *values = context.bind_rows(values, values.len() / COLUMNS, challenge)?;
            }
        }
        Ok(())
    }

    fn validate(&self, len: usize) -> Result<(), CudaError> {
        let valid = match self {
            Self::Split { lo, hi } => {
                lo.len().is_multiple_of(TERMS)
                    && hi.len().is_multiple_of(TERMS)
                    && (lo.len() / TERMS).is_power_of_two()
                    && lo.len() / TERMS >= 2
                    && (hi.len() / TERMS).is_power_of_two()
                    && lo.len() / TERMS * (hi.len() / TERMS) == len
            }
            Self::Dense(values) => values.len() == COLUMNS * len,
        };
        if !valid {
            return Err(invariant("increment weights do not cover the local domain"));
        }
        Ok(())
    }

    #[cfg(feature = "allocative")]
    fn device_bytes(&self) -> usize {
        match self {
            Self::Split { lo, hi } => lo.device_bytes() + hi.device_bytes(),
            Self::Dense(values) => values.device_bytes(),
        }
    }
}

enum Columns {
    Native {
        atoms: Arc<DeviceMemoryColumns>,
        len: usize,
    },
    Dense(DeviceFrVec),
}

impl Columns {
    fn len(&self) -> usize {
        match self {
            Self::Native { len, .. } => *len,
            Self::Dense(values) => values.len() / COLUMNS,
        }
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        match self {
            Self::Native { atoms, len } => {
                let half = CudaKernelContext::count_of(*len / 2)?;
                let mut values = context.alloc(COLUMNS * half as usize)?;
                let r = fr_limbs(challenge);
                let mut builder = context.stream().launch_builder(&context.inc_paired_bind);
                let _ = builder
                    .arg(&atoms.ram_inc)
                    .arg(&atoms.rd_inc)
                    .arg(&half)
                    .arg(&r[0])
                    .arg(&r[1])
                    .arg(&r[2])
                    .arg(&r[3])
                    .arg(values.limbs_mut());
                // SAFETY: IncrementShard::new checks ownership and two u64
                // words per native value. The kernel reads two values per
                // guarded thread and writes two disjoint rows of half fields.
                let _ = unsafe { builder.launch(LaunchConfig::for_num_elems(half)) }?;
                *self = Self::Dense(values);
            }
            Self::Dense(values) => {
                *values = context.bind_rows(values, values.len() / COLUMNS, challenge)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "allocative")]
    fn device_bytes(&self) -> usize {
        match self {
            Self::Native { atoms, .. } => atoms.device_bytes(),
            Self::Dense(values) => values.device_bytes(),
        }
    }
}

pub(super) struct IncrementShard {
    ordinal: usize,
    columns: Columns,
    weights: Weights,
}

// CUDA allocations do not implement Allocative; visit their owned device bytes.
#[cfg(feature = "allocative")]
impl Allocative for IncrementShard {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            Key::new("device"),
            self.columns.device_bytes() + self.weights.device_bytes(),
        );
        visitor.exit();
    }
}

impl IncrementShard {
    pub(super) fn new<F: JoltField>(
        context: &CudaKernelContext,
        window: &CycleWindow,
        atoms: Arc<DeviceMemoryColumns>,
        points: [&[F]; TERMS],
        gamma: F,
    ) -> Result<Self, CudaError> {
        if !window.len.is_power_of_two() || window.len < 2 {
            return Err(invariant("increment shard requires at least two cycles"));
        }
        let _ = CudaKernelContext::count_of(window.len)?;
        for values in [&atoms.ram_inc, &atoms.rd_inc] {
            context.require_owned(values.ordinal())?;
            if values.len() < 2 * window.len {
                return Err(invariant("increment column is shorter than its window"));
            }
        }
        let weights = Weights::new(context, window, points, gamma)?;
        weights.validate(window.len)?;
        Ok(Self {
            ordinal: context.ordinal(),
            columns: Columns::Native {
                atoms,
                len: window.len,
            },
            weights,
        })
    }

    pub(super) fn message(&self) -> Result<[Fr; COLUMNS], CudaError> {
        let context = device(self.ordinal)?;
        let len = CudaKernelContext::count_of(self.columns.len())?;
        if len < 2 {
            return Err(invariant("increment message has no remaining pairs"));
        }
        self.weights.validate(len as usize)?;
        let blocks = (len / 2).div_ceil(BLOCK);
        let mut partials = context.alloc(COLUMNS * blocks as usize)?;
        let mut builder = match (&self.columns, &self.weights) {
            (Columns::Native { atoms, .. }, Weights::Split { lo, hi }) => {
                let mut builder = context
                    .stream()
                    .launch_builder(&context.inc_paired_native_round);
                let _ = builder.arg(&atoms.ram_inc).arg(&atoms.rd_inc).arg(&len);
                let _ = builder.arg(lo.limbs()).arg(hi.limbs());
                builder
            }
            (Columns::Dense(values), Weights::Split { lo, hi }) => {
                let mut builder = context
                    .stream()
                    .launch_builder(&context.inc_paired_split_round);
                let _ = builder.arg(values.limbs()).arg(&len);
                let _ = builder.arg(lo.limbs()).arg(hi.limbs());
                builder
            }
            (Columns::Dense(values), Weights::Dense(weights)) => {
                let mut builder = context
                    .stream()
                    .launch_builder(&context.inc_paired_dense_round);
                let _ = builder.arg(values.limbs()).arg(&len).arg(weights.limbs());
                builder
            }
            (Columns::Native { .. }, Weights::Dense(_)) => {
                return Err(invariant("native increments require split weights"));
            }
        };
        let (lo_len, hi_len) = match &self.weights {
            Weights::Split { lo, hi } => (
                CudaKernelContext::count_of(lo.len() / TERMS)?,
                CudaKernelContext::count_of(hi.len() / TERMS)?,
            ),
            Weights::Dense(_) => (1, len),
        };
        if matches!(self.weights, Weights::Split { .. }) {
            let _ = builder.arg(&lo_len).arg(&hi_len);
        }
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: construction checks native column ownership/length; binds
        // preserve two equally sized dense rows on that device. validate checks
        // all weight rows against len. Kernels guard len/2 threads and write
        // two block partials using BLOCK * LIMBS u64 words of shared scratch.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * 8,
            })
        }?;
        let values = reduce_lanes(context, partials, COLUMNS as u32, blocks)?.to_host()?;
        let [zero, two] = values.as_slice() else {
            return Err(invariant("increment message requires two evaluations"));
        };
        Ok([*zero, *two])
    }

    pub(super) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        if self.columns.len() < 2 {
            return Err(invariant("increment shard is already fully bound"));
        }
        let context = device(self.ordinal)?;
        self.columns.bind(context, challenge)?;
        self.weights.bind(context, challenge)
    }

    pub(super) fn ready_to_consolidate(&self) -> bool {
        self.columns.len() <= CONSOLIDATE_AT && matches!(self.weights, Weights::Dense(_))
    }

    pub(super) fn consolidate(shards: &[Self]) -> Result<Self, CudaError> {
        let mut columns = [Vec::new(), Vec::new()];
        let mut weights = [Vec::new(), Vec::new()];
        for shard in shards {
            let (Columns::Dense(values), Weights::Dense(eqs)) = (&shard.columns, &shard.weights)
            else {
                return Err(invariant("increment consolidation requires dense tables"));
            };
            let len = shard.columns.len();
            shard.weights.validate(len)?;
            let values = values.to_host()?;
            let eqs = eqs.to_host()?;
            for c in 0..COLUMNS {
                columns[c].extend_from_slice(&values[c * len..(c + 1) * len]);
                weights[c].extend_from_slice(&eqs[c * len..(c + 1) * len]);
            }
        }
        let context = device(0)?;
        Ok(Self {
            ordinal: 0,
            columns: Columns::Dense(context.upload(&columns.concat())?),
            weights: Weights::Dense(context.upload(&weights.concat())?),
        })
    }

    pub(super) fn final_values(&self) -> Result<[Fr; COLUMNS], CudaError> {
        let Columns::Dense(values) = &self.columns else {
            return Err(invariant("increment claims require bound columns"));
        };
        let values = values.to_host()?;
        let [ram, rd] = values.as_slice() else {
            return Err(invariant("increment claims require two final values"));
        };
        Ok([*ram, *rd])
    }
}

pub(super) fn device(ordinal: usize) -> Result<&'static CudaKernelContext, CudaError> {
    context_for(ordinal).ok_or(invariant("missing increment reduction device"))
}

pub(super) const fn invariant(reason: &'static str) -> CudaError {
    CudaError::InvariantViolation { reason }
}
