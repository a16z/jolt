//! Donor split-LT value reductions over the register seed's existing allocation.
use std::mem::MaybeUninit;
use std::sync::Arc;

use bytemuck_derive::{Pod, Zeroable};
use jolt_akita::AkitaField as F;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::registers_read_write::metal::SharedRdIncrements;
use super::*;

const KERNEL: &str = "registers_val_round";
const THREADS: usize = 256;
const MIN_PREFIX: usize = 1 << 16;

struct Gpu {
    device: Arc<Device>,
    library: ShaderLibrary,
    #[cfg(test)]
    rounds: std::sync::atomic::AtomicUsize,
}

pub(crate) struct MetalRegistersVal {
    gpu: Arc<Gpu>,
    min_prefix: usize,
    cpu_tail: usize,
}

impl MetalRegistersVal {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self, MetalError> {
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec
            .source("registers_val.metal", include_str!("value.metal"))
            .instantiate::<F>(KERNEL);
        let library = ShaderLibrary::compile(&device, &spec)?;
        let pipeline = library.pipeline(&host_name::<F>(KERNEL))?;
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < THREADS
            || pipeline.static_threadgroup_memory_length()
                > device.limits().max_threadgroup_memory_length
        {
            return Err(invalid("registers value reduction exceeds device limits"));
        }
        Ok(Self {
            gpu: Arc::new(Gpu {
                device,
                library,
                #[cfg(test)]
                rounds: std::sync::atomic::AtomicUsize::new(0),
            }),
            min_prefix: MIN_PREFIX,
            cpu_tail: 1 << 12,
        })
    }
}

impl PrepareKernel<F, RegistersValEvaluation<F>> for MetalRegistersVal {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.trace_dimensions().log_t();
        if !(5..=31).contains(&log_t) {
            drop(session.take::<SharedRdIncrements>());
            return OptimizedRegistersValEvaluation.prepare(session, witness, inputs);
        }
        let carry = session.state::<SharedRdIncrements>();
        if let Some(carry) = carry {
            if carry.cycles != 1usize << log_t
                || !Arc::ptr_eq(&carry.device, &self.gpu.device)
                || !witness
                    .random_access()
                    .is_some_and(|source| source.same_source(&carry.source))
            {
                return Err(KernelError::InvariantViolation {
                    reason: "register increment carry belongs to a different source/device/domain",
                });
            }
        }
        let rd = session.state::<SharedRdIndices>();
        let admitted = match (carry, rd) {
            (Some(carry), Some(rd)) if rd.cycles == carry.cycles => {
                let prefix = rd.indices.len();
                let inc = carry
                    .values
                    .lock()
                    .map_err(|_| compute_error(invalid("increment source lock poisoned")))?;
                if inc.len() > prefix {
                    return Err(KernelError::InvariantViolation {
                        reason: "register increment prefix exceeds write-selector prefix",
                    });
                }
                let bytes = prefix.div_ceil(2).checked_mul(2 * size_of::<F>());
                prefix >= self.min_prefix
                    && prefix != 0
                    && bytes
                        .is_some_and(|bytes| bytes <= self.gpu.device.limits().max_buffer_length)
            }
            _ => false,
        };
        if !admitted {
            drop(session.take::<SharedRdIncrements>());
            return OptimizedRegistersValEvaluation.prepare(session, witness, inputs);
        }
        let mut cpu = OptimizedRegistersValEvaluation::prepare_state(session, witness, inputs)?;
        let carry =
            session
                .take::<SharedRdIncrements>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "admitted register increment carry disappeared",
                })?;
        let WaState::Indices { rd, eq_address } =
            std::mem::replace(&mut cpu.wa, WaState::Bound(Vec::new()))
        else {
            return Err(KernelError::InvariantViolation {
                reason: "register write selectors were bound before GPU prepare",
            });
        };
        let prefix = rd.len();
        let span = tracing::info_span!("metal_registers_val_selectors", prefix);
        let _entered = span.enter();
        let mut selectors = DeviceBuffer::<MaybeUninit<u8>>::uninit(&self.gpu.device, prefix)
            .map_err(compute_error)?;
        let destination = selectors.write_uninit().map_err(compute_error)?;
        #[cfg(feature = "parallel")]
        destination
            .par_iter_mut()
            .zip(rd.par_iter())
            .for_each(|(dst, index)| {
                let _ = dst.write(index.unwrap_or(u8::MAX));
            });
        #[cfg(not(feature = "parallel"))]
        for (dst, index) in destination.iter_mut().zip(&rd) {
            let _ = dst.write(index.unwrap_or(u8::MAX));
        }
        // SAFETY: the canonical selector prefix initialized every destination.
        let rd = unsafe { selectors.assume_init() }.map_err(compute_error)?;
        let eq_address =
            DeviceBuffer::from_slice(&self.gpu.device, &eq_address).map_err(compute_error)?;
        cpu.inc = IncState::Dense(PrefixColumn {
            values: Vec::new(),
            tail: [F::zero(); 2],
        });
        Ok(Box::new(MetalValueKernel {
            cpu,
            gpu: Arc::clone(&self.gpu),
            prefix,
            // Restore before the final low-LT bind changes the split representation.
            cpu_tail: self.cpu_tail.max(1usize << (log_t - log_t / 2 + 1)),
            state: Some(Resident::Native {
                source: carry,
                rd,
                eq_address,
            }),
        }))
    }
}

enum Resident {
    Native {
        source: SharedRdIncrements,
        rd: DeviceBuffer<u8>,
        eq_address: DeviceBuffer<F>,
    },
    Dense(DeviceBuffer<F>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalValueKernel {
    cpu: ValEvaluationKernel<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<Gpu>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    state: Option<Resident>,
    prefix: usize,
    cpu_tail: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    previous_prefix: u32,
    prefix: u32,
    inc_prefix: u32,
    rd_prefix: u32,
    lo_length: u32,
    mode: u32,
}

impl MetalValueKernel {
    fn restore(&mut self) -> Result<(), SumcheckError<F>> {
        let Some(Resident::Dense(mut dense)) = self.state.take() else {
            return Err(compute_error(invalid(
                "register value handoff precedes materialization",
            )));
        };
        let _span =
            tracing::info_span!("metal_registers_val_restore", rows = self.prefix).entered();
        let pairs = dense.read().map_err(compute_error)?;
        let mut increments = Vec::with_capacity(self.prefix);
        let mut writes = Vec::with_capacity(self.prefix);
        for pair in pairs.chunks_exact(2) {
            increments.push(pair[0]);
            writes.push(pair[1]);
        }
        self.cpu.inc = IncState::Dense(PrefixColumn {
            values: increments,
            tail: [F::zero(); 2],
        });
        self.cpu.wa = WaState::Bound(writes);
        Ok(())
    }

    fn message(&mut self, bind: Option<F>) -> Result<[F; 3], SumcheckError<F>> {
        let previous_prefix = self.prefix;
        if let Some(r) = bind {
            self.cpu.lt.bind(r);
            self.cpu.progress.advance();
            self.prefix = self.prefix.div_ceil(2);
        }
        let SplitLt::Split {
            lt_lo,
            lt_hi,
            eq_hi,
        } = &self.cpu.lt
        else {
            return Err(compute_error(invalid(
                "register value GPU outlived split LT",
            )));
        };
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| compute_error(invalid("missing register value resident state")))?;
        let native = match state {
            Resident::Native { source, .. } => Some(
                source
                    .values
                    .lock()
                    .map_err(|_| compute_error(invalid("increment source lock poisoned")))?,
            ),
            Resident::Dense(_) => None,
        };
        let device = &self.gpu.device;
        let dummy_inc = DeviceBuffer::from_slice(device, &[0i128]).map_err(compute_error)?;
        let dummy_rd = DeviceBuffer::from_slice(device, &[u8::MAX]).map_err(compute_error)?;
        let dummy_field = DeviceBuffer::from_slice(device, &[F::zero()]).map_err(compute_error)?;
        let (rd, eq_address, previous, mode) = match state {
            Resident::Native { rd, eq_address, .. } => {
                (rd, eq_address, &dummy_field, u32::from(bind.is_some()))
            }
            Resident::Dense(previous) => (&dummy_rd, &dummy_field, previous, 2),
        };
        let raw_inc = native.as_deref().unwrap_or(&dummy_inc);
        let params = Params {
            previous_prefix: word(previous_prefix)?,
            prefix: word(self.prefix)?,
            inc_prefix: word(raw_inc.len())?,
            rd_prefix: word(rd.len())?,
            lo_length: word(lt_lo.len())?,
            mode,
        };
        let groups = self.prefix.div_ceil(lt_lo.len());
        if lt_lo.len() < 2 || groups > lt_hi.len() || groups > eq_hi.len() {
            return Err(compute_error(invalid(
                "split LT does not cover register prefix",
            )));
        }
        let output_len = if mode == 0 {
            0
        } else {
            self.prefix
                .checked_mul(2)
                .ok_or_else(|| compute_error(invalid("register value output overflow")))?
        };
        let output =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, output_len).map_err(compute_error)?;
        let partials =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, groups * 3).map_err(compute_error)?;
        let lo = DeviceBuffer::from_slice(device, lt_lo).map_err(compute_error)?;
        let hi = DeviceBuffer::from_slice(device, lt_hi).map_err(compute_error)?;
        let eq = DeviceBuffer::from_slice(device, eq_hi).map_err(compute_error)?;
        let challenge = bind.unwrap_or_else(F::zero);
        let span = tracing::info_span!(
            "metal_registers_val_dispatch",
            mode,
            prefix = self.prefix,
            gpu_s = tracing::field::Empty
        );
        let _entered = span.enter();
        let mut batch = Batch::new(device).map_err(compute_error)?;
        // SAFETY: each high block owns disjoint output rows and three partials;
        // odd endpoints and native prefixes are guarded. All256 threads reach
        // reductions. Mode0 binds an initialized placeholder instead of output.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<F>(KERNEL))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(raw_inc),
                        Binding::buffer(rd),
                        Binding::buffer(eq_address),
                        Binding::buffer(previous),
                        if mode == 0 {
                            Binding::buffer(&dummy_field)
                        } else {
                            Binding::buffer(&output)
                        },
                        Binding::buffer(&lo),
                        Binding::buffer(&hi),
                        Binding::buffer(&eq),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                        Binding::value(&challenge),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(compute_error)?;
        #[cfg(test)]
        let _ = self
            .gpu
            .rounds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        // SAFETY: every dispatched group wrote its three scalar partials.
        let mut partials = unsafe { partials.assume_init() }.map_err(compute_error)?;
        let mut totals = [F::zero(); 3];
        for values in partials.read().map_err(compute_error)?.chunks_exact(3) {
            for (total, value) in totals.iter_mut().zip(values) {
                *total += *value;
            }
        }
        drop(native);
        if mode != 0 {
            // SAFETY: every live row wrote both factors before completion.
            self.state = Some(Resident::Dense(
                unsafe { output.assume_init() }.map_err(compute_error)?,
            ));
        }
        Ok(totals)
    }
}

impl ProveRounds<F> for MetalValueKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if self.state.is_some() {
            if round != self.cpu.progress.bound() + usize::from(bind.is_some())
                || (round == 0) != bind.is_none()
            {
                return Err(compute_error(invalid(
                    "register value round progression mismatch",
                )));
            }
            if matches!(self.state, Some(Resident::Dense(_))) && self.prefix <= self.cpu_tail {
                self.restore()?;
            }
        }
        if self.state.is_none() {
            return self.cpu.prove_round(bind, round, claim);
        }
        let evals = self.message(bind)?;
        Ok(UnivariatePoly::from_evals_and_hint(claim, &evals))
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        if self.state.is_some() {
            self.restore()?;
        }
        self.cpu.finish_rounds(bind)
    }
}
impl SumcheckKernel<F> for MetalValueKernel {
    type Relation = RegistersValEvaluation<F>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        self.cpu.output_claims(inputs)
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        inputs: &SumcheckInputPoints<F, Self::Relation>,
        outputs: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.cpu
            .validate_derived_tables(relation, inputs, outputs, challenges)
    }
}
fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: KERNEL.into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<F> {
    SumcheckError::Compute {
        backend: "Metal registers value",
        reason: error.to_string(),
    }
}
fn word(value: usize) -> Result<u32, SumcheckError<F>> {
    u32::try_from(value).map_err(|_| compute_error(invalid("register value extent exceeds u32")))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test oracle fails immediately")]
mod tests;
