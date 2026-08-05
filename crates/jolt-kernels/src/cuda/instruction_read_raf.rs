use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::combine::{combine_terms, CombineTerm};
use super::context::{CudaKernelContext, BLOCK};
use super::device::{fr_into, require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::{require_context, CudaBackend};
use crate::reference::instruction_read_raf::{InstructionReadRafKernel, InstructionReadRafWitness};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const CHUNK_LEN: usize = 8;
const ADDRESS_BITS: usize = 128;
const NO_PREFIX: u32 = u32::MAX;

pub struct DeviceInstructionReadRaf<F: Field> {
    host: InstructionReadRafKernel<F>,
    context: &'static CudaKernelContext,
}

struct TermLayout {
    scales: Vec<u32>,
    prefix_ids: Vec<u32>,
    suffix_slots: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    suffix_bases: Vec<u32>,
}

fn term_layout(terms: &[(usize, Vec<CombineTerm>)]) -> TermLayout {
    let mut layout = TermLayout {
        scales: Vec::new(),
        prefix_ids: Vec::new(),
        suffix_slots: Vec::new(),
        offsets: Vec::new(),
        counts: Vec::new(),
        suffix_bases: Vec::new(),
    };
    let mut suffix_base = 0u32;
    for (suffix_count, table_terms) in terms {
        layout.offsets.push(layout.scales.len() as u32);
        layout.counts.push(table_terms.len() as u32);
        layout.suffix_bases.push(suffix_base);
        suffix_base += *suffix_count as u32;
        for term in table_terms {
            layout.scales.push(term.scale as u32);
            layout
                .prefix_ids
                .push(term.prefix.map_or(NO_PREFIX, |prefix| prefix as u32));
            layout.suffix_slots.push(term.suffix as u32);
        }
    }
    layout
}

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn address_message(&self) -> Result<[F; 3], CudaError> {
        let host = &self.host;
        let half = host.prefix_tables[0].evals().len() / 2;

        let mut prefix_handles = Vec::with_capacity(host.prefix_tables.len());
        for table in &host.prefix_tables {
            prefix_handles.push(self.context.upload(require_fr_slice(table.evals())?)?);
        }

        let mut suffix_handles = Vec::new();
        let mut terms = Vec::with_capacity(host.suffix_tables.len());
        for (table, columns) in &host.suffix_tables {
            terms.push((columns.len(), combine_terms(*table)?));
            for column in columns {
                suffix_handles.push(self.context.upload(require_fr_slice(column.evals())?)?);
            }
        }
        let layout = term_layout(&terms);

        let mut raf_handles = Vec::new();
        for raf in [&host.raf_left, &host.raf_right, &host.raf_identity] {
            for table in [&raf.prefix, &raf.q_shift, &raf.q_value] {
                raf_handles.push(self.context.upload(require_fr_slice(table.evals())?)?);
            }
        }
        let raf_count = 3u32;

        let prefix_pointers = self
            .context
            .device_pointers(&prefix_handles.iter().collect::<Vec<_>>())?;
        let suffix_pointers = self
            .context
            .device_pointers(&suffix_handles.iter().collect::<Vec<_>>())?;
        let raf_pointers = self
            .context
            .device_pointers(&raf_handles.iter().collect::<Vec<_>>())?;

        let scales = self.context.upload_u32_slice(&layout.scales)?;
        let prefix_ids = self.context.upload_u32_slice(&layout.prefix_ids)?;
        let suffix_slots = self.context.upload_u32_slice(&layout.suffix_slots)?;
        let offsets = self.context.upload_u32_slice(&layout.offsets)?;
        let counts = self.context.upload_u32_slice(&layout.counts)?;
        let suffix_bases = self.context.upload_u32_slice(&layout.suffix_bases)?;

        let table_count = CudaKernelContext::count_of(host.suffix_tables.len())?;
        let half_count = CudaKernelContext::count_of(half)?;
        let lanes = 3 * (1 + raf_count);
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = self.context.alloc(lanes as usize * blocks as usize)?;

        let mut builder = self
            .context
            .stream()
            .launch_builder(self.context.irr_address_message());
        let _ = builder.arg(&prefix_pointers);
        let _ = builder.arg(&prefix_ids);
        let _ = builder.arg(&suffix_slots);
        let _ = builder.arg(&scales);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&counts);
        let _ = builder.arg(&suffix_pointers);
        let _ = builder.arg(&suffix_bases);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(&raf_pointers);
        let _ = builder.arg(&raf_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `b < half` reads `evals[b]`/`evals[b + half]` of tables
        // holding `2 * half` elements this round, reached through pointer arrays
        // sized to the handles uploaded above, which outlive the launch. Term
        // indices come from `term_layout`: `prefix_ids` are `Prefixes`
        // discriminants (`NO_PREFIX` is never dereferenced) and
        // `suffix_bases[t] + suffix_slots[term]` indexes the flattened handle
        // list. Thread 0 writes `partials[lane * gridDim.x + blockIdx.x]` of
        // `lanes * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        self.context.stream().synchronize()?;

        let totals = self.reduce_lanes(partials, lanes, blocks)?.to_host()?;
        let gamma = self.host.gamma;
        let gamma_sqr = gamma * gamma;
        let mut evals = [F::zero(); 3];
        for (c, eval) in evals.iter_mut().enumerate() {
            let base = c * (1 + raf_count as usize);
            let read: F = Self::field(totals[base])?;
            let left: F = Self::field(totals[base + 1])?;
            let right: F = Self::field(totals[base + 2])?;
            let identity: F = Self::field(totals[base + 3])?;
            *eval = read + gamma * left + gamma_sqr * (right + identity);
        }
        Ok(evals)
    }

    fn field(value: jolt_field::Fr) -> Result<F, CudaError> {
        fr_into(value).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }

    fn reduce_lanes(
        &self,
        mut partials: DeviceFrVec,
        lanes: u32,
        mut width: u32,
    ) -> Result<DeviceFrVec, CudaError> {
        while width > 1 {
            let next = width.div_ceil(2);
            let mut folded = self.context.alloc(lanes as usize * next as usize)?;
            let mut builder = self
                .context
                .stream()
                .launch_builder(self.context.lane_sum_reduce());
            let _ = builder.arg(partials.limbs());
            let _ = builder.arg(folded.limbs_mut());
            let _ = builder.arg(&lanes);
            let _ = builder.arg(&width);
            let _ = builder.arg(&next);
            // SAFETY: thread `(i < next, lane < lanes)` reads
            // `in[lane * width + i]` and, when `i + next < width`, its mate at
            // `+ next` — both inside `in`'s `lanes * width` elements — and writes
            // only `out[lane * next + i]` of `lanes * next`. Index sets are
            // pairwise disjoint and `out` is a distinct allocation.
            let _ = unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (next.div_ceil(BLOCK), lanes, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            self.context.stream().synchronize()?;
            partials = folded;
            width = next;
        }
        Ok(partials)
    }
}

impl<F: Field> PrepareKernel<F, InstructionReadRaf<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let context = require_context()?;
        let dimensions = inputs.relation.dimensions();
        if dimensions.instruction_address_bits() != ADDRESS_BITS
            || !ADDRESS_BITS.is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "the CUDA instruction read-RAF address phase supports only the \
                         2·XLEN interleaved-operand address width in 8-variable phases",
            });
        }
        let rows: Vec<InstructionReadRafWitness> =
            collect_bundles(witness, 1 << dimensions.log_t())?;
        let host = InstructionReadRafKernel::new(
            dimensions,
            &inputs.points.lookup_output,
            rows,
            inputs.challenges.gamma,
        )?;
        Ok(Box::new(DeviceInstructionReadRaf { host, context }))
    }
}

impl<F: Field> ProveRounds<F> for DeviceInstructionReadRaf<F> {
    fn num_rounds(&self) -> usize {
        self.host.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.host.bind(challenge)?;
        }
        let evals = if self.host.rounds_bound < self.host.address_bits() {
            self.address_message()
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address_message",
                })?
                .to_vec()
        } else {
            self.host.cycle_message()?
        };
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.host.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceInstructionReadRaf<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        inputs: &InstructionReadRafInputClaims<F>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        self.host.output_claims(inputs)
    }
}
