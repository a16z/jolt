//! Native PC and signed-increment pushforwards. Relation construction and the
//! small address sumcheck remain in the canonical implementation.
use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::{shaders::FIELD_HEADERS, MetalError};

use super::super::metal_rows::ResidentInstructionRows;
use super::super::support::map_indices;
use super::*;

const STRIPES: usize = 64;
const MIN_GPU_PREFIX: usize = 1 << 16;
const MAX_HISTOGRAM_BYTES: usize = 128 << 20;

pub(crate) struct MetalBytecodeReadRafAddress(Arc<BytecodeGpu>);

impl MetalBytecodeReadRafAddress {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        let spec = spec
            .source("bytecode_address.metal", include_str!("address.metal"))
            .instantiate::<AkitaField>("bytecode_pushforward")
            .instantiate::<AkitaField>("bytecode_pushforward_finish");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(BytecodeGpu { device, library })))
    }
}

struct BytecodeGpu {
    device: Device,
    library: ShaderLibrary,
}

/// Kept separate from the common 32-byte selectors: only Bytecode consumes this
/// signed column. The carry owns both buffers through the address/cycle boundary.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BytecodeSources {
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    selectors: Arc<ResidentInstructionRows>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    magnitude: DeviceBuffer<u64>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    signs: DeviceBuffer<u64>,
}

impl BytecodeSources {
    fn upload(
        session: &mut ProofSession,
        device: &Device,
        rows: &Arc<ConstantSuffix<InstructionCycleRow>>,
    ) -> Result<Self, KernelError<AkitaField>> {
        let selectors = ResidentInstructionRows::shared(session, rows)?;
        let _span = tracing::info_span!("metal_bytecode_increment_upload").entered();
        let n = rows.prefix_len();
        let magnitude = map_indices(n, |i| rows[i].fused_inc_parts().0);
        let signs = map_indices(n.div_ceil(64), |word| {
            let mut bits = 0u64;
            for offset in 0..64 {
                let i = word * 64 + offset;
                if i < n && rows[i].fused_inc_parts().1 {
                    bits |= 1 << offset;
                }
            }
            bits
        });
        Ok(Self {
            rows: Arc::clone(rows),
            selectors,
            magnitude: DeviceBuffer::from_slice(device, &magnitude).map_err(compute_error)?,
            signs: DeviceBuffer::from_slice(device, &signs).map_err(compute_error)?,
        })
    }
}

impl PrepareKernel<AkitaField, BytecodeReadRafAddressPhase<AkitaField>>
    for MetalBytecodeReadRafAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafAddressPhase<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        OptimizedBytecodeReadRafAddress::prepare_with_pushforwards(
            session,
            witness,
            inputs,
            |session, rows, base, weighted, addresses| {
                if rows.prefix_len() < MIN_GPU_PREFIX
                    || !admitted(rows.prefix_len(), addresses, base.len() + weighted.len())
                {
                    return Ok(stage_pushforwards(
                        base,
                        weighted,
                        rows.representatives(),
                        addresses,
                        InstructionCycleRow::bytecode_pc,
                        InstructionCycleRow::fused_inc::<AkitaField>,
                    ));
                }
                let sources = BytecodeSources::upload(session, &self.0.device, rows)?;
                let result = self
                    .0
                    .pushforwards(&sources, base, weighted, addresses)
                    .map_err(compute_error)?;
                session.park(sources);
                Ok(result)
            },
        )
    }
}

fn admitted(prefix: usize, addresses: usize, stages: usize) -> bool {
    prefix <= u32::MAX as usize
        && addresses < u32::MAX as usize
        && stages <= u32::MAX as usize
        && addresses
            .checked_mul(stages)
            .and_then(|n| n.checked_mul(STRIPES * 5 * 4))
            .is_some_and(|bytes| bytes <= MAX_HISTOGRAM_BYTES)
}

impl BytecodeGpu {
    fn pushforwards(
        &self,
        source: &BytecodeSources,
        base: &[Vec<AkitaField>],
        weighted: &[Vec<AkitaField>],
        addresses: usize,
    ) -> Result<Vec<Vec<AkitaField>>, MetalError> {
        let Some(point) = base.first().or_else(|| weighted.first()) else {
            return Ok(Vec::new());
        };
        let prefix = source.rows.prefix_len();
        let stages = base.len() + weighted.len();
        if !admitted(prefix, addresses, stages) {
            return Err(invalid(
                "pushforward shape exceeds scratch or indexing bound",
            ));
        }
        if point.len() > 32 || base.iter().chain(weighted).any(|p| p.len() != point.len()) {
            return Err(invalid("pushforward points have unsupported geometry"));
        }
        if prefix == 0 {
            return Ok(finish_pushforwards(
                base,
                weighted,
                0,
                &source.rows[0],
                &InstructionCycleRow::bytecode_pc,
                &InstructionCycleRow::fused_inc::<AkitaField>,
                vec![vec![AkitaField::zero(); addresses]; stages],
            ));
        }
        let lo_bits = point.len() / 2;
        let hi_bits = point.len() - lo_bits;
        let in_len = 1usize << lo_bits;
        let out_len = 1usize << hi_bits;
        let group_rows = in_len.min(256);
        let groups = prefix.div_ceil(group_rows);
        let _span =
            tracing::info_span!("metal_bytecode_pushforwards", prefix, addresses, stages).entered();
        let e_lo: Vec<_> = base
            .iter()
            .chain(weighted)
            .flat_map(|p| eq_table(&p[hi_bits..]))
            .collect();
        let e_hi: Vec<_> = base
            .iter()
            .chain(weighted)
            .flat_map(|p| eq_table(&p[..hi_bits]))
            .collect();
        let e_lo = DeviceBuffer::from_slice(&self.device, &e_lo)?;
        let e_hi = DeviceBuffer::from_slice(&self.device, &e_hi)?;
        let bins = DeviceBuffer::<u32>::zeroed(&self.device, STRIPES * stages * addresses * 5)?;
        let mut result = DeviceBuffer::<AkitaField>::zeroed(&self.device, stages * addresses)?;
        let resident = source.selectors.resident(&self.device)?;
        let rows = resident
            .as_ref()
            .ok_or_else(|| invalid("missing resident PC rows"))?;
        if rows.prefix.len() != prefix {
            return Err(invalid("PC and increment prefixes differ"));
        }
        let pipeline = self
            .library
            .pipeline(&host_name::<AkitaField>("bytecode_pushforward"))?;
        let finish = self
            .library
            .pipeline(&host_name::<AkitaField>("bytecode_pushforward_finish"))?;
        if pipeline.max_total_threads_per_threadgroup() < 256 {
            return Err(invalid("256-lane pushforward groups unsupported"));
        }
        let mut batch = Batch::new(&self.device)?;
        for first in (0..stages).step_by(2) {
            let params = [
                prefix as u32,
                addresses as u32,
                stages as u32,
                base.len() as u32,
                in_len as u32,
                out_len as u32,
                group_rows as u32,
                first as u32,
                STRIPES as u32,
            ];
            // SAFETY: groups cover only explicit rows, each hash has at most 256
            // keys in 512 slots, and all lanes reach both local barriers. Global
            // sums are atomic; the following dispatch runs after these writes.
            unsafe {
                batch.dispatch_unchecked(
                    pipeline,
                    &[
                        Binding::buffer(&rows.prefix),
                        Binding::buffer(&source.magnitude),
                        Binding::buffer(&source.signs),
                        Binding::buffer(&e_lo),
                        Binding::buffer(&e_hi),
                        Binding::buffer(&bins),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * 256, 256),
                )?;
            }
        }
        let params = [addresses as u32, stages as u32, STRIPES as u32];
        // SAFETY: one thread owns one output; the shader guards rounded lanes.
        unsafe {
            batch.dispatch_unchecked(
                finish,
                &[
                    Binding::buffer(&bins),
                    Binding::buffer(&result),
                    Binding::value(&params),
                ],
                Grid::linear(stages * addresses, 128),
            )?;
        }
        let span = tracing::info_span!(
            "metal_bytecode_pushforward_dispatch",
            gpu_s = tracing::field::Empty
        );
        let _entered = span.enter();
        let elapsed = batch.commit_and_wait()?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        let values = result
            .read()?
            .chunks_exact(addresses)
            .map(<[AkitaField]>::to_vec)
            .collect();
        Ok(finish_pushforwards(
            base,
            weighted,
            prefix,
            &source.rows[prefix],
            &InstructionCycleRow::bytecode_pc,
            &InstructionCycleRow::fused_inc::<AkitaField>,
            values,
        ))
    }
}

fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "bytecode read RAF".into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal bytecode read RAF",
        reason: error.to_string(),
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn oracle_pushforwards(
    points: &[Vec<AkitaField>],
    base: usize,
    rows: &[(usize, i128)],
    addresses: usize,
) -> Vec<Vec<AkitaField>> {
    use jolt_witness::witnesses::FusedInc;
    let gpu = MetalBytecodeReadRafAddress::new().expect("Metal bytecode pipelines");
    let rows = Arc::new(ConstantSuffix::new(
        rows.iter()
            .map(|&(pc, inc)| InstructionCycleRow::new(0, None, false, pc, None, FusedInc(inc)))
            .collect(),
        1 << points[0].len(),
    ));
    let mut session = ProofSession::default();
    let source = BytecodeSources::upload(&mut session, &gpu.0.device, &rows)
        .expect("native bytecode sources");
    gpu.0
        .pushforwards(&source, &points[..base], &points[base..], addresses)
        .expect("Metal bytecode pushforwards")
}
