use std::sync::Arc;

use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::*;

// Initial conservative admission; tune from complete-proof measurements.
const MIN_GPU_PREFIX: usize = 1 << 16;

pub(crate) struct MetalOuterRemainder(Arc<OuterGpu>);

impl MetalOuterRemainder {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec
            .source("outer_product.metal", include_str!("outer_product.metal"))
            .instantiate::<AkitaField>("outer_product");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(OuterGpu { device, library })))
    }
}

impl PrepareKernel<AkitaField, OuterRemainder<AkitaField>> for MetalOuterRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, OuterRemainder<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = OuterRemainder<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let carry = session.take::<SpartanOuterCarry<AkitaField>>().ok_or(
            KernelError::InvariantViolation {
                reason: "the outer uni-skip slot parked no carry for the remainder member",
            },
        )?;
        let OuterRemainderKernel { az, bz, state } = OuterRemainderKernel::prepare(carry, &inputs)?;
        Ok(Box::new(MetalOuterKernel {
            state,
            columns: Columns::Cpu { az, bz },
            gpu: Arc::clone(&self.0),
        }))
    }
}

struct OuterGpu {
    device: Device,
    library: ShaderLibrary,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Columns {
    Cpu {
        az: PrefixColumn<AkitaField>,
        bz: PrefixColumn<AkitaField>,
    },
    Gpu(#[cfg_attr(feature = "allocative", allocative(skip))] GpuColumns),
}

struct GpuColumns {
    az: DeviceBuffer<AkitaField>,
    bz: DeviceBuffer<AkitaField>,
    tails: [AkitaField; 2],
}

impl GpuColumns {
    fn upload(
        gpu: &OuterGpu,
        az: &PrefixColumn<AkitaField>,
        bz: &PrefixColumn<AkitaField>,
    ) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_outer_upload").entered();
        debug_assert_eq!(az.tail[0], az.tail[1]);
        debug_assert_eq!(bz.tail[0], bz.tail[1]);
        debug_assert_eq!(az.values.len(), bz.values.len());
        Ok(Self {
            az: DeviceBuffer::from_slice(&gpu.device, &az.values)?,
            bz: DeviceBuffer::from_slice(&gpu.device, &bz.values)?,
            tails: [az.tail[0], bz.tail[0]],
        })
    }

    fn download(&mut self) -> Result<Columns, MetalError> {
        Ok(Columns::Cpu {
            az: PrefixColumn {
                values: self.az.read()?.to_vec(),
                tail: [self.tails[0]; 2],
            },
            bz: PrefixColumn {
                values: self.bz.read()?.to_vec(),
                tail: [self.tails[1]; 2],
            },
        })
    }

    fn round(
        &mut self,
        gpu: &OuterGpu,
        bind: Option<AkitaField>,
        eq: &GruenSplitEqPolynomial<AkitaField>,
    ) -> Result<[AkitaField; 3], MetalError> {
        let span = tracing::info_span!("metal_outer_dispatch", gpu_s = tracing::field::Empty);
        let _span = span.enter();
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "outer_product".to_owned(),
            reason: reason.to_owned(),
        };
        let old_len = u32::try_from(self.az.len())
            .map_err(|_| invalid("prefix exceeds 32-bit dispatch indexing"))?;
        let len = if bind.is_some() {
            self.az.len().div_ceil(2)
        } else {
            self.az.len()
        };
        let pairs = len.div_ceil(2);
        let e_in = DeviceBuffer::from_slice(&gpu.device, eq.e_in_current())?;
        let e_out = DeviceBuffer::from_slice(&gpu.device, eq.e_out_current())?;
        let in_len = u32::try_from(e_in.len())
            .map_err(|_| invalid("equality factor exceeds 32-bit indexing"))?;
        if in_len == 0 || pairs.div_ceil(e_in.len()) > e_out.len() {
            return Err(invalid("equality factors do not cover the explicit prefix"));
        }
        let next = bind
            .map(|_| -> Result<_, MetalError> {
                Ok((
                    DeviceBuffer::<AkitaField>::zeroed(&gpu.device, len)?,
                    DeviceBuffer::<AkitaField>::zeroed(&gpu.device, len)?,
                ))
            })
            .transpose()?;
        let (next_az, next_bz) = next
            .as_ref()
            .map_or((&self.az, &self.bz), |(az, bz)| (az, bz));
        let pipeline = gpu
            .library
            .pipeline(&host_name::<AkitaField>("outer_product"))?;
        let width = pipeline.thread_execution_width();
        if width == 0 {
            return Err(invalid("zero SIMD-group width"));
        }
        let group_size = pipeline.max_total_threads_per_threadgroup().min(256) / width * width;
        if group_size == 0 || group_size / width > 32 || group_size / width > width {
            return Err(invalid("unsupported SIMD-group geometry"));
        }
        let groups = pairs.div_ceil(e_in.len()).max(1);
        let partial_len = groups
            .checked_mul(3)
            .filter(|len| u32::try_from(*len).is_ok())
            .ok_or_else(|| invalid("partial output exceeds 32-bit indexing"))?;
        let mut partials = DeviceBuffer::<AkitaField>::zeroed(&gpu.device, partial_len)?;
        let do_bind = u32::from(bind.is_some());
        let need_one =
            u32::from(eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero());
        let challenge = bind.unwrap_or_else(AkitaField::zero);
        let mut batch = Batch::new(&gpu.device)?;
        // SAFETY: every input read is prefix-guarded; output indices are below
        // len, equality indices are checked above, and partials has three cells
        // per group. Bind outputs are disjoint from inputs. The no-bind dispatch
        // never writes its aliased outputs. Whole SIMD groups reach all barriers.
        unsafe {
            batch.dispatch_unchecked(
                pipeline,
                &[
                    Binding::buffer(&self.az),
                    Binding::buffer(&self.bz),
                    Binding::buffer(next_az),
                    Binding::buffer(next_bz),
                    Binding::buffer(&e_in),
                    Binding::buffer(&e_out),
                    Binding::buffer(&partials),
                    Binding::value(&old_len),
                    Binding::value(&in_len),
                    Binding::value(&do_bind),
                    Binding::value(&need_one),
                    Binding::value(&challenge),
                    Binding::value(&self.tails[0]),
                    Binding::value(&self.tails[1]),
                ],
                Grid::linear(groups * group_size, group_size),
            )?;
        }
        let gpu_time = batch.commit_and_wait()?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        let mut result = [AkitaField::zero(); 3];
        for partial in partials.read()?.chunks_exact(3) {
            for (sum, value) in result.iter_mut().zip(partial) {
                *sum += *value;
            }
        }
        let suffix = eq.suffix_weight(pairs) * self.tails[0] * self.tails[1];
        result[0] += suffix;
        result[2] += suffix;
        if let Some((az, bz)) = next {
            self.az = az;
            self.bz = bz;
        }
        Ok(result)
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalOuterKernel {
    state: OuterRemainderState<AkitaField>,
    columns: Columns,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<OuterGpu>,
}

fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal",
        reason: error.to_string(),
    }
}

impl ProveRounds<AkitaField> for MetalOuterKernel {
    fn num_rounds(&self) -> usize {
        self.state.challenges.total()
    }

    fn prove_round(
        &mut self,
        mut bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if let Columns::Gpu(columns) = &mut self.columns {
            if columns.az.len().div_ceil(2) < MIN_GPU_PREFIX {
                self.columns = columns.download().map_err(compute_error)?;
            }
        }
        if let Columns::Cpu { az, bz } = &mut self.columns {
            if self.state.challenges.as_slice().is_empty()
                && bind.is_some()
                && az.values.len().div_ceil(2) >= MIN_GPU_PREFIX
            {
                if let Some(challenge) = bind.take() {
                    self.state.bind(az, bz, challenge);
                }
                let columns = GpuColumns::upload(&self.gpu, az, bz).map_err(compute_error)?;
                self.columns = Columns::Gpu(columns);
            }
        }
        match &mut self.columns {
            Columns::Cpu { az, bz } => self.state.prove_round(az, bz, bind, round, previous_claim),
            Columns::Gpu(columns) => {
                if let Some(challenge) = bind {
                    self.state.bind_metadata(challenge);
                }
                let [q0, qi, q1] = columns
                    .round(&self.gpu, bind, &self.state.split_eq)
                    .map_err(compute_error)?;
                self.state
                    .split_eq
                    .checked_cubic(q0, qi, previous_claim, round, || q1)
            }
        }
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if let Columns::Gpu(columns) = &mut self.columns {
            self.columns = columns.download().map_err(compute_error)?;
        }
        if let Columns::Cpu { az, bz } = &mut self.columns {
            self.state.bind(az, bz, bind);
        }
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalOuterKernel {
    type Relation = OuterRemainder<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.state.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.state
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }

    #[cfg(feature = "field-inline")]
    fn park_residue(mut self: Box<Self>, session: &mut ProofSession) {
        session.park(FieldSpartanCarry(std::mem::take(
            &mut self.state.field_rows,
        )));
        crate::mem::drop_in_background_thread(self);
    }
}

#[cfg(all(test, target_os = "macos"))]
#[expect(clippy::unwrap_used, reason = "tests assert exact field arithmetic")]
mod tests {
    use jolt_field::Ring;
    use jolt_poly::Polynomial;

    use super::*;

    #[test]
    fn prefix_rounds_match_dense_multilinears() {
        let gpu = MetalOuterRemainder::new().unwrap();
        let bits = 10;
        for prefix in [0, 1, 3, 257, 1023, 1024] {
            for exceptional in [false, true] {
                let point: Vec<_> = (0..bits)
                    .map(|i| {
                        AkitaField::from_u64(if exceptional && i % 2 == 0 {
                            0
                        } else {
                            17 + i as u64
                        })
                    })
                    .collect();
                let make = |seed: u128| PrefixColumn {
                    values: (0..prefix)
                        .map(|i| AkitaField::from_u128(u128::MAX - seed * (i + 1) as u128))
                        .collect(),
                    tail: [AkitaField::from_u128(u128::MAX - seed); 2],
                };
                let az = make(23);
                let bz = make(41);
                let dense = |column: &PrefixColumn<AkitaField>| {
                    Polynomial::new((0..1 << bits).map(|i| column.value(i)).collect())
                };
                let mut dense_az = dense(&az);
                let mut dense_bz = dense(&bz);
                let mut columns = GpuColumns::upload(&gpu.0, &az, &bz).unwrap();
                let mut eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
                let mut bind = None;
                for round in 0..bits {
                    if let Some(challenge) = bind {
                        dense_az.bind_with_order(challenge, BindingOrder::LowToHigh);
                        dense_bz.bind_with_order(challenge, BindingOrder::LowToHigh);
                        eq.bind(challenge);
                    }
                    let got = columns.round(&gpu.0, bind, &eq).unwrap();
                    let mut expected = [AkitaField::zero(); 3];
                    for (row, weight) in
                        EqPolynomial::<AkitaField>::evals(&point[..bits - round - 1], None)
                            .into_iter()
                            .enumerate()
                    {
                        let [a0, a1] = [dense_az.evals()[2 * row], dense_az.evals()[2 * row + 1]];
                        let [b0, b1] = [dense_bz.evals()[2 * row], dense_bz.evals()[2 * row + 1]];
                        expected[0] += weight * a0 * b0;
                        expected[1] += weight * (a1 - a0) * (b1 - b0);
                        expected[2] += weight * a1 * b1;
                    }
                    assert_eq!(&got[..2], &expected[..2], "prefix={prefix}, round={round}");
                    if eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero() {
                        assert_eq!(got[2], expected[2]);
                    }
                    let len = columns.az.len();
                    assert_eq!(columns.az.read().unwrap(), &dense_az.evals()[..len]);
                    assert_eq!(columns.bz.read().unwrap(), &dense_bz.evals()[..len]);
                    bind = Some(AkitaField::from_u64([0, 1, 43][round % 3]));
                }
            }
        }
    }
}
