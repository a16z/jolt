use jolt_field::{One, Ring, Zero};

use super::*;
use crate::optimized::eq_prefix::EqTerms;

const CPU_TAIL: usize = 4096;
const GROUP_ROWS: usize = 64;

pub(crate) struct MetalBytecodeReadRafCycle(pub(super) Arc<BytecodeGpu>, pub(super) usize);

impl PrepareKernel<AkitaField, BytecodeReadRafCycle<AkitaField>> for MetalBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafCycle<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let recipe = OptimizedBytecodeReadRafCycle::prepare_recipe(session, witness, inputs)?;
        let carry = session.take::<BytecodeSources>();
        if !self.0.cycle_supported
            || recipe.prefix_len < self.1
            || recipe.num_vars > 31
            || recipe.ra.num_polys() > 8
            || recipe.ra.num_polys() == 0
        {
            return Ok(Box::new(recipe.into_cpu()));
        }
        let LazyFoldedRa::Lazy {
            source, factors: 1, ..
        } = &recipe.ra
        else {
            return Ok(Box::new(recipe.into_cpu()));
        };
        let rows = Arc::clone(&source.rows);
        let sources = if let Some(carry) = carry {
            if !Arc::ptr_eq(&rows, &carry.rows) {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode cycle changed native source owner",
                });
            }
            carry
        } else {
            BytecodeSources::upload(session, &self.0.device, &rows)?
        };
        Ok(Box::new(MetalCycle {
            gpu: Arc::clone(&self.0),
            cpu_tail: if self.1 == 0 { 0 } else { CPU_TAIL },
            cpu: None,
            state: Some(CycleState {
                progress: recipe.progress,
                degree: recipe.degree,
                prefix: recipe.prefix_len,
                ra_tail: recipe.ra.tail_values().to_vec(),
                inc_tail: recipe.fused_inc.tail_value(),
                ra: Some(recipe.ra),
                inc: Some(recipe.fused_inc),
                device_ra: None,
                device_inc: None,
                coefficients: None,
                terms: EqTerms::new(recipe.num_vars, recipe.terms),
                fused_terms: EqTerms::new(recipe.num_vars, recipe.fused_terms),
                entry: recipe.entry,
                sources: Some(sources),
                output_openings: recipe.output_openings,
            }),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalCycle {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<BytecodeGpu>,
    cpu_tail: usize,
    cpu: Option<CycleKernel<AkitaField>>,
    state: Option<CycleState>,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct CycleState {
    progress: RoundProgress,
    degree: usize,
    prefix: usize,
    ra: Option<LazyFoldedRa<AkitaField, BytecodePcChunks>>,
    inc: Option<LazyFusedInc<AkitaField>>,
    ra_tail: Vec<AkitaField>,
    inc_tail: AkitaField,
    terms: EqTerms<AkitaField>,
    fused_terms: EqTerms<AkitaField>,
    entry: AkitaField,
    sources: Option<BytecodeSources>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    device_ra: Option<DeviceBuffer<AkitaField>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    device_inc: Option<DeviceBuffer<AkitaField>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    coefficients: Option<DeviceBuffer<AkitaField>>,
    output_openings: Vec<JoltOpeningId>,
}

impl CycleState {
    fn restore(mut self) -> Result<CycleKernel<AkitaField>, SumcheckError<AkitaField>> {
        let mut coefficients = self
            .coefficients
            .take()
            .ok_or_else(|| compute_error(invalid("missing resident bytecode coefficients")))?;
        let coefficients = coefficients.read().map_err(compute_error)?;
        let combined = EqPrefix::from_bound_values(
            coefficients.iter().step_by(2).copied().collect(),
            self.terms,
        );
        let fused_combined = EqPrefix::from_bound_values(
            coefficients.iter().skip(1).step_by(2).copied().collect(),
            self.fused_terms,
        );
        let ra = if let Some(ra) = self.ra {
            ra
        } else {
            let mut values = self
                .device_ra
                .take()
                .ok_or_else(|| compute_error(invalid("missing resident bytecode RA")))?;
            LazyFoldedRa::Rows {
                values: values.read().map_err(compute_error)?.to_vec(),
                tail: self.ra_tail,
            }
        };
        let fused_inc = if let Some(inc) = self.inc {
            inc
        } else {
            let mut values = self
                .device_inc
                .take()
                .ok_or_else(|| compute_error(invalid("missing resident bytecode increment")))?;
            LazyFusedInc::Prefix {
                values: values.read().map_err(compute_error)?.to_vec(),
                tail: self.inc_tail,
            }
        };
        Ok(CycleKernel {
            progress: self.progress,
            degree: self.degree,
            ra,
            combined,
            fused_inc,
            fused_combined,
            output_openings: self.output_openings,
        })
    }

    fn round(
        &mut self,
        gpu: &BytecodeGpu,
        bind: Option<AkitaField>,
        claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let previous_prefix = self.prefix;
        let old_boundary = [
            self.terms.value(previous_prefix),
            self.fused_terms.value(previous_prefix),
        ];
        let ra_mode = if self.device_ra.is_some() {
            2
        } else if bind.is_some()
            && self
                .ra
                .as_ref()
                .and_then(LazyFoldedRa::next_materialized_len)
                .is_some()
        {
            1
        } else {
            0
        };
        let inc_mode = if self.device_inc.is_some() {
            2
        } else if bind.is_some()
            && matches!(&self.inc, Some(LazyFusedInc::Lazy { branch_weights, .. }) if branch_weights.len() == 8)
        {
            1
        } else {
            0
        };
        if let Some(r) = bind {
            if ra_mode == 0 {
                self.ra
                    .as_mut()
                    .ok_or_else(|| compute_error(invalid("missing lazy RA")))?
                    .bind(r);
            }
            if inc_mode == 0 {
                self.inc
                    .as_mut()
                    .ok_or_else(|| compute_error(invalid("missing lazy increment")))?
                    .bind(r);
            }
            self.terms.bind(r);
            self.fused_terms.bind(r);
            self.entry *= AkitaField::one() - r;
            self.prefix = self.prefix.div_ceil(2);
            self.progress.advance();
        } else if ra_mode == 2 || inc_mode == 2 {
            return Err(compute_error(invalid(
                "resident bytecode round requires a bind",
            )));
        }
        let coefficient_mode = if self.coefficients.is_some() {
            2
        } else {
            usize::from(bind.is_some())
        };
        let columns = self.ra_tail.len();
        let (ra_width, radix, shifts, flat_ra) = match &self.ra {
            Some(LazyFoldedRa::Lazy {
                tables,
                width,
                source,
                factors: 1,
                ..
            }) => (
                *width,
                tables[0].len() / width,
                source
                    .selectors
                    .iter()
                    .map(|selector| selector.shift() as u32)
                    .collect::<Vec<_>>(),
                tables.iter().flatten().copied().collect::<Vec<_>>(),
            ),
            None => (1, 1, vec![0], vec![AkitaField::zero()]),
            _ => return Err(compute_error(invalid("unexpected bytecode RA state"))),
        };
        let weights = match &self.inc {
            Some(LazyFusedInc::Lazy { branch_weights, .. }) => branch_weights.as_slice(),
            None => &[AkitaField::zero()],
            _ => {
                return Err(compute_error(invalid(
                    "unexpected bytecode increment state",
                )))
            }
        };
        let inc_width = weights.len();
        let zero = [AkitaField::zero()];
        let remaining = self.progress.total() - self.progress.bound();
        let lo_bits = remaining / 2;
        let hi_bits = remaining - lo_bits;
        let in_len = 1usize << lo_bits;
        let out_len = 1usize << hi_bits;
        let base_count = self.terms.terms().len();
        let fused_count = self.fused_terms.terms().len();
        let (mut e_lo, mut e_hi) = (Vec::new(), Vec::new());
        if coefficient_mode < 2 {
            for (point, weight) in self.terms.terms().iter().chain(self.fused_terms.terms()) {
                e_lo.extend(eq_table(&point[hi_bits..]));
                e_hi.extend(EqPolynomial::evals(&point[..hi_bits], Some(*weight)));
            }
        }
        let device = &gpu.device;
        let ra_tables = DeviceBuffer::from_slice(device, &flat_ra).map_err(compute_error)?;
        let shifts = DeviceBuffer::from_slice(device, &shifts).map_err(compute_error)?;
        let weights = DeviceBuffer::from_slice(device, weights).map_err(compute_error)?;
        let e_lo = DeviceBuffer::from_slice(device, if e_lo.is_empty() { &zero } else { &e_lo })
            .map_err(compute_error)?;
        let e_hi = DeviceBuffer::from_slice(device, if e_hi.is_empty() { &zero } else { &e_hi })
            .map_err(compute_error)?;
        let mut tails = self.ra_tail.clone();
        tails.push(self.inc_tail);
        tails.extend(old_boundary);
        tails.extend([
            self.terms.value(self.prefix),
            self.fused_terms.value(self.prefix),
            self.entry,
        ]);
        let tails = DeviceBuffer::from_slice(device, &tails).map_err(compute_error)?;
        let dummy_field = DeviceBuffer::<AkitaField>::zeroed(device, 1).map_err(compute_error)?;
        let dummy_rows = DeviceBuffer::<[u64; 4]>::zeroed(device, 1).map_err(compute_error)?;
        let dummy_u64 = DeviceBuffer::<u64>::zeroed(device, 1).map_err(compute_error)?;
        let next_ra = DeviceBuffer::<AkitaField>::zeroed(
            device,
            if ra_mode > 0 {
                self.prefix * columns
            } else {
                1
            },
        )
        .map_err(compute_error)?;
        let next_inc =
            DeviceBuffer::<AkitaField>::zeroed(device, if inc_mode > 0 { self.prefix } else { 1 })
                .map_err(compute_error)?;
        let next_coefficients = DeviceBuffer::<AkitaField>::zeroed(
            device,
            if coefficient_mode > 0 {
                self.prefix * 2
            } else {
                1
            },
        )
        .map_err(compute_error)?;
        let pairs = self.prefix.div_ceil(2);
        let groups = pairs.div_ceil(GROUP_ROWS);
        let mut partials = DeviceBuffer::<AkitaField>::zeroed(device, groups * self.degree)
            .map_err(compute_error)?;
        let resident = self
            .sources
            .as_ref()
            .map(|source| source.selectors.resident(device))
            .transpose()
            .map_err(compute_error)?;
        let uploaded = resident.as_ref().and_then(|guard| guard.as_ref());
        let (source_prefix, tail_row) =
            uploaded.map_or((0, [0; 4]), |rows| (rows.prefix.len(), rows.tail));
        let source = self.sources.as_ref();
        let (tail_magnitude, tail_negative) = source.map_or((0, false), |source| {
            source.rows[source.rows.prefix_len()].fused_inc_parts()
        });
        let source_tail = [
            tail_row[2] & 0x00ff_ffff_ffff_ffff,
            tail_magnitude,
            u64::from(tail_negative),
            0,
        ];
        let params = [
            source_prefix,
            previous_prefix,
            self.prefix,
            columns,
            ra_width,
            inc_width,
            radix,
            ra_mode,
            inc_mode,
            coefficient_mode,
            in_len,
            out_len,
            base_count,
            fused_count,
            self.degree,
        ]
        .map(|n| {
            u32::try_from(n)
                .map_err(|_| compute_error(invalid("bytecode shape exceeds 32-bit indexing")))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let params: [u32; 15] = params
            .try_into()
            .map_err(|_| compute_error(invalid("invalid bytecode parameter count")))?;
        let challenge = bind.unwrap_or_else(AkitaField::zero);
        let pipeline = gpu
            .library
            .pipeline(&host_name::<AkitaField>("bytecode_cycle_round"))
            .map_err(compute_error)?;
        if !gpu.cycle_supported
            || pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < 64
        {
            return Err(compute_error(invalid(
                "unsupported bytecode reduction geometry",
            )));
        }
        let span = tracing::info_span!(
            "metal_bytecode_cycle_dispatch",
            round = self.progress.bound(),
            prefix = self.prefix,
            ra_mode,
            inc_mode,
            coefficient_mode,
            gpu_s = tracing::field::Empty
        );
        let _entered = span.enter();
        let mut batch = Batch::new(device).map_err(compute_error)?;
        // SAFETY: each group owns disjoint output rows and partials; native
        // gathers substitute the compact suffix. Missing odd dense endpoints
        // use the old equality boundary or constant RA/increment tails.
        unsafe {
            batch
                .dispatch_unchecked(
                    pipeline,
                    &[
                        Binding::buffer(uploaded.map_or(&dummy_rows, |rows| &rows.prefix)),
                        Binding::buffer(source.map_or(&dummy_u64, |s| &s.magnitude)),
                        Binding::buffer(source.map_or(&dummy_u64, |s| &s.signs)),
                        Binding::buffer(&ra_tables),
                        Binding::buffer(&shifts),
                        Binding::buffer(&weights),
                        Binding::buffer(self.device_ra.as_ref().unwrap_or(&dummy_field)),
                        Binding::buffer(self.device_inc.as_ref().unwrap_or(&dummy_field)),
                        Binding::buffer(self.coefficients.as_ref().unwrap_or(&dummy_field)),
                        Binding::buffer(&next_ra),
                        Binding::buffer(&next_inc),
                        Binding::buffer(&next_coefficients),
                        Binding::buffer(&e_lo),
                        Binding::buffer(&e_hi),
                        Binding::buffer(&tails),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                        Binding::value(&source_tail),
                        Binding::value(&challenge),
                    ],
                    Grid::linear(groups * 64, 64),
                )
                .map_err(compute_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        let mut evals = vec![AkitaField::zero(); self.degree];
        for values in partials
            .read()
            .map_err(compute_error)?
            .chunks_exact(self.degree)
        {
            for (sum, value) in evals.iter_mut().zip(values) {
                *sum += *value;
            }
        }
        // Release guards before moving/dropping the source owner.
        drop(resident);
        if ra_mode > 0 {
            self.ra = None;
            self.device_ra = Some(next_ra);
        }
        if inc_mode > 0 {
            self.inc = None;
            self.device_inc = Some(next_inc);
        }
        if coefficient_mode > 0 {
            self.coefficients = Some(next_coefficients);
        }
        if self.ra.is_none() && self.inc.is_none() {
            self.sources = None;
        }
        let (a, b) = self.terms.suffix_pair_sums(pairs);
        let (c, d) = self.fused_terms.suffix_pair_sums(pairs);
        let tail: AkitaField = self.ra_tail.iter().copied().product();
        let lo = tail * (a + self.inc_tail * c);
        let step = tail * (b + self.inc_tail * d) - lo;
        for (slot, value) in evals.iter_mut().enumerate() {
            *value +=
                lo + AkitaField::from_u64(if slot == 0 { 0 } else { (slot + 1) as u64 }) * step;
        }
        Ok(round_poly_from_skipped_evals(&evals, claim))
    }
}

impl ProveRounds<AkitaField> for MetalCycle {
    fn num_rounds(&self) -> usize {
        self.cpu.as_ref().map_or_else(
            || self.state.as_ref().map_or(0, |s| s.progress.total()),
            ProveRounds::num_rounds,
        )
    }
    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self
            .state
            .as_ref()
            .is_some_and(|state| state.prefix <= self.cpu_tail)
        {
            self.cpu = Some(
                self.state
                    .take()
                    .ok_or_else(|| compute_error(invalid("missing bytecode cycle state")))?
                    .restore()?,
            );
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.prove_round(bind, round, claim)
        } else {
            self.state
                .as_mut()
                .ok_or_else(|| compute_error(invalid("missing bytecode cycle state")))?
                .round(&self.gpu, bind, claim)
        }
    }
    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if let Some(state) = self.state.take() {
            self.cpu = Some(state.restore()?);
        }
        self.cpu
            .as_mut()
            .ok_or_else(|| compute_error(invalid("missing bytecode final state")))?
            .finish_rounds(bind)
    }
}
impl SumcheckKernel<AkitaField> for MetalCycle {
    type Relation = BytecodeReadRafCycle<AkitaField>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        if let Some(state) = &self.state {
            state.progress.require_complete()?;
        }
        self.cpu
            .as_mut()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "bytecode claims requested before CPU extraction handoff",
            })?
            .output_claims(inputs)
    }
}
