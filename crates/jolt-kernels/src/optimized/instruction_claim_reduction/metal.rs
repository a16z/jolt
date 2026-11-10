use super::super::native_operands::{compute_error, invalid, ForClaimReduction};
use super::super::spartan_product::metal::{ProductGpu, ProductParams, ResidentProduct};
use super::*;
use jolt_akita::AkitaField as F;
use jolt_field::Zero;
use jolt_metal::runtime::{host_name, Batch, Binding, DeviceBuffer, Grid};
use std::mem::MaybeUninit;
use std::sync::Arc;

const KERNEL: &str = "spartan_operand_combined";
const THREADS: usize = 256;
const CPU_TAIL: usize = 1 << 12;
pub(crate) struct MetalInstructionClaimReduction(pub(crate) Arc<ProductGpu>);
impl PrepareKernel<F, InstructionClaimReduction<F>> for MetalInstructionClaimReduction {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let Some(ForClaimReduction(source)) = session.take::<ForClaimReduction>() else {
            return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
        };
        let tau = inputs.relation.tau_low();
        {
            let core = source
                .core
                .lock()
                .map_err(|_| compute_error(invalid("native core lock poisoned")))?;
            if core.cycles != 1usize << tau.len()
                || !witness
                    .random_access()
                    .is_some_and(|identity| identity.same_source(&core.identity))
            {
                return Err(KernelError::InvariantViolation {
                    reason: "instruction claims received a different native source/domain",
                });
            }
        }
        let rows = BundleStore::resolve(witness, 1 << tau.len())?;
        if rows.access().constant_suffix_start(1 << tau.len())? == 0 {
            return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
        }
        let native = ResidentProduct {
            gpu: Arc::clone(&self.0),
            source,
        };
        let mut prepared = None;
        let cpu = OptimizedInstructionClaimReductionKernel::new_with_producer(
            tau,
            rows,
            inputs.challenges.gamma,
            |coefficients, rows, prefix, eq| {
                let tail = coefficients.combine(&rows.row(prefix.min((1 << tau.len()) - 1))?);
                let mut state = Combined {
                    native,
                    values: None,
                    prefix: prefix.div_ceil(2) * 2,
                    tail,
                    weights: coefficients.gamma_powers,
                    pending: None,
                };
                state.pending = Some(state.step(None, prefix, eq)?);
                prepared = Some(state);
                Ok(Vec::new())
            },
        )?;
        let device = prepared.ok_or(KernelError::InvariantViolation {
            reason: "native instruction claim preparation did not run",
        })?;
        Ok(Box::new(MetalClaimKernel { cpu, device }))
    }
}
struct Combined {
    native: ResidentProduct,
    values: Option<DeviceBuffer<F>>,
    prefix: usize,
    tail: F,
    weights: [F; 5],
    pending: Option<[F; 3]>,
}
impl Combined {
    fn step(
        &mut self,
        challenge: Option<F>,
        previous_prefix: usize,
        eq: &GruenSplitEqPolynomial<F>,
    ) -> Result<[F; 3], SumcheckError<F>> {
        let device = &self.native.gpu.device;
        let core = self
            .native
            .source
            .core
            .lock()
            .map_err(|_| compute_error(invalid("native core lock poisoned")))?;
        let lookup = self
            .native
            .source
            .lookup
            .lock()
            .map_err(|_| compute_error(invalid("native lookup lock poisoned")))?;
        let weights = DeviceBuffer::from_slice(device, &self.weights).map_err(compute_error)?;
        let dummy = DeviceBuffer::<F>::zeroed(device, 1).map_err(compute_error)?;
        let previous = self.values.as_ref().unwrap_or(&dummy);
        let inner = DeviceBuffer::from_slice(device, eq.e_in_current()).map_err(compute_error)?;
        let outer = DeviceBuffer::from_slice(device, eq.e_out_current()).map_err(compute_error)?;
        let pairs = self.prefix.div_ceil(2);
        let groups = pairs.div_ceil(inner.len());
        if groups > outer.len() || (challenge.is_none() && previous_prefix > core.prefix) {
            return Err(compute_error(invalid(
                "native instruction claim extent exceeds source/equality",
            )));
        }
        let output =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, self.prefix).map_err(compute_error)?;
        let partials =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, groups * 3).map_err(compute_error)?;
        let word = |n| {
            u32::try_from(n)
                .map_err(|_| compute_error(invalid("instruction claim extent exceeds u32")))
        };
        let params = ProductParams {
            first: 0,
            count: word(self.prefix)?,
            in_len: word(inner.len())?,
            outputs: 3,
            mode: u32::from(challenge.is_some()),
            prefix: word(previous_prefix)?,
            pairs: word(pairs)?,
            need_one: 0,
        };
        let span = tracing::info_span!(
            "metal_instruction_claim_round",
            native = challenge.is_none(),
            prefix = self.prefix,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let challenge_value = challenge.unwrap_or_else(F::zero);
        let mut batch = Batch::new(device).map_err(compute_error)?;
        // SAFETY: checked source/equality domains cover every read. Each pair
        // owns disjoint initialized outputs; all groups write three partials.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.native
                        .gpu
                        .library
                        .pipeline(&host_name::<F>(KERNEL))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&core.buffer),
                        Binding::buffer(&*lookup),
                        Binding::buffer(&weights),
                        Binding::buffer(previous),
                        Binding::buffer(&output),
                        Binding::buffer(&inner),
                        Binding::buffer(&outer),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                        Binding::value(&self.native.gpu.terms),
                        Binding::value(&self.tail),
                        Binding::value(&challenge_value),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        // SAFETY: successful completion initialized every output and partial.
        let mut partials = unsafe { partials.assume_init() }.map_err(compute_error)?;
        let mut sums = [eq.suffix_weight(pairs) * self.tail; 3];
        for partial in partials.read().map_err(compute_error)?.chunks_exact(3) {
            for (sum, value) in sums.iter_mut().zip(partial) {
                *sum += *value;
            }
        }
        // SAFETY: the completed dispatch wrote every allocated output element.
        self.values = Some(unsafe { output.assume_init() }.map_err(compute_error)?);
        Ok(sums)
    }
    fn restore(&mut self) -> Result<Vec<F>, SumcheckError<F>> {
        let _span =
            tracing::info_span!("metal_instruction_claim_restore", prefix = self.prefix).entered();
        let mut values = self
            .values
            .take()
            .ok_or_else(|| compute_error(invalid("instruction claim values already restored")))?;
        Ok(values.read().map_err(compute_error)?.to_vec())
    }
}
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalClaimKernel {
    cpu: OptimizedInstructionClaimReductionKernel<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    device: Combined,
}
impl ProveRounds<F> for MetalClaimKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if self.device.values.is_some()
            && self.device.prefix <= CPU_TAIL
            && self.device.pending.is_none()
        {
            self.cpu.combined.values = self.device.restore()?;
        }
        if self.device.values.is_none() {
            return self.cpu.prove_round(bind, round, claim);
        }
        if round != self.cpu.progress.bound() + usize::from(bind.is_some())
            || (round == 0) != bind.is_none()
        {
            return Err(compute_error(invalid(
                "instruction claim progression mismatch",
            )));
        }
        let mut evaluations = if let Some(challenge) = bind {
            self.cpu.bind(challenge);
            let previous = self.device.prefix;
            self.device.prefix = previous.div_ceil(2);
            self.device
                .step(Some(challenge), previous, &self.cpu.gruen)?
        } else {
            self.device.pending.take().ok_or_else(|| {
                compute_error(invalid("instruction claim initial message missing"))
            })?
        };
        self.cpu
            .gruen
            .checked_round_poly(&mut evaluations, claim, round)
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        if self.device.values.is_some() {
            self.cpu.combined.values = self.device.restore()?;
        }
        self.cpu.finish_rounds(bind)?;
        let point: Vec<_> = self.cpu.bound_challenges.iter().rev().copied().collect();
        self.device
            .native
            .openings(&point)
            .map(|_| ())
            .map_err(compute_error)
    }
}
impl SumcheckKernel<F> for MetalClaimKernel {
    type Relation = InstructionClaimReduction<F>;
    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.cpu.progress.require_complete()?;
        let point: Vec<_> = self.cpu.bound_challenges.iter().rev().copied().collect();
        let values = self.device.native.cached_openings(&point)?;
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: values[4],
            left_lookup_operand: values[8],
            right_lookup_operand: values[9],
            left_instruction_input: values[0],
            right_instruction_input: values[1],
        })
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
