//! Metal stage-2 Spartan product package.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, SpartanProductVirtualizationPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::S256;
use jolt_field::{Accumulator as _, Fr, Ring, WithAccumulator};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::spartan_outer::{dispatch_round, PairTables};
use super::{num_threadgroups, DeviceRound, Partials};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer};
use crate::metal::field::fr_as_u32s;
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::spartan_product::{
    extension_coefficients, OptimizedProductRemainder, OptimizedProductUniskip,
};
use crate::optimized::trace_record::TraceRecord;
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "spartan_product";
const DOMAIN: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

pub struct MetalProductUniskip {
    fallback: OptimizedProductUniskip,
}

impl MetalProductUniskip {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedProductUniskip,
        }
    }
}

impl Default for MetalProductUniskip {
    fn default() -> Self {
        Self::new()
    }
}

struct MetalProductCarry {
    log_t: usize,
    tau_low: Vec<Fr>,
    t1_values: Vec<Fr>,
    record: Arc<TraceRecord>,
    context: &'static MetalContext,
}

impl UniskipKernel<Fr, ProductRemainder<Fr>> for MetalProductUniskip {
    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[Fr],
        witness: &dyn JoltWitnessPlane<Fr>,
    ) -> Result<(), KernelError<Fr>> {
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t < 2 {
            return self.fallback.prepare(session, log_t, tau_low, witness);
        }
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product tau_low must carry log_t challenges",
            });
        }
        let record = TraceRecord::shared(session, witness, log_t)?;
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return OptimizedProductUniskip::prepare_from_record(
                    session, log_t, tau_low, record,
                );
            }
        };
        match dispatch_t1(context, &record, tau_low) {
            Ok(t1_values) => {
                session.park(MetalProductCarry {
                    log_t,
                    tau_low: tau_low.to_vec(),
                    t1_values,
                    record,
                    context,
                });
                Ok(())
            }
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device prepare failed; using optimized fallback");
                OptimizedProductUniskip::prepare_from_record(session, log_t, tau_low, record)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[Fr],
    ) -> Result<UnivariatePoly<Fr>, KernelError<Fr>> {
        let Some(carry) = session.state::<MetalProductCarry>() else {
            return self.fallback.first_round_poly(session, late_tau);
        };
        let &[tau_high] = late_tau else {
            return Err(KernelError::InvariantViolation {
                reason: "the product uni-skip first-round polynomial expects one late challenge",
            });
        };
        let kernel_values = centered_lagrange_evals::<Fr>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &carry.t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

pub struct MetalProductRemainder {
    fallback: OptimizedProductRemainder,
}

impl MetalProductRemainder {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedProductRemainder,
        }
    }
}

impl Default for MetalProductRemainder {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, ProductRemainder<Fr>> for MetalProductRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, ProductRemainder<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ProductRemainder<Fr>>>, KernelError<Fr>> {
        let Some(carry) = session.take::<MetalProductCarry>() else {
            return self.fallback.prepare(session, witness, inputs);
        };
        let fallback_log_t = carry.log_t;
        let fallback_tau_low = carry.tau_low.clone();
        let fallback_record = Arc::clone(&carry.record);
        match MetalProductRemainderKernel::prepare(carry, &inputs)? {
            Ok(kernel) => Ok(Box::new(kernel)),
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device materialization failed; using optimized fallback");
                OptimizedProductUniskip::prepare_from_record(
                    session,
                    fallback_log_t,
                    &fallback_tau_low,
                    fallback_record,
                )?;
                self.fallback.prepare(session, witness, inputs)
            }
        }
    }
}

struct MetalProductRemainderKernel {
    rounds: usize,
    tables: PairTables,
    split_eq: GruenSplitEqPolynomial<Fr>,
    pending_endpoints: Option<(Fr, Fr)>,
    challenges: Vec<Fr>,
    record: Arc<TraceRecord>,
    lagrange_weights: Vec<Fr>,
    partials: Partials,
    device: DeviceRound,
}

impl MetalProductRemainderKernel {
    fn prepare(
        carry: MetalProductCarry,
        inputs: &ProverInputs<'_, Fr, ProductRemainder<Fr>>,
    ) -> Result<Result<Self, MetalError>, KernelError<Fr>> {
        let MetalProductCarry {
            log_t,
            tau_low,
            record,
            context,
            ..
        } = carry;
        let rounds = inputs.relation.rounds();
        if rounds != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "product remainder rounds disagree with the uni-skip carry's log_t",
            });
        }
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let tau_high = inputs.relation.tau_high();
        let lagrange_weights = centered_lagrange_evals::<Fr>(DOMAIN, uniskip_challenge)?;
        let scale = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            &tau_low,
            BindingOrder::LowToHigh,
            Some(scale),
        );
        let cycles = 1usize << log_t;
        let result = (|| {
            let tables = PairTables::new(context, cycles)?;
            let partials = Partials::new(context, 2, cycles / 2)?;
            let endpoints = dispatch_lr(
                context,
                &record,
                &lagrange_weights,
                &split_eq,
                &tables.cur,
                &partials,
            )?;
            Ok(Self {
                rounds,
                tables,
                split_eq,
                pending_endpoints: Some((endpoints[0], endpoints[1])),
                challenges: Vec::with_capacity(rounds),
                record,
                lagrange_weights,
                partials,
                device: DeviceRound::new(context, KIND),
            })
        })();
        Ok(result)
    }

    fn endpoints_cpu(&self) -> (Fr, Fr) {
        let len = self.tables.len;
        let (left, right) = self.tables.cur.as_slice().split_at(len);
        let e_out = self.split_eq.e_out_current();
        let e_in = self.split_eq.e_in_current();
        let in_len = e_in.len();
        debug_assert_eq!(e_out.len() * in_len * 2, len);
        let block = |x_out: usize| {
            let mut q0 = Fr::from_u64(0);
            let mut qinf = Fr::from_u64(0);
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * in_len + x_in);
                q0 += e * left[pair] * right[pair];
                qinf += e * (left[pair + 1] - left[pair]) * (right[pair + 1] - right[pair]);
            }
            (e_out[x_out] * q0, e_out[x_out] * qinf)
        };
        let add = |left: (Fr, Fr), right: (Fr, Fr)| (left.0 + right.0, left.1 + right.1);
        #[cfg(feature = "parallel")]
        {
            (0..e_out.len())
                .into_par_iter()
                .map(block)
                .reduce(|| (Fr::from_u64(0), Fr::from_u64(0)), add)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..e_out.len())
                .map(block)
                .fold((Fr::from_u64(0), Fr::from_u64(0)), add)
        }
    }

    fn bind_and_endpoints(&mut self, challenge: Fr) -> (Fr, Fr) {
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
        let groups = self.tables.len / 4;
        let Some(context) = (groups != 0)
            .then(|| self.device.gated(self.tables.len))
            .flatten()
        else {
            self.tables.bind_cpu(challenge);
            return self.endpoints_cpu();
        };
        match dispatch_round(
            context,
            &self.tables,
            &self.split_eq,
            &self.partials,
            challenge,
            groups,
        ) {
            Ok(endpoints) => {
                self.tables.swap();
                (endpoints[0], endpoints[1])
            }
            Err(error) => {
                self.device.failed(&error);
                self.tables.bind_cpu(challenge);
                self.endpoints_cpu()
            }
        }
    }
}

impl ProveRounds<Fr> for MetalProductRemainderKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let (q_zero, q_infinity) = match bind {
            Some(challenge) => self.bind_and_endpoints(challenge),
            None => {
                self.pending_endpoints
                    .take()
                    .ok_or(SumcheckError::MissingEvaluationSource {
                        kind: "product endpoints",
                    })?
            }
        };
        Ok(self
            .split_eq
            .gruen_poly_deg_3(q_zero, q_infinity, previous_claim))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.split_eq.bind(bind);
        self.challenges.push(bind);
        self.tables.bind_cpu(bind);
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalProductRemainderKernel {
    type Relation = ProductRemainder<Fr>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<Fr, Self::Relation>, SumcheckKernelError<Fr>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let values = claimed_inputs(&self.record, &self.challenges);
        let ids = [
            left_instruction_input_product(),
            right_instruction_input_product(),
            jump_flag_product(),
            write_lookup_output_to_rd_product(),
            lookup_output_product(),
            branch_flag_product(),
            next_is_noop_product(),
            virtual_instruction_product(),
        ];
        let claims: BTreeMap<JoltOpeningId, Fr> = ids.into_iter().zip(values).collect();
        SumcheckOutputClaims::<Fr, Self::Relation>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let ids = std::iter::once(SpartanProductVirtualizationPublic::TauKernel)
            .chain((0..DOMAIN).map(SpartanProductVirtualizationPublic::LagrangeWeight));
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanProductVirtualizationPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanProductVirtualizationPublic::LagrangeWeight(index) => {
                    self.lagrange_weights[index]
                }
                SpartanProductVirtualizationPublic::UniskipLagrangeWeight(_) => continue,
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

fn claimed_inputs(record: &TraceRecord, challenges: &[Fr]) -> [Fr; 8] {
    let reversed: Vec<Fr> = challenges.iter().rev().copied().collect();
    // `eq(reversed, t) = e_hi[t >> lo_bits] · e_lo[t & mask]` computed on the
    // fly — the exact field value of the full T-sized eq table
    // (multiplication regrouping only), without its T-sized materialization
    // (4.3 GiB @2^27). Word-valued lanes accumulate unreduced (`fmadd_s256`,
    // one Barrett reduce per block ≡ the same sum mod p); 0/1 flags stay on
    // the small-scalar window (the word lanes would overflow it).
    let hi_bits = reversed.len() / 2;
    let lo_bits = reversed.len() - hi_bits;
    let e_hi = EqPolynomial::<Fr>::evals(&reversed[..hi_bits], None);
    let e_lo = EqPolynomial::<Fr>::evals(&reversed[hi_bits..], None);
    let lo_mask = (1usize << lo_bits) - 1;
    let block_size = 1usize << 12;
    let blocks = record.len().div_ceil(block_size);
    let block = |index: usize| {
        let start = index * block_size;
        let end = (start + block_size).min(record.len());
        let mut out = [Fr::from_u64(0); 8];
        // Per `e_hi`-run factoring: rows sharing `t >> lo_bits` accumulate
        // under their `e_lo` weight alone; one `e_hi` scale per run
        // (`e_hi·Σ e_lo·v = Σ (e_hi·e_lo)·v` exactly). Blocks and runs are
        // both power-of-two aligned, so a production block sits inside one
        // run.
        let mut t = start;
        while t < end {
            let hi = t >> lo_bits;
            let run_end = end.min((hi + 1) << lo_bits);
            let mut words: [<Fr as WithAccumulator>::SignedProductAccumulator; 3] =
                Default::default();
            let mut flags: [<Fr as WithAccumulator>::SmallScalarAccumulator; 5] =
                Default::default();
            for t in t..run_end {
                let weight = e_lo[t & lo_mask];
                words[0].fmadd_s256(weight, &S256::from_u64(record.left_instruction_input[t]));
                words[1].fmadd_s256(weight, &S256::from_i128(record.right_instruction_input[t]));
                words[2].fmadd_s256(weight, &S256::from_u64(record.lookup_output[t]));
                flags[0].fmadd_u64(
                    weight,
                    u64::from(record.circuit_flag(t, CircuitFlags::Jump)),
                );
                flags[1].fmadd_u64(
                    weight,
                    u64::from(record.circuit_flag(t, CircuitFlags::WriteLookupOutputToRD)),
                );
                flags[2].fmadd_u64(
                    weight,
                    u64::from(record.instruction_flag(t, InstructionFlags::Branch)),
                );
                flags[3].fmadd_u64(weight, u64::from(record.next_is_noop(t)));
                flags[4].fmadd_u64(
                    weight,
                    u64::from(record.circuit_flag(t, CircuitFlags::VirtualInstruction)),
                );
            }
            let [left_input, right_input, lookup_output] = words;
            let [jump, write_lookup, branch, noop, virtual_instruction] = flags;
            let e_hi_eval = e_hi[hi];
            let run = [
                left_input.reduce(),
                right_input.reduce(),
                jump.reduce(),
                write_lookup.reduce(),
                lookup_output.reduce(),
                branch.reduce(),
                noop.reduce(),
                virtual_instruction.reduce(),
            ];
            for (out, run) in out.iter_mut().zip(run) {
                *out += e_hi_eval * run;
            }
            t = run_end;
        }
        out
    };
    let add = |mut left: [Fr; 8], right: [Fr; 8]| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    {
        (0..blocks)
            .into_par_iter()
            .map(block)
            .reduce(|| [Fr::from_u64(0); 8], add)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..blocks).map(block).fold([Fr::from_u64(0); 8], add)
    }
}

fn record_buffers<'a>(
    context: &MetalContext,
    record: &'a TraceRecord,
) -> Result<[DeviceBuffer<'a>; 4], MetalError> {
    let buffers = [
        context.wrap_slice(record.left_instruction_input.as_slice())?,
        context.wrap_slice(record.right_instruction_input.as_slice())?,
        context.wrap_slice(record.lookup_output.as_slice())?,
        context.wrap_slice(record.flags.as_slice())?,
    ];
    testing::note_copied_buffers(
        buffers
            .iter()
            .map(|buffer| u64::from(buffer.was_copied()))
            .sum(),
    );
    Ok(buffers)
}

fn dispatch_t1(
    context: &'static MetalContext,
    record: &TraceRecord,
    tau_low: &[Fr],
) -> Result<Vec<Fr>, MetalError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<Fr>::evals(out_point, None);
    let e_in = EqPolynomial::<Fr>::evals(in_point, None);
    let coefficients: Vec<i64> = extension_coefficients().into_iter().flatten().collect();
    let groups = record.len();
    let num_tgs = num_threadgroups(groups);
    let partials = Partials::new(context, EXTENDED_SIZE, groups)?;
    let record_buffers = record_buffers(context, record)?;
    let e_in_buffer = context.wrap_slice(fr_as_u32s(&e_in))?;
    let e_out_buffer = context.wrap_slice(fr_as_u32s(&e_out))?;
    let coefficient_buffer = context.wrap_slice(&coefficients)?;
    let partial_buffer = partials.buffer().device_buffer();
    let mut buffers: Vec<&DeviceBuffer<'_>> = record_buffers.iter().collect();
    buffers.extend([
        &e_in_buffer,
        &e_out_buffer,
        &coefficient_buffer,
        &partial_buffer,
    ]);
    let params = [groups as u32, num_tgs as u32, e_in.len().trailing_zeros()];
    let mut pass = context.begin_pass()?;
    pass.dispatch(KernelId::ProductT1, &params, &buffers, groups);
    pass.run()?;
    testing::note_device_round();
    Ok(partials.sums(num_tgs))
}

fn dispatch_lr(
    context: &'static MetalContext,
    record: &TraceRecord,
    lagrange: &[Fr],
    split_eq: &GruenSplitEqPolynomial<Fr>,
    tables: &OwnedDeviceBuffer<Fr>,
    partials: &Partials,
) -> Result<Vec<Fr>, MetalError> {
    let len = record.len();
    let groups = len / 2;
    let num_tgs = num_threadgroups(groups);
    let record_buffers = record_buffers(context, record)?;
    let lagrange_buffer = context.wrap_slice(fr_as_u32s(lagrange))?;
    let e_in_buffer = context.wrap_slice(fr_as_u32s(split_eq.e_in_current()))?;
    let e_out_buffer = context.wrap_slice(fr_as_u32s(split_eq.e_out_current()))?;
    let table_buffer = tables.device_buffer();
    let partial_buffer = partials.buffer().device_buffer();
    let mut buffers: Vec<&DeviceBuffer<'_>> = record_buffers.iter().collect();
    buffers.extend([
        &lagrange_buffer,
        &e_in_buffer,
        &e_out_buffer,
        &table_buffer,
        &partial_buffer,
    ]);
    let params = [
        len as u32,
        num_tgs as u32,
        split_eq.e_in_current().len().trailing_zeros(),
    ];
    let mut pass = context.begin_pass()?;
    pass.dispatch(KernelId::ProductLr, &params, &buffers, groups);
    pass.run()?;
    testing::note_device_round();
    Ok(partials.sums(num_tgs))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::NoChallenges;
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainderInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::with_booleanity_backend;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn run_package_parity(log_t: usize, seed: u64) {
        with_booleanity_backend(log_t, 8, |backend, _| {
            let tau_low: Vec<Fr> = (0..log_t).map(|i| fr(31 + seed + 17 * i as u64)).collect();
            let tau_high = fr(1009 + seed);
            let mut optimized_session = ProofSession::default();
            OptimizedProductUniskip
                .prepare(&mut optimized_session, log_t, &tau_low, backend)
                .unwrap();
            let optimized_uniskip = OptimizedProductUniskip
                .first_round_poly(&mut optimized_session, &[tau_high])
                .unwrap();

            let before = device_probe_count();
            let mut metal_session = ProofSession::default();
            let metal_front = MetalProductUniskip::new();
            metal_front
                .prepare(&mut metal_session, log_t, &tau_low, backend)
                .unwrap();
            let metal_uniskip = metal_front
                .first_round_poly(&mut metal_session, &[tau_high])
                .unwrap();
            assert_eq!(metal_uniskip, optimized_uniskip);

            let r0 = fr(2971 + seed);
            let input_claim = fr(31_337 + seed);
            let relation = ProductRemainder::new(
                SpartanProductDimensions::new(log_t),
                r0,
                tau_high,
                tau_low.clone(),
            );
            let claims = product_remainder_input_values_from_uniskip_output(input_claim);
            let points = ProductRemainderInputClaims::<Vec<Fr>>::default();
            let no_challenges = NoChallenges::<Fr>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            };
            let mut optimized = OptimizedProductRemainder
                .prepare(&mut optimized_session, backend, inputs())
                .unwrap();
            let mut metal = MetalProductRemainder::new()
                .prepare(&mut metal_session, backend, inputs())
                .unwrap();
            let challenges: Vec<Fr> = (0..log_t).map(|i| fr(4241 + seed + 3 * i as u64)).collect();
            let mut bind = None;
            let mut previous = input_claim;
            for (round, &challenge) in challenges.iter().enumerate() {
                let optimized_round = optimized.prove_round(bind, round, previous).unwrap();
                let metal_round = metal.prove_round(bind, round, previous).unwrap();
                assert_eq!(metal_round.coefficients(), optimized_round.coefficients());
                previous = optimized_round.evaluate(challenge);
                bind = Some(challenge);
            }
            let last = bind.unwrap();
            optimized.finish_rounds(last).unwrap();
            metal.finish_rounds(last).unwrap();
            assert_eq!(
                metal.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&challenges, &points)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
            metal
                .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    fn force_device() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_SPARTAN_PRODUCT", "0");
    }

    #[test]
    fn product_package_matches_optimized_even_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_package_parity(6, 11);
    }

    #[test]
    fn product_package_matches_optimized_odd_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_package_parity(5, 87);
    }
}
