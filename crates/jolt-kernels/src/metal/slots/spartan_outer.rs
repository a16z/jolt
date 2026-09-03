//! Metal stage-1 Spartan outer package. The shared trace record stays the
//! source of truth; exact row arithmetic, Az/Bz materialization, and the
//! shrinking remainder rounds run on device with host assembly unchanged.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId, SpartanOuterPublic};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::{Fr, Ring};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, own_device_frs, DeviceRound, Partials};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::spartan_outer::{
    claimed_inputs_from_record, extension_coefficients, OptimizedOuterRemainder,
    OptimizedOuterUniskip,
};
use crate::optimized::trace_record::TraceRecord;
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "spartan_outer";
const DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);
const VARIABLE_COUNT: usize = 35;
const CLAIM_TILES: usize = VARIABLE_COUNT.div_ceil(4);
const CLAIMS_MIN_TERMS: usize = 1 << 24;

pub struct MetalOuterUniskip {
    fallback: OptimizedOuterUniskip,
}

impl MetalOuterUniskip {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedOuterUniskip,
        }
    }
}

impl Default for MetalOuterUniskip {
    fn default() -> Self {
        Self::new()
    }
}

struct MetalOuterCarry {
    log_t: usize,
    tau: Vec<Fr>,
    t1_values: Vec<Fr>,
    record: Arc<TraceRecord>,
    context: &'static MetalContext,
}

impl UniskipKernel<Fr, OuterRemainder<Fr>> for MetalOuterUniskip {
    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[Fr],
        witness: &dyn JoltWitnessPlane<Fr>,
    ) -> Result<(), KernelError<Fr>> {
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t == 0 {
            return self.fallback.prepare(session, log_t, tau, witness);
        }
        if tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer tau must carry log_t + 2 challenges",
            });
        }
        let record = TraceRecord::shared(session, witness, log_t)?;
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return OptimizedOuterUniskip::prepare_from_record(session, log_t, tau, record);
            }
        };
        match dispatch_t1(context, &record, &tau[..=log_t]) {
            Ok(t1_values) => {
                session.park(MetalOuterCarry {
                    log_t,
                    tau: tau.to_vec(),
                    t1_values,
                    record,
                    context,
                });
                Ok(())
            }
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device prepare failed; using optimized fallback");
                OptimizedOuterUniskip::prepare_from_record(session, log_t, tau, record)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[Fr],
    ) -> Result<UnivariatePoly<Fr>, KernelError<Fr>> {
        let Some(carry) = session.state::<MetalOuterCarry>() else {
            return self.fallback.first_round_poly(session, late_tau);
        };
        let tau_high = carry.tau[carry.log_t + 1];
        let kernel_values = centered_lagrange_evals::<Fr>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &carry.t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

pub struct MetalOuterRemainder {
    fallback: OptimizedOuterRemainder,
}

impl MetalOuterRemainder {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedOuterRemainder,
        }
    }
}

impl Default for MetalOuterRemainder {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, OuterRemainder<Fr>> for MetalOuterRemainder {
    #[tracing::instrument(skip_all, name = "MetalOuterRemainder::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, OuterRemainder<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = OuterRemainder<Fr>>>, KernelError<Fr>> {
        let Some(carry) = session.take::<MetalOuterCarry>() else {
            return self.fallback.prepare(session, witness, inputs);
        };
        let fallback_log_t = carry.log_t;
        let fallback_tau = carry.tau.clone();
        let fallback_record = Arc::clone(&carry.record);
        match MetalOuterRemainderKernel::prepare(carry, &inputs)? {
            Ok(kernel) => Ok(Box::new(kernel)),
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device materialization failed; using optimized fallback");
                OptimizedOuterUniskip::prepare_from_record(
                    session,
                    fallback_log_t,
                    &fallback_tau,
                    fallback_record,
                )?;
                self.fallback.prepare(session, witness, inputs)
            }
        }
    }
}

struct DerivedWeights {
    az_weights: [Vec<Fr>; 2],
    bz_weights: [Vec<Fr>; 2],
    az_constant: [Fr; 2],
    bz_constant: [Fr; 2],
}

fn derived_weights(
    uniskip_challenge: Fr,
    variable_count: usize,
) -> Result<DerivedWeights, KernelError<Fr>> {
    let matrices = spartan_outer_constraints::<Fr>();
    let columns: Vec<usize> = (1..=variable_count).collect();
    let mut az_weights = [Vec::new(), Vec::new()];
    let mut bz_weights = [Vec::new(), Vec::new()];
    let mut az_constant = [Fr::from_u64(0); 2];
    let mut bz_constant = [Fr::from_u64(0); 2];
    for (index, stream) in [Fr::from_u64(0), Fr::from_u64(1)].into_iter().enumerate() {
        let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
        let weighted = matrices.weighted_columns(&weights, &columns)?;
        az_weights[index] = weighted.a;
        bz_weights[index] = weighted.b;
        let constants = matrices.public_column_contributions(&weights, 0, Fr::from_u64(1))?;
        az_constant[index] = constants.a;
        bz_constant[index] = constants.b;
    }
    Ok(DerivedWeights {
        az_weights,
        bz_weights,
        az_constant,
        bz_constant,
    })
}

pub(super) struct PairTables {
    pub(super) cur: OwnedDeviceBuffer<Fr>,
    pub(super) nxt: OwnedDeviceBuffer<Fr>,
    pub(super) len: usize,
}

impl PairTables {
    pub(super) fn new(context: &'static MetalContext, len: usize) -> Result<Self, MetalError> {
        Ok(Self {
            cur: own_device_frs(context, 2 * len)?,
            nxt: own_device_frs(context, len)?,
            len,
        })
    }

    pub(super) fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.nxt);
        self.len /= 2;
    }

    pub(super) fn bind_cpu(&mut self, challenge: Fr) {
        let len = self.len;
        let half = len / 2;
        for table in 0..2 {
            let source = &self.cur.as_slice()[table * len..(table + 1) * len];
            let target = &mut self.nxt.as_mut_slice()[table * half..(table + 1) * half];
            #[cfg(feature = "parallel")]
            target
                .par_iter_mut()
                .zip(source.par_chunks_exact(2))
                .for_each(|(out, pair)| {
                    *out = pair[0] + challenge * (pair[1] - pair[0]);
                });
            #[cfg(not(feature = "parallel"))]
            for (out, pair) in target.iter_mut().zip(source.chunks_exact(2)) {
                *out = pair[0] + challenge * (pair[1] - pair[0]);
            }
        }
        self.swap();
    }
}

struct MetalOuterRemainderKernel {
    rounds: usize,
    tables: PairTables,
    split_eq: GruenSplitEqPolynomial<Fr>,
    pending_endpoints: Option<(Fr, Fr)>,
    challenges: Vec<Fr>,
    record: Arc<TraceRecord>,
    opening_ids: Vec<JoltOpeningId>,
    derived: DerivedWeights,
    partials: Partials,
    device: DeviceRound,
    context: &'static MetalContext,
}

impl MetalOuterRemainderKernel {
    fn prepare(
        carry: MetalOuterCarry,
        inputs: &ProverInputs<'_, Fr, OuterRemainder<Fr>>,
    ) -> Result<Result<Self, MetalError>, KernelError<Fr>> {
        let MetalOuterCarry {
            log_t,
            tau,
            record,
            context,
            ..
        } = carry;
        let rounds = inputs.relation.rounds();
        if rounds != log_t + 1 {
            return Err(KernelError::InvariantViolation {
                reason: "outer remainder rounds disagree with the uni-skip carry's log_t",
            });
        }
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let tau_high = tau[log_t + 1];
        let tau_low = &tau[..=log_t];
        let lagrange = centered_lagrange_evals::<Fr>(DOMAIN, uniskip_challenge)?;
        let kernel_scale = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            tau_low,
            BindingOrder::LowToHigh,
            Some(kernel_scale),
        );
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let opening_ids: Vec<JoltOpeningId> = dimensions
            .variables()
            .iter()
            .map(|&variable| outer_opening(variable))
            .collect();
        let derived = derived_weights(uniskip_challenge, opening_ids.len())?;
        let cycles = 1usize << log_t;
        let result = (|| {
            let alloc_span = tracing::info_span!("outer_pair_tables_alloc").entered();
            let tables = PairTables::new(context, 2 * cycles)?;
            let partials = Partials::new(context, 2, cycles)?;
            drop(alloc_span);
            let endpoints = dispatch_azbz(
                context,
                &record,
                &lagrange,
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
                opening_ids,
                derived,
                partials,
                device: DeviceRound::new(context, KIND),
                context,
            })
        })();
        Ok(result)
    }

    fn endpoints_cpu(&self) -> (Fr, Fr) {
        let len = self.tables.len;
        let tables = self.tables.cur.as_slice();
        let (az, bz) = tables.split_at(len);
        let e_out = self.split_eq.e_out_current();
        let e_in = self.split_eq.e_in_current();
        let in_len = e_in.len();
        debug_assert_eq!(e_out.len() * in_len * 2, len);
        let block = |x_out: usize| {
            let mut q0 = Fr::from_u64(0);
            let mut qinf = Fr::from_u64(0);
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * in_len + x_in);
                q0 += e * az[pair] * bz[pair];
                qinf += e * (az[pair + 1] - az[pair]) * (bz[pair + 1] - bz[pair]);
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
        let device = if groups == 0 {
            None
        } else {
            self.device.gated(self.tables.len)
        };
        let Some(context) = device else {
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

impl ProveRounds<Fr> for MetalOuterRemainderKernel {
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
                        kind: "outer endpoints",
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

impl SumcheckKernel<Fr> for MetalOuterRemainderKernel {
    type Relation = OuterRemainder<Fr>;

    #[tracing::instrument(skip_all, name = "SpartanOuter::output_claims")]
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<Fr, Self::Relation>, SumcheckKernelError<Fr>> {
        let remaining = self.rounds - self.challenges.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let values = if self.record.len() < CLAIMS_MIN_TERMS {
            claimed_inputs_from_record(Arc::clone(&self.record), &self.challenges)
        } else {
            match dispatch_claimed_inputs(self.context, &self.record, &self.challenges) {
                Ok(values) => values,
                Err(error) => {
                    tracing::warn!(slot = KIND, %error, "device claims failed; using optimized fallback");
                    claimed_inputs_from_record(Arc::clone(&self.record), &self.challenges)
                }
            }
        };
        let claims: BTreeMap<JoltOpeningId, Fr> =
            self.opening_ids.iter().copied().zip(values).collect();
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
        let stream = self.challenges[0];
        let blend = |pair: [&Fr; 2]| *pair[0] + stream * (*pair[1] - *pair[0]);
        let variable_count = self.opening_ids.len();
        let ids = std::iter::once(SpartanOuterPublic::TauKernel)
            .chain((0..variable_count).map(SpartanOuterPublic::AzWeight))
            .chain((0..variable_count).map(SpartanOuterPublic::BzWeight))
            .chain([
                SpartanOuterPublic::AzConstant,
                SpartanOuterPublic::BzConstant,
            ]);
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanOuterPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanOuterPublic::AzWeight(index) => blend([
                    &self.derived.az_weights[0][index],
                    &self.derived.az_weights[1][index],
                ]),
                SpartanOuterPublic::BzWeight(index) => blend([
                    &self.derived.bz_weights[0][index],
                    &self.derived.bz_weights[1][index],
                ]),
                SpartanOuterPublic::AzConstant => {
                    blend([&self.derived.az_constant[0], &self.derived.az_constant[1]])
                }
                SpartanOuterPublic::BzConstant => {
                    blend([&self.derived.bz_constant[0], &self.derived.bz_constant[1]])
                }
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

/// Semantic `2^256 mod p`: multiplying a standard-form device sum by this
/// returns it to the canonical Montgomery representation.
fn mont_form_fix() -> Fr {
    let x = Fr::from_u64(1u64 << 32);
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x4
}

/// Columns whose lazy-kernel partial stays in Montgomery form (boolean flag
/// openings fold the weight directly); mirrors `jk_outer_claim_is_bool`.
fn outer_claim_is_bool(column: usize) -> bool {
    matches!(column, 3 | 17 | 18) || column >= 20
}

fn record_buffers<'a>(
    context: &MetalContext,
    record: &'a TraceRecord,
) -> Result<[DeviceBuffer<'a>; 17], MetalError> {
    let buffers = [
        context.wrap_slice(record.pc.as_slice())?,
        context.wrap_slice(record.unexpanded_pc.as_slice())?,
        context.wrap_slice(record.imm.as_slice())?,
        context.wrap_slice(record.registers.rs1_value.as_slice())?,
        context.wrap_slice(record.registers.rs2_value.as_slice())?,
        context.wrap_slice(record.registers.rd_post_value.as_slice())?,
        context.wrap_slice(record.ram_address.as_slice())?,
        context.wrap_slice(record.ram_values.pre_values.as_slice())?,
        context.wrap_slice(record.ram_values.post_values.as_slice())?,
        context.wrap_slice(record.left_lookup_operand.as_slice())?,
        context.wrap_slice(record.right_lookup_operand.as_slice())?,
        context.wrap_slice(record.left_instruction_input.as_slice())?,
        context.wrap_slice(record.right_instruction_input.as_slice())?,
        context.wrap_slice(record.product_magnitude_lo.as_slice())?,
        context.wrap_slice(record.product_magnitude_hi.as_slice())?,
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

#[tracing::instrument(skip_all, name = "outer_claims")]
fn dispatch_claimed_inputs(
    context: &'static MetalContext,
    record: &TraceRecord,
    challenges: &[Fr],
) -> Result<Vec<Fr>, MetalError> {
    let reversed: Vec<Fr> = challenges[1..].iter().rev().copied().collect();
    let hi_bits = reversed.len() / 2;
    let e_out = jolt_poly::EqPolynomial::<Fr>::evals(&reversed[..hi_bits], None);
    let e_in = jolt_poly::EqPolynomial::<Fr>::evals(&reversed[hi_bits..], None);
    if e_out.len() * e_in.len() != record.len() {
        return Err(MetalError::UnsupportedShape("outer claims geometry"));
    }
    let partials = Partials::new(context, VARIABLE_COUNT, e_out.len() * 256)?;
    let record_buffers = record_buffers(context, record)?;
    let e_in_buffer = context.wrap_slice(fr_as_u32s(&e_in))?;
    let partial_buffer = partials.buffer().device_buffer();
    let mut buffers: Vec<&DeviceBuffer<'_>> = record_buffers.iter().collect();
    buffers.extend([&e_in_buffer, &partial_buffer]);
    let params = [record.len() as u32, e_in.len() as u32, e_out.len() as u32];
    let threads = CLAIM_TILES * e_out.len() * 256;
    let mut pass = context.begin_pass()?;
    pass.dispatch(KernelId::OuterClaimsLazy, &params, &buffers, threads);
    pass.run()?;
    testing::note_device_round();

    let form_fix = mont_form_fix();
    let raw = partials.buffer().as_slice();
    Ok((0..VARIABLE_COUNT)
        .map(|column| {
            let sum = raw[column * e_out.len()..(column + 1) * e_out.len()]
                .iter()
                .zip(&e_out)
                .fold(Fr::from_u64(0), |sum, (partial, weight)| {
                    sum + *weight * *partial
                });
            if outer_claim_is_bool(column) {
                sum
            } else {
                sum * form_fix
            }
        })
        .collect())
}

#[tracing::instrument(skip_all, name = "outer_t1")]
fn dispatch_t1(
    context: &'static MetalContext,
    record: &TraceRecord,
    tau_low: &[Fr],
) -> Result<Vec<Fr>, MetalError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = jolt_poly::EqPolynomial::<Fr>::evals(out_point, None);
    let e_in = jolt_poly::EqPolynomial::<Fr>::evals(in_point, None);
    let coefficients: Vec<i64> = extension_coefficients()
        .into_iter()
        .flat_map(|(_, row)| row)
        .collect();
    let groups = record.len();
    let num_tgs = num_threadgroups(groups);
    let partials = Partials::new(context, 9, groups)?;
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
    pass.dispatch(KernelId::OuterT1Lazy, &params, &buffers, groups);
    pass.run()?;
    testing::note_device_round();
    let form_fix = mont_form_fix();
    let reduced = partials.sums(num_tgs);
    let mut values = vec![Fr::from_u64(0); EXTENDED_SIZE];
    for (((position, _), value), _) in extension_coefficients().into_iter().zip(reduced).zip(0..9) {
        values[position] = value * form_fix;
    }
    Ok(values)
}

#[tracing::instrument(skip_all, name = "outer_azbz")]
fn dispatch_azbz(
    context: &'static MetalContext,
    record: &TraceRecord,
    lagrange: &[Fr],
    split_eq: &GruenSplitEqPolynomial<Fr>,
    tables: &OwnedDeviceBuffer<Fr>,
    partials: &Partials,
) -> Result<Vec<Fr>, MetalError> {
    let groups = record.len();
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
        groups as u32,
        num_tgs as u32,
        split_eq.e_in_current().len().trailing_zeros(),
    ];
    let mut pass = context.begin_pass()?;
    pass.dispatch(KernelId::OuterAzbzLazy, &params, &buffers, groups);
    pass.run()?;
    testing::note_device_round();
    Ok(partials.sums(num_tgs))
}

#[tracing::instrument(skip_all, name = "outer_round")]
pub(super) fn dispatch_round(
    context: &'static MetalContext,
    tables: &PairTables,
    split_eq: &GruenSplitEqPolynomial<Fr>,
    partials: &Partials,
    challenge: Fr,
    groups: usize,
) -> Result<Vec<Fr>, MetalError> {
    let num_tgs = num_threadgroups(groups);
    let e_in_buffer = context.wrap_slice(fr_as_u32s(split_eq.e_in_current()))?;
    let e_out_buffer = context.wrap_slice(fr_as_u32s(split_eq.e_out_current()))?;
    let cur = tables.cur.device_buffer();
    let nxt = tables.nxt.device_buffer();
    let partial_buffer = partials.buffer().device_buffer();
    let mut params = vec![
        groups as u32,
        num_tgs as u32,
        split_eq.e_in_current().len().trailing_zeros(),
        tables.len as u32,
    ];
    params.extend_from_slice(&fr_to_u32_limbs(challenge));
    let mut pass = context.begin_pass()?;
    pass.dispatch(
        KernelId::OuterRound,
        &params,
        &[&cur, &nxt, &e_in_buffer, &e_out_buffer, &partial_buffer],
        groups,
    );
    pass.run()?;
    testing::note_device_round();
    Ok(partials.sums(num_tgs))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::NoChallenges;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainderInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::with_booleanity_backend;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn run_package_parity(log_t: usize, seed: u64) {
        with_booleanity_backend(log_t, 8, |backend, _| {
            let tau: Vec<Fr> = (0..log_t + 2)
                .map(|i| fr(29 + seed + 13 * i as u64))
                .collect();
            let mut optimized_session = ProofSession::default();
            OptimizedOuterUniskip
                .prepare(&mut optimized_session, log_t, &tau, backend)
                .unwrap();
            let optimized_uniskip = OptimizedOuterUniskip
                .first_round_poly(&mut optimized_session, &[])
                .unwrap();

            let before = device_probe_count();
            let mut metal_session = ProofSession::default();
            let metal_front = MetalOuterUniskip::new();
            metal_front
                .prepare(&mut metal_session, log_t, &tau, backend)
                .unwrap();
            let metal_uniskip = metal_front
                .first_round_poly(&mut metal_session, &[])
                .unwrap();
            assert_eq!(metal_uniskip, optimized_uniskip);

            let r0 = fr(40961 + seed);
            let input_claim = fr(7_770_001 + seed);
            let relation =
                OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
            let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
            let points = OuterRemainderInputClaims::<Vec<Fr>>::default();
            let no_challenges = NoChallenges::<Fr>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            };
            let mut optimized = OptimizedOuterRemainder
                .prepare(&mut optimized_session, backend, inputs())
                .unwrap();
            let mut metal = MetalOuterRemainder::new()
                .prepare(&mut metal_session, backend, inputs())
                .unwrap();
            let challenges: Vec<Fr> = (0..=log_t)
                .map(|i| fr(7919 + seed + 31 * i as u64))
                .collect();
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
        std::env::set_var("JOLT_METAL_MIN_TERMS_SPARTAN_OUTER", "0");
    }

    #[test]
    fn outer_package_matches_optimized_even_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_package_parity(6, 5);
    }

    #[test]
    fn outer_package_matches_optimized_odd_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_package_parity(5, 91);
    }
}
