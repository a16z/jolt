//! Metal stage-2 instruction claim reduction.

use std::sync::Arc;

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::signed::S256;
use jolt_field::{Accumulator as _, Fr, Ring, WithAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, own_uninit_frs, DeviceRound, Partials};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use crate::optimized::trace_record::TraceRecord;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "instruction_claim_reduction";

pub struct MetalInstructionClaimReduction {
    pub fallback: OptimizedInstructionClaimReduction,
}

impl PrepareKernel<Fr, InstructionClaimReduction<Fr>> for MetalInstructionClaimReduction {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, InstructionClaimReduction<Fr>>,
    ) -> Result<
        Box<dyn SumcheckKernel<Fr, Relation = InstructionClaimReduction<Fr>>>,
        KernelError<Fr>,
    > {
        let tau_low = inputs.relation.tau_low();
        let len = 1usize << tau_low.len();
        if metal_gate(KIND, len) {
            match MetalContext::global() {
                Ok(context) => {
                    let record = TraceRecord::shared(session, witness, tau_low.len())?;
                    match MetalInstructionClaimReductionKernel::new(
                        context,
                        tau_low,
                        record,
                        inputs.challenges.gamma,
                    ) {
                        Ok(kernel) => return Ok(Box::new(kernel)),
                        Err(error) => tracing::warn!(
                            slot = KIND,
                            %error,
                            "device prepare failed; using optimized fallback"
                        ),
                    }
                }
                Err(error) => tracing::warn!(
                    slot = KIND,
                    %error,
                    "no device context; using optimized fallback"
                ),
            }
        }
        self.fallback.prepare(session, witness, inputs)
    }
}

struct MetalInstructionClaimReductionKernel {
    log_t: usize,
    gamma: Fr,
    len: usize,
    initialized: bool,
    rounds_bound: usize,
    gruen: GruenSplitEqPolynomial<Fr>,
    bound_challenges: Vec<Fr>,
    record: Arc<TraceRecord>,
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
    partials: Partials,
    device: DeviceRound,
}

impl MetalInstructionClaimReductionKernel {
    fn new(
        context: &'static MetalContext,
        tau_low: &[Fr],
        record: Arc<TraceRecord>,
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let log_t = tau_low.len();
        let len = 1usize << log_t;
        if record.len() != len || len < 2 {
            return Err(MetalError::UnsupportedShape(
                "instruction claim-reduction record shape",
            ));
        }
        let alloc = |elements| -> Result<OwnedDeviceBuffer<Fr>, MetalError> {
            match own_uninit_frs(context, elements)? {
                Some(buffer) => Ok(buffer),
                None => {
                    context.own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), elements))
                }
            }
        };
        Ok(Self {
            log_t,
            gamma,
            len,
            initialized: false,
            rounds_bound: 0,
            gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            bound_challenges: Vec::with_capacity(log_t),
            record,
            cur: alloc(len)?,
            nxt: alloc(len / 2)?,
            partials: Partials::new(context, 3, len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn gamma_powers(&self) -> [Fr; 5] {
        let gamma_sqr = self.gamma * self.gamma;
        [
            Fr::from_u64(1),
            self.gamma,
            gamma_sqr,
            gamma_sqr * self.gamma,
            gamma_sqr * gamma_sqr,
        ]
    }

    fn cpu_init(&mut self) {
        let powers = self.gamma_powers();
        let record = &self.record;
        let combined_value = |t: usize| {
            powers[0] * Fr::from_u64(record.lookup_output[t])
                + powers[1] * Fr::from_u64(record.left_lookup_operand[t])
                + powers[2] * Fr::from_u128(record.right_lookup_operand[t])
                + powers[3] * Fr::from_u64(record.left_instruction_input[t])
                + powers[4] * Fr::from_i128(record.right_instruction_input[t])
        };
        let values = &mut self.cur.as_mut_slice()[..self.len];
        #[cfg(feature = "parallel")]
        values
            .par_iter_mut()
            .enumerate()
            .for_each(|(t, value)| *value = combined_value(t));
        #[cfg(not(feature = "parallel"))]
        for (t, value) in values.iter_mut().enumerate() {
            *value = combined_value(t);
        }
        self.initialized = true;
    }

    fn cpu_evals(&self) -> [Fr; 3] {
        let values = &self.cur.as_slice()[..self.len];
        let e_out = self.gruen.e_out_current();
        let e_in = self.gruen.e_in_current();
        let in_len = e_in.len();
        debug_assert_eq!(e_out.len() * in_len * 2, self.len);
        let block = |x_out: usize| {
            let mut out = [Fr::from_u64(0); 3];
            for (x_in, &eq) in e_in.iter().enumerate() {
                let pair = 2 * (x_out * in_len + x_in);
                let lo = values[pair];
                let hi = values[pair + 1];
                out[0] += eq * lo;
                out[1] += eq * hi;
                out[2] += eq * (hi + hi - lo);
            }
            out.map(|value| e_out[x_out] * value)
        };
        let add = |mut left: [Fr; 3], right: [Fr; 3]| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };
        #[cfg(feature = "parallel")]
        {
            (0..e_out.len())
                .into_par_iter()
                .map(block)
                .reduce(|| [Fr::from_u64(0); 3], add)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..e_out.len()).map(block).fold([Fr::from_u64(0); 3], add)
        }
    }

    fn cpu_bind(&mut self, challenge: Fr) {
        let half = self.len / 2;
        let source = &self.cur.as_slice()[..self.len];
        let target = &mut self.nxt.as_mut_slice()[..half];
        #[cfg(feature = "parallel")]
        target
            .par_iter_mut()
            .zip(source.par_chunks_exact(2))
            .for_each(|(out, pair)| *out = pair[0] + challenge * (pair[1] - pair[0]));
        #[cfg(not(feature = "parallel"))]
        for (out, pair) in target.iter_mut().zip(source.chunks_exact(2)) {
            *out = pair[0] + challenge * (pair[1] - pair[0]);
        }
        std::mem::swap(&mut self.cur, &mut self.nxt);
        self.len = half;
        self.rounds_bound += 1;
    }

    fn dispatch_init(&self, context: &MetalContext) -> Result<[Fr; 3], MetalError> {
        let groups = self.len / 2;
        let num_tgs = num_threadgroups(groups);
        let lookup_output = context.wrap_slice(self.record.lookup_output.as_slice())?;
        let left_lookup = context.wrap_slice(self.record.left_lookup_operand.as_slice())?;
        let right_lookup = context.wrap_slice(self.record.right_lookup_operand.as_slice())?;
        let left_input = context.wrap_slice(self.record.left_instruction_input.as_slice())?;
        let right_input = context.wrap_slice(self.record.right_instruction_input.as_slice())?;
        let gamma = self.gamma_powers();
        let gamma_buffer = context.wrap_slice(fr_as_u32s(&gamma))?;
        let e_in_buffer = context.wrap_slice(fr_as_u32s(self.gruen.e_in_current()))?;
        let e_out_buffer = context.wrap_slice(fr_as_u32s(self.gruen.e_out_current()))?;
        testing::note_copied_buffers(
            [
                lookup_output.was_copied(),
                left_lookup.was_copied(),
                right_lookup.was_copied(),
                left_input.was_copied(),
                right_input.was_copied(),
                gamma_buffer.was_copied(),
                e_in_buffer.was_copied(),
                e_out_buffer.was_copied(),
            ]
            .into_iter()
            .map(u64::from)
            .sum(),
        );
        let cur = self.cur.device_buffer();
        let partials = self.partials.buffer().device_buffer();
        let params = [
            self.len as u32,
            num_tgs as u32,
            self.gruen.e_in_current().len().trailing_zeros(),
        ];
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IcrInit,
            &params,
            &[
                &lookup_output,
                &left_lookup,
                &right_lookup,
                &left_input,
                &right_input,
                &gamma_buffer,
                &e_in_buffer,
                &e_out_buffer,
                &cur,
                &partials,
            ],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1], sums[2]])
    }

    fn dispatch_round(
        &self,
        context: &MetalContext,
        challenge: Fr,
        groups: usize,
    ) -> Result<[Fr; 3], MetalError> {
        let num_tgs = num_threadgroups(groups);
        let e_in_buffer = context.wrap_slice(fr_as_u32s(self.gruen.e_in_current()))?;
        let e_out_buffer = context.wrap_slice(fr_as_u32s(self.gruen.e_out_current()))?;
        let cur = self.cur.device_buffer();
        let nxt = self.nxt.device_buffer();
        let partials = self.partials.buffer().device_buffer();
        let mut params = vec![
            groups as u32,
            num_tgs as u32,
            self.gruen.e_in_current().len().trailing_zeros(),
        ];
        params.extend_from_slice(&fr_to_u32_limbs(challenge));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IcrRound,
            &params,
            &[&cur, &nxt, &e_in_buffer, &e_out_buffer, &partials],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1], sums[2]])
    }

    fn initial_evals(&mut self) -> [Fr; 3] {
        if let Some(context) = self.device.gated(self.len) {
            match self.dispatch_init(context) {
                Ok(evals) => {
                    self.initialized = true;
                    return evals;
                }
                Err(error) => self.device.failed(&error),
            }
        }
        self.cpu_init();
        self.cpu_evals()
    }

    fn binding_evals(&mut self, challenge: Fr) -> [Fr; 3] {
        let groups = self.len / 4;
        if groups != 0 {
            if let Some(context) = self.device.gated(self.len) {
                match self.dispatch_round(context, challenge, groups) {
                    Ok(evals) => {
                        std::mem::swap(&mut self.cur, &mut self.nxt);
                        self.len /= 2;
                        self.rounds_bound += 1;
                        return evals;
                    }
                    Err(error) => self.device.failed(&error),
                }
            }
        }
        self.cpu_bind(challenge);
        self.cpu_evals()
    }

    fn assemble_message(
        &self,
        q_evals: [Fr; 3],
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let (l_at_0, l_at_1) = self.gruen.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        let evals = q_evals.map(|q| {
            let value = l_eval * q;
            l_eval += l_step;
            value
        });
        let actual = evals[0] + evals[1];
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn operand_claims(&self) -> [Fr; 5] {
        let reversed: Vec<Fr> = self.bound_challenges.iter().rev().copied().collect();
        // `eq(reversed, t) = e_hi[t >> lo_bits] · e_lo[t & mask]` computed on
        // the fly — the exact field value of the full T-sized eq table
        // (multiplication regrouping only), without its T-sized
        // materialization (4.3 GiB @2^27). Word-valued lanes accumulate
        // unreduced (`fmadd_s256`, one Barrett reduce per block ≡ the same
        // sum mod p) instead of one Montgomery conversion + field multiply
        // per lane per cycle.
        let hi_bits = reversed.len() / 2;
        let lo_bits = reversed.len() - hi_bits;
        let e_hi = EqPolynomial::<Fr>::evals(&reversed[..hi_bits], None);
        let e_lo = EqPolynomial::<Fr>::evals(&reversed[hi_bits..], None);
        let lo_mask = (1usize << lo_bits) - 1;
        let block_size = 1usize << 12;
        let record = &self.record;
        let blocks = record.len().div_ceil(block_size);
        let block = |index: usize| {
            let start = index * block_size;
            let end = (start + block_size).min(record.len());
            let mut out = [Fr::from_u64(0); 5];
            // Per `e_hi`-run factoring: rows sharing `t >> lo_bits`
            // accumulate under their `e_lo` weight alone; one `e_hi` scale
            // per run (`e_hi·Σ e_lo·v = Σ (e_hi·e_lo)·v` exactly). Blocks
            // and runs are both power-of-two aligned, so a production block
            // sits inside one run.
            let mut t = start;
            while t < end {
                let hi = t >> lo_bits;
                let run_end = end.min((hi + 1) << lo_bits);
                let mut sums: [<Fr as WithAccumulator>::SignedProductAccumulator; 5] =
                    Default::default();
                for t in t..run_end {
                    let weight = e_lo[t & lo_mask];
                    sums[0].fmadd_s256(weight, &S256::from_u64(record.lookup_output[t]));
                    sums[1].fmadd_s256(weight, &S256::from_u64(record.left_lookup_operand[t]));
                    sums[2].fmadd_s256(weight, &S256::from_u128(record.right_lookup_operand[t]));
                    sums[3].fmadd_s256(weight, &S256::from_u64(record.left_instruction_input[t]));
                    sums[4].fmadd_s256(weight, &S256::from_i128(record.right_instruction_input[t]));
                }
                let e_hi_eval = e_hi[hi];
                for (out, sum) in out.iter_mut().zip(sums) {
                    *out += e_hi_eval * sum.reduce();
                }
                t = run_end;
            }
            out
        };
        let add = |mut left: [Fr; 5], right: [Fr; 5]| {
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
                .reduce(|| [Fr::from_u64(0); 5], add)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..blocks).map(block).fold([Fr::from_u64(0); 5], add)
        }
    }
}

impl ProveRounds<Fr> for MetalInstructionClaimReductionKernel {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let q_evals = if let Some(challenge) = bind {
            self.gruen.bind(challenge);
            self.bound_challenges.push(challenge);
            self.binding_evals(challenge)
        } else {
            self.initial_evals()
        };
        self.assemble_message(q_evals, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.gruen.bind(bind);
        self.bound_challenges.push(bind);
        if !self.initialized {
            self.cpu_init();
        }
        self.cpu_bind(bind);
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalInstructionClaimReductionKernel {
    type Relation = InstructionClaimReduction<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let [lookup_output, left_lookup_operand, right_lookup_operand, left_instruction_input, right_instruction_input] =
            self.operand_claims();
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output,
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input,
            right_instruction_input,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::TraceDimensions;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::with_booleanity_backend;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn run_parity(log_t: usize, seed: u64) {
        with_booleanity_backend(log_t, 8, |backend, _| {
            let tau_low: Vec<Fr> = (0..log_t).map(|i| fr(500 + seed + 41 * i as u64)).collect();
            let gamma = fr(0xC0FF_EE11 + seed);
            let relation =
                InstructionClaimReduction::new(TraceDimensions::new(log_t), tau_low.clone());
            let claims = InstructionClaimReductionInputClaims {
                lookup_output: fr(0),
                left_lookup_operand: fr(0),
                right_lookup_operand: fr(0),
                left_instruction_input: fr(0),
                right_instruction_input: fr(0),
            };
            let points = InstructionClaimReductionInputClaims {
                lookup_output: Vec::new(),
                left_lookup_operand: Vec::new(),
                right_lookup_operand: Vec::new(),
                left_instruction_input: Vec::new(),
                right_instruction_input: Vec::new(),
            };
            let relation_challenges = InstructionClaimReductionChallenges { gamma };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &relation_challenges,
            };

            let mut record_session = ProofSession::default();
            let record = TraceRecord::shared::<Fr>(&mut record_session, backend, log_t).unwrap();
            let powers = [
                fr(1),
                gamma,
                gamma * gamma,
                gamma * gamma * gamma,
                gamma * gamma * gamma * gamma,
            ];
            let eq = EqPolynomial::<Fr>::evals(&tau_low, None);
            let mut claim = fr(0);
            for (t, weight) in eq.into_iter().enumerate() {
                let combo = powers[0] * Fr::from_u64(record.lookup_output[t])
                    + powers[1] * Fr::from_u64(record.left_lookup_operand[t])
                    + powers[2] * Fr::from_u128(record.right_lookup_operand[t])
                    + powers[3] * Fr::from_u64(record.left_instruction_input[t])
                    + powers[4] * Fr::from_i128(record.right_instruction_input[t]);
                claim += weight * combo;
            }

            let mut optimized_session = ProofSession::default();
            let mut optimized = OptimizedInstructionClaimReduction
                .prepare(&mut optimized_session, backend, inputs())
                .unwrap();
            let before = device_probe_count();
            let mut metal_session = ProofSession::default();
            let mut metal = MetalInstructionClaimReduction {
                fallback: OptimizedInstructionClaimReduction,
            }
            .prepare(&mut metal_session, backend, inputs())
            .unwrap();
            let challenges: Vec<Fr> = (0..log_t)
                .map(|i| fr(0xA5A5_1234 + seed + 19 * i as u64))
                .collect();
            let mut bind = None;
            let mut previous = claim;
            for (round, &challenge) in challenges.iter().enumerate() {
                let expected = optimized.prove_round(bind, round, previous).unwrap();
                let got = metal.prove_round(bind, round, previous).unwrap();
                assert_eq!(got.coefficients(), expected.coefficients());
                previous = expected.evaluate(challenge);
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
            metal
                .validate_derived_tables(&relation, &points, &output_points, &relation_challenges)
                .unwrap();
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    fn force_device() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_INSTRUCTION_CLAIM_REDUCTION", "0");
    }

    #[test]
    fn instruction_claim_reduction_matches_optimized_even_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_parity(6, 7);
    }

    #[test]
    fn instruction_claim_reduction_matches_optimized_odd_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_parity(5, 99);
    }
}
