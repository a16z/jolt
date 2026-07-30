//! Metal stage-3 Spartan shift slot.

use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::fr_as_u32s;
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::spartan_shift::OptimizedSpartanShift;
use crate::optimized::trace_record::TraceRecord;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "spartan_shift";

pub struct MetalSpartanShift {
    fallback: OptimizedSpartanShift,
}

impl MetalSpartanShift {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedSpartanShift,
        }
    }
}

impl Default for MetalSpartanShift {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, SpartanShift<Fr>> for MetalSpartanShift {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, SpartanShift<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = SpartanShift<Fr>>>, KernelError<Fr>> {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t < 2 {
            return self.fallback.prepare(session, witness, inputs);
        }
        let r_outer = relation.product_uniskip_tau_low();
        let r_product = relation.product_remainder_opening_point();
        if r_outer.len() != log_t || r_product.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan shift eq+1 point has the wrong variable count",
            });
        }
        let record = TraceRecord::shared(session, witness, log_t)?;
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        let gamma = inputs.challenges.gamma;
        let mut gamma_powers = [Fr::from_u64(1); 5];
        for i in 1..5 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        let pairs = match dispatch_q(context, &record, &outer, &product, &gamma_powers) {
            Ok(q) => [
                (outer.prefix_0, q[0].clone()),
                (outer.prefix_1, q[1].clone()),
                (product.prefix_0, q[2].clone()),
                (product.prefix_1, q[3].clone()),
            ],
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device prepare failed; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        Ok(Box::new(MetalShiftKernel {
            log_t,
            gamma_powers,
            r_outer: r_outer.to_vec(),
            r_product: r_product.to_vec(),
            record,
            context,
            phase: Phase::PrefixSuffix { pairs },
            bound_challenges: Vec::with_capacity(log_t),
        }))
    }
}

enum Phase {
    PrefixSuffix {
        pairs: [(Vec<Fr>, Vec<Fr>); 4],
    },
    Dense {
        eq_plus_one_outer: Vec<Fr>,
        eq_plus_one_product: Vec<Fr>,
        unexpanded_pc: Vec<Fr>,
        pc: Vec<Fr>,
        is_virtual: Vec<Fr>,
        is_first_in_sequence: Vec<Fr>,
        is_noop: Vec<Fr>,
    },
}

struct MetalShiftKernel {
    log_t: usize,
    gamma_powers: [Fr; 5],
    r_outer: Vec<Fr>,
    r_product: Vec<Fr>,
    record: Arc<TraceRecord>,
    context: &'static MetalContext,
    phase: Phase,
    bound_challenges: Vec<Fr>,
}

impl MetalShiftKernel {
    fn rounds_bound(&self) -> usize {
        self.bound_challenges.len()
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<Fr>> {
        let remaining = self.log_t - self.rounds_bound();
        if remaining == 0 {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound { remaining })
        }
    }

    fn cpu_refold(&self, eq_prefix: &[Fr], remaining: usize) -> Vec<[Fr; 5]> {
        let chunk = eq_prefix.len();
        let fold = |chunk_index: usize| {
            let mut out = [Fr::from_u64(0); 5];
            for (offset, &eq) in eq_prefix.iter().enumerate() {
                let t = chunk_index * chunk + offset;
                out[0] += eq * Fr::from_u64(self.record.unexpanded_pc[t]);
                out[1] += eq * Fr::from_u64(self.record.pc[t]);
                if self
                    .record
                    .circuit_flag(t, CircuitFlags::VirtualInstruction)
                {
                    out[2] += eq;
                }
                if self.record.circuit_flag(t, CircuitFlags::IsFirstInSequence) {
                    out[3] += eq;
                }
                if self.record.instruction_flag(t, InstructionFlags::IsNoop) {
                    out[4] += eq;
                }
            }
            out
        };
        #[cfg(feature = "parallel")]
        {
            (0..remaining).into_par_iter().map(fold).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..remaining).map(fold).collect()
        }
    }

    fn transition_to_dense(&mut self) {
        let bound = self.rounds_bound();
        let r_prefix: Vec<Fr> = self.bound_challenges.iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<Fr>::evals(&r_prefix, None);
        let remaining = 1usize << (self.log_t - bound);
        let folds = match dispatch_refold(self.context, &self.record, &eq_prefix, remaining) {
            Ok(folds) => folds,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device refold failed; finishing on CPU");
                self.cpu_refold(&eq_prefix, remaining)
            }
        };
        let recombine = |point: &[Fr]| -> Vec<Fr> {
            let split = EqPlusOnePrefixSuffix::new(point);
            let prefix_0_eval = Polynomial::new(split.prefix_0).evaluate(&r_prefix);
            let prefix_1_eval = Polynomial::new(split.prefix_1).evaluate(&r_prefix);
            split
                .suffix_0
                .iter()
                .zip(&split.suffix_1)
                .map(|(&suffix_0, &suffix_1)| prefix_0_eval * suffix_0 + prefix_1_eval * suffix_1)
                .collect()
        };
        self.phase = Phase::Dense {
            eq_plus_one_outer: recombine(&self.r_outer),
            eq_plus_one_product: recombine(&self.r_product),
            unexpanded_pc: folds.iter().map(|fold| fold[0]).collect(),
            pc: folds.iter().map(|fold| fold[1]).collect(),
            is_virtual: folds.iter().map(|fold| fold[2]).collect(),
            is_first_in_sequence: folds.iter().map(|fold| fold[3]).collect(),
            is_noop: folds.iter().map(|fold| fold[4]).collect(),
        };
    }

    fn bind(&mut self, challenge: Fr) {
        self.bound_challenges.push(challenge);
        if matches!(&self.phase, Phase::PrefixSuffix { pairs } if pairs[0].0.len() == 2) {
            self.transition_to_dense();
            return;
        }
        let bind_table = |table: &mut Vec<Fr>| {
            let half = table.len() / 2;
            for y in 0..half {
                let lo = table[2 * y];
                table[y] = lo + challenge * (table[2 * y + 1] - lo);
            }
            table.truncate(half);
        };
        match &mut self.phase {
            Phase::PrefixSuffix { pairs } => {
                for (p, q) in pairs {
                    bind_table(p);
                    bind_table(q);
                }
            }
            Phase::Dense {
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            } => {
                for table in [
                    eq_plus_one_outer,
                    eq_plus_one_product,
                    unexpanded_pc,
                    pc,
                    is_virtual,
                    is_first_in_sequence,
                    is_noop,
                ] {
                    bind_table(table);
                }
            }
        }
    }
}

impl ProveRounds<Fr> for MetalShiftKernel {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let evals = match &self.phase {
            Phase::PrefixSuffix { pairs } => {
                let mut evals = [Fr::from_u64(0); 2];
                for (p, q) in pairs {
                    for y in 0..p.len() / 2 {
                        let (p_0, p_1) = (p[2 * y], p[2 * y + 1]);
                        let (q_0, q_1) = (q[2 * y], q[2 * y + 1]);
                        evals[0] += p_0 * q_0;
                        evals[1] += (p_1 + p_1 - p_0) * (q_1 + q_1 - q_0);
                    }
                }
                evals
            }
            Phase::Dense {
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            } => {
                let mut evals = [Fr::from_u64(0); 2];
                let pair = |table: &[Fr], y: usize| (table[2 * y], table[2 * y + 1]);
                let extend = |(lo, hi): (Fr, Fr)| hi + hi - lo;
                for y in 0..eq_plus_one_outer.len() / 2 {
                    let eq1o = pair(eq_plus_one_outer, y);
                    let eq1p = pair(eq_plus_one_product, y);
                    let upc = pair(unexpanded_pc, y);
                    let pcs = pair(pc, y);
                    let virt = pair(is_virtual, y);
                    let first = pair(is_first_in_sequence, y);
                    let noop = pair(is_noop, y);
                    evals[0] += eq1o.0
                        * (upc.0
                            + self.gamma_powers[1] * pcs.0
                            + self.gamma_powers[2] * virt.0
                            + self.gamma_powers[3] * first.0);
                    evals[0] += eq1p.0 * self.gamma_powers[4] * (Fr::from_u64(1) - noop.0);
                    evals[1] += extend(eq1o)
                        * (extend(upc)
                            + self.gamma_powers[1] * extend(pcs)
                            + self.gamma_powers[2] * extend(virt)
                            + self.gamma_powers[3] * extend(first));
                    evals[1] +=
                        extend(eq1p) * self.gamma_powers[4] * (Fr::from_u64(1) - extend(noop));
                }
                evals
            }
        };
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalShiftKernel {
    type Relation = SpartanShift<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SpartanShiftOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        self.require_fully_bound()?;
        let Phase::Dense {
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        Ok(SpartanShiftOutputClaims {
            unexpanded_pc: unexpanded_pc[0],
            pc: pc[0],
            is_virtual: is_virtual[0],
            is_first_in_sequence: is_first_in_sequence[0],
            is_noop: is_noop[0],
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        self.require_fully_bound()?;
        let Phase::Dense {
            eq_plus_one_outer,
            eq_plus_one_product,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        for (public, got) in [
            (SpartanShiftPublic::EqPlusOneOuter, eq_plus_one_outer[0]),
            (SpartanShiftPublic::EqPlusOneProduct, eq_plus_one_product[0]),
        ] {
            let id = JoltDerivedId::from(public);
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

fn output_buffer(context: &MetalContext, len: usize) -> Result<OwnedDeviceBuffer<Fr>, MetalError> {
    context.own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), len))
}

fn dispatch_q(
    context: &'static MetalContext,
    record: &TraceRecord,
    outer: &EqPlusOnePrefixSuffix<Fr>,
    product: &EqPlusOnePrefixSuffix<Fr>,
    gamma_powers: &[Fr; 5],
) -> Result<[Vec<Fr>; 4], MetalError> {
    let prefix_len = outer.prefix_0.len();
    let hi_len = outer.suffix_0.len();
    if prefix_len * hi_len != record.len() || product.suffix_0.len() != hi_len {
        return Err(MetalError::UnsupportedShape("Spartan shift prefix/suffix"));
    }
    let output = output_buffer(context, 4 * prefix_len)?;
    let upc = context.wrap_slice(record.unexpanded_pc.as_slice())?;
    let pc = context.wrap_slice(record.pc.as_slice())?;
    let flags = context.wrap_slice(record.flags.as_slice())?;
    let outer_0 = context.wrap_slice(fr_as_u32s(&outer.suffix_0))?;
    let outer_1 = context.wrap_slice(fr_as_u32s(&outer.suffix_1))?;
    let product_0 = context.wrap_slice(fr_as_u32s(&product.suffix_0))?;
    let product_1 = context.wrap_slice(fr_as_u32s(&product.suffix_1))?;
    let gammas = context.wrap_slice(fr_as_u32s(gamma_powers))?;
    testing::note_copied_buffers(
        [
            &upc, &pc, &flags, &outer_0, &outer_1, &product_0, &product_1, &gammas,
        ]
        .into_iter()
        .map(|buffer| u64::from(buffer.was_copied()))
        .sum(),
    );
    let out = output.device_buffer();
    let params = [prefix_len as u32, hi_len as u32];
    let mut pass = context.begin_pass()?;
    pass.dispatch(
        KernelId::ShiftQ,
        &params,
        &[
            &upc, &pc, &flags, &outer_0, &outer_1, &product_0, &product_1, &gammas, &out,
        ],
        prefix_len,
    );
    pass.run()?;
    testing::note_device_round();
    Ok(core::array::from_fn(|lane| {
        output.as_slice()[lane * prefix_len..(lane + 1) * prefix_len].to_vec()
    }))
}

fn dispatch_refold(
    context: &'static MetalContext,
    record: &TraceRecord,
    eq_prefix: &[Fr],
    remaining: usize,
) -> Result<Vec<[Fr; 5]>, MetalError> {
    if remaining * eq_prefix.len() != record.len() {
        return Err(MetalError::UnsupportedShape("Spartan shift refold"));
    }
    let output = output_buffer(context, 5 * remaining)?;
    let upc = context.wrap_slice(record.unexpanded_pc.as_slice())?;
    let pc = context.wrap_slice(record.pc.as_slice())?;
    let flags = context.wrap_slice(record.flags.as_slice())?;
    let eq = context.wrap_slice(fr_as_u32s(eq_prefix))?;
    testing::note_copied_buffers(
        [&upc, &pc, &flags, &eq]
            .into_iter()
            .map(|buffer| u64::from(buffer.was_copied()))
            .sum(),
    );
    let out = output.device_buffer();
    let params = [remaining as u32, eq_prefix.len() as u32];
    let mut pass = context.begin_pass()?;
    pass.dispatch(
        KernelId::ShiftRefold,
        &params,
        &[&upc, &pc, &flags, &eq, &out],
        remaining,
    );
    pass.run()?;
    testing::note_device_round();
    Ok(output
        .as_slice()
        .chunks_exact(5)
        .map(|fold| [fold[0], fold[1], fold[2], fold[3], fold[4]])
        .collect())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::TraceDimensions;
    use jolt_verifier::stages::stage3::spartan_shift::{
        SpartanShiftChallenges, SpartanShiftInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::harness::{probe_input_claim, synthetic_point};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn run_parity(log_t: usize, seed: u64) {
        with_booleanity_backend(log_t, 8, |backend, _| {
            let relation = SpartanShift::new(
                TraceDimensions::new(log_t),
                synthetic_point(log_t, 5 + seed),
                synthetic_point(log_t, 9 + seed),
            );
            let challenges = SpartanShiftChallenges {
                gamma: fr(1747 + seed),
            };
            let claims = SpartanShiftInputClaims::<Fr>::default();
            let points = SpartanShiftInputClaims::<Vec<Fr>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let mut optimized = OptimizedSpartanShift
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();
            let before = device_probe_count();
            let mut metal = MetalSpartanShift::new()
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();
            let mut probe = OptimizedSpartanShift
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();
            let claim = probe_input_claim(probe.as_mut());
            let sumcheck_challenges: Vec<Fr> =
                (0..log_t).map(|i| fr(8887 + seed + 7 * i as u64)).collect();
            let mut bind = None;
            let mut previous = claim;
            for (round, &challenge) in sumcheck_challenges.iter().enumerate() {
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
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    fn force_device() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_SPARTAN_SHIFT", "0");
    }

    #[test]
    fn spartan_shift_matches_optimized_even_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_parity(6, 3);
    }

    #[test]
    fn spartan_shift_matches_optimized_odd_log_t() {
        let _lock = gpu_lock();
        force_device();
        run_parity(5, 71);
    }
}
