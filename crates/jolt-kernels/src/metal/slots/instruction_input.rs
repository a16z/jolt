//! Metal instruction input-virtualization (stage 3): device twin of
//! [`OptimizedInstructionInput`], byte-identical round polynomials by
//! construction.
//!
//! Round 0 reduces three coefficients on the device directly from the shared
//! trace record: Boolean endpoint selection yields q(0), q(1), while flag
//! transitions times operand slopes yield the quadratic coefficient; the
//! host reconstructs q(2), q(3). The first bind then pays the optimized
//! tier's big bill (eight dense `T/2`
//! tables materialized through per-row field promotion): one
//! `jk_instr_input_bind_native` dispatch reads the record's native lanes
//! in place (u64 values, i128 immediates, packed flag words — all
//! zero-copy-eligible at production sizes), promotes them in-register
//! (`mont_mul` by R², the same canonical residues `ToField` produces),
//! folds with the challenge, writes the eight tables table-major into one
//! device buffer, and accumulates that round's four `q(t)` sums. Later
//! rounds are one fused `jk_instr_input_round` fold+eval dispatch each over
//! the table-major ping-pong pair.
//!
//! The device's flat `e_out·e_in·q` products regroup the CPU's nested
//! `e_out·(Σ e_in·q)` sums and its integer-pipeline round 0 maps through
//! the ring homomorphism ℤ → Fr; both are exact, so every round polynomial
//! (assembled host-side by the shared [`assemble_message`]) is
//! byte-identical — pinned by the lockstep parity tests below and by
//! `byte_diff`'s metal arms.

use std::sync::Arc;

use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_riscv::InstructionFlags;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, own_uninit_frs, wrap_eq, DeviceRound, Partials};
use crate::metal::buffers::OwnedDeviceBuffer;
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::instruction_input::{
    assemble_message, native_q_evals, InstructionInputRow, OptimizedInstructionInput, NUM_TABLES,
};
use crate::optimized::trace_record::{RecordRows, TraceRecord};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};

const KIND: &str = "instruction_input";

fn native_q0_params(gruen: &GruenSplitEqPolynomial<Fr>, gamma: Fr, len: usize) -> Vec<u32> {
    let groups = len / 2;
    let mut params = vec![
        groups as u32,
        num_threadgroups(groups) as u32,
        gruen.e_in_current().len().trailing_zeros(),
        TraceRecord::instruction_flag_bit(InstructionFlags::LeftOperandIsRs1Value),
        TraceRecord::instruction_flag_bit(InstructionFlags::LeftOperandIsPC),
        TraceRecord::instruction_flag_bit(InstructionFlags::RightOperandIsRs2Value),
        TraceRecord::instruction_flag_bit(InstructionFlags::RightOperandIsImm),
    ];
    params.extend_from_slice(&fr_to_u32_limbs(gamma));
    params
}

/// Slot front: device kernel above the [`metal_gate`] threshold, the
/// optimized fallback below it or on any device failure.
pub struct MetalInstructionInput {
    pub fallback: OptimizedInstructionInput,
}

impl PrepareKernel<Fr, InstructionInput<Fr>> for MetalInstructionInput {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, InstructionInput<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = InstructionInput<Fr>>>, KernelError<Fr>> {
        let r_product = inputs.relation.product_remainder_opening_point();
        if metal_gate(KIND, 1usize << r_product.len()) {
            match MetalContext::global() {
                Ok(context) => {
                    // Structural errors propagate — the fallback would fail
                    // identically; only device failures fall back.
                    let record = TraceRecord::shared(session, witness, r_product.len())?;
                    match MetalInstructionInputKernel::new(
                        context,
                        r_product,
                        record,
                        inputs.challenges.gamma,
                    ) {
                        Ok(kernel) => return Ok(Box::new(kernel)),
                        Err(error) => tracing::warn!(
                            slot = KIND,
                            %error,
                            "device prepare failed; using the optimized fallback"
                        ),
                    }
                }
                Err(error) => tracing::warn!(
                    slot = KIND,
                    %error,
                    "no device context; using the optimized fallback"
                ),
            }
        }
        self.fallback.prepare(session, witness, inputs)
    }
}

/// The column state: the shared record until the first bind, the table-major
/// device pair afterwards.
enum State {
    Native(Arc<TraceRecord>),
    Dense,
}

struct MetalInstructionInputKernel {
    log_t: usize,
    gamma: Fr,
    rounds_bound: usize,
    /// Per-table logical length of the live representation: `T` while
    /// native, halving with every bind once dense.
    len: usize,
    gruen: GruenSplitEqPolynomial<Fr>,
    state: State,
    /// Table-major ping-pong pair: table `i`'s `len` evaluations at element
    /// `i·len`. `cur` is written whole by the first bind (uninit until
    /// then); capacities alternate between `8·T/2` and `8·T/4`.
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
    partials: Partials,
    device: DeviceRound,
}

impl MetalInstructionInputKernel {
    fn new(
        context: &'static MetalContext,
        r_product: &[Fr],
        record: Arc<TraceRecord>,
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let log_t = r_product.len();
        let t = 1usize << log_t;
        if record.len() != t {
            // The fallback's structural TableSizeMismatch reports it.
            return Err(MetalError::UnsupportedShape(
                "trace record length differs from the sumcheck domain",
            ));
        }
        let alloc = |len| {
            own_uninit_frs(context, len)?.ok_or(MetalError::UnsupportedShape(
                "instruction-input tables below the no-copy size",
            ))
        };
        Ok(Self {
            log_t,
            gamma,
            rounds_bound: 0,
            len: t,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            state: State::Native(record),
            cur: alloc(NUM_TABLES * (t / 2))?,
            nxt: alloc(NUM_TABLES * (t / 4))?,
            partials: Partials::new(context, 4, t / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    /// A never-device twin for the lockstep tests' CPU reference.
    #[cfg(test)]
    fn disabled(
        context: &'static MetalContext,
        r_product: &[Fr],
        record: Arc<TraceRecord>,
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let mut kernel = Self::new(context, r_product, record, gamma)?;
        kernel.device = DeviceRound::disabled(KIND);
        Ok(kernel)
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    fn expand_q0(q0: Fr, q1: Fr, quadratic: Fr) -> [Fr; 4] {
        let q2 = q1 + q1 - q0 + quadratic + quadratic;
        let q3 = q1 + q1 + q1 - q0 - q0
            + quadratic
            + quadratic
            + quadratic
            + quadratic
            + quadratic
            + quadratic;
        [q0, q1, q2, q3]
    }

    /// Round 0 over native lanes: endpoint selection plus one quadratic
    /// coefficient, reduced on device and expanded to q(0..=3) on host.
    fn dispatch_native_q0(
        &self,
        context: &MetalContext,
        record: &TraceRecord,
    ) -> Result<[Fr; 4], MetalError> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let groups = self.len / 2;
        let num_tgs = num_threadgroups(groups);
        let params = native_q0_params(&self.gruen, self.gamma, self.len);

        let flags = context.wrap_slice(record.flags.as_slice())?;
        let rs1 = context.wrap_slice(record.registers.rs1_value.as_slice())?;
        let upc = context.wrap_slice(record.unexpanded_pc.as_slice())?;
        let rs2 = context.wrap_slice(record.registers.rs2_value.as_slice())?;
        let imm = context.wrap_slice(record.imm.as_slice())?;
        let e_in_buffer = context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = context.wrap_slice(fr_as_u32s(e_out))?;
        testing::note_copied_buffers(
            u64::from(flags.was_copied())
                + u64::from(rs1.was_copied())
                + u64::from(upc.was_copied())
                + u64::from(rs2.was_copied())
                + u64::from(imm.was_copied())
                + u64::from(e_in_buffer.was_copied())
                + u64::from(e_out_buffer.was_copied()),
        );
        let partials = self.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::InstrInputQ0,
            &params,
            &[
                &flags,
                &rs1,
                &upc,
                &rs2,
                &imm,
                &e_in_buffer,
                &e_out_buffer,
                &partials,
            ],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        let sums = self.partials.sums(num_tgs);
        Ok(Self::expand_q0(sums[0], sums[1], sums[2]))
    }

    fn round0_evals(&mut self) -> [Fr; 4] {
        let record = match &self.state {
            State::Native(record) => Arc::clone(record),
            State::Dense => return self.cpu_dense_evals(),
        };
        if let Some(context) = self.device.gated(self.len / 2) {
            match self.dispatch_native_q0(context, &record) {
                Ok(evals) => return evals,
                Err(error) => self.device.failed(&error),
            }
        }
        native_q_evals(&self.gruen, self.gamma, &RecordRows::Record(record))
    }

    /// The fused first-bind dispatch: native lanes → dense `cur` plus the
    /// round's q sums. The eq levels are the CURRENT (post-`gruen.bind`)
    /// ones — the caller binds eq first.
    fn dispatch_bind_native(
        &self,
        context: &MetalContext,
        record: &TraceRecord,
        r: Fr,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let num_tgs = num_threadgroups(groups);
        let mut params = vec![
            groups as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            (self.len / 2) as u32,
            TraceRecord::instruction_flag_bit(InstructionFlags::LeftOperandIsRs1Value),
            TraceRecord::instruction_flag_bit(InstructionFlags::LeftOperandIsPC),
            TraceRecord::instruction_flag_bit(InstructionFlags::RightOperandIsRs2Value),
            TraceRecord::instruction_flag_bit(InstructionFlags::RightOperandIsImm),
        ];
        params.extend_from_slice(&fr_to_u32_limbs(r));
        params.extend_from_slice(&fr_to_u32_limbs(self.gamma));

        let flags = context.wrap_slice(record.flags.as_slice())?;
        let rs1 = context.wrap_slice(record.registers.rs1_value.as_slice())?;
        let upc = context.wrap_slice(record.unexpanded_pc.as_slice())?;
        let rs2 = context.wrap_slice(record.registers.rs2_value.as_slice())?;
        let imm = context.wrap_slice(record.imm.as_slice())?;
        let e_in_buffer = context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = context.wrap_slice(fr_as_u32s(e_out))?;
        testing::note_copied_buffers(
            u64::from(flags.was_copied())
                + u64::from(rs1.was_copied())
                + u64::from(upc.was_copied())
                + u64::from(rs2.was_copied())
                + u64::from(imm.was_copied())
                + u64::from(e_in_buffer.was_copied())
                + u64::from(e_out_buffer.was_copied()),
        );
        let dense = self.cur.device_buffer();
        let partials = self.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::InstrInputBindNative,
            &params,
            &[
                &flags,
                &rs1,
                &upc,
                &rs2,
                &imm,
                &dense,
                &e_in_buffer,
                &e_out_buffer,
                &partials,
            ],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// One fused dense round: fold all eight tables and accumulate the four
    /// q sums, one dispatch, one command buffer, one wait.
    fn dispatch_dense_round(
        &self,
        context: &MetalContext,
        r: Fr,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let num_tgs = num_threadgroups(groups);
        let mut params = vec![
            groups as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            self.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(r));
        params.extend_from_slice(&fr_to_u32_limbs(self.gamma));
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        let cur = self.cur.device_buffer();
        let nxt = self.nxt.device_buffer();
        let partials = self.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::InstrInputRound,
            &params,
            &[&cur, &nxt, &e_in_buffer, &e_out_buffer, &partials],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// Host materialization of the first bind — recovery and tail path,
    /// mirroring the optimized kernel's `v₀ + r·(v₁ − v₀)` per table.
    fn cpu_bind0(&mut self, record: &Arc<TraceRecord>, r: Fr) {
        let half = self.len / 2;
        let rows = RecordRows::<InstructionInputRow>::Record(Arc::clone(record));
        let dense = &mut self.cur.as_mut_slice()[..NUM_TABLES * half];
        let fill = |table: usize, y: usize| {
            let even = rows.row(2 * y).field_values::<Fr>()[table];
            let odd = rows.row(2 * y + 1).field_values::<Fr>()[table];
            even + r * (odd - even)
        };
        #[cfg(feature = "parallel")]
        dense
            .par_chunks_exact_mut(half)
            .enumerate()
            .for_each(|(table, out)| {
                out.par_iter_mut()
                    .enumerate()
                    .for_each(|(y, slot)| *slot = fill(table, y));
            });
        #[cfg(not(feature = "parallel"))]
        for (table, out) in dense.chunks_exact_mut(half).enumerate() {
            for (y, slot) in out.iter_mut().enumerate() {
                *slot = fill(table, y);
            }
        }
        self.state = State::Dense;
        self.bind_bookkeeping();
    }

    /// Host fold of every table's `cur[..len]` into `nxt` — the CPU twin of
    /// the device bind, for below-threshold tail rounds and post-failure
    /// recovery (a failed dispatch never corrupts `cur`).
    fn cpu_dense_bind(&mut self, r: Fr) {
        let len = self.len;
        let half = len / 2;
        let src = &self.cur.as_slice()[..NUM_TABLES * len];
        let dst = &mut self.nxt.as_mut_slice()[..NUM_TABLES * half];
        for (table, out) in dst.chunks_exact_mut(half).enumerate() {
            let pairs = &src[table * len..(table + 1) * len];
            #[cfg(feature = "parallel")]
            out.par_iter_mut()
                .zip(pairs.par_chunks_exact(2))
                .for_each(|(slot, pair)| *slot = pair[0] + r * (pair[1] - pair[0]));
            #[cfg(not(feature = "parallel"))]
            for (slot, pair) in out.iter_mut().zip(pairs.chunks_exact(2)) {
                *slot = pair[0] + r * (pair[1] - pair[0]);
            }
        }
        std::mem::swap(&mut self.cur, &mut self.nxt);
        self.bind_bookkeeping();
    }

    /// The dense `q` sums over the unified-memory tables, mirroring the
    /// optimized kernel's `dense_q_evals` (table-major reads; same exact
    /// field sums under regrouping).
    fn cpu_dense_evals(&self) -> [Fr; 4] {
        let len = self.len;
        let gamma = self.gamma;
        let tables = self.cur.as_slice();
        self.gruen.par_fold_out_in(
            || [Fr::from_u64(0); 4],
            |acc, y, _x_in, e_in| {
                let mut evals = [Fr::from_u64(0); NUM_TABLES];
                let mut steps = [Fr::from_u64(0); NUM_TABLES];
                for i in 0..NUM_TABLES {
                    let lo = tables[i * len + 2 * y];
                    evals[i] = lo;
                    steps[i] = tables[i * len + 2 * y + 1] - lo;
                }
                for value in acc.iter_mut() {
                    let right = evals[4] * evals[5] + evals[6] * evals[7];
                    let left = evals[0] * evals[1] + evals[2] * evals[3];
                    *value += e_in * (right + gamma * left);
                    for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                        *eval += *step;
                    }
                }
            },
            |_x_out, e_out, acc| acc.map(|value| e_out * value),
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        )
    }

    /// The binding rounds' q sums: device fused fold+eval when healthy and
    /// above gate, host twins otherwise. The caller has already bound eq.
    fn binding_round_evals(&mut self, r: Fr) -> [Fr; 4] {
        let groups = self.len / 4;
        let eq_tiles = self.gruen.e_out_current_len() * self.gruen.e_in_current_len() == groups;
        let device = if groups == 0 || !eq_tiles {
            None
        } else {
            self.device.gated(self.len)
        };
        let native_record = match &self.state {
            State::Native(record) => Some(Arc::clone(record)),
            State::Dense => None,
        };
        if let Some(record) = native_record {
            if let Some(context) = device {
                match self.dispatch_bind_native(context, &record, r, groups) {
                    Ok(sums) => {
                        self.state = State::Dense;
                        self.bind_bookkeeping();
                        return [sums[0], sums[1], sums[2], sums[3]];
                    }
                    Err(error) => self.device.failed(&error),
                }
            }
            self.cpu_bind0(&record, r);
        } else {
            if let Some(context) = device {
                match self.dispatch_dense_round(context, r, groups) {
                    Ok(sums) => {
                        std::mem::swap(&mut self.cur, &mut self.nxt);
                        self.bind_bookkeeping();
                        return [sums[0], sums[1], sums[2], sums[3]];
                    }
                    Err(error) => self.device.failed(&error),
                }
            }
            self.cpu_dense_bind(r);
        }
        self.cpu_dense_evals()
    }
}

impl ProveRounds<Fr> for MetalInstructionInputKernel {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        // The eq factor binds host-side FIRST (O(1) scalar work) — exactly
        // once per challenge, so a device failure below cannot re-bind it.
        if let Some(challenge) = bind {
            self.gruen.bind(challenge);
        }
        let q_evals = match bind {
            None => self.round0_evals(),
            Some(challenge) => self.binding_round_evals(challenge),
        };
        assemble_message(&self.gruen, q_evals, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.gruen.bind(bind);
        let native_record = match &self.state {
            State::Native(record) => Some(Arc::clone(record)),
            State::Dense => None,
        };
        match native_record {
            Some(record) => self.cpu_bind0(&record, bind),
            None => self.cpu_dense_bind(bind),
        }
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalInstructionInputKernel {
    type Relation = InstructionInput<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let values: [Fr; NUM_TABLES] = match &self.state {
            // Bindless extraction happens only for `log_t = 0` geometries.
            State::Native(record) => RecordRows::<InstructionInputRow>::Record(Arc::clone(record))
                .row(0)
                .field_values(),
            State::Dense => {
                let tables = self.cur.as_slice();
                core::array::from_fn(|i| tables[i * self.len])
            }
        };
        let [left_operand_is_rs1, rs1_value, left_operand_is_pc, unexpanded_pc, right_operand_is_rs2, rs2_value, right_operand_is_imm, imm] =
            values;
        Ok(InstructionInputOutputClaims {
            left_operand_is_rs1,
            rs1_value,
            left_operand_is_pc,
            unexpanded_pc,
            right_operand_is_rs2,
            rs2_value,
            right_operand_is_imm,
            imm,
        })
    }

    /// The split-eq scalar against the verifier's `derive_output_term` — the
    /// same drift detector the optimized kernel runs.
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
        let id = JoltDerivedId::from(InstructionInputPublic::EqProduct);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

/// Lockstep parity against the optimized kernel over a synthetic trace
/// record, device path forced and probed — the same drive as the optimized
/// tier's own parity test plus the device/CPU handoff cases.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::TraceDimensions;
    use jolt_verifier::stages::relations::ConcreteSumcheck;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::mmap_vec::MmapVec;
    use crate::optimized::lifetime_trace::LifetimeTag;
    use crate::optimized::ram_trace::{RamAccessColumns, RamAccessValues};
    use crate::optimized::trace_record::RegisterLanes;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x1357_9BDF_2468_ACE0 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A synthetic record exercising every lane this slot reads: random
    /// flag bits at the four packed positions, full-range u64 values, and
    /// signed immediates spanning the i128 lane.
    fn synthetic_record(log_t: usize, seed: u64) -> Arc<TraceRecord> {
        let t = 1usize << log_t;
        let mut state = seed;
        let mut flags = Vec::with_capacity(t);
        let mut imm = Vec::with_capacity(t);
        let mut rs1_value = Vec::with_capacity(t);
        let mut rs2_value = Vec::with_capacity(t);
        let mut unexpanded_pc = Vec::with_capacity(t);
        for index in 0..t {
            let raw = splitmix(&mut state);
            let bit = |flag: InstructionFlags, on: bool| {
                u32::from(on) << TraceRecord::instruction_flag_bit(flag)
            };
            flags.push(
                bit(InstructionFlags::LeftOperandIsRs1Value, raw & 1 != 0)
                    | bit(InstructionFlags::LeftOperandIsPC, raw & 2 != 0)
                    | bit(InstructionFlags::RightOperandIsRs2Value, raw & 4 != 0)
                    | bit(InstructionFlags::RightOperandIsImm, raw & 8 != 0),
            );
            rs1_value.push(splitmix(&mut state));
            unexpanded_pc.push(splitmix(&mut state));
            rs2_value.push(splitmix(&mut state));
            let wide = ((splitmix(&mut state) as i128) << 64) | splitmix(&mut state) as i128;
            imm.push(if index % 3 == 0 { -wide } else { wide });
        }
        Arc::new(TraceRecord {
            pc: MmapVec::zeroed(t),
            unexpanded_pc: unexpanded_pc.into_iter().collect(),
            imm: imm.into_iter().collect(),
            registers: Arc::new(RegisterLanes {
                rs1_value: rs1_value.into_iter().collect(),
                rs2_value: rs2_value.into_iter().collect(),
                rd_pre_value: MmapVec::zeroed(t),
                rd_post_value: MmapVec::zeroed(t),
                rs1_index: MmapVec::zeroed(t),
                rs2_index: MmapVec::zeroed(t),
                rd_index: MmapVec::zeroed(t),
                _lifetime: LifetimeTag::new("RegisterLanes(test)", t * 35),
            }),
            ram_address: MmapVec::zeroed(t),
            left_lookup_operand: MmapVec::zeroed(t),
            right_lookup_operand: MmapVec::zeroed(t),
            left_instruction_input: MmapVec::zeroed(t),
            right_instruction_input: MmapVec::zeroed(t),
            product_magnitude_lo: MmapVec::zeroed(t),
            product_magnitude_hi: MmapVec::zeroed(t),
            lookup_output: MmapVec::zeroed(t),
            flags: flags.into_iter().collect(),
            ram: Arc::new(RamAccessColumns {
                addresses: Vec::new(),
            }),
            ram_values: Arc::new(RamAccessValues {
                pre_values: Vec::new(),
                post_values: Vec::new(),
            }),
            _lifetime: LifetimeTag::new("TraceRecord(test)", t * 116),
        })
    }

    /// Drive device and disabled twins in lockstep; returns the device
    /// dispatch count observed.
    fn parity(log_t: usize, seed: u64) -> u64 {
        let context = MetalContext::global().unwrap();
        let record = synthetic_record(log_t, seed);
        let r_product: Vec<Fr> = (0..log_t).map(|i| fr(900 + 53 * i as u64)).collect();
        let gamma = fr(0xDADA_CAFE);

        let mut reference =
            MetalInstructionInputKernel::disabled(context, &r_product, Arc::clone(&record), gamma)
                .unwrap();
        let mut device =
            MetalInstructionInputKernel::new(context, &r_product, Arc::clone(&record), gamma)
                .unwrap();
        // The disabled twin IS the optimized computation (shared seams); pin
        // that against the real optimized kernel too.
        let mut optimized =
            crate::optimized::instruction_input::OptimizedInstructionInputKernel::new(
                &r_product,
                RecordRows::Record(Arc::clone(&record)),
                gamma,
            )
            .unwrap();

        // True input claim: the full hypercube sum of the summand.
        let rows = RecordRows::<InstructionInputRow>::Record(Arc::clone(&record));
        let eq = crate::reference::views::eq_table(&r_product);
        let mut claim = fr(0);
        for (j, eq_j) in eq.iter().enumerate() {
            let v: [Fr; NUM_TABLES] = rows.row(j).field_values();
            let right = v[4] * v[5] + v[6] * v[7];
            let left = v[0] * v[1] + v[2] * v[3];
            claim += *eq_j * (right + gamma * left);
        }

        let probes_before = device_probe_count();
        let rounds = device.num_rounds();
        let mut drawn = Vec::new();
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let expected = reference.prove_round(bind, round, claim).unwrap();
            let actual = device.prove_round(bind, round, claim).unwrap();
            let baseline = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(expected, actual, "round {round} polynomial mismatch");
            assert_eq!(baseline, actual, "round {round} drifted from optimized");
            let r = challenge(round);
            claim = expected.evaluate(r);
            drawn.push(r);
        }
        let last = challenge(rounds - 1);
        reference.finish_rounds(last).unwrap();
        device.finish_rounds(last).unwrap();
        optimized.finish_rounds(last).unwrap();

        let claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let outputs = device.output_claims(&claims).unwrap();
        assert_eq!(reference.output_claims(&claims).unwrap(), outputs);
        assert_eq!(optimized.output_claims(&claims).unwrap(), outputs);

        let relation = InstructionInput::<Fr>::new(TraceDimensions::new(log_t), r_product.clone());
        let output_points = relation.derive_opening_points(&drawn, &points).unwrap();
        let challenges = InstructionInputChallenges { gamma };
        device
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
        device_probe_count() - probes_before
    }

    #[test]
    fn matches_optimized() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        // log_t = 10: q0 + bind0 + 8 dense device rounds (the last
        // prove_round folds 4 → 2 pairs; finish_rounds is host-side).
        assert_eq!(parity(10, 21), 10);
    }

    #[test]
    fn matches_optimized_odd_rounds() {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        assert_eq!(parity(9, 8080), 9);
    }

    /// Tail handoff: the gate admits q0, bind0 at `len = 1024`, and the first
    /// dense round at `len = 512`, then hands the shrinking tail to the CPU.
    #[test]
    fn tail_rounds_hand_off_to_cpu() {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_INSTRUCTION_INPUT", "512");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        assert_eq!(parity(10, 7), 3);
    }
}
