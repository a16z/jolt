use std::error::Error;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputInputClaims;
use jolt_field::signed::{S192, S256, S64};
use jolt_field::{
    AkitaField, AkitaSignedProductAccumulator, FromPrimitiveInt, SignedProductAccumulator as _,
};
use jolt_kernels::metal::solinas::{
    InstructionInputSequence, InstructionInputSequenceConfig, SolinasMetal, SpartanOuterUniskipRow,
    INSTRUCTION_INPUT_TABLES,
};
use jolt_kernels::optimized::instruction_input::{
    InstructionInputRow, OptimizedInstructionInputKernel,
};
use jolt_kernels::SumcheckKernel;
use jolt_poly::thread::unsafe_allocate_zero_vec;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, ProveRounds, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};
use rayon::prelude::*;

pub type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

pub const TABLES: usize = INSTRUCTION_INPUT_TABLES;
pub const SAMPLES: usize = 4;
pub const DESCRIPTORS: usize = 3;

const FLAG_LOAD: u32 = 0;
const FLAG_IMM_POSITIVE: u32 = 18;
const FLAG_LEFT_OPERAND_IS_RS1: u32 = 20;
const FLAG_LEFT_OPERAND_IS_PC: u32 = 21;
const FLAG_RIGHT_OPERAND_IS_RS2: u32 = 22;
const FLAG_RIGHT_OPERAND_IS_IMM: u32 = 23;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceDispatch {
    pub native_message: usize,
    pub native_transition: usize,
    pub dense_transition: usize,
}

impl SequenceDispatch {
    pub const fn config(self) -> InstructionInputSequenceConfig {
        InstructionInputSequenceConfig {
            native_message_threads_per_threadgroup: Some(self.native_message),
            native_transition_threads_per_threadgroup: Some(self.native_transition),
            dense_transition_threads_per_threadgroup: Some(self.dense_transition),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundState {
    pub is_dense: bool,
    pub elements: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Trace {
    pub q_evals: Vec<[AkitaField; SAMPLES]>,
    pub round_polys: Vec<UnivariatePoly<AkitaField>>,
    pub challenges: Vec<AkitaField>,
    pub states: Vec<RoundState>,
    pub cutoff_tables: Option<Vec<AkitaField>>,
    pub final_claims: [AkitaField; TABLES],
    pub final_sumcheck_claim: AkitaField,
    pub derived_eq_cycle: AkitaField,
    pub transcript_state: [u8; 32],
}

pub struct TimedTrace {
    pub trace: Trace,
    pub wall: Duration,
    pub reset: Duration,
    pub gpu_wall: Duration,
    pub host_rounds: Duration,
    pub readback: Duration,
    pub cpu_tail: Duration,
    pub gpu_active: Duration,
    pub resident_rows_stable: bool,
    pub static_device_buffers_stable: bool,
    pub readbacks: usize,
    pub preallocated_readback_bytes: usize,
}

pub struct ActualOptimizedTrace {
    pub round_polys: Vec<UnivariatePoly<AkitaField>>,
    pub challenges: Vec<AkitaField>,
    pub final_claims: [AkitaField; TABLES],
    pub final_sumcheck_claim: AkitaField,
    pub transcript_state: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capture {
    pub cutoff_tables: bool,
}

impl Capture {
    pub const TARGET: Self = Self {
        cutoff_tables: false,
    };
    pub const VALIDATION: Self = Self {
        cutoff_tables: true,
    };
}

pub type CpuInstructionInputRow = InstructionInputRow;

fn row_fields(row: CpuInstructionInputRow) -> [AkitaField; TABLES] {
    let imm = AkitaField::from_u128(row.imm.0.unsigned_abs());
    let imm = if row.imm.0 < 0 { -imm } else { imm };
    [
        AkitaField::from_u64(u64::from(row.is_rs1.0)),
        AkitaField::from_u64(row.rs1_value.0),
        AkitaField::from_u64(u64::from(row.is_pc.0)),
        AkitaField::from_u64(row.unexpanded_pc.0),
        AkitaField::from_u64(u64::from(row.is_rs2.0)),
        AkitaField::from_u64(row.rs2_value.0),
        AkitaField::from_u64(u64::from(row.is_imm.0)),
        imm,
    ]
}

pub struct Workload {
    pub log_n: usize,
    pub cpu_rows: Arc<Vec<CpuInstructionInputRow>>,
    resident_seed: Option<u64>,
    pub point: Vec<AkitaField>,
    pub gamma: AkitaField,
    pub initial_claim: AkitaField,
}

impl Workload {
    pub fn new(log_n: usize, seed: u64) -> EvalResult<Self> {
        if !(2..usize::BITS as usize).contains(&log_n) {
            return Err("InstructionInput evaluator log size is outside its domain".into());
        }
        let rows = 1usize << log_n;
        let cpu_rows = Arc::new(
            (0..rows)
                .into_par_iter()
                .map(|index| make_cpu_row(index, seed))
                .collect::<Vec<_>>(),
        );
        let (point, gamma, initial_claim) = protocol(log_n, &cpu_rows, seed)?;
        Ok(Self {
            log_n,
            cpu_rows,
            resident_seed: Some(seed),
            point,
            gamma,
            initial_claim,
        })
    }

    pub const fn rows(&self) -> usize {
        1usize << self.log_n
    }

    pub fn cpu_rows_identity(&self) -> *const CpuInstructionInputRow {
        self.cpu_rows.as_ptr()
    }

    pub fn retarget(&self, seed: u64) -> EvalResult<Self> {
        let (point, gamma, initial_claim) = protocol(self.log_n, &self.cpu_rows, seed)?;
        Ok(Self {
            log_n: self.log_n,
            cpu_rows: Arc::clone(&self.cpu_rows),
            resident_seed: None,
            point,
            gamma,
            initial_claim,
        })
    }

    pub fn prepare_sequence(
        &mut self,
        context: &SolinasMetal,
        dispatch: SequenceDispatch,
    ) -> EvalResult<InstructionInputSequence> {
        let seed = self
            .resident_seed
            .take()
            .ok_or("InstructionInput resident source was already consumed")?;
        let rows = self
            .cpu_rows
            .par_iter()
            .enumerate()
            .map(|(index, row)| resident_row(index, *row, seed))
            .collect::<Vec<_>>();
        let sequence = context.prepare_instruction_input_sequence(&rows, dispatch.config())?;
        drop(rows);
        Ok(sequence)
    }
}

fn protocol(
    log_n: usize,
    cpu_rows: &Arc<Vec<CpuInstructionInputRow>>,
    seed: u64,
) -> EvalResult<(Vec<AkitaField>, AkitaField, AkitaField)> {
    let point = (0..log_n)
        .map(|round| field_value(seed ^ 0x6a09_e667_f3bc_c909, round))
        .collect::<Vec<_>>();
    let gamma = field_value(seed ^ 0xbb67_ae85_84ca_a73b, log_n);
    let control = CpuInstructionInput::new(Arc::clone(cpu_rows));
    let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let q = control.q_evals(&gruen, gamma)?;
    let (l_at_0, l_at_1) = gruen.current_linear_evals();
    Ok((point, gamma, l_at_0 * q[0] + l_at_1 * q[1]))
}

enum CpuState {
    Native(Arc<Vec<CpuInstructionInputRow>>),
    Dense(Vec<Polynomial<AkitaField>>),
}

struct CpuInstructionInput {
    state: CpuState,
    bind_scratch: Vec<AkitaField>,
}

impl CpuInstructionInput {
    fn new(rows: Arc<Vec<CpuInstructionInputRow>>) -> Self {
        Self {
            state: CpuState::Native(rows),
            bind_scratch: Vec::new(),
        }
    }

    fn from_dense(flat_tables: &[AkitaField], elements: usize) -> EvalResult<Self> {
        if elements == 0 || !elements.is_power_of_two() || flat_tables.len() != TABLES * elements {
            return Err("invalid InstructionInput dense-tail shape".into());
        }
        Ok(Self {
            state: CpuState::Dense(
                flat_tables
                    .chunks_exact(elements)
                    .map(|table| Polynomial::new(table.to_vec()))
                    .collect(),
            ),
            bind_scratch: Vec::new(),
        })
    }

    fn state(&self) -> RoundState {
        match &self.state {
            CpuState::Native(rows) => RoundState {
                is_dense: false,
                elements: rows.len(),
            },
            CpuState::Dense(tables) => RoundState {
                is_dense: true,
                elements: tables[0].evals().len(),
            },
        }
    }

    fn q_evals(
        &self,
        gruen: &GruenSplitEqPolynomial<AkitaField>,
        gamma: AkitaField,
    ) -> EvalResult<[AkitaField; SAMPLES]> {
        match &self.state {
            CpuState::Native(rows) => Ok(native_q_evals(rows, gruen, gamma)),
            CpuState::Dense(tables) => Ok(dense_q_evals(tables, gruen, gamma)),
        }
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.state = match std::mem::replace(&mut self.state, CpuState::Dense(Vec::new())) {
            CpuState::Native(rows) => {
                let half = rows.len() / 2;
                let materialize = |table: usize| {
                    let mut values: Vec<AkitaField> = unsafe_allocate_zero_vec(half);
                    values.par_iter_mut().enumerate().for_each(|(pair, slot)| {
                        let even = row_fields(rows[2 * pair])[table];
                        let odd = row_fields(rows[2 * pair + 1])[table];
                        *slot = even + challenge * (odd - even);
                    });
                    Polynomial::new(values)
                };
                CpuState::Dense((0..TABLES).map(materialize).collect())
            }
            CpuState::Dense(mut tables) => {
                for table in &mut tables {
                    table.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
                }
                CpuState::Dense(tables)
            }
        };
    }

    fn flatten_dense(&self) -> EvalResult<Vec<AkitaField>> {
        match &self.state {
            CpuState::Dense(tables) => Ok(tables
                .iter()
                .flat_map(|table| table.evals())
                .copied()
                .collect()),
            CpuState::Native(_) => Err("InstructionInput tables are still native".into()),
        }
    }

    fn final_claims(&self) -> EvalResult<[AkitaField; TABLES]> {
        match &self.state {
            CpuState::Dense(tables)
                if tables.len() == TABLES
                    && tables.iter().all(|table| table.evals().len() == 1) =>
            {
                Ok(std::array::from_fn(|table| tables[table].evals()[0]))
            }
            CpuState::Native(rows) if rows.len() == 1 => Ok(row_fields(rows[0])),
            _ => Err("InstructionInput final claims requested before full binding".into()),
        }
    }
}

fn native_q_evals(
    rows: &[CpuInstructionInputRow],
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    gamma: AkitaField,
) -> [AkitaField; SAMPLES] {
    gruen.par_fold_out_in(
        || {
            (
                [AkitaSignedProductAccumulator::default(); SAMPLES],
                [AkitaSignedProductAccumulator::default(); SAMPLES],
            )
        },
        |(right_acc, left_acc), y, _x_in, e_in| {
            let even = rows[2 * y];
            let odd = rows[2 * y + 1];
            let (is_rs1, is_rs1_m) = ext_flag(even.is_rs1.0, odd.is_rs1.0);
            let (is_pc, is_pc_m) = ext_flag(even.is_pc.0, odd.is_pc.0);
            let (is_rs2, is_rs2_m) = ext_flag(even.is_rs2.0, odd.is_rs2.0);
            let (is_imm, is_imm_m) = ext_flag(even.is_imm.0, odd.is_imm.0);
            let (rs1, rs1_m) = ext_u64(even.rs1_value.0, odd.rs1_value.0);
            let (upc, upc_m) = ext_u64(even.unexpanded_pc.0, odd.unexpanded_pc.0);
            let (rs2, rs2_m) = ext_u64(even.rs2_value.0, odd.rs2_value.0);
            let imm_even = S192::from_i128(even.imm.0);
            let imm_odd = S192::from_i128(odd.imm.0);
            for t in 0..SAMPLES as i64 {
                let f_rs1 = is_rs1 + t * is_rs1_m;
                let f_pc = is_pc + t * is_pc_m;
                let f_rs2 = is_rs2 + t * is_rs2_m;
                let f_imm = is_imm + t * is_imm_m;
                let left = i128::from(f_rs1) * (rs1 + i128::from(t) * rs1_m)
                    + i128::from(f_pc) * (upc + i128::from(t) * upc_m);
                let mut right = S256::from_i128(i128::from(f_rs2) * (rs2 + i128::from(t) * rs2_m));
                S64::from_i64(f_imm * (1 - t)).fmadd_trunc::<3, 4>(&imm_even, &mut right);
                S64::from_i64(f_imm * t).fmadd_trunc::<3, 4>(&imm_odd, &mut right);
                right_acc[t as usize].fmadd_s256(e_in, &right);
                left_acc[t as usize].fmadd_s256(e_in, &S256::from_i128(left));
            }
        },
        |_x_out, e_out, (right_acc, left_acc)| {
            let mut output = [AkitaField::zero(); SAMPLES];
            for (slot, (right, left)) in output.iter_mut().zip(right_acc.into_iter().zip(left_acc))
            {
                *slot = e_out * (right.reduce() + gamma * left.reduce());
            }
            output
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(&right) {
                *left += *right;
            }
            left
        },
    )
}

fn dense_q_evals(
    tables: &[Polynomial<AkitaField>],
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    gamma: AkitaField,
) -> [AkitaField; SAMPLES] {
    gruen.par_fold_out_in(
        || {
            (
                [AkitaField::zero(); SAMPLES],
                [AkitaField::zero(); TABLES],
                [AkitaField::zero(); TABLES],
            )
        },
        |(acc, evals, steps), row, _x_in, e_in| {
            for ((table, eval), step) in tables.iter().zip(evals.iter_mut()).zip(steps.iter_mut()) {
                let table = table.evals();
                let low = table[2 * row];
                *eval = low;
                *step = table[2 * row + 1] - low;
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
        |_x_out, e_out, (mut acc, _, _)| {
            for value in &mut acc {
                *value *= e_out;
            }
            acc
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(&right) {
                *left += *right;
            }
            left
        },
    )
}

fn message_poly(
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    q_evals: &[AkitaField; SAMPLES],
    previous_claim: AkitaField,
    round: usize,
) -> EvalResult<UnivariatePoly<AkitaField>> {
    let (l_at_0, l_at_1) = gruen.current_linear_evals();
    let l_step = l_at_1 - l_at_0;
    let mut l_eval = l_at_0;
    let mut evals = [AkitaField::zero(); SAMPLES];
    for (eval, q) in evals.iter_mut().zip(q_evals) {
        *eval = l_eval * *q;
        l_eval += l_step;
    }
    let actual = evals[0] + evals[1];
    if actual != previous_claim {
        return Err(format!(
            "InstructionInput round {round} claim mismatch: expected {previous_claim:?}, got {actual:?}"
        )
        .into());
    }
    Ok(UnivariatePoly::from_evals(&evals))
}

fn descriptor_grid(descriptors: [AkitaField; DESCRIPTORS]) -> [AkitaField; SAMPLES] {
    let [q_at_0, q_at_1, quadratic] = descriptors;
    let twice_quadratic = quadratic + quadratic;
    let q_at_2 = q_at_1 + q_at_1 - q_at_0 + twice_quadratic;
    let q_at_3 = q_at_2 + q_at_1 - q_at_0 + twice_quadratic + twice_quadratic;
    [q_at_0, q_at_1, q_at_2, q_at_3]
}

fn transcript(initial_claim: AkitaField) -> EvalTranscript {
    let mut transcript = EvalTranscript::new(b"metal-instruction-input-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    transcript
}

fn absorb_round(
    transcript: &mut EvalTranscript,
    polynomial: &UnivariatePoly<AkitaField>,
) -> AkitaField {
    CompressedLabeledRoundPoly::sumcheck(polynomial).append_to_transcript(transcript);
    transcript.challenge()
}

#[expect(
    clippy::too_many_arguments,
    reason = "the trace bundles all semantically checked round state"
)]
fn finish_trace(
    q_evals: Vec<[AkitaField; SAMPLES]>,
    round_polys: Vec<UnivariatePoly<AkitaField>>,
    challenges: Vec<AkitaField>,
    states: Vec<RoundState>,
    cutoff_tables: Option<Vec<AkitaField>>,
    control: &CpuInstructionInput,
    claim: AkitaField,
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    transcript: &EvalTranscript,
) -> EvalResult<Trace> {
    Ok(Trace {
        q_evals,
        round_polys,
        challenges,
        states,
        cutoff_tables,
        final_claims: control.final_claims()?,
        final_sumcheck_claim: claim,
        derived_eq_cycle: gruen.current_scalar(),
        transcript_state: transcript.state(),
    })
}

pub fn run_actual_optimized(workload: &Workload) -> EvalResult<ActualOptimizedTrace> {
    let mut kernel = OptimizedInstructionInputKernel::new(
        &workload.point,
        workload.cpu_rows.as_ref().clone(),
        workload.gamma,
    )?;
    let mut transcript = transcript(workload.initial_claim);
    let mut claim = workload.initial_claim;
    let mut round_polys = Vec::with_capacity(workload.log_n);
    let mut challenges = Vec::with_capacity(workload.log_n);
    let mut previous_challenge = None;
    for round in 0..workload.log_n {
        let polynomial = kernel.prove_round(previous_challenge, round, claim)?;
        let challenge = absorb_round(&mut transcript, &polynomial);
        claim = polynomial.evaluate(challenge);
        round_polys.push(polynomial);
        challenges.push(challenge);
        previous_challenge = Some(challenge);
    }
    let final_challenge = previous_challenge.ok_or("InstructionInput has no rounds")?;
    kernel.finish_rounds(final_challenge)?;
    let input_claims = InstructionInputInputClaims {
        right_instruction_input: AkitaField::zero(),
        left_instruction_input: AkitaField::zero(),
    };
    let outputs = kernel.output_claims(&input_claims)?;
    Ok(ActualOptimizedTrace {
        round_polys,
        challenges,
        final_claims: [
            outputs.left_operand_is_rs1,
            outputs.rs1_value,
            outputs.left_operand_is_pc,
            outputs.unexpanded_pc,
            outputs.right_operand_is_rs2,
            outputs.rs2_value,
            outputs.right_operand_is_imm,
            outputs.imm,
        ],
        final_sumcheck_claim: claim,
        transcript_state: transcript.state(),
    })
}

pub fn run_cpu(workload: &Workload, cutoff: usize, capture: Capture) -> EvalResult<TimedTrace> {
    validate_cutoff(workload.rows(), cutoff)?;
    let started = Instant::now();
    let mut control = CpuInstructionInput::new(Arc::clone(&workload.cpu_rows));
    let mut gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
    let mut transcript = transcript(workload.initial_claim);
    let mut claim = workload.initial_claim;
    let mut q_evals = Vec::with_capacity(workload.log_n);
    let mut round_polys = Vec::with_capacity(workload.log_n);
    let mut challenges = Vec::with_capacity(workload.log_n);
    let mut states = Vec::with_capacity(workload.log_n);
    let mut cutoff_tables = None;

    while control.state().elements > 1 {
        let state = control.state();
        if capture.cutoff_tables
            && state.is_dense
            && state.elements == cutoff
            && cutoff_tables.is_none()
        {
            cutoff_tables = Some(control.flatten_dense()?);
        }
        let q = control.q_evals(&gruen, workload.gamma)?;
        let polynomial = message_poly(&gruen, &q, claim, states.len())?;
        let challenge = absorb_round(&mut transcript, &polynomial);
        claim = polynomial.evaluate(challenge);
        q_evals.push(q);
        round_polys.push(polynomial);
        challenges.push(challenge);
        states.push(state);
        gruen.bind(challenge);
        control.bind(challenge);
    }
    let trace = finish_trace(
        q_evals,
        round_polys,
        challenges,
        states,
        cutoff_tables,
        &control,
        claim,
        &gruen,
        &transcript,
    )?;
    let wall = started.elapsed();
    Ok(TimedTrace {
        trace,
        wall,
        reset: Duration::ZERO,
        gpu_wall: Duration::ZERO,
        host_rounds: Duration::ZERO,
        readback: Duration::ZERO,
        cpu_tail: wall,
        gpu_active: Duration::ZERO,
        resident_rows_stable: true,
        static_device_buffers_stable: true,
        readbacks: 0,
        preallocated_readback_bytes: 0,
    })
}

pub fn run_hybrid(
    sequence: &mut InstructionInputSequence,
    workload: &Workload,
    cutoff: usize,
    capture: Capture,
) -> EvalResult<TimedTrace> {
    validate_cutoff(workload.rows(), cutoff)?;
    let mut cutoff_readback = vec![AkitaField::zero(); TABLES * cutoff];
    let preallocated_readback_bytes = cutoff_readback.len() * size_of::<AkitaField>();
    let total_started = Instant::now();
    let resident_identity = sequence.resident_row_identity();
    let static_identity = sequence.static_buffer_identity();
    let reset_started = Instant::now();
    sequence.reset();
    let reset = reset_started.elapsed();

    let mut gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
    let mut transcript = transcript(workload.initial_claim);
    let mut claim = workload.initial_claim;
    let mut q_evals = Vec::with_capacity(workload.log_n);
    let mut round_polys = Vec::with_capacity(workload.log_n);
    let mut challenges = Vec::with_capacity(workload.log_n);
    let mut states = Vec::with_capacity(workload.log_n);
    let mut gpu_wall = Duration::ZERO;
    let mut host_rounds = Duration::ZERO;
    let mut readback = Duration::ZERO;
    let mut cpu_tail = Duration::ZERO;
    let mut readbacks = 0;

    let gpu_started = Instant::now();
    let descriptors =
        sequence.message(workload.gamma, gruen.e_in_current(), gruen.e_out_current())?;
    gpu_wall += gpu_started.elapsed();
    let mut q = descriptor_grid(descriptors);

    let (mut tail, cutoff_tables) = loop {
        let state = RoundState {
            is_dense: sequence.is_dense(),
            elements: sequence.current_elements(),
        };
        let host_started = Instant::now();
        let polynomial = message_poly(&gruen, &q, claim, states.len())?;
        let challenge = absorb_round(&mut transcript, &polynomial);
        claim = polynomial.evaluate(challenge);
        q_evals.push(q);
        round_polys.push(polynomial);
        challenges.push(challenge);
        states.push(state);
        host_rounds += host_started.elapsed();

        if state.is_dense && state.elements == cutoff {
            let read_started = Instant::now();
            sequence.read_current_tables(&mut cutoff_readback)?;
            readback += read_started.elapsed();
            readbacks += 1;
            let tail_started = Instant::now();
            let mut tail = CpuInstructionInput::from_dense(&cutoff_readback, cutoff)?;
            gruen.bind(challenge);
            tail.bind(challenge);
            cpu_tail += tail_started.elapsed();
            let captured = capture.cutoff_tables.then_some(cutoff_readback);
            break (tail, captured);
        }

        let host_started = Instant::now();
        gruen.bind(challenge);
        host_rounds += host_started.elapsed();
        let gpu_started = Instant::now();
        let descriptors = sequence.bind_and_message(
            challenge,
            workload.gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
        )?;
        gpu_wall += gpu_started.elapsed();
        q = descriptor_grid(descriptors);
    };

    let tail_started = Instant::now();
    while tail.state().elements > 1 {
        let state = tail.state();
        let q = tail.q_evals(&gruen, workload.gamma)?;
        let polynomial = message_poly(&gruen, &q, claim, states.len())?;
        let challenge = absorb_round(&mut transcript, &polynomial);
        claim = polynomial.evaluate(challenge);
        q_evals.push(q);
        round_polys.push(polynomial);
        challenges.push(challenge);
        states.push(state);
        gruen.bind(challenge);
        tail.bind(challenge);
    }
    cpu_tail += tail_started.elapsed();

    let trace = finish_trace(
        q_evals,
        round_polys,
        challenges,
        states,
        cutoff_tables,
        &tail,
        claim,
        &gruen,
        &transcript,
    )?;
    Ok(TimedTrace {
        trace,
        wall: total_started.elapsed(),
        reset,
        gpu_wall,
        host_rounds,
        readback,
        cpu_tail,
        gpu_active: sequence.gpu_active_time(),
        resident_rows_stable: sequence.resident_row_identity() == resident_identity,
        static_device_buffers_stable: sequence.static_buffer_identity() == static_identity,
        readbacks,
        preallocated_readback_bytes,
    })
}

pub fn derived_eq_cycle_is_exact(workload: &Workload, trace: &Trace) -> bool {
    let reversed = trace.challenges.iter().rev().copied().collect::<Vec<_>>();
    trace.derived_eq_cycle == EqPolynomial::<AkitaField>::mle(&workload.point, &reversed)
}

pub fn final_relation_is_exact(workload: &Workload, trace: &Trace) -> bool {
    let claims = trace.final_claims;
    let right = claims[4] * claims[5] + claims[6] * claims[7];
    let left = claims[0] * claims[1] + claims[2] * claims[3];
    trace.final_sumcheck_claim == trace.derived_eq_cycle * (right + workload.gamma * left)
}

pub fn expected_states(log_n: usize) -> Vec<RoundState> {
    (0..log_n)
        .map(|round| RoundState {
            is_dense: round != 0,
            elements: 1usize << (log_n - round),
        })
        .collect()
}

pub fn median(values: &[Duration]) -> Duration {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn validate_cutoff(rows: usize, cutoff: usize) -> EvalResult<()> {
    if cutoff < 2 || !cutoff.is_power_of_two() || cutoff > rows / 2 {
        return Err("InstructionInput cutoff must be a dense power-of-two tail".into());
    }
    Ok(())
}

#[inline]
fn ext_u64(even: u64, odd: u64) -> (i128, i128) {
    (i128::from(even), i128::from(odd) - i128::from(even))
}

#[inline]
fn ext_flag(even: bool, odd: bool) -> (i64, i64) {
    (i64::from(even), i64::from(odd) - i64::from(even))
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn field_value(seed: u64, index: usize) -> AkitaField {
    let low = splitmix(seed.wrapping_add(2 * index as u64));
    let high = splitmix(seed.wrapping_add(2 * index as u64 + 1)) & 0x3fff_ffff_ffff_ffff;
    AkitaField::from_u128(u128::from(low) | (u128::from(high) << 64))
}

fn make_cpu_row(index: usize, seed: u64) -> CpuInstructionInputRow {
    let key = seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let selector = splitmix(key ^ 0x243f_6a88_85a3_08d3);
    let wide = u128::from(splitmix(key ^ 0x1319_8a2e_0370_7344))
        | (u128::from(splitmix(key ^ 0xa409_3822_299f_31d0)) << 64);
    let imm = match index {
        0 => i128::MIN,
        1 => i128::MAX,
        _ => wide as i128,
    };
    let load = index % 31 == 7;
    InstructionInputRow {
        is_rs1: InstructionFlag(selector & 1 != 0),
        rs1_value: Rs1Value(splitmix(key ^ 0x082e_fa98_ec4e_6c89)),
        is_pc: InstructionFlag(selector & 2 != 0),
        unexpanded_pc: UnexpandedPc(splitmix(key ^ 0x4528_21e6_38d0_1377)),
        is_rs2: InstructionFlag(selector & 4 != 0),
        rs2_value: Rs2Value(if load {
            0
        } else {
            splitmix(key ^ 0xbe54_66cf_34e9_0c6c)
        }),
        is_imm: InstructionFlag(selector & 8 != 0),
        imm: Imm(imm),
    }
}

fn resident_row(index: usize, row: CpuInstructionInputRow, seed: u64) -> SpartanOuterUniskipRow {
    let load = index % 31 == 7;
    let mut words = [0u64; 20];
    words[6] = row.unexpanded_pc.0;
    let magnitude = row.imm.0.unsigned_abs();
    words[7] = magnitude as u64;
    words[8] = (magnitude >> 64) as u64;
    words[9] = row.rs1_value.0;
    words[10] = if load {
        splitmix(seed ^ index as u64 ^ 0xc0ac_29b7_c97c_50dd)
    } else {
        row.rs2_value.0
    };
    let mut flags = 0u64;
    let mut set = |bit: u32, value: bool| flags |= u64::from(value) << bit;
    set(FLAG_LOAD, load);
    set(FLAG_IMM_POSITIVE, row.imm.0 >= 0);
    set(FLAG_LEFT_OPERAND_IS_RS1, row.is_rs1.0);
    set(FLAG_LEFT_OPERAND_IS_PC, row.is_pc.0);
    set(FLAG_RIGHT_OPERAND_IS_RS2, row.is_rs2.0);
    set(FLAG_RIGHT_OPERAND_IS_IMM, row.is_imm.0);
    words[19] = flags;
    SpartanOuterUniskipRow::from_words(words)
}

const _: () = assert!(TABLES == 8);
const _: () = assert!(size_of::<AkitaField>() == 16);
const _: () = assert!(size_of::<SpartanOuterUniskipRow>() == 160);
