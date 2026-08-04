use std::error::Error;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, RingAccumulator,
};
use jolt_kernels::metal::solinas::{
    InstructionRaLookupPlane, InstructionRaMaterializeWidth, InstructionRaSequence,
    InstructionRaSequenceConfig, SolinasMetal,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;

pub type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

pub const FACTORS: usize = 16;
pub const GROUPS: usize = 4;
pub const FACTORS_PER_GROUP: usize = 4;
pub const BINS: usize = 256;
pub const CPU_MATERIALIZE_WIDTH: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceDispatch {
    pub message_threads: usize,
    pub materialize_threads: usize,
    pub materialize_width: InstructionRaMaterializeWidth,
    pub reuse_inverse_for_dense: bool,
}

impl SequenceDispatch {
    pub fn config(self) -> InstructionRaSequenceConfig {
        InstructionRaSequenceConfig {
            message_threads_per_threadgroup: Some(self.message_threads),
            materialize_threads_per_threadgroup: Some(self.materialize_threads),
            materialize_width: self.materialize_width,
            reuse_inverse_for_dense: self.reuse_inverse_for_dense,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundState {
    pub branch_width: usize,
    pub is_dense: bool,
    pub elements: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Trace {
    pub q_evals: Vec<[AkitaField; 4]>,
    pub round_polys: Vec<UnivariatePoly<AkitaField>>,
    pub challenges: Vec<AkitaField>,
    pub states: Vec<RoundState>,
    pub scheduled_tables: Option<Vec<AkitaField>>,
    pub cutoff_tables: Option<Vec<AkitaField>>,
    pub raw_final_claims: Vec<AkitaField>,
    pub final_claims: Vec<AkitaField>,
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
    pub resident_plane_zero_copy: bool,
    pub static_device_buffers_stable: bool,
    pub inverse_dense_b_handoff_exact: bool,
    pub preallocated_readback_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capture {
    pub scheduled_elements: Option<usize>,
    pub cutoff_tables: bool,
}

impl Capture {
    pub const TARGET: Self = Self {
        scheduled_elements: None,
        cutoff_tables: false,
    };

    pub const fn validation(rows: usize, materialize_width: usize) -> Self {
        Self {
            scheduled_elements: Some(rows / materialize_width),
            cutoff_tables: true,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuInstructionRow {
    lookup_lo: u64,
    lookup_hi: u64,
    _stage_5_columns: [u64; 3],
}

impl CpuInstructionRow {
    fn new(lookup: u128) -> Self {
        Self {
            lookup_lo: lookup as u64,
            lookup_hi: (lookup >> 64) as u64,
            _stage_5_columns: [0; 3],
        }
    }

    fn lookup(self) -> u128 {
        u128::from(self.lookup_lo) | (u128::from(self.lookup_hi) << 64)
    }
}

pub struct Workload {
    pub log_n: usize,
    pub cycle_rows: Arc<Vec<CpuInstructionRow>>,
    pub table_major_lookups: Vec<u128>,
    pub cycle_to_table_major: Vec<u32>,
    pub chunk_tables: Vec<AkitaField>,
    pub point: Vec<AkitaField>,
    pub gamma: AkitaField,
    pub initial_claim: AkitaField,
}

impl Workload {
    pub fn new(log_n: usize, seed: u64) -> EvalResult<Self> {
        if !(5..usize::BITS as usize).contains(&log_n) {
            return Err("Instruction RA evaluator log size is outside its domain".into());
        }
        let rows = 1usize << log_n;
        let (cycle_rows, table_major_lookups, cycle_to_table_major) =
            lookup_layout(rows, log_n, seed);
        let gamma = AkitaField::from_u64(7);
        let chunk_tables = chunk_tables(gamma, seed ^ 0xa54f_f53a_5f1d_36f1);
        let point = (0..log_n)
            .map(|round| field_value(seed ^ 0x6a09_e667_f3bc_c909, round))
            .collect::<Vec<_>>();
        let cycle_rows = Arc::new(cycle_rows);
        let cpu = CpuInstructionRa::new(Arc::clone(&cycle_rows), &chunk_tables);
        let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let (q_at_zero, q_at_one) = cpu.q_at_zero_and_one(&gruen);
        let (l_at_zero, l_at_one) = gruen.current_linear_evals();
        let initial_claim = l_at_zero * q_at_zero + l_at_one * q_at_one;
        Ok(Self {
            log_n,
            cycle_rows,
            table_major_lookups,
            cycle_to_table_major,
            chunk_tables,
            point,
            gamma,
            initial_claim,
        })
    }

    pub const fn rows(&self) -> usize {
        1usize << self.log_n
    }

    pub fn prepare_plane(&self, context: &SolinasMetal) -> EvalResult<InstructionRaLookupPlane> {
        Ok(context.prepare_instruction_ra_lookup_plane(
            &self.table_major_lookups,
            &self.cycle_to_table_major,
        )?)
    }

    pub fn prepare_sequence(
        &self,
        context: &SolinasMetal,
        plane: InstructionRaLookupPlane,
        dispatch: SequenceDispatch,
    ) -> EvalResult<InstructionRaSequence> {
        let gruen = GruenSplitEqPolynomial::new(&self.point, BindingOrder::LowToHigh);
        Ok(context.prepare_instruction_ra_sequence(
            plane,
            &self.chunk_tables,
            gruen.e_in_current().len(),
            gruen.e_out_current().len(),
            dispatch.config(),
        )?)
    }

    pub fn release_table_major_layout(&mut self) {
        self.table_major_lookups = Vec::new();
        self.cycle_to_table_major = Vec::new();
    }
}

enum CpuTableState {
    Lazy {
        tables: Vec<Vec<AkitaField>>,
        width: usize,
    },
    Dense(Vec<Vec<AkitaField>>),
}

struct CpuInstructionRa {
    rows: usize,
    lookups: Arc<Vec<CpuInstructionRow>>,
    state: CpuTableState,
}

impl CpuInstructionRa {
    fn new(lookups: Arc<Vec<CpuInstructionRow>>, chunk_tables: &[AkitaField]) -> Self {
        let tables = chunk_tables
            .chunks_exact(BINS)
            .map(<[AkitaField]>::to_vec)
            .collect::<Vec<_>>();
        debug_assert_eq!(tables.len(), FACTORS);
        Self {
            rows: lookups.len(),
            lookups,
            state: CpuTableState::Lazy { tables, width: 1 },
        }
    }

    fn from_dense(flat_tables: &[AkitaField], elements: usize) -> EvalResult<Self> {
        if elements == 0 || flat_tables.len() != FACTORS * elements {
            return Err("invalid Instruction RA dense-tail shape".into());
        }
        Ok(Self {
            rows: elements,
            lookups: Arc::new(Vec::new()),
            state: CpuTableState::Dense(
                flat_tables
                    .chunks_exact(elements)
                    .map(<[AkitaField]>::to_vec)
                    .collect(),
            ),
        })
    }

    fn state(&self) -> RoundState {
        match &self.state {
            CpuTableState::Lazy { width, .. } => RoundState {
                branch_width: *width,
                is_dense: false,
                elements: self.rows / *width,
            },
            CpuTableState::Dense(tables) => RoundState {
                branch_width: CPU_MATERIALIZE_WIDTH,
                is_dense: true,
                elements: tables[0].len(),
            },
        }
    }

    fn message(&self, gruen: &GruenSplitEqPolynomial<AkitaField>) -> [AkitaField; 4] {
        struct Scratch {
            lanes: [AkitaAccumulator; 4],
            row_lanes: [AkitaAccumulator; 4],
            pairs: [(AkitaField, AkitaField); FACTORS],
        }

        let block_lanes = gruen.par_fold_out_in(
            || Scratch {
                lanes: [AkitaAccumulator::default(); 4],
                row_lanes: [AkitaAccumulator::default(); 4],
                pairs: [(AkitaField::zero(), AkitaField::zero()); FACTORS],
            },
            |scratch, row, _x_in, e_in| {
                self.lo_hi_all(row, &mut scratch.pairs);
                scratch.row_lanes = [AkitaAccumulator::default(); 4];
                for factors in scratch.pairs.chunks_exact(FACTORS_PER_GROUP) {
                    let left = quadratic_grid(factors[0], factors[1]);
                    let right = quadratic_grid(factors[2], factors[3]);
                    for ((lane, left), right) in scratch.row_lanes.iter_mut().zip(left).zip(right) {
                        lane.fmadd(left, right);
                    }
                }
                for (lane, row_lane) in scratch.lanes.iter_mut().zip(&scratch.row_lanes) {
                    lane.fmadd(e_in, row_lane.reduce());
                }
            },
            |_x_out, e_out, scratch| {
                let mut out = [AkitaAccumulator::default(); 4];
                for (out, lane) in out.iter_mut().zip(scratch.lanes) {
                    out.fmadd(e_out, lane.reduce());
                }
                out
            },
            |mut lhs, rhs| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                    lhs.merge(rhs);
                }
                lhs
            },
        );
        block_lanes.map(AdditiveAccumulator::reduce)
    }

    fn q_at_zero_and_one(
        &self,
        gruen: &GruenSplitEqPolynomial<AkitaField>,
    ) -> (AkitaField, AkitaField) {
        let blocks = gruen.par_fold_out_in(
            || {
                (
                    [AkitaAccumulator::default(); 2],
                    [(AkitaField::zero(), AkitaField::zero()); FACTORS],
                )
            },
            |(lanes, pairs), row, _x_in, e_in| {
                self.lo_hi_all(row, pairs);
                let mut sums = [AkitaField::zero(); 2];
                for factors in pairs.chunks_exact(FACTORS_PER_GROUP) {
                    let mut at_zero = factors[0].0;
                    let mut at_one = factors[0].1;
                    for factor in &factors[1..] {
                        at_zero *= factor.0;
                        at_one *= factor.1;
                    }
                    sums[0] += at_zero;
                    sums[1] += at_one;
                }
                for (lane, sum) in lanes.iter_mut().zip(sums) {
                    lane.fmadd(e_in, sum);
                }
            },
            |_x_out, e_out, (lanes, _)| {
                let mut out = [AkitaAccumulator::default(); 2];
                for (out, lane) in out.iter_mut().zip(lanes) {
                    out.fmadd(e_out, lane.reduce());
                }
                out
            },
            |mut lhs, rhs| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                    lhs.merge(rhs);
                }
                lhs
            },
        );
        let [at_zero, at_one] = blocks.map(AdditiveAccumulator::reduce);
        (at_zero, at_one)
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.state = match std::mem::replace(&mut self.state, CpuTableState::Dense(Vec::new())) {
            CpuTableState::Lazy { tables, width } => {
                let next_width = 2 * width;
                let tables = tables
                    .into_par_iter()
                    .map(|table| double_branches(table, challenge))
                    .collect::<Vec<_>>();
                if next_width < CPU_MATERIALIZE_WIDTH {
                    CpuTableState::Lazy {
                        tables,
                        width: next_width,
                    }
                } else {
                    CpuTableState::Dense(materialize(&tables, &self.lookups, next_width))
                }
            }
            CpuTableState::Dense(mut tables) => {
                tables.par_iter_mut().for_each(|table| {
                    let bound_len = table.len() / 2;
                    for index in 0..bound_len {
                        let lo = table[2 * index];
                        let hi = table[2 * index + 1];
                        table[index] = lo + challenge * (hi - lo);
                    }
                    table.truncate(bound_len);
                });
                CpuTableState::Dense(tables)
            }
        };
    }

    fn flatten_dense(&self) -> EvalResult<Vec<AkitaField>> {
        match &self.state {
            CpuTableState::Dense(tables) => Ok(tables.iter().flatten().copied().collect()),
            CpuTableState::Lazy { .. } => {
                Err("Instruction RA tables are still in the lazy prefix".into())
            }
        }
    }

    fn final_raw_claims(&self) -> EvalResult<Vec<AkitaField>> {
        match &self.state {
            CpuTableState::Dense(tables)
                if tables.len() == FACTORS && tables.iter().all(|table| table.len() == 1) =>
            {
                Ok(tables.iter().map(|table| table[0]).collect())
            }
            _ => Err("Instruction RA final claims requested before full binding".into()),
        }
    }

    fn lo_hi_all(&self, row: usize, output: &mut [(AkitaField, AkitaField); FACTORS]) {
        match &self.state {
            CpuTableState::Lazy { tables, width } => {
                for (factor, (output, table)) in output.iter_mut().zip(tables).enumerate() {
                    *output = (
                        gather(table, *width, &self.lookups, factor, 2 * row),
                        gather(table, *width, &self.lookups, factor, 2 * row + 1),
                    );
                }
            }
            CpuTableState::Dense(tables) => {
                for (output, table) in output.iter_mut().zip(tables) {
                    *output = (table[2 * row], table[2 * row + 1]);
                }
            }
        }
    }
}

fn quadratic_grid(
    first: (AkitaField, AkitaField),
    second: (AkitaField, AkitaField),
) -> [AkitaField; 4] {
    let at_zero = first.0 * second.0;
    let at_one = first.1 * second.1;
    let at_infinity = (first.1 - first.0) * (second.1 - second.0);
    let twice_at_infinity = at_infinity + at_infinity;
    let at_two = at_one + at_one - at_zero + twice_at_infinity;
    let at_three = at_two + at_one - at_zero + twice_at_infinity + twice_at_infinity;
    [at_one, at_two, at_three, at_infinity]
}

fn double_branches(table: Vec<AkitaField>, challenge: AkitaField) -> Vec<AkitaField> {
    let mut next = Vec::with_capacity(2 * table.len());
    let one_minus = AkitaField::one() - challenge;
    next.extend(table.iter().map(|value| one_minus * *value));
    next.extend(table.iter().map(|value| challenge * *value));
    next
}

fn materialize(
    tables: &[Vec<AkitaField>],
    lookups: &[CpuInstructionRow],
    width: usize,
) -> Vec<Vec<AkitaField>> {
    let elements = lookups.len() / width;
    (0..FACTORS)
        .into_par_iter()
        .map(|factor| {
            let table = &tables[factor];
            (0..elements)
                .map(|index| gather(table, width, lookups, factor, index))
                .collect()
        })
        .collect()
}

fn gather(
    table: &[AkitaField],
    width: usize,
    lookups: &[CpuInstructionRow],
    factor: usize,
    index: usize,
) -> AkitaField {
    let stride = table.len() / width;
    let mut value = AkitaField::zero();
    for offset in 0..width {
        let bin = lookup_byte(lookups[index * width + offset].lookup(), factor);
        value += table[offset * stride + bin];
    }
    value
}

fn lookup_byte(lookup: u128, factor: usize) -> usize {
    ((lookup >> (8 * (FACTORS - 1 - factor))) & 0xff) as usize
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn lookup(cycle: usize, seed: u64) -> u128 {
    match cycle {
        0 => 0x0001_0203_0405_0607_0809_0a0b_0c0d_0e0f,
        1 => 0xf0e1_d2c3_b4a5_9687_7869_5a4b_3c2d_1e0f,
        2 => 0xff00_aa55_cc33_9966_1234_5678_9abc_def0,
        _ => {
            let counter = seed.wrapping_add(2 * cycle as u64);
            u128::from(splitmix(counter)) | (u128::from(splitmix(counter + 1)) << 64)
        }
    }
}

fn permute(index: usize, log_n: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - log_n)
}

fn lookup_layout(
    rows: usize,
    log_n: usize,
    seed: u64,
) -> (Vec<CpuInstructionRow>, Vec<u128>, Vec<u32>) {
    let cycle_order = (0..rows)
        .into_par_iter()
        .map(|cycle| CpuInstructionRow::new(lookup(cycle, seed)))
        .collect();
    let table_major = (0..rows)
        .into_par_iter()
        .map(|slot| lookup(permute(slot, log_n), seed))
        .collect();
    let inverse = (0..rows)
        .into_par_iter()
        .map(|cycle| permute(cycle, log_n) as u32)
        .collect();
    (cycle_order, table_major, inverse)
}

fn field_value(seed: u64, index: usize) -> AkitaField {
    let low = splitmix(seed.wrapping_add(2 * index as u64));
    let high = splitmix(seed.wrapping_add(2 * index as u64 + 1)) & 0x3fff_ffff_ffff_ffff;
    AkitaField::from_u128(u128::from(low) | (u128::from(high) << 64))
}

fn chunk_tables(gamma: AkitaField, seed: u64) -> Vec<AkitaField> {
    let mut gamma_power = AkitaField::one();
    let mut tables = Vec::with_capacity(FACTORS * BINS);
    for factor in 0..FACTORS {
        let point = (0..8)
            .map(|bit| field_value(seed ^ 0x94d0_49bb_1331_11eb, 8 * factor + bit))
            .collect::<Vec<_>>();
        let mut table = EqPolynomial::<AkitaField>::evals(&point, None);
        if factor.is_multiple_of(FACTORS_PER_GROUP) {
            for value in &mut table {
                *value *= gamma_power;
            }
        }
        tables.extend(table);
        if (factor + 1).is_multiple_of(FACTORS_PER_GROUP) {
            gamma_power *= gamma;
        }
    }
    tables
}

fn transcript(initial_claim: AkitaField) -> EvalTranscript {
    let mut transcript = EvalTranscript::new(b"metal-instruction-ra-eval");
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

fn unscale_final_claims(
    raw_claims: &[AkitaField],
    gamma: AkitaField,
) -> EvalResult<Vec<AkitaField>> {
    let Some(gamma_inv) = gamma.inverse() else {
        return Err("Instruction RA gamma must be invertible".into());
    };
    let mut claims = raw_claims.to_vec();
    let mut power_inv = AkitaField::one();
    for group in 0..GROUPS {
        claims[group * FACTORS_PER_GROUP] *= power_inv;
        power_inv *= gamma_inv;
    }
    Ok(claims)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the trace bundles all semantically checked round state"
)]
fn finish_trace(
    q_evals: Vec<[AkitaField; 4]>,
    round_polys: Vec<UnivariatePoly<AkitaField>>,
    challenges: Vec<AkitaField>,
    states: Vec<RoundState>,
    scheduled_tables: Option<Vec<AkitaField>>,
    cutoff_tables: Option<Vec<AkitaField>>,
    tables: &CpuInstructionRa,
    gamma: AkitaField,
    final_sumcheck_claim: AkitaField,
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    transcript: &EvalTranscript,
) -> EvalResult<Trace> {
    let raw_final_claims = tables.final_raw_claims()?;
    let final_claims = unscale_final_claims(&raw_final_claims, gamma)?;
    Ok(Trace {
        q_evals,
        round_polys,
        challenges,
        states,
        scheduled_tables,
        cutoff_tables,
        raw_final_claims,
        final_claims,
        final_sumcheck_claim,
        derived_eq_cycle: gruen.current_scalar(),
        transcript_state: transcript.state(),
    })
}

pub fn run_cpu(workload: &Workload, cutoff: usize, capture: Capture) -> EvalResult<TimedTrace> {
    validate_cutoff(workload.rows(), cutoff, CPU_MATERIALIZE_WIDTH)?;
    let started = Instant::now();
    let mut tables =
        CpuInstructionRa::new(Arc::clone(&workload.cycle_rows), &workload.chunk_tables);
    let mut gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
    let mut transcript = transcript(workload.initial_claim);
    let mut claim = workload.initial_claim;
    let mut q_evals = Vec::with_capacity(workload.log_n);
    let mut round_polys = Vec::with_capacity(workload.log_n);
    let mut challenges = Vec::with_capacity(workload.log_n);
    let mut states = Vec::with_capacity(workload.log_n);
    let mut scheduled_tables = None;
    let mut cutoff_tables = None;

    while tables.state().elements > 1 {
        let state = tables.state();
        if state.is_dense
            && scheduled_tables.is_none()
            && capture.scheduled_elements == Some(state.elements)
        {
            scheduled_tables = Some(tables.flatten_dense()?);
        }
        if capture.cutoff_tables
            && state.is_dense
            && state.elements == cutoff
            && cutoff_tables.is_none()
        {
            cutoff_tables = Some(tables.flatten_dense()?);
        }
        let q = tables.message(&gruen);
        let polynomial = gruen.gruen_poly_from_evals(&q, claim);
        let challenge = absorb_round(&mut transcript, &polynomial);
        claim = polynomial.evaluate(challenge);
        q_evals.push(q);
        round_polys.push(polynomial);
        challenges.push(challenge);
        states.push(state);
        gruen.bind(challenge);
        tables.bind(challenge);
    }
    let trace = finish_trace(
        q_evals,
        round_polys,
        challenges,
        states,
        scheduled_tables,
        cutoff_tables,
        &tables,
        workload.gamma,
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
        resident_plane_zero_copy: true,
        static_device_buffers_stable: true,
        inverse_dense_b_handoff_exact: true,
        preallocated_readback_bytes: 0,
    })
}

pub fn run_hybrid(
    sequence: &mut InstructionRaSequence,
    plane: InstructionRaLookupPlane,
    workload: &Workload,
    cutoff: usize,
    capture: Capture,
) -> EvalResult<TimedTrace> {
    validate_cutoff(workload.rows(), cutoff, sequence.materialize_width())?;
    let mut scheduled_readback = capture
        .scheduled_elements
        .map(|elements| vec![AkitaField::zero(); FACTORS * elements]);
    let mut cutoff_readback = vec![AkitaField::zero(); FACTORS * cutoff];
    let preallocated_readback_bytes = (cutoff_readback.len()
        + scheduled_readback.as_ref().map_or(0, Vec::len))
        * size_of::<AkitaField>();
    let total_started = Instant::now();
    let gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
    let static_buffer_identity = sequence.static_buffer_identity();
    let inverse_buffer_identity = plane.inverse_buffer_identity();
    let reuses_inverse = sequence.reuses_inverse_for_dense();
    let reset_started = Instant::now();
    sequence.reset(plane, &workload.chunk_tables)?;
    let reset = reset_started.elapsed();
    let resident_plane_zero_copy = sequence.lookup_plane_is_resident();

    let mut gruen = gruen;
    let mut transcript = transcript(workload.initial_claim);
    let mut claim = workload.initial_claim;
    let mut q_evals = Vec::with_capacity(workload.log_n);
    let mut round_polys = Vec::with_capacity(workload.log_n);
    let mut challenges = Vec::with_capacity(workload.log_n);
    let mut states = Vec::with_capacity(workload.log_n);
    let mut scheduled_tables = None;
    let mut gpu_wall = Duration::ZERO;
    let mut host_rounds = Duration::ZERO;
    let mut readback = Duration::ZERO;
    let mut cpu_tail = Duration::ZERO;

    let gpu_started = Instant::now();
    let mut q = sequence.message(gruen.e_in_current(), gruen.e_out_current())?;
    gpu_wall += gpu_started.elapsed();

    let (mut tail, cutoff_tables) = loop {
        let state = RoundState {
            branch_width: sequence.branch_width(),
            is_dense: sequence.is_dense(),
            elements: sequence.current_elements(),
        };
        if state.is_dense
            && scheduled_tables.is_none()
            && capture.scheduled_elements == Some(state.elements)
        {
            let read_started = Instant::now();
            let tables = scheduled_readback
                .as_mut()
                .ok_or("scheduled Instruction RA readback storage is missing")?;
            sequence.read_current_tables(tables)?;
            readback += read_started.elapsed();
            scheduled_tables = scheduled_readback.take();
        }

        let host_started = Instant::now();
        let polynomial = gruen.gruen_poly_from_evals(&q, claim);
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
            let tail_started = Instant::now();
            let mut tail = CpuInstructionRa::from_dense(&cutoff_readback, state.elements)?;
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
        q = sequence.bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())?;
        gpu_wall += gpu_started.elapsed();
    };

    let tail_started = Instant::now();
    while tail.state().elements > 1 {
        let state = tail.state();
        let q = tail.message(&gruen);
        let polynomial = gruen.gruen_poly_from_evals(&q, claim);
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
        scheduled_tables,
        cutoff_tables,
        &tail,
        workload.gamma,
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
        resident_plane_zero_copy: resident_plane_zero_copy && !sequence.lookup_plane_is_resident(),
        static_device_buffers_stable: sequence.static_buffer_identity() == static_buffer_identity,
        inverse_dense_b_handoff_exact: !reuses_inverse
            || sequence.dense_b_identity() == Some(inverse_buffer_identity),
        preallocated_readback_bytes,
    })
}

fn validate_cutoff(rows: usize, cutoff: usize, materialize_width: usize) -> EvalResult<()> {
    if cutoff < 2 || !cutoff.is_power_of_two() || cutoff > rows / materialize_width {
        return Err("Instruction RA cutoff must be a dense power-of-two tail".into());
    }
    Ok(())
}

pub fn first_factor_only_gamma_unscale(trace: &Trace, gamma: AkitaField) -> bool {
    if trace.raw_final_claims.len() != FACTORS || trace.final_claims.len() != FACTORS {
        return false;
    }
    let Some(gamma_inv) = gamma.inverse() else {
        return false;
    };
    let mut power_inv = AkitaField::one();
    for index in 0..FACTORS {
        let expected = if index.is_multiple_of(FACTORS_PER_GROUP) {
            let expected = trace.raw_final_claims[index] * power_inv;
            power_inv *= gamma_inv;
            expected
        } else {
            trace.raw_final_claims[index]
        };
        if trace.final_claims[index] != expected {
            return false;
        }
    }
    true
}

pub fn derived_eq_cycle_is_exact(workload: &Workload, trace: &Trace) -> bool {
    let reversed = trace.challenges.iter().rev().copied().collect::<Vec<_>>();
    trace.derived_eq_cycle == EqPolynomial::<AkitaField>::mle(&workload.point, &reversed)
}

pub fn final_relation_is_exact(trace: &Trace) -> bool {
    if trace.raw_final_claims.len() != FACTORS {
        return false;
    }
    let mut q = AkitaField::zero();
    for factors in trace.raw_final_claims.chunks_exact(FACTORS_PER_GROUP) {
        let mut product = AkitaField::one();
        for factor in factors {
            product *= *factor;
        }
        q += product;
    }
    trace.final_sumcheck_claim == trace.derived_eq_cycle * q
}

pub fn expected_cpu_states(log_n: usize) -> Vec<RoundState> {
    (0..log_n)
        .map(|round| RoundState {
            branch_width: if round < CPU_MATERIALIZE_WIDTH.ilog2() as usize {
                1 << round
            } else {
                CPU_MATERIALIZE_WIDTH
            },
            is_dense: round >= CPU_MATERIALIZE_WIDTH.ilog2() as usize,
            elements: 1 << (log_n - round),
        })
        .collect()
}

pub fn expected_hybrid_states(
    log_n: usize,
    materialize_width: usize,
    cutoff: usize,
) -> Vec<RoundState> {
    let materialize_round = materialize_width.ilog2() as usize;
    let cutoff_round = log_n - cutoff.ilog2() as usize;
    (0..log_n)
        .map(|round| {
            let on_gpu = round <= cutoff_round;
            RoundState {
                branch_width: if on_gpu {
                    (1 << round).min(materialize_width)
                } else {
                    CPU_MATERIALIZE_WIDTH
                },
                is_dense: !on_gpu || round >= materialize_round,
                elements: 1 << (log_n - round),
            }
        })
        .collect()
}

pub fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

const _: () = assert!(size_of::<CpuInstructionRow>() == 40);
