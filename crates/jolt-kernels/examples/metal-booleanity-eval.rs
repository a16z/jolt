#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{
    env,
    error::Error,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, RingAccumulator,
};
use jolt_kernels::metal::solinas::{
    BooleanityRow, BooleanitySelector, BooleanitySequence, BooleanitySequenceConfig, SolinasMetal,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

const K: usize = 256;
const CHUNK_BITS: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Trace {
    messages: Vec<UnivariatePoly<AkitaField>>,
    challenges: Vec<AkitaField>,
    final_tables: Vec<AkitaField>,
    final_claim: AkitaField,
    transcript_state: [u8; 32],
}

struct TimedTrace {
    trace: Trace,
    wall: Duration,
    gpu: Duration,
    reset: Duration,
    gpu_wall: Duration,
    host_rounds: Duration,
    readback: Duration,
    cpu_tail: Duration,
    round_wall: Vec<Duration>,
}

enum CpuTables {
    Lazy {
        branches: Vec<Vec<AkitaField>>,
        width: usize,
    },
    Dense(Vec<Polynomial<AkitaField>>),
}

struct CpuBooleanity {
    rows: Arc<Vec<BooleanityRow>>,
    selectors: Arc<Vec<BooleanitySelector>>,
    rho: Arc<Vec<AkitaField>>,
    tables: CpuTables,
}

impl CpuBooleanity {
    fn new(
        rows: Arc<Vec<BooleanityRow>>,
        selectors: Arc<Vec<BooleanitySelector>>,
        rho: Arc<Vec<AkitaField>>,
        base_tables: &[AkitaField],
    ) -> Self {
        let branches = base_tables.chunks_exact(K).map(<[_]>::to_vec).collect();
        Self {
            rows,
            selectors,
            rho,
            tables: CpuTables::Lazy { branches, width: 1 },
        }
    }

    fn current_elements(&self) -> usize {
        match &self.tables {
            CpuTables::Lazy { width, .. } => self.rows.len() / width,
            CpuTables::Dense(polys) => polys[0].len(),
        }
    }

    fn message(&self, gruen: &GruenSplitEqPolynomial<AkitaField>) -> [AkitaField; 2] {
        relation_message(&self.rho, gruen, |poly, row| self.lo_hi(poly, row))
    }

    fn lo_hi(&self, poly: usize, row: usize) -> (AkitaField, AkitaField) {
        match &self.tables {
            CpuTables::Lazy { branches, width } => (
                gather(
                    &branches[poly],
                    *width,
                    &self.rows,
                    self.selectors[poly],
                    2 * row,
                ),
                gather(
                    &branches[poly],
                    *width,
                    &self.rows,
                    self.selectors[poly],
                    2 * row + 1,
                ),
            ),
            CpuTables::Dense(polys) => {
                let evals = polys[poly].evals();
                (evals[2 * row], evals[2 * row + 1])
            }
        }
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.tables = match std::mem::replace(&mut self.tables, CpuTables::Dense(Vec::new())) {
            CpuTables::Lazy { branches, width } => {
                let branches = branches
                    .into_par_iter()
                    .map(|table| {
                        let one_minus = AkitaField::one() - challenge;
                        let mut next = Vec::with_capacity(2 * table.len());
                        next.extend(table.iter().map(|value| one_minus * *value));
                        next.extend(table.iter().map(|value| challenge * *value));
                        next
                    })
                    .collect::<Vec<_>>();
                if width < 8 {
                    CpuTables::Lazy {
                        branches,
                        width: 2 * width,
                    }
                } else {
                    CpuTables::Dense(materialize(
                        &branches,
                        2 * width,
                        &self.rows,
                        &self.selectors,
                    ))
                }
            }
            CpuTables::Dense(mut polys) => {
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                CpuTables::Dense(polys)
            }
        };
    }

    fn final_values(&self) -> Vec<AkitaField> {
        match &self.tables {
            CpuTables::Lazy { .. } => (0..self.rho.len())
                .map(|poly| self.lo_hi(poly, 0).0)
                .collect(),
            CpuTables::Dense(polys) => polys.iter().map(|poly| poly.evals()[0]).collect(),
        }
    }
}

struct CpuTail {
    a: Vec<AkitaField>,
    b: Vec<AkitaField>,
    polys: usize,
    elements: usize,
    source_in_a: bool,
}

impl CpuTail {
    fn new(polys: usize, capacity: usize) -> Self {
        Self {
            a: vec![AkitaField::zero(); polys * capacity],
            b: vec![AkitaField::zero(); polys * capacity / 2],
            polys,
            elements: capacity,
            source_in_a: true,
        }
    }

    fn load(&mut self, sequence: &BooleanitySequence) -> EvalResult<()> {
        self.elements = sequence.current_elements();
        self.source_in_a = true;
        let length = self.polys * self.elements;
        sequence.read_current_tables(&mut self.a[..length])?;
        Ok(())
    }

    fn message(
        &self,
        rho: &[AkitaField],
        gruen: &GruenSplitEqPolynomial<AkitaField>,
    ) -> [AkitaField; 2] {
        let source = self.source();
        relation_message(rho, gruen, |poly, row| {
            let base = poly * self.elements + 2 * row;
            (source[base], source[base + 1])
        })
    }

    fn bind(&mut self, challenge: AkitaField) {
        let source_elements = self.elements;
        let destination_elements = source_elements / 2;
        if self.source_in_a {
            bind_flat_tables(
                &self.a,
                &mut self.b,
                self.polys,
                source_elements,
                destination_elements,
                challenge,
            );
        } else {
            bind_flat_tables(
                &self.b,
                &mut self.a,
                self.polys,
                source_elements,
                destination_elements,
                challenge,
            );
        }
        self.elements = destination_elements;
        self.source_in_a = !self.source_in_a;
    }

    fn source(&self) -> &[AkitaField] {
        let length = self.polys * self.elements;
        if self.source_in_a {
            &self.a[..length]
        } else {
            &self.b[..length]
        }
    }

    fn final_values(&self) -> Vec<AkitaField> {
        (0..self.polys)
            .map(|poly| self.source()[poly * self.elements])
            .collect()
    }
}

fn relation_message(
    rho: &[AkitaField],
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    lo_hi: impl Fn(usize, usize) -> (AkitaField, AkitaField) + Sync,
) -> [AkitaField; 2] {
    let polys = rho.len();
    struct Scratch {
        lanes: [AkitaAccumulator; 2],
        pairs: Vec<(AkitaField, AkitaField)>,
    }

    let lanes = gruen.par_fold_out_in(
        || Scratch {
            lanes: [AkitaAccumulator::default(); 2],
            pairs: vec![(AkitaField::zero(), AkitaField::zero()); polys],
        },
        |scratch, row, _x_in, e_in| {
            for (poly, pair) in scratch.pairs.iter_mut().enumerate() {
                *pair = lo_hi(poly, row);
            }
            let mut constant = AkitaAccumulator::default();
            let mut leading = AkitaAccumulator::default();
            for ((h_0, h_1), rho) in scratch.pairs.iter().zip(rho) {
                let delta = *h_1 - *h_0;
                constant.fmadd(*h_0, *h_0 - *rho);
                leading.fmadd(delta, delta);
            }
            scratch.lanes[0].fmadd(e_in, constant.reduce());
            scratch.lanes[1].fmadd(e_in, leading.reduce());
        },
        |_x_out, e_out, scratch| {
            let mut output = [AkitaAccumulator::default(); 2];
            output[0].fmadd(e_out, scratch.lanes[0].reduce());
            output[1].fmadd(e_out, scratch.lanes[1].reduce());
            output
        },
        |mut lhs, rhs| {
            lhs[0].merge(rhs[0]);
            lhs[1].merge(rhs[1]);
            lhs
        },
    );
    lanes.map(AdditiveAccumulator::reduce)
}

fn initial_claim(state: &CpuBooleanity, gruen: &GruenSplitEqPolynomial<AkitaField>) -> AkitaField {
    struct Scratch {
        lanes: [AkitaAccumulator; 2],
        pairs: Vec<(AkitaField, AkitaField)>,
    }

    let polys = state.rho.len();
    let endpoints = gruen
        .par_fold_out_in(
            || Scratch {
                lanes: [AkitaAccumulator::default(); 2],
                pairs: vec![(AkitaField::zero(), AkitaField::zero()); polys],
            },
            |scratch, row, _x_in, e_in| {
                for (poly, pair) in scratch.pairs.iter_mut().enumerate() {
                    *pair = state.lo_hi(poly, row);
                }
                let mut at_zero = AkitaAccumulator::default();
                let mut at_one = AkitaAccumulator::default();
                for ((h_0, h_1), rho) in scratch.pairs.iter().zip(state.rho.iter()) {
                    at_zero.fmadd(*h_0, *h_0 - *rho);
                    at_one.fmadd(*h_1, *h_1 - *rho);
                }
                scratch.lanes[0].fmadd(e_in, at_zero.reduce());
                scratch.lanes[1].fmadd(e_in, at_one.reduce());
            },
            |_x_out, e_out, scratch| {
                let mut output = [AkitaAccumulator::default(); 2];
                output[0].fmadd(e_out, scratch.lanes[0].reduce());
                output[1].fmadd(e_out, scratch.lanes[1].reduce());
                output
            },
            |mut lhs, rhs| {
                lhs[0].merge(rhs[0]);
                lhs[1].merge(rhs[1]);
                lhs
            },
        )
        .map(AdditiveAccumulator::reduce);
    let (eq_at_zero, eq_at_one) = gruen.current_linear_evals();
    eq_at_zero * endpoints[0] + eq_at_one * endpoints[1]
}

fn bind_flat_tables(
    source: &[AkitaField],
    destination: &mut [AkitaField],
    polys: usize,
    source_elements: usize,
    destination_elements: usize,
    challenge: AkitaField,
) {
    for poly in 0..polys {
        let source = &source[poly * source_elements..(poly + 1) * source_elements];
        let destination =
            &mut destination[poly * destination_elements..(poly + 1) * destination_elements];
        let bind = |(index, output): (usize, &mut AkitaField)| {
            let lo = source[2 * index];
            let hi = source[2 * index + 1];
            *output = lo + challenge * (hi - lo);
        };
        if destination_elements >= 1024 {
            destination.par_iter_mut().enumerate().for_each(bind);
        } else {
            destination
                .iter_mut()
                .enumerate()
                .for_each(|(index, output)| {
                    let lo = source[2 * index];
                    let hi = source[2 * index + 1];
                    *output = lo + challenge * (hi - lo);
                });
        }
    }
}

fn materialize(
    branches: &[Vec<AkitaField>],
    width: usize,
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
) -> Vec<Polynomial<AkitaField>> {
    let elements = rows.len() / width;
    (0..branches.len())
        .into_par_iter()
        .map(|poly| {
            Polynomial::new(
                (0..elements)
                    .map(|index| gather(&branches[poly], width, rows, selectors[poly], index))
                    .collect(),
            )
        })
        .collect()
}

fn gather(
    table: &[AkitaField],
    width: usize,
    rows: &[BooleanityRow],
    selector: BooleanitySelector,
    index: usize,
) -> AkitaField {
    let mut value = AkitaField::zero();
    for offset in 0..width {
        if let Some(hot) = hot_index(rows[index * width + offset], selector) {
            value += table[offset * K + hot];
        }
    }
    value
}

fn hot_index(row: BooleanityRow, selector: BooleanitySelector) -> Option<usize> {
    let words = row.words();
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let lookup = u128::from(words[0]) | (u128::from(words[1]) << 64);
            Some(((lookup >> shift) as usize) & (K - 1))
        }
        BooleanitySelector::Bytecode { shift } => {
            let pc_plus_one = words[4] & ((1 << 56) - 1);
            pc_plus_one
                .checked_sub(1)
                .map(|pc| ((pc >> shift) as usize) & (K - 1))
        }
        BooleanitySelector::Ram { shift } => words[2]
            .checked_sub(1)
            .map(|address| ((address >> shift) as usize) & (K - 1)),
        BooleanitySelector::FusedInc { shift } => {
            let standard = ((biased_fused_inc(words) >> shift) as usize) & (K - 1);
            Some((standard + K / 2) & (K - 1))
        }
        BooleanitySelector::FusedIncMsb => {
            let carry = biased_fused_inc(words) >> 64;
            Some(carry.rem_euclid(K as i128) as usize)
        }
    }
}

fn biased_fused_inc(words: [u64; 5]) -> i128 {
    let magnitude = i128::from(words[3]);
    let value = if words[4] >> 63 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let radix = 1i128 << CHUNK_BITS;
    let bias = (radix / 2) * (((1i128 << 64) - 1) / (radix - 1));
    value + bias
}

fn selectors() -> Vec<BooleanitySelector> {
    let mut selectors = (0..16)
        .map(|index| BooleanitySelector::Lookup {
            shift: (CHUNK_BITS * index) as u32,
        })
        .collect::<Vec<_>>();
    selectors.push(BooleanitySelector::Bytecode { shift: 0 });
    selectors.extend([0, 8, 56].map(|shift| BooleanitySelector::Ram { shift }));
    selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc {
        shift: (CHUNK_BITS * index) as u32,
    }));
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn rows(count: usize, seed: u64) -> EvalResult<Vec<BooleanityRow>> {
    let mut state = u128::from(seed) | (u128::from(!seed) << 64);
    (0..count)
        .map(|row| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            let mapped_pc = (row % 7 != 0).then_some(((state >> 49) as u64) & ((1 << 55) - 2));
            let ram_address = (row % 11 != 0).then_some((state as u64) & (u64::MAX - 1));
            let fused_inc = match row % 6 {
                0 => -(u64::MAX as i128),
                1 => -((1i128 << 63) + row as i128),
                2 => u64::MAX as i128 - row as i128,
                3 => (1i128 << 63) + row as i128,
                4 => row as i128,
                _ => -(row as i128),
            };
            BooleanityRow::new(state, mapped_pc, ram_address, fused_inc)
                .map_err(|error| error.into())
        })
        .collect()
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let low = z ^ (z >> 31);
            AkitaField::from_u128(u128::from(low) | (u128::from(!low) << 64 & (u128::MAX >> 1)))
        })
        .collect()
}

fn base_tables(
    selectors: usize,
    address_point: &[AkitaField],
    gamma: AkitaField,
) -> (Vec<AkitaField>, Vec<AkitaField>) {
    let eq_address = EqPolynomial::<AkitaField>::evals(address_point, None);
    let mut rho = Vec::with_capacity(selectors);
    let mut tables = Vec::with_capacity(selectors * K);
    let mut power = AkitaField::one();
    for _ in 0..selectors {
        rho.push(power);
        tables.extend(eq_address.iter().map(|value| power * *value));
        power *= gamma;
    }
    (rho, tables)
}

fn absorb_round(transcript: &mut EvalTranscript, poly: &UnivariatePoly<AkitaField>) -> AkitaField {
    CompressedLabeledRoundPoly::sumcheck(poly).append_to_transcript(transcript);
    transcript.challenge()
}

fn run_cpu(
    rows: Arc<Vec<BooleanityRow>>,
    selectors: Arc<Vec<BooleanitySelector>>,
    rho: Arc<Vec<AkitaField>>,
    base_tables: &[AkitaField],
    point: &[AkitaField],
    initial_claim: AkitaField,
) -> TimedTrace {
    let started = Instant::now();
    let mut state = CpuBooleanity::new(rows, selectors, rho, base_tables);
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = EvalTranscript::new(b"metal-booleanity-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    let mut messages = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());
    let mut round_wall = Vec::with_capacity(point.len());

    while state.current_elements() > 1 {
        let round_started = Instant::now();
        let relation = state.message(&gruen);
        let poly = gruen.gruen_poly_deg_3(relation[0], relation[1], claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        gruen.bind(challenge);
        state.bind(challenge);
        round_wall.push(round_started.elapsed());
    }
    let wall = started.elapsed();
    TimedTrace {
        trace: Trace {
            messages,
            challenges,
            final_tables: state.final_values(),
            final_claim: claim,
            transcript_state: transcript.state(),
        },
        wall,
        gpu: Duration::ZERO,
        reset: Duration::ZERO,
        gpu_wall: Duration::ZERO,
        host_rounds: Duration::ZERO,
        readback: Duration::ZERO,
        cpu_tail: wall,
        round_wall,
    }
}

fn run_hybrid(
    sequence: &mut BooleanitySequence,
    tail: &mut CpuTail,
    rho: &[AkitaField],
    base_tables: &[AkitaField],
    point: &[AkitaField],
    initial_claim: AkitaField,
    cutoff: usize,
) -> EvalResult<TimedTrace> {
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = EvalTranscript::new(b"metal-booleanity-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    let mut messages = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());
    let mut round_wall = Vec::with_capacity(point.len());
    let mut gpu_wall = Duration::ZERO;
    let mut host_rounds = Duration::ZERO;
    let mut readback = Duration::ZERO;
    let mut cpu_tail = Duration::ZERO;

    let started = Instant::now();
    let reset_started = Instant::now();
    sequence.reset(base_tables)?;
    let reset = reset_started.elapsed();

    let mut round_started = Instant::now();
    let gpu_started = Instant::now();
    let mut relation = sequence.message(gruen.e_in_current(), gruen.e_out_current())?;
    gpu_wall += gpu_started.elapsed();
    loop {
        let host_started = Instant::now();
        let poly = gruen.gruen_poly_deg_3(relation[0], relation[1], claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        host_rounds += host_started.elapsed();
        round_wall.push(round_started.elapsed());

        if sequence.current_elements() <= cutoff || sequence.current_elements() == 2 {
            let readback_started = Instant::now();
            tail.load(sequence)?;
            readback += readback_started.elapsed();
            let tail_started = Instant::now();
            gruen.bind(challenge);
            tail.bind(challenge);
            cpu_tail += tail_started.elapsed();
            break;
        }

        let host_started = Instant::now();
        gruen.bind(challenge);
        host_rounds += host_started.elapsed();
        round_started = Instant::now();
        let gpu_started = Instant::now();
        relation =
            sequence.bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())?;
        gpu_wall += gpu_started.elapsed();
    }

    while tail.elements > 1 {
        let round_started = Instant::now();
        let tail_started = Instant::now();
        let relation = tail.message(rho, &gruen);
        let poly = gruen.gruen_poly_deg_3(relation[0], relation[1], claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        gruen.bind(challenge);
        tail.bind(challenge);
        cpu_tail += tail_started.elapsed();
        round_wall.push(round_started.elapsed());
    }

    Ok(TimedTrace {
        trace: Trace {
            messages,
            challenges,
            final_tables: tail.final_values(),
            final_claim: claim,
            transcript_state: transcript.state(),
        },
        wall: started.elapsed(),
        gpu: sequence.gpu_active_time(),
        reset,
        gpu_wall,
        host_rounds,
        readback,
        cpu_tail,
        round_wall,
    })
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn round_medians(samples: &[Vec<Duration>]) -> Vec<f64> {
    (0..samples[0].len())
        .map(|round| {
            let mut values = samples
                .iter()
                .map(|sample| sample[round])
                .collect::<Vec<_>>();
            median(&mut values).as_secs_f64()
        })
        .collect()
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 22)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 7)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let cutoff_log2 = env_usize("JOLT_METAL_CUTOFF_LOG2", 10)?;
    let threads = env_usize("JOLT_METAL_BOOLEANITY_THREADS", 256)?;
    let dense_threads = env_usize("JOLT_METAL_BOOLEANITY_DENSE_THREADS", threads)?;
    let materialize_width = env_usize("JOLT_METAL_BOOLEANITY_MATERIALIZE_WIDTH", 8)?;
    if !(8..32).contains(&log_n)
        || repeats < 3
        || repeats.is_multiple_of(2)
        || cutoff_log2 < 1
        || cutoff_log2 > log_n - 4
        || !(1..=32).contains(&materialize_width)
        || !materialize_width.is_power_of_two()
    {
        return Err("log_n, repeats, or cutoff is outside the evaluator domain".into());
    }

    let elements = 1usize << log_n;
    let cutoff = 1usize << cutoff_log2;
    let selectors = Arc::new(selectors());
    let rows = Arc::new(rows(elements, seed)?);
    let point = values(log_n, seed ^ 0x6a09_e667_f3bc_c909);
    let address_point = values(CHUNK_BITS, seed ^ 0xbb67_ae85_84ca_a73b);
    let gamma = values(1, seed ^ 0x3c6e_f372_fe94_f82b)[0];
    let (rho, base_tables) = base_tables(selectors.len(), &address_point, gamma);
    let rho = Arc::new(rho);
    let initial_gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let claim_state = CpuBooleanity::new(
        Arc::clone(&rows),
        Arc::clone(&selectors),
        Arc::clone(&rho),
        &base_tables,
    );
    let initial_claim = initial_claim(&claim_state, &initial_gruen);

    let context = SolinasMetal::for_akita()?;
    let prepare_started = Instant::now();
    let mut sequence = context.prepare_booleanity_sequence(
        &rows,
        &selectors,
        &base_tables,
        &rho,
        K,
        initial_gruen.e_in_current().len(),
        initial_gruen.e_out_current().len(),
        BooleanitySequenceConfig {
            threads_per_threadgroup: Some(threads),
            dense_threads_per_threadgroup: Some(dense_threads),
            materialize_width,
        },
    )?;
    let prepare = prepare_started.elapsed();
    let mut tail = CpuTail::new(selectors.len(), cutoff);

    let cpu_reference = run_cpu(
        Arc::clone(&rows),
        Arc::clone(&selectors),
        Arc::clone(&rho),
        &base_tables,
        &point,
        initial_claim,
    );
    let hybrid_reference = run_hybrid(
        &mut sequence,
        &mut tail,
        &rho,
        &base_tables,
        &point,
        initial_claim,
        cutoff,
    )?;
    let exact_messages = cpu_reference.trace.messages == hybrid_reference.trace.messages;
    let exact_challenges = cpu_reference.trace.challenges == hybrid_reference.trace.challenges;
    let exact_final_state = cpu_reference.trace.final_tables == hybrid_reference.trace.final_tables
        && cpu_reference.trace.final_claim == hybrid_reference.trace.final_claim
        && cpu_reference.trace.transcript_state == hybrid_reference.trace.transcript_state;
    if !exact_messages || !exact_challenges || !exact_final_state {
        return Err("hybrid Booleanity trace differs from the optimized CPU oracle".into());
    }

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut hybrid_times = Vec::with_capacity(repeats);
    let mut reset_times = Vec::with_capacity(repeats);
    let mut gpu_wall_times = Vec::with_capacity(repeats);
    let mut host_round_times = Vec::with_capacity(repeats);
    let mut readback_times = Vec::with_capacity(repeats);
    let mut cpu_tail_times = Vec::with_capacity(repeats);
    let mut cpu_round_times = Vec::with_capacity(repeats);
    let mut hybrid_round_times = Vec::with_capacity(repeats);
    let mut gpu_time = Duration::ZERO;
    for repeat in 0..repeats {
        let run_cpu_sample = || {
            black_box(run_cpu(
                Arc::clone(&rows),
                Arc::clone(&selectors),
                Arc::clone(&rho),
                &base_tables,
                &point,
                initial_claim,
            ))
        };
        if repeat.is_multiple_of(2) {
            let cpu = run_cpu_sample();
            cpu_times.push(cpu.wall);
            cpu_round_times.push(cpu.round_wall);
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                &mut tail,
                &rho,
                &base_tables,
                &point,
                initial_claim,
                cutoff,
            )?);
            record_hybrid(
                hybrid,
                &mut hybrid_times,
                &mut reset_times,
                &mut gpu_wall_times,
                &mut host_round_times,
                &mut readback_times,
                &mut cpu_tail_times,
                &mut hybrid_round_times,
                &mut gpu_time,
            );
        } else {
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                &mut tail,
                &rho,
                &base_tables,
                &point,
                initial_claim,
                cutoff,
            )?);
            record_hybrid(
                hybrid,
                &mut hybrid_times,
                &mut reset_times,
                &mut gpu_wall_times,
                &mut host_round_times,
                &mut readback_times,
                &mut cpu_tail_times,
                &mut hybrid_round_times,
                &mut gpu_time,
            );
            let cpu = run_cpu_sample();
            cpu_times.push(cpu.wall);
            cpu_round_times.push(cpu.round_wall);
        }
    }

    let cpu_round_medians = round_medians(&cpu_round_times);
    let hybrid_round_medians = round_medians(&hybrid_round_times);
    let cpu_median = median(&mut cpu_times);
    let hybrid_median = median(&mut hybrid_times);
    let reset_median = median(&mut reset_times);
    let gpu_wall_median = median(&mut gpu_wall_times);
    let host_round_median = median(&mut host_round_times);
    let readback_median = median(&mut readback_times);
    let cpu_tail_median = median(&mut cpu_tail_times);
    let speedup = cpu_median.as_secs_f64() / hybrid_median.as_secs_f64();
    let polys = selectors.len() as u128;
    let n = elements as u128;
    let useful = |width: u128| {
        polys * (2 * n + n / width - 3) + 2 * n - 2 + 2 * polys * K as u128 * (width - 1)
    };
    let cpu_useful_multiplications = useful(16);
    let initial_pair_precompute_multiplications =
        polys * ((K + 1) as u128 + ((K + 1) as u128).pow(2));
    let metal_useful_multiplications =
        useful(materialize_width as u128) - polys * n + initial_pair_precompute_multiplications;
    let lazy_scans = materialize_width.ilog2() as u128 + 1;
    let optimistic_unique_bytes =
        40 * n * lazy_scans + 64 * polys * n / materialize_width as u128 - 48 * polys;
    let logical_table_cache_bytes = 16 * polys * n * lazy_scans;
    let info = context.device_info();
    let output = json!({
        "schema_version": 1,
        "kernel": "booleanity_cycle",
        "metrics": {
            "hybrid_speedup": speedup,
            "useful_cpu_gmul_per_second": cpu_useful_multiplications as f64 / cpu_median.as_secs_f64() / 1e9,
            "useful_hybrid_gmul_per_second": metal_useful_multiplications as f64 / hybrid_median.as_secs_f64() / 1e9
        },
        "timings": {
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "hybrid_median_seconds": hybrid_median.as_secs_f64(),
            "prepare_once_seconds": prepare.as_secs_f64(),
            "base_table_reset_median_seconds": reset_median.as_secs_f64(),
            "gpu_dispatch_wall_median_seconds": gpu_wall_median.as_secs_f64(),
            "host_round_median_seconds": host_round_median.as_secs_f64(),
            "readback_median_seconds": readback_median.as_secs_f64(),
            "cpu_tail_median_seconds": cpu_tail_median.as_secs_f64(),
            "cpu_round_median_seconds": cpu_round_medians,
            "hybrid_round_median_seconds": hybrid_round_medians,
            "gpu_active_total_seconds": gpu_time.as_secs_f64(),
            "repeats": repeats
        },
        "guards": {
            "exact_messages": exact_messages,
            "exact_challenges": exact_challenges,
            "exact_final_state": exact_final_state,
            "no_round_allocations": sequence.round_device_buffer_allocations() == 0
        },
        "analytical": {
            "cpu_useful_field_multiplications": cpu_useful_multiplications.to_string(),
            "metal_useful_field_multiplications": metal_useful_multiplications.to_string(),
            "optimistic_unique_device_bytes": optimistic_unique_bytes.to_string(),
            "logical_cache_table_bytes": logical_table_cache_bytes.to_string(),
            "initial_pair_table_bytes": (16 * initial_pair_precompute_multiplications).to_string(),
            "compute_floor_seconds_at_16_4_gmul_s": metal_useful_multiplications as f64 / 16.4e9,
            "unique_memory_floor_seconds_at_420_gib_s": optimistic_unique_bytes as f64 / (420.0 * 1024.0_f64.powi(3))
        },
        "resources": {
            "gpu_seconds": gpu_time.as_secs_f64()
        },
        "workload": {
            "log_n": log_n,
            "elements": elements,
            "polynomials": selectors.len(),
            "k": K,
            "cutoff_log2": cutoff_log2,
            "threads": threads,
            "dense_threads": dense_threads,
            "materialize_width": materialize_width,
            "initial_pair_tables": true,
            "host_fiat_shamir": true,
            "resident_row_handoff": true,
            "base_table_reset_in_primary_metric": true,
            "final_readback_in_primary_metric": true
        },
        "fingerprint": {
            "device": info.name,
            "max_buffer_length": info.max_buffer_length,
            "max_threadgroup_memory_length": info.max_threadgroup_memory_length,
            "cpu_threads": std::thread::available_parallelism()?.get()
        }
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the evaluator records each timed component"
)]
fn record_hybrid(
    hybrid: TimedTrace,
    hybrid_times: &mut Vec<Duration>,
    reset_times: &mut Vec<Duration>,
    gpu_wall_times: &mut Vec<Duration>,
    host_round_times: &mut Vec<Duration>,
    readback_times: &mut Vec<Duration>,
    cpu_tail_times: &mut Vec<Duration>,
    hybrid_round_times: &mut Vec<Vec<Duration>>,
    gpu_time: &mut Duration,
) {
    hybrid_times.push(hybrid.wall);
    reset_times.push(hybrid.reset);
    gpu_wall_times.push(hybrid.gpu_wall);
    host_round_times.push(hybrid.host_rounds);
    readback_times.push(hybrid.readback);
    cpu_tail_times.push(hybrid.cpu_tail);
    hybrid_round_times.push(hybrid.round_wall);
    *gpu_time += hybrid.gpu;
}
