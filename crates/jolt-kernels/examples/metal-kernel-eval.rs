#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::env;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, RingAccumulator,
};
use jolt_kernels::metal::solinas::{
    Product5Sequence, Product5SequenceConfig, SolinasMetal, PRODUCT5_FACTORS,
};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

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
    handoff: Duration,
    gpu_wall: Duration,
    host_rounds: Duration,
    readback: Duration,
    cpu_tail: Duration,
}

struct CpuTables {
    a: Vec<AkitaField>,
    b: Vec<AkitaField>,
    elements: usize,
    source_in_a: bool,
}

impl CpuTables {
    fn from_initial(initial: &[AkitaField], elements: usize) -> Self {
        Self {
            a: initial.to_vec(),
            b: vec![AkitaField::zero(); PRODUCT5_FACTORS * elements / 2],
            elements,
            source_in_a: true,
        }
    }

    fn for_tail(max_elements: usize) -> Self {
        Self {
            a: vec![AkitaField::zero(); PRODUCT5_FACTORS * max_elements],
            b: vec![AkitaField::zero(); PRODUCT5_FACTORS * max_elements / 2],
            elements: max_elements,
            source_in_a: true,
        }
    }

    fn load_from_sequence(&mut self, sequence: &Product5Sequence) -> EvalResult<()> {
        self.elements = sequence.current_elements();
        self.source_in_a = true;
        let length = PRODUCT5_FACTORS * self.elements;
        sequence.read_current_tables(&mut self.a[..length])?;
        Ok(())
    }

    fn message(
        &self,
        gruen: &GruenSplitEqPolynomial<AkitaField>,
    ) -> [AkitaField; PRODUCT5_FACTORS] {
        dense_message(self.source(), self.elements, gruen)
    }

    fn bind(&mut self, challenge: AkitaField) {
        let source_elements = self.elements;
        let destination_elements = source_elements / 2;
        if self.source_in_a {
            bind_tables(
                &self.a,
                &mut self.b,
                source_elements,
                destination_elements,
                challenge,
            );
        } else {
            bind_tables(
                &self.b,
                &mut self.a,
                source_elements,
                destination_elements,
                challenge,
            );
        }
        self.elements = destination_elements;
        self.source_in_a = !self.source_in_a;
    }

    fn source(&self) -> &[AkitaField] {
        let length = PRODUCT5_FACTORS * self.elements;
        if self.source_in_a {
            &self.a[..length]
        } else {
            &self.b[..length]
        }
    }
}

fn bind_tables(
    source: &[AkitaField],
    destination: &mut [AkitaField],
    source_elements: usize,
    destination_elements: usize,
    challenge: AkitaField,
) {
    destination[..PRODUCT5_FACTORS * destination_elements]
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let factor = index / destination_elements;
            let position = index % destination_elements;
            let source_index = factor * source_elements + 2 * position;
            let lo = source[source_index];
            let hi = source[source_index + 1];
            *output = lo + challenge * (hi - lo);
        });
}

fn dense_message(
    tables: &[AkitaField],
    elements: usize,
    gruen: &GruenSplitEqPolynomial<AkitaField>,
) -> [AkitaField; PRODUCT5_FACTORS] {
    struct Scratch {
        lanes: [AkitaAccumulator; PRODUCT5_FACTORS],
        evals: [AkitaField; PRODUCT5_FACTORS],
        steps: [AkitaField; PRODUCT5_FACTORS],
    }

    let block_lanes = gruen.par_fold_out_in(
        || Scratch {
            lanes: [AkitaAccumulator::default(); PRODUCT5_FACTORS],
            evals: [AkitaField::zero(); PRODUCT5_FACTORS],
            steps: [AkitaField::zero(); PRODUCT5_FACTORS],
        },
        |scratch, row, _x_in, e_in| {
            for factor in 0..PRODUCT5_FACTORS {
                let base = factor * elements + 2 * row;
                let mut lo = tables[base];
                let mut hi = tables[base + 1];
                if factor == 0 {
                    lo *= e_in;
                    hi *= e_in;
                }
                scratch.evals[factor] = hi;
                scratch.steps[factor] = hi - lo;
            }
            accumulate_product(&scratch.evals, &mut scratch.lanes[0]);
            for lane in 1..PRODUCT5_FACTORS - 1 {
                for (eval, step) in scratch.evals.iter_mut().zip(scratch.steps) {
                    *eval += step;
                }
                accumulate_product(&scratch.evals, &mut scratch.lanes[lane]);
            }
            accumulate_product(&scratch.steps, &mut scratch.lanes[PRODUCT5_FACTORS - 1]);
        },
        |_x_out, e_out, scratch| {
            let mut out = [AkitaAccumulator::default(); PRODUCT5_FACTORS];
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

#[inline]
fn accumulate_product(factors: &[AkitaField; PRODUCT5_FACTORS], lane: &mut AkitaAccumulator) {
    let mut product = factors[0];
    for factor in &factors[1..PRODUCT5_FACTORS - 1] {
        product *= *factor;
    }
    lane.fmadd(product, factors[PRODUCT5_FACTORS - 1]);
}

fn absorb_round(transcript: &mut EvalTranscript, poly: &UnivariatePoly<AkitaField>) -> AkitaField {
    CompressedLabeledRoundPoly::sumcheck(poly).append_to_transcript(transcript);
    transcript.challenge()
}

fn run_cpu(initial: &[AkitaField], point: &[AkitaField], initial_claim: AkitaField) -> TimedTrace {
    let mut tables = CpuTables::from_initial(initial, 1usize << point.len());
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = EvalTranscript::new(b"metal-cycle-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    let mut messages = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());

    let started = Instant::now();
    while tables.elements > 1 {
        let q_evals = tables.message(&gruen);
        let poly = gruen.gruen_poly_from_evals(&q_evals, claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        gruen.bind(challenge);
        tables.bind(challenge);
    }
    let wall = started.elapsed();

    TimedTrace {
        trace: Trace {
            messages,
            challenges,
            final_tables: tables.source().to_vec(),
            final_claim: claim,
            transcript_state: transcript.state(),
        },
        wall,
        gpu: Duration::ZERO,
        handoff: Duration::ZERO,
        gpu_wall: Duration::ZERO,
        host_rounds: Duration::ZERO,
        readback: Duration::ZERO,
        cpu_tail: wall,
    }
}

fn run_hybrid(
    sequence: &mut Product5Sequence,
    tail: &mut CpuTables,
    initial: &[AkitaField],
    point: &[AkitaField],
    initial_claim: AkitaField,
    cutoff: usize,
) -> EvalResult<TimedTrace> {
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = EvalTranscript::new(b"metal-cycle-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    let mut messages = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());
    let mut gpu_wall = Duration::ZERO;
    let mut host_rounds = Duration::ZERO;
    let mut readback = Duration::ZERO;
    let mut cpu_tail = Duration::ZERO;

    let started = Instant::now();
    let handoff_started = Instant::now();
    sequence.reset(initial)?;
    let handoff = handoff_started.elapsed();
    let gpu_started = Instant::now();
    let mut q_evals = sequence.message(gruen.e_in_current(), gruen.e_out_current())?;
    gpu_wall += gpu_started.elapsed();

    loop {
        let host_started = Instant::now();
        let poly = gruen.gruen_poly_from_evals(&q_evals, claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        host_rounds += host_started.elapsed();

        if sequence.current_elements() <= cutoff || sequence.current_elements() == 2 {
            let readback_started = Instant::now();
            tail.load_from_sequence(sequence)?;
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
        let gpu_started = Instant::now();
        q_evals =
            sequence.bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())?;
        gpu_wall += gpu_started.elapsed();
    }

    let tail_started = Instant::now();
    finish_cpu_tail(
        tail,
        &mut gruen,
        &mut transcript,
        &mut claim,
        &mut messages,
        &mut challenges,
    );
    cpu_tail += tail_started.elapsed();
    let wall = started.elapsed();
    Ok(TimedTrace {
        trace: Trace {
            messages,
            challenges,
            final_tables: tail.source().to_vec(),
            final_claim: claim,
            transcript_state: transcript.state(),
        },
        wall,
        gpu: sequence.gpu_active_time(),
        handoff,
        gpu_wall,
        host_rounds,
        readback,
        cpu_tail,
    })
}

fn finish_cpu_tail(
    tables: &mut CpuTables,
    gruen: &mut GruenSplitEqPolynomial<AkitaField>,
    transcript: &mut EvalTranscript,
    claim: &mut AkitaField,
    messages: &mut Vec<UnivariatePoly<AkitaField>>,
    challenges: &mut Vec<AkitaField>,
) {
    while tables.elements > 1 {
        let q_evals = tables.message(gruen);
        let poly = gruen.gruen_poly_from_evals(&q_evals, *claim);
        let challenge = absorb_round(transcript, &poly);
        *claim = poly.evaluate(challenge);
        messages.push(poly);
        challenges.push(challenge);
        gruen.bind(challenge);
        tables.bind(challenge);
    }
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

fn main() -> EvalResult<()> {
    let kernel = env::args().nth(1).unwrap_or_default();
    if kernel != "instruction-read-raf-cycle" {
        return Err(format!("unknown evaluator kernel: {kernel}").into());
    }

    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 20)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 9)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let cutoff_log2 = env_usize("JOLT_METAL_CUTOFF_LOG2", 16)?;
    let message_threads = env_usize("JOLT_METAL_MESSAGE_THREADS", 128)?;
    let transition_threads = env_usize("JOLT_METAL_TRANSITION_THREADS", 64)?;
    if !(2..32).contains(&log_n) || repeats < 3 || cutoff_log2 >= log_n {
        return Err("log_n, repeats, or cutoff is outside the evaluator domain".into());
    }
    let elements = 1usize << log_n;
    let cutoff = 1usize << cutoff_log2;
    let initial = values(PRODUCT5_FACTORS * elements, seed);
    let point = values(log_n, seed ^ 0x6a09_e667_f3bc_c909);
    let initial_claim = AkitaField::from_u64(seed ^ 0xa54f_f53a_5f1d_36f1);
    let context = SolinasMetal::for_akita()?;
    let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let mut sequence = context.prepare_product5_sequence(
        &initial,
        elements,
        gruen.e_in_current(),
        gruen.e_out_current(),
        Product5SequenceConfig {
            message_threads_per_threadgroup: Some(message_threads),
            transition_threads_per_threadgroup: Some(transition_threads),
        },
    )?;
    let mut tail = CpuTables::for_tail(cutoff);

    let cpu_reference = run_cpu(&initial, &point, initial_claim);
    let hybrid_reference = run_hybrid(
        &mut sequence,
        &mut tail,
        &initial,
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
        return Err("hybrid dense-cycle trace differs from the CPU oracle".into());
    }

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut hybrid_times = Vec::with_capacity(repeats);
    let mut direct_handoff_times = Vec::with_capacity(repeats);
    let mut handoff_times = Vec::with_capacity(repeats);
    let mut gpu_wall_times = Vec::with_capacity(repeats);
    let mut host_round_times = Vec::with_capacity(repeats);
    let mut readback_times = Vec::with_capacity(repeats);
    let mut cpu_tail_times = Vec::with_capacity(repeats);
    let mut gpu_time = Duration::ZERO;
    for repeat in 0..repeats {
        if repeat.is_multiple_of(2) {
            let cpu = black_box(run_cpu(&initial, &point, initial_claim));
            cpu_times.push(cpu.wall);
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                &mut tail,
                &initial,
                &point,
                initial_claim,
                cutoff,
            )?);
            hybrid_times.push(hybrid.wall);
            direct_handoff_times.push(hybrid.wall.saturating_sub(hybrid.handoff));
            handoff_times.push(hybrid.handoff);
            gpu_wall_times.push(hybrid.gpu_wall);
            host_round_times.push(hybrid.host_rounds);
            readback_times.push(hybrid.readback);
            cpu_tail_times.push(hybrid.cpu_tail);
            gpu_time += hybrid.gpu;
        } else {
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                &mut tail,
                &initial,
                &point,
                initial_claim,
                cutoff,
            )?);
            hybrid_times.push(hybrid.wall);
            direct_handoff_times.push(hybrid.wall.saturating_sub(hybrid.handoff));
            handoff_times.push(hybrid.handoff);
            gpu_wall_times.push(hybrid.gpu_wall);
            host_round_times.push(hybrid.host_rounds);
            readback_times.push(hybrid.readback);
            cpu_tail_times.push(hybrid.cpu_tail);
            gpu_time += hybrid.gpu;
            let cpu = black_box(run_cpu(&initial, &point, initial_claim));
            cpu_times.push(cpu.wall);
        }
    }
    let cpu_median = median(&mut cpu_times);
    let hybrid_median = median(&mut hybrid_times);
    let direct_handoff_median = median(&mut direct_handoff_times);
    let handoff_median = median(&mut handoff_times);
    let gpu_wall_median = median(&mut gpu_wall_times);
    let host_round_median = median(&mut host_round_times);
    let readback_median = median(&mut readback_times);
    let cpu_tail_median = median(&mut cpu_tail_times);
    let speedup = cpu_median.as_secs_f64() / hybrid_median.as_secs_f64();
    let direct_handoff_speedup = cpu_median.as_secs_f64() / direct_handoff_median.as_secs_f64();
    let info = context.device_info();
    let output = json!({
        "schema_version": 1,
        "kernel": "instruction_read_raf_cycle",
        "metrics": {
            "hybrid_speedup": speedup,
            "direct_handoff_speedup": direct_handoff_speedup
        },
        "timings": {
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "hybrid_median_seconds": hybrid_median.as_secs_f64(),
            "direct_handoff_median_seconds": direct_handoff_median.as_secs_f64(),
            "handoff_median_seconds": handoff_median.as_secs_f64(),
            "gpu_dispatch_wall_median_seconds": gpu_wall_median.as_secs_f64(),
            "host_round_median_seconds": host_round_median.as_secs_f64(),
            "readback_median_seconds": readback_median.as_secs_f64(),
            "cpu_tail_median_seconds": cpu_tail_median.as_secs_f64(),
            "gpu_active_total_seconds": gpu_time.as_secs_f64(),
            "repeats": repeats
        },
        "guards": {
            "exact_messages": exact_messages,
            "exact_challenges": exact_challenges,
            "exact_final_state": exact_final_state,
            "no_round_allocations": sequence.round_device_buffer_allocations() == 0
        },
        "resources": {
            "gpu_seconds": gpu_time.as_secs_f64()
        },
        "workload": {
            "log_n": log_n,
            "elements": elements,
            "cutoff_log2": cutoff_log2,
            "message_threads": message_threads,
            "transition_threads": transition_threads,
            "field_factors": PRODUCT5_FACTORS,
            "host_fiat_shamir": true,
            "handoff_in_primary_metric": true,
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
