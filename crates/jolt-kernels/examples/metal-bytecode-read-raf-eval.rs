#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::BytecodeCycleSequenceConfig;
use jolt_kernels::metal::{
    BytecodeReadRafMetalConfig, BytecodeReadRafResidentRows, MetalBackend, MetalConfig,
};
use jolt_kernels::optimized::{BytecodeCycleAlgebra, OptimizedBytecodeReadRafCycle};
use jolt_kernels::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle, BytecodeReadRafCycleOutputClaims,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
};
use jolt_witness::testing::with_sample_backend_at_log_t;
use jolt_witness::JoltWitnessPlane;
use serde_json::json;
use tracing::field::{Field as TracingField, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::Layer;

type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;
type OutputClaims = BytecodeReadRafCycleOutputClaims<AkitaField>;
type OutputPoints = BytecodeReadRafCycleOutputClaims<Vec<AkitaField>>;

const PREFIX: &str = "MetalBytecodeReadRafCycle::";

#[derive(Clone, Copy)]
enum Arm {
    Cpu,
    Metal,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct PhaseObservation {
    counts: BTreeMap<String, usize>,
    allocation: Option<BTreeMap<String, u64>>,
    readback: Option<BTreeMap<String, u64>>,
}

impl PhaseObservation {
    fn count(&self, suffix: &str) -> usize {
        self.counts
            .get(&format!("{PREFIX}{suffix}"))
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Clone, Default)]
struct PhaseObserver(Arc<Mutex<PhaseObservation>>);

impl PhaseObserver {
    fn clear(&self) {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = PhaseObservation::default();
    }

    fn snapshot(&self) -> PhaseObservation {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

struct PhaseLayer(PhaseObserver);

#[derive(Default)]
struct NumericFields(BTreeMap<String, u64>);

impl Visit for NumericFields {
    fn record_u64(&mut self, field: &TracingField, value: u64) {
        let _ = self.0.insert(field.name().to_owned(), value);
    }

    fn record_i64(&mut self, field: &TracingField, value: i64) {
        if let Ok(value) = u64::try_from(value) {
            let _ = self.0.insert(field.name().to_owned(), value);
        }
    }

    fn record_debug(&mut self, _field: &TracingField, _value: &dyn std::fmt::Debug) {}
}

impl<S: Subscriber> Layer<S> for PhaseLayer {
    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let name = attributes.metadata().name();
        if !name.starts_with(PREFIX) {
            return;
        }
        let mut fields = NumericFields::default();
        attributes.record(&mut fields);
        let mut observation = self
            .0
             .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *observation.counts.entry(name.to_owned()).or_default() += 1;
        if name == format!("{PREFIX}allocation_plan") {
            observation.allocation = Some(fields.0);
        } else if name == format!("{PREFIX}readback") {
            observation.readback = Some(fields.0);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MemberTrace {
    round_polys: Vec<UnivariatePoly<AkitaField>>,
    challenges: Vec<AkitaField>,
    final_claim: AkitaField,
    output_points: OutputPoints,
    output_claims: OutputClaims,
    transcript_state: [u8; 32],
}

struct MemberRun {
    trace: MemberTrace,
    prepare: Duration,
    rounds: Vec<Duration>,
    finish: Duration,
    output_claims: Duration,
    host_fs: Duration,
    phases: PhaseObservation,
}

impl MemberRun {
    fn member_time(&self) -> Duration {
        self.prepare
            + self.rounds.iter().copied().sum::<Duration>()
            + self.finish
            + self.output_claims
    }

    fn core_time(&self) -> Duration {
        self.rounds.iter().copied().sum::<Duration>() + self.finish + self.output_claims
    }
}

struct Instance {
    relation: BytecodeReadRafCycle<AkitaField>,
    claims: BytecodeReadRafInputClaims<AkitaField>,
    points: BytecodeReadRafInputClaims<Vec<AkitaField>>,
    challenges: BytecodeReadRafCyclePhaseCommittedChallenges<AkitaField>,
}

#[derive(Clone, Copy)]
struct Tuning {
    message_threads: usize,
    transition_threads: usize,
    max_threadgroups: usize,
    cutoff_log2: usize,
    trace_cutoff_log2: usize,
}

fn field(seed: u64, index: usize) -> AkitaField {
    let value = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(2 * index as u64 + 3);
    AkitaField::from_u64(value)
}

fn point(length: usize, seed: u64) -> Vec<AkitaField> {
    (0..length).map(|index| field(seed, index)).collect()
}

fn instance(log_t: usize, seed: u64) -> Instance {
    let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
    let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
        dimensions,
        r_address: point(13, seed ^ 0x243f_6a88_85a3_08d3),
        stage_cycle_points: std::array::from_fn(|stage| {
            point(log_t, seed ^ (0x1319_8a2e_0370_7344 + stage as u64))
        }),
        entry_bytecode_index: 17,
        committed_chunk_bits: 8,
        val_stages: (0..NUM_BYTECODE_VAL_STAGES)
            .map(|stage| field(seed ^ 0xa409_3822_299f_31d0, stage))
            .collect(),
    });
    Instance {
        relation,
        claims: BytecodeReadRafInputClaims::default(),
        points: BytecodeReadRafInputClaims::default(),
        challenges: BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: field(seed ^ 0x082e_fa98_ec4e_6c89, 0),
        },
    }
}

fn transcript(initial_claim: AkitaField) -> EvalTranscript {
    let mut transcript = EvalTranscript::new(b"metal-bytecode-read-raf-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    transcript
}

fn run_member(
    arm: Arm,
    backend: &MetalBackend,
    resident_rows: &BytecodeReadRafResidentRows,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    instance: &Instance,
    challenge_tape: Option<&[AkitaField]>,
    observer: &PhaseObserver,
) -> EvalResult<MemberRun> {
    observer.clear();
    let mut session = ProofSession::default();
    if matches!(arm, Arm::Metal) {
        resident_rows.install(&mut session);
    }
    let inputs = || ProverInputs {
        relation: &instance.relation,
        claims: &instance.claims,
        points: &instance.points,
        challenges: &instance.challenges,
    };

    let prepare_started = Instant::now();
    let mut kernel: Box<
        dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafCycle<AkitaField>>,
    > = match arm {
        Arm::Cpu => OptimizedBytecodeReadRafCycle::new(BytecodeCycleAlgebra::Q10).prepare(
            &mut session,
            witness,
            inputs(),
        )?,
        Arm::Metal => <MetalBackend as PrepareKernel<
            AkitaField,
            BytecodeReadRafCycle<AkitaField>,
        >>::prepare(backend, &mut session, witness, inputs())?,
    };
    let prepare = prepare_started.elapsed();
    if kernel.num_rounds() != instance.relation.rounds() {
        return Err("Bytecode evaluator kernel round count disagrees with the relation".into());
    }

    let mut transcript = transcript(instance.claims.address_phase);
    let mut claim = instance.claims.address_phase;
    let mut round_polys = Vec::with_capacity(kernel.num_rounds());
    let mut challenges = Vec::with_capacity(kernel.num_rounds());
    let mut rounds = Vec::with_capacity(kernel.num_rounds());
    let mut host_fs = Duration::ZERO;
    for round in 0..kernel.num_rounds() {
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        let round_started = Instant::now();
        let polynomial = black_box(kernel.prove_round(bind, round, claim)?);
        rounds.push(round_started.elapsed());

        let host_started = Instant::now();
        CompressedLabeledRoundPoly::sumcheck(&polynomial).append_to_transcript(&mut transcript);
        let challenge = transcript.challenge();
        if let Some(tape) = challenge_tape {
            if tape.get(round) != Some(&challenge) {
                return Err(
                    format!("Bytecode evaluator challenge diverged at round {round}").into(),
                );
            }
        }
        claim = polynomial.evaluate(challenge);
        host_fs += host_started.elapsed();
        challenges.push(challenge);
        round_polys.push(polynomial);
    }

    let final_bind = *challenges
        .last()
        .ok_or("Bytecode evaluator produced no challenges")?;
    let finish_started = Instant::now();
    kernel.finish_rounds(final_bind)?;
    let finish = finish_started.elapsed();
    let output_started = Instant::now();
    let output_claims = kernel.output_claims(&instance.claims)?;
    let output_claims_time = output_started.elapsed();
    let output_points = instance
        .relation
        .derive_opening_points(&challenges, &instance.points)?;
    let phases = observer.snapshot();
    Ok(MemberRun {
        trace: MemberTrace {
            round_polys,
            challenges,
            final_claim: claim,
            output_points,
            output_claims,
            transcript_state: transcript.state(),
        },
        prepare,
        rounds,
        finish,
        output_claims: output_claims_time,
        host_fs,
        phases,
    })
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    median(
        &values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>(),
    )
}

fn ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn phase_json(observation: &PhaseObservation) -> serde_json::Value {
    json!({
        "counts": observation.counts,
        "allocation": observation.allocation,
        "readback": observation.readback,
    })
}

fn exact_metal_schedule(observation: &PhaseObservation, log_n: usize, cutoff_log2: usize) -> bool {
    let dense_rounds = log_n - cutoff_log2 - 1;
    let allocation = observation.allocation.as_ref();
    let current = allocation.and_then(|fields| fields.get("current_device_bytes"));
    let planned = allocation.and_then(|fields| fields.get("planned_device_bytes"));
    let recommended = allocation.and_then(|fields| fields.get("recommended_device_bytes"));
    let admitted =
        current
            .zip(planned)
            .zip(recommended)
            .is_some_and(|((current, planned), recommended)| {
                current
                    .checked_add(*planned)
                    .is_some_and(|total| total <= *recommended)
            });
    observation.count("prepare") == 1
        && observation.count("allocation_plan") == 1
        && observation.count("first_message") == 1
        && observation.count("first_bind") == 1
        && observation.count("dense_round") == dense_rounds
        && observation.count("readback") == 1
        && observation.count("cpu_tail") == cutoff_log2
        && observation.count("invalid_round") == 0
        && allocation.and_then(|fields| fields.get("device_buffers")) == Some(&17)
        && observation
            .readback
            .as_ref()
            .and_then(|fields| fields.get("bytes"))
            == Some(&(5 * (1u64 << cutoff_log2) * 16))
        && admitted
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 26)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 3)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let tuning = Tuning {
        message_threads: env_usize("JOLT_METAL_BYTECODE_MESSAGE_THREADS", 256)?,
        transition_threads: env_usize("JOLT_METAL_BYTECODE_TRANSITION_THREADS", 128)?,
        max_threadgroups: env_usize("JOLT_METAL_BYTECODE_MAX_THREADGROUPS", 8192)?,
        cutoff_log2: env_usize("JOLT_METAL_BYTECODE_CUTOFF_LOG2", 16)?,
        trace_cutoff_log2: env_usize("JOLT_METAL_BYTECODE_TRACE_CUTOFF_LOG2", 18)?,
    };
    if !(5..=28).contains(&log_n)
        || repeats < 3
        || repeats.is_multiple_of(2)
        || tuning.cutoff_log2 < 1
        || tuning.cutoff_log2 > log_n - 2
        || tuning.trace_cutoff_log2 > log_n
        || !(32..=1024).contains(&tuning.message_threads)
        || !tuning.message_threads.is_power_of_two()
        || !(32..=1024).contains(&tuning.transition_threads)
        || !tuning.transition_threads.is_power_of_two()
        || tuning.max_threadgroups < 1
        || !tuning.max_threadgroups.is_power_of_two()
    {
        return Err("Bytecode evaluator parameters are outside the fixed domain".into());
    }

    let observer = PhaseObserver::default();
    let subscriber = tracing_subscriber::registry().with(PhaseLayer(observer.clone()));
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);
    let trace_elements = 1usize << log_n;
    let cutoff_elements = 1usize << tuning.cutoff_log2;
    let trace_cutoff_elements = 1usize << tuning.trace_cutoff_log2;
    let backend = MetalBackend::new(MetalConfig {
        bytecode_read_raf_cycle: BytecodeReadRafMetalConfig {
            trace_cutoff_elements,
            cutoff_elements,
            dispatch: BytecodeCycleSequenceConfig {
                message_threads_per_threadgroup: Some(tuning.message_threads),
                transition_threads_per_threadgroup: Some(tuning.transition_threads),
                max_threadgroups: tuning.max_threadgroups,
            },
            cpu_tail_algebra: BytecodeCycleAlgebra::Q10,
        },
        ..Default::default()
    })?;
    let instance = instance(log_n, seed);

    with_sample_backend_at_log_t(log_n, 8, |witness| -> EvalResult<()> {
        let upload_started = Instant::now();
        let resident_rows =
            backend.prepare_bytecode_read_raf_resident_rows(witness, trace_elements)?;
        let resident_upload = upload_started.elapsed();
        let oracle = run_member(
            Arm::Cpu,
            &backend,
            &resident_rows,
            witness,
            &instance,
            None,
            &observer,
        )?;

        let mut cpu_runs = Vec::with_capacity(repeats);
        let mut metal_runs = Vec::with_capacity(repeats);
        let mut orders = Vec::with_capacity(repeats);
        for repeat in 0..repeats {
            let cpu_first = repeat.is_multiple_of(2);
            orders.push(if cpu_first {
                ["optimized", "metal"]
            } else {
                ["metal", "optimized"]
            });
            let run_cpu = || {
                run_member(
                    Arm::Cpu,
                    &backend,
                    &resident_rows,
                    witness,
                    &instance,
                    Some(&oracle.trace.challenges),
                    &observer,
                )
            };
            let run_metal = || {
                run_member(
                    Arm::Metal,
                    &backend,
                    &resident_rows,
                    witness,
                    &instance,
                    Some(&oracle.trace.challenges),
                    &observer,
                )
            };
            let (cpu, metal) = if cpu_first {
                (run_cpu()?, run_metal()?)
            } else {
                let metal = run_metal()?;
                let cpu = run_cpu()?;
                (cpu, metal)
            };
            cpu_runs.push(cpu);
            metal_runs.push(metal);
        }

        let exact_cpu = cpu_runs.iter().all(|run| run.trace == oracle.trace);
        let exact_metal = metal_runs.iter().all(|run| run.trace == oracle.trace);
        let cpu_phases_absent = cpu_runs
            .iter()
            .all(|run| run.phases == PhaseObservation::default());
        let metal_schedules = metal_runs
            .iter()
            .all(|run| exact_metal_schedule(&run.phases, log_n, tuning.cutoff_log2));
        let cpu_ns = cpu_runs
            .iter()
            .map(|run| ns(run.member_time()))
            .collect::<Vec<_>>();
        let metal_ns = metal_runs
            .iter()
            .map(|run| ns(run.member_time()))
            .collect::<Vec<_>>();
        let paired_speedups = cpu_ns
            .iter()
            .zip(&metal_ns)
            .map(|(cpu, metal)| *cpu as f64 / *metal as f64)
            .collect::<Vec<_>>();
        let median_speedup = median(&paired_speedups);
        let guards = json!({
            "exact_round_polys": exact_cpu && exact_metal,
            "exact_challenges": exact_cpu && exact_metal,
            "exact_final_claim": exact_cpu && exact_metal,
            "exact_opening_points": exact_cpu && exact_metal,
            "exact_output_claims": exact_cpu && exact_metal,
            "exact_transcript_state": exact_cpu && exact_metal,
            "host_fiat_shamir": true,
            "optimized_q10_cpu_control": true,
            "cpu_metal_spans_absent": cpu_phases_absent,
            "metal_backend_exercised": metal_schedules,
            "exact_metal_schedule": metal_schedules,
            "resident_rows_prepared_outside_timer": true,
            "alternating_pair_order": true,
            "all_exact": exact_cpu && exact_metal && cpu_phases_absent && metal_schedules,
        });
        let result = json!({
            "schema_version": 1,
            "kernel": "bytecode_read_raf_cycle",
            "metrics": {
                "hybrid_speedup": median_speedup,
                "paired_speedups": paired_speedups,
                "paired_speedup_mad": mad(&paired_speedups),
                "cpu_member_ms_median": median(&cpu_ns.iter().map(|value| *value as f64 / 1e6).collect::<Vec<_>>()),
                "metal_member_ms_median": median(&metal_ns.iter().map(|value| *value as f64 / 1e6).collect::<Vec<_>>()),
                "cpu_member_ns_samples": cpu_ns,
                "metal_member_ns_samples": metal_ns,
                "cpu_core_ns_samples": cpu_runs.iter().map(|run| ns(run.core_time())).collect::<Vec<_>>(),
                "metal_core_ns_samples": metal_runs.iter().map(|run| ns(run.core_time())).collect::<Vec<_>>(),
                "cpu_prepare_ns_samples": cpu_runs.iter().map(|run| ns(run.prepare)).collect::<Vec<_>>(),
                "metal_prepare_ns_samples": metal_runs.iter().map(|run| ns(run.prepare)).collect::<Vec<_>>(),
                "cpu_host_fs_ns_samples": cpu_runs.iter().map(|run| ns(run.host_fs)).collect::<Vec<_>>(),
                "metal_host_fs_ns_samples": metal_runs.iter().map(|run| ns(run.host_fs)).collect::<Vec<_>>(),
            },
            "guards": guards,
            "phase_samples": metal_runs.iter().map(|run| phase_json(&run.phases)).collect::<Vec<_>>(),
            "resources": {
                "gpu_seconds": metal_runs.iter().map(|run| run.member_time().as_secs_f64()).sum::<f64>(),
                "resident_upload_ns": ns(resident_upload),
                "resident_row_bytes": 40u64 * trace_elements as u64,
            },
            "fingerprint": {
                "log_n": log_n,
                "trace_elements": trace_elements,
                "seed": seed,
                "repeats": repeats,
                "cpu_algebra": "q10",
                "message_threads": tuning.message_threads,
                "transition_threads": tuning.transition_threads,
                "max_threadgroups": tuning.max_threadgroups,
                "cutoff_log2": tuning.cutoff_log2,
                "cutoff_elements": cutoff_elements,
                "trace_cutoff_log2": tuning.trace_cutoff_log2,
                "trace_cutoff_elements": trace_cutoff_elements,
                "orders": orders,
                "fixture": "two-row real TraceBackend with padded cycle domain",
                "initial_claim": "zero synthetic stage input",
            },
        });
        println!("{result}");
        Ok(())
    })
}
