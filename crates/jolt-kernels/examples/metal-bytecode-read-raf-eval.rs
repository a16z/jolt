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
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
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
    BytecodeReadRafCycle, BytecodeReadRafCycleInputs, BytecodeReadRafCycleOutputClaims,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
    BytecodeReadRafTableFoldInputs,
};
use jolt_witness::testing::with_diverse_sample_backend_at_geometry;
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

    fn member_with_host_fs_time(&self) -> Duration {
        self.member_time() + self.host_fs
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

fn instance(
    log_t: usize,
    seed: u64,
    witness: &dyn JoltWitnessPlane<AkitaField>,
) -> EvalResult<Instance> {
    let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
    let r_address = point(13, seed ^ 0x243f_6a88_85a3_08d3);
    let stage_cycle_points =
        std::array::from_fn(|stage| point(log_t, seed ^ (0x1319_8a2e_0370_7344 + stage as u64)));
    let register_read_write_point = point(REGISTER_ADDRESS_BITS, seed ^ 0x4528_21e6_38d0_1377);
    let register_val_evaluation_point = point(REGISTER_ADDRESS_BITS, seed ^ 0xbe54_66cf_34e9_0c6c);
    let address_challenges = BytecodeReadRafAddressPhaseChallenges {
        gamma: field(seed ^ 0xc0ac_29b7_c97c_50dd, 0),
        stage1_gamma: field(seed ^ 0x3f84_d5b5_b547_0917, 0),
        stage2_gamma: field(seed ^ 0x9216_d5d9_8979_fb1b, 0),
        stage3_gamma: field(seed ^ 0xd131_0ba6_98df_b5ac, 0),
        stage4_gamma: field(seed ^ 0x2ffd_72db_d01a_dfb7, 0),
        stage5_gamma: field(seed ^ 0xb8e1_afed_6a26_7e96, 0),
    };
    let stage_gammas = address_challenges.stage_gamma_powers();
    let cycle_gamma = address_challenges.gamma;
    let program = witness.program_preprocessing();
    if program.bytecode.bytecode.len() != 1 << 13 {
        return Err("Bytecode evaluator fixture has the wrong program domain".into());
    }
    let entry_bytecode_index = program
        .bytecode
        .entry_bytecode_index()
        .ok_or("Bytecode evaluator fixture has no entry bytecode index")?;
    let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
        dimensions,
        r_address,
        stage_cycle_points,
        entry_bytecode_index,
        committed_chunk_bits: 8,
        table_fold: Some(BytecodeReadRafTableFoldInputs {
            bytecode: &program.bytecode.bytecode,
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            stage_gammas: std::array::from_fn(|stage| stage_gammas[stage].as_slice()),
        }),
    })?;
    Ok(Instance {
        relation,
        claims: BytecodeReadRafInputClaims::default(),
        points: BytecodeReadRafInputClaims::default(),
        challenges: BytecodeReadRafCyclePhaseCommittedChallenges { gamma: cycle_gamma },
    })
}

fn transcript(initial_claim: AkitaField) -> EvalTranscript {
    let mut transcript = EvalTranscript::new(b"metal-bytecode-read-raf-eval");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    transcript
}

fn run_member(
    arm: Arm,
    backend: &MetalBackend,
    resident_rows: Option<&BytecodeReadRafResidentRows>,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    instance: &Instance,
    challenge_tape: Option<&[AkitaField]>,
    observer: &PhaseObserver,
) -> EvalResult<MemberRun> {
    observer.clear();
    let mut session = ProofSession::default();
    if matches!(arm, Arm::Metal) {
        resident_rows
            .ok_or("Metal Bytecode evaluator arm has no resident rows")?
            .install(&mut session);
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

    let host_started = Instant::now();
    let mut transcript = transcript(instance.claims.address_phase);
    let mut host_fs = host_started.elapsed();
    let mut claim = instance.claims.address_phase;
    let mut round_polys = Vec::with_capacity(kernel.num_rounds());
    let mut challenges = Vec::with_capacity(kernel.num_rounds());
    let mut rounds = Vec::with_capacity(kernel.num_rounds());
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
    let output_points = instance
        .relation
        .derive_opening_points(&challenges, &instance.points)?;
    let output_started = Instant::now();
    let output_claims = kernel.output_claims(&instance.claims)?;
    let output_claims_time = output_started.elapsed();
    let expected_output = instance.relation.expected_output(
        &instance.points,
        &output_claims,
        &output_points,
        &instance.challenges,
    )?;
    if expected_output != claim {
        return Err("Bytecode evaluator final claim disagrees with expected output".into());
    }
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
    with_diverse_sample_backend_at_geometry(log_n, 13, 8, |witness| -> EvalResult<()> {
        let mut instance = instance(log_n, seed, witness)?;
        let input_claim_started = Instant::now();
        let input_claim = MetalBackend::bytecode_read_raf_input_claim(
            witness,
            &instance.relation,
            &instance.challenges,
        )?;
        let input_claim_time = input_claim_started.elapsed();
        if input_claim == AkitaField::zero() {
            return Err("Bytecode evaluator honest input claim is zero".into());
        }
        instance.claims.address_phase = input_claim;
        let oracle = run_member(
            Arm::Cpu,
            &backend,
            None,
            witness,
            &instance,
            None,
            &observer,
        )?;
        let mut cpu_controls = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            cpu_controls.push(run_member(
                Arm::Cpu,
                &backend,
                None,
                witness,
                &instance,
                Some(&oracle.trace.challenges),
                &observer,
            )?);
        }

        let upload_started = Instant::now();
        let resident_rows =
            backend.prepare_bytecode_read_raf_resident_rows(witness, trace_elements)?;
        let resident_upload = upload_started.elapsed();

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
                    None,
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
                    Some(&resident_rows),
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

        let exact_cpu_control = cpu_controls.iter().all(|run| run.trace == oracle.trace);
        let exact_cpu = cpu_runs.iter().all(|run| run.trace == oracle.trace);
        let exact_metal = metal_runs.iter().all(|run| run.trace == oracle.trace);
        let cpu_phases_absent = cpu_controls
            .iter()
            .chain(&cpu_runs)
            .all(|run| run.phases == PhaseObservation::default());
        let metal_schedules = metal_runs
            .iter()
            .all(|run| exact_metal_schedule(&run.phases, log_n, tuning.cutoff_log2));
        let cpu_ns = cpu_runs
            .iter()
            .map(|run| ns(run.member_with_host_fs_time()))
            .collect::<Vec<_>>();
        let metal_ns = metal_runs
            .iter()
            .map(|run| ns(run.member_with_host_fs_time()))
            .collect::<Vec<_>>();
        let cpu_control_ns = cpu_controls
            .iter()
            .map(|run| ns(run.member_with_host_fs_time()))
            .collect::<Vec<_>>();
        let kernel_only_cpu_ns = cpu_runs
            .iter()
            .map(|run| ns(run.member_time()))
            .collect::<Vec<_>>();
        let kernel_only_metal_ns = metal_runs
            .iter()
            .map(|run| ns(run.member_time()))
            .collect::<Vec<_>>();
        let paired_speedups = cpu_ns
            .iter()
            .zip(&metal_ns)
            .map(|(cpu, metal)| *cpu as f64 / *metal as f64)
            .collect::<Vec<_>>();
        let kernel_only_speedups = kernel_only_cpu_ns
            .iter()
            .zip(&kernel_only_metal_ns)
            .map(|(cpu, metal)| *cpu as f64 / *metal as f64)
            .collect::<Vec<_>>();
        let median_speedup = median(&paired_speedups);
        let cpu_control_median = median(
            &cpu_control_ns
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>(),
        );
        let paired_cpu_median =
            median(&cpu_ns.iter().map(|value| *value as f64).collect::<Vec<_>>());
        let cpu_denominator_ratio =
            cpu_control_median.max(paired_cpu_median) / cpu_control_median.min(paired_cpu_median);
        let cpu_denominator_stable = cpu_denominator_ratio <= 1.10;
        let guards = json!({
            "honest_input_claim_nonzero": true,
            "expected_output_matches_final_claim": true,
            "full_program_relation": true,
            "diverse_pc_and_fused_inc_fixture": true,
            "exact_round_polys": exact_cpu_control && exact_cpu && exact_metal,
            "exact_challenges": exact_cpu_control && exact_cpu && exact_metal,
            "exact_final_claim": exact_cpu_control && exact_cpu && exact_metal,
            "exact_opening_points": exact_cpu_control && exact_cpu && exact_metal,
            "exact_output_claims": exact_cpu_control && exact_cpu && exact_metal,
            "exact_transcript_state": exact_cpu_control && exact_cpu && exact_metal,
            "host_fiat_shamir": true,
            "optimized_q10_cpu_control": true,
            "cpu_no_resident_control_exact": exact_cpu_control,
            "cpu_denominator_stable": cpu_denominator_stable,
            "cpu_metal_spans_absent": cpu_phases_absent,
            "metal_backend_exercised": metal_schedules,
            "exact_metal_schedule": metal_schedules,
            "resident_rows_prepared_outside_timer": true,
            "alternating_pair_order": true,
            "all_exact": exact_cpu_control && exact_cpu && exact_metal && cpu_phases_absent && metal_schedules,
        });
        let result = json!({
            "schema_version": 1,
            "kernel": "bytecode_read_raf_cycle",
            "metrics": {
                "hybrid_speedup": median_speedup,
                "paired_speedups": paired_speedups,
                "paired_speedup_mad": mad(&paired_speedups),
                "kernel_only_hybrid_speedup": median(&kernel_only_speedups),
                "kernel_only_paired_speedups": kernel_only_speedups,
                "cpu_member_ms_median": median(&cpu_ns.iter().map(|value| *value as f64 / 1e6).collect::<Vec<_>>()),
                "metal_member_ms_median": median(&metal_ns.iter().map(|value| *value as f64 / 1e6).collect::<Vec<_>>()),
                "cpu_member_ns_samples": cpu_ns,
                "metal_member_ns_samples": metal_ns,
                "cpu_no_resident_member_ns_samples": cpu_control_ns,
                "cpu_denominator_ratio": cpu_denominator_ratio,
                "cpu_core_ns_samples": cpu_runs.iter().map(|run| ns(run.core_time())).collect::<Vec<_>>(),
                "metal_core_ns_samples": metal_runs.iter().map(|run| ns(run.core_time())).collect::<Vec<_>>(),
                "cpu_round_ns_samples": cpu_runs.iter().map(|run| run.rounds.iter().map(|round| ns(*round)).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "metal_round_ns_samples": metal_runs.iter().map(|run| run.rounds.iter().map(|round| ns(*round)).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "cpu_prepare_ns_samples": cpu_runs.iter().map(|run| ns(run.prepare)).collect::<Vec<_>>(),
                "metal_prepare_ns_samples": metal_runs.iter().map(|run| ns(run.prepare)).collect::<Vec<_>>(),
                "cpu_host_fs_ns_samples": cpu_runs.iter().map(|run| ns(run.host_fs)).collect::<Vec<_>>(),
                "metal_host_fs_ns_samples": metal_runs.iter().map(|run| ns(run.host_fs)).collect::<Vec<_>>(),
            },
            "guards": guards,
            "phase_samples": metal_runs.iter().map(|run| phase_json(&run.phases)).collect::<Vec<_>>(),
            "resources": {
                "gpu_seconds": metal_runs.iter().map(|run| run.member_with_host_fs_time().as_secs_f64()).sum::<f64>(),
                "metal_hybrid_wall_seconds": metal_runs.iter().map(|run| run.member_with_host_fs_time().as_secs_f64()).sum::<f64>(),
                "input_claim_precompute_ns": ns(input_claim_time),
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
                "entry_bytecode_index": instance.relation.entry_bytecode_index(),
                "fixture": "address-diverse TraceBackend in a full 8192-row program and padded cycle domain",
                "fixture_trace_rows": trace_elements.min((1usize << 13) - 1),
                "fixture_program_rows": 1usize << 13,
                "covers_high_ra_chunk": true,
                "fused_inc_fixture": "mixed rd and RAM signed deltas",
                "relation_variant": "full-program",
                "initial_claim": "independent direct cycle-domain sum",
                "primary_metric_includes_host_fs": true,
            },
        });
        println!("{result}");
        Ok(())
    })
}
