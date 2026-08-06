use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    instruction_claim_reduction::{
        InstructionClaimKernelConfig, InstructionClaimOperandPlanes, InstructionClaimRightInput,
        InstructionClaimRightLookup, InstructionClaimSequence, InstructionClaimTiming,
    },
    SolinasMetal,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const TARGET_ELEMENTS: usize = 1 << 26;
const TARGET_CPU_REFERENCE: Duration = Duration::from_nanos(306_683_705);

struct PreparedWeights {
    messages: Vec<(Vec<AkitaField>, Vec<AkitaField>)>,
    opening_e_in: Vec<AkitaField>,
    opening_e_out: Vec<AkitaField>,
    challenges: Vec<AkitaField>,
}

#[derive(Clone, Copy)]
enum OpeningPath {
    Aliased,
    Standalone,
}

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let config = InstructionClaimKernelConfig {
        materialize_threads_per_threadgroup: setting(
            "JOLT_METAL_INSTRUCTION_CLAIM_MATERIALIZE_THREADS",
            128,
        ),
        transition_threads_per_threadgroup: setting(
            "JOLT_METAL_INSTRUCTION_CLAIM_TRANSITION_THREADS",
            64,
        ),
        opening_threads_per_threadgroup: setting(
            "JOLT_METAL_INSTRUCTION_CLAIM_OPENING_THREADS",
            128,
        ),
    };
    let opening_path = match env::var("JOLT_METAL_INSTRUCTION_CLAIM_OPENING").as_deref() {
        Ok("standalone") => OpeningPath::Standalone,
        Ok("aliased") | Err(_) => OpeningPath::Aliased,
        Ok(value) => panic!(
            "JOLT_METAL_INSTRUCTION_CLAIM_OPENING must be `aliased` or `standalone`, got `{value}`"
        ),
    };
    let opening_name = match opening_path {
        OpeningPath::Aliased => "aliased",
        OpeningPath::Standalone => "standalone",
    };
    let cutoff = setting("JOLT_METAL_INSTRUCTION_CLAIM_CUTOFF", 1 << 16);
    assert!(
        cutoff >= 2 && cutoff.is_power_of_two(),
        "instruction claim cutoff must be a power of two at least two"
    );
    let control_cutoff = env::var("JOLT_METAL_INSTRUCTION_CLAIM_CONTROL_CUTOFF")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("JOLT_METAL_INSTRUCTION_CLAIM_CONTROL_CUTOFF should be a positive integer")
        });
    assert!(
        control_cutoff.is_none_or(|value| value >= 2 && value.is_power_of_two()),
        "instruction claim control cutoff must be a power of two at least two"
    );
    let gamma = AkitaField::from_u64(0x1_0000_01b3);
    let mut group = c.benchmark_group("metal_sumcheck/instruction_claim_reduction");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4));

    for elements in cases() {
        let setup_started = Instant::now();
        let planes = operand_planes(elements);
        let weights = prepared_weights(elements);
        let mut sequence = context
            .prepare_instruction_claim_sequence(&planes, gamma, config)
            .expect("instruction claim sequence should prepare");
        drop(planes);
        let setup_wall = setup_started.elapsed();
        assert_eq!(sequence.round_device_buffer_allocations(), 0);
        let allocations = sequence.allocation_identities();

        let first = run_service(&mut sequence, &weights, opening_path, cutoff);
        let warm = run_service(&mut sequence, &weights, opening_path, cutoff);
        assert_eq!(first.0, warm.0);
        assert_eq!(sequence.allocation_identities(), allocations);
        let ratio = (elements == TARGET_ELEMENTS)
            .then(|| TARGET_CPU_REFERENCE.as_secs_f64() / warm.1.wall.as_secs_f64());
        eprintln!(
            "instruction-claim-reduction n={elements} opening={opening_name} cutoff={cutoff} setup={setup_wall:?} first-wall={:?} first-active={:?} warm-wall={:?} warm-active={:?} resident-bytes={} cpu-reference={:?} resident-speedup={ratio:?}",
            first.1.wall,
            first.1.gpu_active,
            warm.1.wall,
            warm.1.gpu_active,
            sequence.storage_layout().resident_bytes(),
            (elements == TARGET_ELEMENTS).then_some(TARGET_CPU_REFERENCE),
        );
        if let Some(control_cutoff) = control_cutoff.filter(|&control| control != cutoff) {
            let mut candidate_wall = Vec::with_capacity(10);
            let mut candidate_active = Vec::with_capacity(10);
            let mut control_wall = Vec::with_capacity(10);
            let mut control_active = Vec::with_capacity(10);
            for pair in 0usize..10 {
                let mut sample = |sample_cutoff: usize| {
                    run_service(&mut sequence, &weights, opening_path, sample_cutoff).1
                };
                let (candidate, control) = if pair.is_multiple_of(2) {
                    (sample(cutoff), sample(control_cutoff))
                } else {
                    let control = sample(control_cutoff);
                    (sample(cutoff), control)
                };
                candidate_wall.push(candidate.wall);
                candidate_active.push(candidate.gpu_active);
                control_wall.push(control.wall);
                control_active.push(control.gpu_active);
            }
            let candidate_wall_median = median(&candidate_wall);
            let candidate_active_median = median(&candidate_active);
            let control_wall_median = median(&control_wall);
            let control_active_median = median(&control_active);
            eprintln!(
                "instruction-claim-reduction-paired n={elements} candidate-cutoff={cutoff} control-cutoff={control_cutoff} candidate-wall={candidate_wall:?} control-wall={control_wall:?} candidate-active={candidate_active:?} control-active={control_active:?} candidate-wall-median={candidate_wall_median:?} control-wall-median={control_wall_median:?} candidate-active-median={candidate_active_median:?} control-active-median={control_active_median:?} candidate-vs-control={} candidate-vs-cpu={:?}",
                control_wall_median.as_secs_f64() / candidate_wall_median.as_secs_f64(),
                (elements == TARGET_ELEMENTS).then(|| {
                    TARGET_CPU_REFERENCE.as_secs_f64() / candidate_wall_median.as_secs_f64()
                }),
            );
        }

        let suffix = format!(
            "n{elements}_{opening_name}_cutoff{cutoff}_m{}_t{}_o{}",
            config.materialize_threads_per_threadgroup,
            config.transition_threads_per_threadgroup,
            config.opening_threads_per_threadgroup,
        );
        let _ = group.throughput(Throughput::Elements(elements as u64));
        let _ = group.bench_function(
            BenchmarkId::new("resident_service_wall", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let started = Instant::now();
                    for _ in 0..iterations {
                        let _ =
                            black_box(run_service(&mut sequence, &weights, opening_path, cutoff).0);
                    }
                    started.elapsed()
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("resident_service_active", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut active = Duration::ZERO;
                    for _ in 0..iterations {
                        let (_, timing) =
                            run_service(&mut sequence, &weights, opening_path, cutoff);
                        active += timing.gpu_active;
                    }
                    active
                });
            },
        );
    }
    group.finish();
}

fn run_service(
    sequence: &mut InstructionClaimSequence,
    weights: &PreparedWeights,
    opening_path: OpeningPath,
    cutoff: usize,
) -> ((AkitaField, Vec<AkitaField>), InstructionClaimTiming) {
    sequence.reset();
    let started = Instant::now();
    let _ = sequence
        .message(&weights.messages[0].0, &weights.messages[0].1)
        .expect("instruction claim first message should execute");
    let mut round = 1;
    while round < weights.messages.len() && sequence.current_elements() > cutoff {
        let _ = sequence
            .bind_and_message(
                weights.challenges[round - 1],
                &weights.messages[round].0,
                &weights.messages[round].1,
            )
            .expect("instruction claim transition should execute");
        round += 1;
    }
    let claim = if round < weights.messages.len() {
        let mut tail = sequence
            .handoff_to_cpu()
            .expect("instruction claim state should hand off to the CPU tail");
        assert_eq!(tail.round_device_buffer_allocations(), 0);
        while round < weights.messages.len() {
            let _ = tail
                .bind_and_message(
                    weights.challenges[round - 1],
                    &weights.messages[round].0,
                    &weights.messages[round].1,
                )
                .expect("instruction claim CPU tail should execute");
            round += 1;
        }
        sequence
            .finish_cpu_tail(tail, weights.challenges[weights.challenges.len() - 1])
            .expect("instruction claim CPU tail should finish")
    } else {
        sequence
            .finish(weights.challenges[weights.challenges.len() - 1])
            .expect("instruction claim final pair should bind")
    };
    let openings = match opening_path {
        OpeningPath::Aliased => sequence
            .aliased_openings(&weights.opening_e_in, &weights.opening_e_out)
            .expect("instruction claim aliased opening should execute")
            .to_vec(),
        OpeningPath::Standalone => sequence
            .openings(&weights.opening_e_in, &weights.opening_e_out)
            .expect("instruction claim standalone opening should execute")
            .into_array()
            .to_vec(),
    };
    let mut timing = sequence.timing();
    timing.wall = started.elapsed();
    ((claim, openings), timing)
}

fn prepared_weights(elements: usize) -> PreparedWeights {
    let log_t = elements.trailing_zeros() as usize;
    let point = (0..log_t)
        .map(|index| AkitaField::from_u64(0x101 + 2 * index as u64))
        .collect::<Vec<_>>();
    let challenges = (0..log_t)
        .map(|index| AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1)))
        .collect::<Vec<_>>();
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let mut messages = Vec::with_capacity(log_t);
    messages.push((
        gruen.e_in_current().to_vec(),
        gruen.e_out_current().to_vec(),
    ));
    for &challenge in challenges.iter().take(log_t - 1) {
        gruen.bind(challenge);
        messages.push((
            gruen.e_in_current().to_vec(),
            gruen.e_out_current().to_vec(),
        ));
    }
    let reversed = challenges.iter().rev().copied().collect::<Vec<_>>();
    let (r_hi, r_lo) = reversed.split_at(reversed.len() / 2);
    PreparedWeights {
        messages,
        opening_e_in: EqPolynomial::evals(r_lo, None),
        opening_e_out: EqPolynomial::evals(r_hi, None),
        challenges,
    }
}

fn operand_planes(elements: usize) -> InstructionClaimOperandPlanes {
    let lookup_output = (0..elements)
        .into_par_iter()
        .map(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let left_lookup_operand = (0..elements)
        .into_par_iter()
        .map(|index| (index as u64).rotate_left(17) ^ u64::MAX)
        .collect();
    let right_lookup_operand = (0..elements)
        .into_par_iter()
        .map(|index| {
            InstructionClaimRightLookup::new(
                ((index as u128) << 79) | (index as u128).wrapping_mul(0x1_0000_0000_0000_01b3),
            )
        })
        .collect();
    let left_instruction_input = (0..elements)
        .into_par_iter()
        .map(|index| (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93))
        .collect();
    let right_instruction_input = (0..elements)
        .into_par_iter()
        .map(|index| {
            let magnitude = ((index as i128) << 61) | (index as i128 + 1);
            InstructionClaimRightInput::new(if index.is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            })
        })
        .collect();
    InstructionClaimOperandPlanes::new(
        lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        left_instruction_input,
        right_instruction_input,
    )
    .expect("instruction claim operand planes should have a valid shape")
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}

fn median(values: &[Duration]) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn cases() -> Vec<usize> {
    let cases = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    );
    assert!(
        cases
            .iter()
            .all(|elements| elements.is_power_of_two() && *elements >= 1 << 7),
        "instruction claim benchmark sizes must be powers of two at least 2^7"
    );
    cases
}
