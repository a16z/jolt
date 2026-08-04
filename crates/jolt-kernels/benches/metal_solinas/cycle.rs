use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
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

type CycleTranscript = Blake2bTranscript<AkitaField>;

#[derive(Debug, Eq, PartialEq)]
struct ResultState {
    final_tables: [AkitaField; PRODUCT5_FACTORS],
    final_claim: AkitaField,
    transcript_state: [u8; 32],
}

struct CpuTables {
    a: Vec<AkitaField>,
    b: Vec<AkitaField>,
    initial_elements: usize,
    elements: usize,
    source_in_a: bool,
}

impl CpuTables {
    fn new(max_elements: usize) -> Self {
        Self {
            a: vec![AkitaField::zero(); PRODUCT5_FACTORS * max_elements],
            b: vec![AkitaField::zero(); PRODUCT5_FACTORS * max_elements / 2],
            initial_elements: max_elements,
            elements: max_elements,
            source_in_a: true,
        }
    }

    fn reset(&mut self, initial: &[AkitaField]) {
        self.a.copy_from_slice(initial);
        self.elements = self.initial_elements;
        self.source_in_a = true;
    }

    fn load(&mut self, sequence: &Product5Sequence) {
        self.elements = sequence.current_elements();
        self.source_in_a = true;
        let length = PRODUCT5_FACTORS * self.elements;
        sequence
            .read_current_tables(&mut self.a[..length])
            .expect("resident tables should read back");
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

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let mut group = c.benchmark_group("metal_sumcheck/instruction_read_raf_cycle");
    let _ = group
        .sample_size(12)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let cutoff = cutoff(elements);
        let initial = values(PRODUCT5_FACTORS * elements, 1);
        let point = values(elements.trailing_zeros() as usize, 0x6a09_e667_f3bc_c909);
        let initial_claim = AkitaField::from_u64(0xa54f_f53a_5f1d_36f1);
        let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let mut sequence = context
            .prepare_product5_sequence(
                &initial,
                elements,
                gruen.e_in_current(),
                gruen.e_out_current(),
                dispatch(),
            )
            .expect("resident sequence should prepare");
        let mut cpu = CpuTables::new(elements);
        let mut tail = CpuTables::new(cutoff);

        cpu.reset(&initial);
        let expected = run_cpu(&mut cpu, &point, initial_claim);
        sequence.reset(&initial).expect("sequence should reset");
        let actual = run_hybrid(&mut sequence, &mut tail, &point, initial_claim, cutoff);
        assert_eq!(actual, expected);

        let _ = group.throughput(Throughput::Elements(27 * (elements as u64 - 1)));
        let suffix = format!("n{elements}_cutoff{cutoff}");
        let _ = group.bench_function(BenchmarkId::new("cpu_optimized_resident", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    cpu.reset(&initial);
                    let started = Instant::now();
                    let _ = black_box(run_cpu(&mut cpu, &point, initial_claim));
                    elapsed += started.elapsed();
                }
                elapsed
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_direct_handoff", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    sequence.reset(&initial).expect("sequence should reset");
                    let started = Instant::now();
                    let _ = black_box(run_hybrid(
                        &mut sequence,
                        &mut tail,
                        &point,
                        initial_claim,
                        cutoff,
                    ));
                    elapsed += started.elapsed();
                }
                elapsed
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_copied_handoff", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let started = Instant::now();
                    sequence.reset(&initial).expect("sequence should reset");
                    let _ = black_box(run_hybrid(
                        &mut sequence,
                        &mut tail,
                        &point,
                        initial_claim,
                        cutoff,
                    ));
                    elapsed += started.elapsed();
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn run_cpu(tables: &mut CpuTables, point: &[AkitaField], initial_claim: AkitaField) -> ResultState {
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = CycleTranscript::new(b"metal-cycle-criterion");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    while tables.elements > 1 {
        let q_evals = dense_message(tables.source(), tables.elements, &gruen);
        let poly = gruen.gruen_poly_from_evals(&q_evals, claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        gruen.bind(challenge);
        tables.bind(challenge);
    }
    ResultState {
        final_tables: tables.source().try_into().expect("five final factors"),
        final_claim: claim,
        transcript_state: transcript.state(),
    }
}

fn run_hybrid(
    sequence: &mut Product5Sequence,
    tail: &mut CpuTables,
    point: &[AkitaField],
    initial_claim: AkitaField,
    cutoff: usize,
) -> ResultState {
    let mut gruen = GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh);
    let mut transcript = CycleTranscript::new(b"metal-cycle-criterion");
    append_sumcheck_claim(&mut transcript, &initial_claim);
    let mut claim = initial_claim;
    let mut q_evals = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .expect("initial Metal message should execute");

    loop {
        let poly = gruen.gruen_poly_from_evals(&q_evals, claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        if sequence.current_elements() <= cutoff || sequence.current_elements() == 2 {
            tail.load(sequence);
            gruen.bind(challenge);
            tail.bind(challenge);
            break;
        }
        gruen.bind(challenge);
        q_evals = sequence
            .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
            .expect("Metal transition should execute");
    }

    while tail.elements > 1 {
        let q_evals = dense_message(tail.source(), tail.elements, &gruen);
        let poly = gruen.gruen_poly_from_evals(&q_evals, claim);
        let challenge = absorb_round(&mut transcript, &poly);
        claim = poly.evaluate(challenge);
        gruen.bind(challenge);
        tail.bind(challenge);
    }
    ResultState {
        final_tables: tail.source().try_into().expect("five final factors"),
        final_claim: claim,
        transcript_state: transcript.state(),
    }
}

fn absorb_round(transcript: &mut CycleTranscript, poly: &UnivariatePoly<AkitaField>) -> AkitaField {
    CompressedLabeledRoundPoly::sumcheck(poly).append_to_transcript(transcript);
    transcript.challenge()
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

fn cases() -> Vec<usize> {
    env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| vec![1 << 16, 1 << 20, 1 << 22],
        |value| {
            let elements = value
                .parse::<usize>()
                .expect("benchmark element count should be an integer");
            assert!(elements >= 16 && elements.is_power_of_two());
            vec![elements]
        },
    )
}

fn cutoff(elements: usize) -> usize {
    let log_n = elements.trailing_zeros() as usize;
    let default_log = 16.min(log_n - 4);
    let log_cutoff = env::var("JOLT_METAL_CUTOFF_LOG2").map_or(default_log, |value| {
        value
            .parse::<usize>()
            .expect("Metal cutoff should be an integer")
    });
    assert!(log_cutoff < log_n);
    1usize << log_cutoff
}

fn dispatch() -> Product5SequenceConfig {
    let parse = |name: &str, default| {
        env::var(name).map_or(default, |value| {
            value
                .parse::<usize>()
                .expect("threadgroup width should be an integer")
        })
    };
    Product5SequenceConfig {
        message_threads_per_threadgroup: Some(parse("JOLT_METAL_MESSAGE_THREADS", 128)),
        transition_threads_per_threadgroup: Some(parse("JOLT_METAL_TRANSITION_THREADS", 64)),
    }
}
