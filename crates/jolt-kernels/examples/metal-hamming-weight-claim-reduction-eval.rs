#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{
    env,
    error::Error,
    hint::black_box,
    mem::{self, size_of},
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::lattice::{geometry::balanced_inc_value, UnsignedIncChunking};
use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, RingAccumulator, RingCore,
};
use jolt_kernels::metal::solinas::{
    BooleanityAddressPushforwardConfig, BooleanityRow, BooleanitySelector, MetalError,
    PipelineLimits, SolinasMetal,
};
use jolt_poly::{
    boolean_point_msb, BindingOrder, EqPolynomial, Polynomial, TensorEqTable, UnivariatePoly,
};
use jolt_sumcheck::{append_sumcheck_claim, CompressedLabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

const K: usize = 256;
const CHUNK_BITS: usize = 8;
const POLYS: usize = 29;
const RA_POLYS: usize = 20;
const INC_CHUNKS: usize = 8;
const ROW_BYTES: usize = 40;
const ACCUMULATOR_WORDS: usize = 5;
const SIMD_WIDTH: usize = 32;
const MIN_PROMOTION_LOG_N: usize = 26;
const MIN_PROMOTION_PAIRS: usize = 5;
const MIN_PROMOTION_SPEEDUP: f64 = 4.0;

#[derive(Clone, Debug, Eq, PartialEq)]
struct HammingTrace {
    q_evals: Vec<[AkitaField; 2]>,
    messages: Vec<UnivariatePoly<AkitaField>>,
    challenges: Vec<AkitaField>,
    final_claim: AkitaField,
    final_relation: AkitaField,
    output_claims: Vec<AkitaField>,
    transcript_state: [u8; 32],
}

struct Sample {
    masses: Vec<AkitaField>,
    trace: HammingTrace,
    wall: Duration,
    prepare: Duration,
    host_rounds: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MetalShape {
    resident_row_identity: usize,
    rows: usize,
    polys: usize,
    selectors_per_tile: usize,
    selector_tiles: usize,
    e_in_elements: usize,
    e_out_elements: usize,
    output_elements: usize,
    partial_bytes: u64,
    tile_threads: usize,
    finalize_threads: usize,
    production_specialized: bool,
    tile_limits: PipelineLimits,
    finalize_limits: PipelineLimits,
}

struct MetalSample {
    member: Sample,
    gpu_active: Duration,
    dispatch_wall: Duration,
    readback: Duration,
    shape: MetalShape,
    static_buffer_identities: [usize; 5],
    static_buffers_stable: bool,
    static_buffers_distinct: bool,
}

struct CpuState {
    partial: Vec<AkitaAccumulator>,
    block: Vec<AkitaAccumulator>,
}

#[derive(Clone, Copy)]
struct HammingInputs<'a> {
    reference_cycle: &'a [AkitaField],
    gamma: AkitaField,
    reference_address: &'a [AkitaField],
    virtualization_points: &'a [Vec<AkitaField>],
    ram_hamming_weight: AkitaField,
}

fn selectors() -> Vec<BooleanitySelector> {
    let mut selectors = (0..16)
        .map(|index| BooleanitySelector::Lookup {
            shift: (CHUNK_BITS * (15 - index)) as u32,
        })
        .collect::<Vec<_>>();
    selectors.extend([8, 0].map(|shift| BooleanitySelector::Bytecode { shift }));
    selectors.extend([8, 0].map(|shift| BooleanitySelector::Ram { shift }));
    selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc {
        shift: (CHUNK_BITS * index) as u32,
    }));
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn production_selector_schedule_is_exact(selectors: &[BooleanitySelector]) -> bool {
    selectors.len() == POLYS
        && selectors[..16]
            .iter()
            .copied()
            .enumerate()
            .all(|(index, selector)| {
                selector
                    == BooleanitySelector::Lookup {
                        shift: (CHUNK_BITS * (15 - index)) as u32,
                    }
            })
        && selectors[16..18]
            == [
                BooleanitySelector::Bytecode { shift: 8 },
                BooleanitySelector::Bytecode { shift: 0 },
            ]
        && selectors[18..20]
            == [
                BooleanitySelector::Ram { shift: 8 },
                BooleanitySelector::Ram { shift: 0 },
            ]
        && selectors[20..28]
            .iter()
            .copied()
            .enumerate()
            .all(|(index, selector)| {
                selector
                    == BooleanitySelector::FusedInc {
                        shift: (CHUNK_BITS * index) as u32,
                    }
            })
        && selectors[28] == BooleanitySelector::FusedIncMsb
}

fn rows(count: usize, seed: u64) -> Result<Vec<BooleanityRow>, MetalError> {
    (0..count)
        .into_par_iter()
        .map(|row| {
            let mut state = u128::from(splitmix(seed ^ row as u64))
                | (u128::from(splitmix(!seed ^ row.rotate_left(17) as u64)) << 64);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            let mapped_pc =
                (!row.is_multiple_of(7)).then_some(((state >> 61) as u64) & ((1 << 55) - 2));
            let ram_address = (!row.is_multiple_of(11)).then_some((state as u64) & (u64::MAX - 1));
            let fused_inc = match row % 6 {
                0 => -(u64::MAX as i128),
                1 => -((1i128 << 63) + row as i128),
                2 => u64::MAX as i128 - row as i128,
                3 => (1i128 << 63) + row as i128,
                4 => row as i128,
                _ => -(row as i128),
            };
            BooleanityRow::new(state, mapped_pc, ram_address, fused_inc)
        })
        .collect()
}

fn point(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| {
            let low = splitmix(seed ^ index as u64);
            let high = splitmix(!seed ^ (index as u64).rotate_left(23)) & 0x7fff_ffff_ffff_ffff;
            AkitaField::from_u128(u128::from(low) | (u128::from(high) << 64))
        })
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hot_index(row: BooleanityRow, selector: BooleanitySelector) -> Option<usize> {
    let words = row.words();
    let mask = (K - 1) as u64;
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let word = if shift < 64 { words[0] } else { words[1] };
            let shift = if shift < 64 { shift } else { shift - 64 };
            Some(((word >> shift) & mask) as usize)
        }
        BooleanitySelector::Bytecode { shift } => {
            let plus_one = words[4] & 0x00ff_ffff_ffff_ffff;
            (plus_one != 0).then(|| (((plus_one - 1) >> shift) & mask) as usize)
        }
        BooleanitySelector::Ram { shift } => {
            (words[2] != 0).then(|| (((words[2] - 1) >> shift) & mask) as usize)
        }
        BooleanitySelector::FusedInc { shift } => {
            let (biased, _) = biased_inc(words);
            let standard = (biased >> shift) & mask;
            Some(((standard + (K / 2) as u64) & mask) as usize)
        }
        BooleanitySelector::FusedIncMsb => {
            let (_, carry) = biased_inc(words);
            Some((carry as usize) & (K - 1))
        }
    }
}

fn biased_inc(words: [u64; 5]) -> (u64, i32) {
    let radix = 1u128 << CHUNK_BITS;
    let bias = ((radix / 2) * (u128::from(u64::MAX) / (radix - 1))) as u64;
    let magnitude = words[3];
    if words[4] >> 63 != 0 {
        (
            bias.wrapping_sub(magnitude),
            if magnitude > bias { -1 } else { 0 },
        )
    } else {
        let biased = bias.wrapping_add(magnitude);
        (biased, i32::from(biased < bias))
    }
}

fn cpu_pushforward(
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
    reference_cycle: &[AkitaField],
) -> Vec<AkitaField> {
    let eq = TensorEqTable::new(reference_cycle);
    let e_out = eq.e_out();
    let e_in = eq.e_in();
    let fields = selectors.len() * K;
    let zero = || CpuState {
        partial: vec![AkitaAccumulator::default(); fields],
        block: vec![AkitaAccumulator::default(); fields],
    };
    let scatter = |mut state: CpuState, x_out: usize| {
        let base = x_out * e_in.len();
        for (x_in, weight) in e_in.iter().copied().enumerate() {
            let row = rows[base + x_in];
            for (selector_index, selector) in selectors.iter().copied().enumerate() {
                if let Some(hot) = hot_index(row, selector) {
                    state.block[selector_index * K + hot].add(weight);
                }
            }
        }
        let outer = e_out[x_out];
        for (partial, block) in state.partial.iter_mut().zip(&mut state.block) {
            let value = mem::take(block).reduce();
            if value != AkitaField::zero() {
                partial.fmadd(outer, value);
            }
        }
        state
    };
    let merge = |mut left: CpuState, right: CpuState| {
        for (left, right) in left.partial.iter_mut().zip(right.partial) {
            left.merge(right);
        }
        left
    };
    (0..e_out.len())
        .into_par_iter()
        .fold(zero, scatter)
        .reduce(zero, merge)
        .partial
        .into_iter()
        .map(|accumulator| accumulator.reduce())
        .collect()
}

fn hamming_trace(masses: &[AkitaField], inputs: HammingInputs<'_>) -> HammingTrace {
    assert_eq!(masses.len(), POLYS * K);
    assert!(masses
        .chunks_exact(K)
        .all(|table| table[0] == AkitaField::zero()));
    assert_eq!(inputs.virtualization_points.len(), RA_POLYS);
    let mut g_tables = masses
        .chunks_exact(K)
        .map(|table| Polynomial::new(table.to_vec()))
        .collect::<Vec<_>>();
    let Ok(chunking) = UnsignedIncChunking::new(CHUNK_BITS) else {
        unreachable!("the evaluator's fixed chunk width is valid")
    };
    assert_eq!(chunking.chunk_count(), INC_CHUNKS);
    let ra_terms = 3 * RA_POLYS;
    let decode_power = ra_terms + 2 * (INC_CHUNKS + 1);
    let mut gamma_powers = vec![AkitaField::one(); decode_power + 1];
    for index in 1..gamma_powers.len() {
        gamma_powers[index] = gamma_powers[index - 1] * inputs.gamma;
    }
    let eq_bool = EqPolynomial::<AkitaField>::evals(inputs.reference_address, None);
    let at_default = |point: &[AkitaField]| {
        point.iter().fold(AkitaField::one(), |acc, coordinate| {
            acc * (AkitaField::one() - *coordinate)
        })
    };
    let eq_bool_default = at_default(inputs.reference_address);
    let mut baseline = AkitaField::zero();
    let mut weight_tables = Vec::with_capacity(POLYS);
    for (index, point) in inputs.virtualization_points.iter().enumerate() {
        assert_eq!(point.len(), CHUNK_BITS);
        let eq_virt = EqPolynomial::<AkitaField>::evals(point, None);
        let eq_virt_default = at_default(point);
        let hamming_weight = if index < 18 {
            AkitaField::one()
        } else {
            inputs.ram_hamming_weight
        };
        baseline += hamming_weight
            * (gamma_powers[3 * index]
                + gamma_powers[3 * index + 1] * eq_bool_default
                + gamma_powers[3 * index + 2] * eq_virt_default);
        weight_tables.push(Polynomial::new(
            (0..K)
                .map(|lane| {
                    gamma_powers[3 * index + 1] * (eq_bool[lane] - eq_bool_default)
                        + gamma_powers[3 * index + 2] * (eq_virt[lane] - eq_virt_default)
                })
                .collect(),
        ));
    }
    let balanced_values = (0..K)
        .map(|lane| balanced_inc_value(&boolean_point_msb::<AkitaField>(CHUNK_BITS, lane)))
        .collect::<Vec<AkitaField>>();
    for index in 0..INC_CHUNKS {
        let offset = ra_terms + 2 * index;
        baseline += gamma_powers[offset] + gamma_powers[offset + 1] * eq_bool_default;
        let decode_scale = gamma_powers[decode_power] * chunking.place_value::<AkitaField>(index);
        weight_tables.push(Polynomial::new(
            (0..K)
                .map(|lane| {
                    gamma_powers[offset + 1] * (eq_bool[lane] - eq_bool_default)
                        + decode_scale * balanced_values[lane]
                })
                .collect(),
        ));
    }
    let msb_offset = ra_terms + 2 * INC_CHUNKS;
    baseline += gamma_powers[msb_offset] + gamma_powers[msb_offset + 1] * eq_bool_default;
    let decode_scale = gamma_powers[decode_power] * AkitaField::pow2(64);
    weight_tables.push(Polynomial::new(
        (0..K)
            .map(|lane| {
                gamma_powers[msb_offset + 1] * (eq_bool[lane] - eq_bool_default)
                    + decode_scale * balanced_values[lane]
            })
            .collect(),
    ));
    assert_eq!(weight_tables.len(), POLYS);
    let mut baseline_table = vec![AkitaField::zero(); K];
    baseline_table[0] = baseline;
    let mut baseline_table = Polynomial::new(baseline_table);

    let mut transcript = EvalTranscript::new(b"metal-hamming-weight-eval");
    let mut claim = g_tables
        .iter()
        .zip(&weight_tables)
        .map(|(g, weight)| {
            g.evals()
                .iter()
                .zip(weight.evals())
                .map(|(g, weight)| *g * *weight)
                .sum::<AkitaField>()
        })
        .sum::<AkitaField>()
        + baseline;
    append_sumcheck_claim(&mut transcript, &claim);
    let mut q_evals = Vec::with_capacity(CHUNK_BITS);
    let mut messages = Vec::with_capacity(CHUNK_BITS);
    let mut challenges = Vec::with_capacity(CHUNK_BITS);

    let bind = |challenge: AkitaField,
                g_tables: &mut [Polynomial<AkitaField>],
                weight_tables: &mut [Polynomial<AkitaField>],
                baseline_table: &mut Polynomial<AkitaField>| {
        for table in g_tables.iter_mut().chain(weight_tables) {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        baseline_table.bind_with_order(challenge, BindingOrder::LowToHigh);
    };

    for _ in 0..CHUNK_BITS {
        if let Some(challenge) = challenges.last().copied() {
            bind(
                challenge,
                &mut g_tables,
                &mut weight_tables,
                &mut baseline_table,
            );
        }
        let half = weight_tables[0].len() / 2;
        let evals = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut out = [AkitaField::zero(); 2];
                for (g, weight) in g_tables.iter().zip(&weight_tables) {
                    let g_lo = g.evals()[2 * index];
                    let g_hi = g.evals()[2 * index + 1];
                    let weight_lo = weight.evals()[2 * index];
                    let weight_hi = weight.evals()[2 * index + 1];
                    out[0] += g_lo * weight_lo;
                    out[1] += (g_hi + g_hi - g_lo) * (weight_hi + weight_hi - weight_lo);
                }
                let baseline_lo = baseline_table.evals()[2 * index];
                let baseline_hi = baseline_table.evals()[2 * index + 1];
                out[0] += baseline_lo;
                out[1] += baseline_hi + baseline_hi - baseline_lo;
                out
            })
            .reduce(
                || [AkitaField::zero(); 2],
                |left, right| [left[0] + right[0], left[1] + right[1]],
            );
        let poly = UnivariatePoly::from_evals(&[evals[0], claim - evals[0], evals[1]]);
        CompressedLabeledRoundPoly::sumcheck(&poly).append_to_transcript(&mut transcript);
        let challenge = transcript.challenge();
        claim = poly.evaluate(challenge);
        q_evals.push(evals);
        messages.push(poly);
        challenges.push(challenge);
    }
    let Some(final_challenge) = challenges.last().copied() else {
        unreachable!("the Hamming relation has a fixed positive number of rounds")
    };
    bind(
        final_challenge,
        &mut g_tables,
        &mut weight_tables,
        &mut baseline_table,
    );
    let final_relation = g_tables
        .iter()
        .zip(&weight_tables)
        .map(|(g, weight)| g.evals()[0] * weight.evals()[0])
        .sum::<AkitaField>()
        + baseline_table.evals()[0];
    assert_eq!(final_relation, claim);
    let output_claims = g_tables.iter().map(|table| table.evals()[0]).collect();
    HammingTrace {
        q_evals,
        messages,
        challenges,
        final_claim: claim,
        final_relation,
        output_claims,
        transcript_state: transcript.state(),
    }
}

fn recenter(masses: &mut [AkitaField]) {
    for table in masses.chunks_exact_mut(K) {
        table[0] = AkitaField::zero();
    }
}

fn run_cpu(
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
    inputs: HammingInputs<'_>,
) -> Sample {
    let started = Instant::now();
    let prepare_started = Instant::now();
    let mut masses = cpu_pushforward(rows, selectors, inputs.reference_cycle);
    let prepare = prepare_started.elapsed();
    let rounds_started = Instant::now();
    recenter(&mut masses);
    let trace = hamming_trace(&masses, inputs);
    let host_rounds = rounds_started.elapsed();
    Sample {
        masses,
        trace,
        wall: started.elapsed(),
        prepare,
        host_rounds,
    }
}

fn run_metal(
    context: &SolinasMetal,
    resident_rows: &jolt_kernels::metal::solinas::BooleanityRows,
    selectors: &[BooleanitySelector],
    inputs: HammingInputs<'_>,
    config: BooleanityAddressPushforwardConfig,
) -> EvalResult<MetalSample> {
    let started = Instant::now();
    let prepare_started = Instant::now();
    let invocation = context.prepare_booleanity_address_pushforward(
        resident_rows.clone(),
        selectors,
        inputs.reference_cycle,
        config,
    )?;
    let prepare = prepare_started.elapsed();
    let identities = invocation.static_buffer_identities();
    let shape = MetalShape {
        resident_row_identity: invocation.resident_row_identity(),
        rows: invocation.row_count(),
        polys: invocation.polys(),
        selectors_per_tile: invocation.selectors_per_tile(),
        selector_tiles: invocation.selector_tiles(),
        e_in_elements: invocation.e_in_length(),
        e_out_elements: invocation.e_out_length(),
        output_elements: invocation.output_elements(),
        partial_bytes: invocation.partial_bytes(),
        tile_threads: invocation.tile_threads_per_threadgroup(),
        finalize_threads: invocation.finalize_threads_per_threadgroup(),
        production_specialized: invocation.uses_production_specialization(),
        tile_limits: invocation.tile_pipeline_limits(),
        finalize_limits: invocation.finalize_pipeline_limits(),
    };
    let dispatch_started = Instant::now();
    let gpu_active = invocation.execute_timed()?;
    let dispatch_wall = dispatch_started.elapsed();
    let static_buffers_stable = identities == invocation.static_buffer_identities();
    let readback_started = Instant::now();
    let mut masses = invocation.read_masses()?;
    let readback = readback_started.elapsed();
    let expected_masses = selectors.len() * K;
    if masses.len() != expected_masses {
        return Err(format!(
            "Metal returned {} masses, expected {expected_masses}",
            masses.len()
        )
        .into());
    }
    let rounds_started = Instant::now();
    recenter(&mut masses);
    let trace = hamming_trace(&masses, inputs);
    let host_rounds = rounds_started.elapsed();
    Ok(MetalSample {
        member: Sample {
            masses,
            trace,
            wall: started.elapsed(),
            prepare,
            host_rounds,
        },
        gpu_active,
        dispatch_wall,
        readback,
        shape,
        static_buffer_identities: identities,
        static_buffers_stable,
        static_buffers_distinct: identities
            .iter()
            .enumerate()
            .all(|(index, identity)| !identities[..index].contains(identity)),
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

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn mad(values: &[f64], median: f64) -> f64 {
    let mut deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    median_f64(&mut deviations)
}

fn ns(value: Duration) -> EvalResult<u64> {
    Ok(u64::try_from(value.as_nanos())?)
}

fn ns_samples(values: &[Duration]) -> EvalResult<Vec<u64>> {
    values.iter().copied().map(ns).collect()
}

fn timing_remainder(wall: Duration, exclusive: &[Duration]) -> (Duration, bool) {
    let accounted = exclusive.iter().copied().sum::<Duration>();
    let Some(remainder) = wall.checked_sub(accounted) else {
        return (Duration::ZERO, false);
    };
    (
        remainder,
        accounted
            .checked_add(remainder)
            .is_some_and(|sum| sum == wall),
    )
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 26)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 5)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let trace_cutoff_log2 = env_usize("JOLT_METAL_HAMMING_WEIGHT_TRACE_CUTOFF_LOG2", 18)?;
    let inner_log2 = env_usize("JOLT_METAL_HAMMING_WEIGHT_INNER_LOG2", 15)?;
    let selectors_per_tile = env_usize("JOLT_METAL_HAMMING_WEIGHT_SELECTORS_PER_TILE", 6)?;
    let tile_threads = env_usize("JOLT_METAL_HAMMING_WEIGHT_TILE_THREADS", 512)?;
    let finalize_threads = env_usize("JOLT_METAL_HAMMING_WEIGHT_FINALIZE_THREADS", 1024)?;
    if repeats == 0 || repeats.is_multiple_of(2) {
        return Err("repeats must be a positive odd number".into());
    }
    if !(inner_log2..=28).contains(&log_n) || trace_cutoff_log2 > log_n {
        return Err("log_n, inner_log2, or trace cutoff is outside the evaluator domain".into());
    }

    let elements = 1usize << log_n;
    let selectors = Arc::new(selectors());
    if !production_selector_schedule_is_exact(&selectors) {
        return Err("production selector schedule diverged from the PIOP ABI".into());
    }
    let rows = Arc::new(rows(elements, seed)?);
    let selector_row_opportunities = u64::try_from(elements)?
        .checked_mul(u64::try_from(POLYS)?)
        .ok_or("selector-row opportunity count overflowed")?;
    let nonzero_recentered_contributions = u64::try_from(
        rows.par_iter()
            .map(|row| {
                selectors
                    .iter()
                    .copied()
                    .filter(|selector| hot_index(*row, *selector).is_some_and(|hot| hot != 0))
                    .count()
            })
            .sum::<usize>(),
    )?;
    let reference_cycle = point(log_n, seed ^ 0xc1c1_e5e5);
    let reference_address = point(CHUNK_BITS, seed ^ 0xadd2_e550);
    let virtualization_points = (0..RA_POLYS)
        .map(|index| point(CHUNK_BITS, seed ^ 0x71a0_0000 ^ index as u64))
        .collect::<Vec<_>>();
    let gamma = AkitaField::from_u128(
        u128::from(splitmix(seed ^ 0x6a6d_6d61))
            | (u128::from(splitmix(seed ^ 0xfeed_f00d) & 0x7fff_ffff_ffff_ffff) << 64),
    );
    let ram_hamming_weight = AkitaField::from_u128(
        u128::from(splitmix(seed ^ 0x2a6d_0001))
            | (u128::from(splitmix(seed ^ 0x2a6d_0002) & 0x7fff_ffff_ffff_ffff) << 64),
    );
    let hamming_inputs = HammingInputs {
        reference_cycle: &reference_cycle,
        gamma,
        reference_address: &reference_address,
        virtualization_points: &virtualization_points,
        ram_hamming_weight,
    };
    let config = BooleanityAddressPushforwardConfig {
        inner_log2,
        selectors_per_tile,
        tile_threads_per_threadgroup: Some(tile_threads),
        finalize_threads_per_threadgroup: Some(finalize_threads),
    };
    let context = SolinasMetal::for_akita()?;
    let resident_rows = context.prepare_booleanity_rows(&rows)?;
    let resident_identity = resident_rows.allocation_identity();
    let device_before_reference = context.device_info();
    let expected_masses = POLYS * K;

    let cpu_reference = black_box(run_cpu(&rows, &selectors, hamming_inputs));
    let metal_reference = black_box(run_metal(
        &context,
        &resident_rows,
        &selectors,
        hamming_inputs,
        config,
    )?);
    let reference_mass_lengths_exact = cpu_reference.masses.len() == expected_masses
        && metal_reference.member.masses.len() == expected_masses;
    let exact_masses = cpu_reference.masses == metal_reference.member.masses;
    let recentered_bucket_zero_exact = cpu_reference
        .masses
        .chunks_exact(K)
        .chain(metal_reference.member.masses.chunks_exact(K))
        .all(|table| table[0] == AkitaField::zero());
    let nonzero_recentered_values_present = cpu_reference
        .masses
        .chunks_exact(K)
        .any(|table| table[1..].iter().any(|value| *value != AkitaField::zero()));
    let exact_skipped_q_evals = cpu_reference.trace.q_evals == metal_reference.member.trace.q_evals;
    let exact_round_polynomials =
        cpu_reference.trace.messages == metal_reference.member.trace.messages;
    let exact_host_fiat_shamir_challenges =
        cpu_reference.trace.challenges == metal_reference.member.trace.challenges;
    let exact_final_claim =
        cpu_reference.trace.final_claim == metal_reference.member.trace.final_claim;
    let exact_output_claims =
        cpu_reference.trace.output_claims == metal_reference.member.trace.output_claims;
    let exact_transcript_state =
        cpu_reference.trace.transcript_state == metal_reference.member.trace.transcript_state;
    if !(reference_mass_lengths_exact
        && exact_masses
        && recentered_bucket_zero_exact
        && nonzero_recentered_values_present
        && exact_skipped_q_evals
        && exact_round_polynomials
        && exact_host_fiat_shamir_challenges
        && exact_final_claim
        && exact_output_claims
        && exact_transcript_state)
    {
        return Err(
            "Metal Hamming-weight claim reduction result differs from the optimized CPU control"
                .into(),
        );
    }

    let mut cpu_samples = Vec::with_capacity(repeats);
    let mut cpu_prepare = Vec::with_capacity(repeats);
    let mut cpu_host_rounds = Vec::with_capacity(repeats);
    let mut cpu_unattributed = Vec::with_capacity(repeats);
    let mut metal_samples = Vec::with_capacity(repeats);
    let mut metal_prepare = Vec::with_capacity(repeats);
    let mut metal_dispatch = Vec::with_capacity(repeats);
    let mut metal_readback = Vec::with_capacity(repeats);
    let mut metal_host_rounds = Vec::with_capacity(repeats);
    let mut gpu_active = Vec::with_capacity(repeats);
    let mut metal_unattributed = Vec::with_capacity(repeats);
    let mut paired_speedups = Vec::with_capacity(repeats);
    let mut cpu_component_timings_reconciled = true;
    let mut metal_component_timings_reconciled = true;
    let mut gpu_timestamps_nested = true;
    let mut timed_mass_lengths_exact = true;
    let mut timed_samples_exact = true;
    let mut static_buffers_stable = true;
    let mut static_buffers_distinct = true;
    let mut resident_rows_reused = true;
    let mut metal_shape_stable = true;

    for repeat in 0..repeats {
        let cpu = || black_box(run_cpu(&rows, &selectors, hamming_inputs));
        let metal = || {
            black_box(run_metal(
                &context,
                &resident_rows,
                &selectors,
                hamming_inputs,
                config,
            ))
        };
        let (cpu, metal) = if repeat.is_multiple_of(2) {
            (cpu(), metal()?)
        } else {
            let metal = metal()?;
            (cpu(), metal)
        };
        timed_mass_lengths_exact &=
            cpu.masses.len() == expected_masses && metal.member.masses.len() == expected_masses;
        timed_samples_exact &= cpu.masses == cpu_reference.masses
            && cpu.trace == cpu_reference.trace
            && metal.member.masses == cpu_reference.masses
            && metal.member.trace == cpu_reference.trace;
        if !timed_mass_lengths_exact || !timed_samples_exact {
            return Err("a timed sample drifted from the exact reference".into());
        }
        let (cpu_remainder, cpu_reconciled) =
            timing_remainder(cpu.wall, &[cpu.prepare, cpu.host_rounds]);
        let (metal_remainder, metal_reconciled) = timing_remainder(
            metal.member.wall,
            &[
                metal.member.prepare,
                metal.dispatch_wall,
                metal.readback,
                metal.member.host_rounds,
            ],
        );
        cpu_component_timings_reconciled &= cpu_reconciled;
        metal_component_timings_reconciled &= metal_reconciled;
        gpu_timestamps_nested &=
            metal.gpu_active > Duration::ZERO && metal.gpu_active <= metal.dispatch_wall;
        paired_speedups.push(cpu.wall.as_secs_f64() / metal.member.wall.as_secs_f64());
        cpu_samples.push(cpu.wall);
        cpu_prepare.push(cpu.prepare);
        cpu_host_rounds.push(cpu.host_rounds);
        cpu_unattributed.push(cpu_remainder);
        metal_samples.push(metal.member.wall);
        metal_prepare.push(metal.member.prepare);
        metal_dispatch.push(metal.dispatch_wall);
        metal_readback.push(metal.readback);
        metal_host_rounds.push(metal.member.host_rounds);
        gpu_active.push(metal.gpu_active);
        metal_unattributed.push(metal_remainder);
        static_buffers_stable &= metal.static_buffers_stable;
        static_buffers_distinct &= metal.static_buffers_distinct;
        resident_rows_reused &= metal.shape.resident_row_identity == resident_identity;
        metal_shape_stable &= metal.shape == metal_reference.shape;
    }

    let cpu_ns_samples = ns_samples(&cpu_samples)?;
    let cpu_prepare_ns_samples = ns_samples(&cpu_prepare)?;
    let cpu_host_rounds_ns_samples = ns_samples(&cpu_host_rounds)?;
    let cpu_unattributed_ns_samples = ns_samples(&cpu_unattributed)?;
    let metal_ns_samples = ns_samples(&metal_samples)?;
    let metal_prepare_ns_samples = ns_samples(&metal_prepare)?;
    let metal_dispatch_ns_samples = ns_samples(&metal_dispatch)?;
    let metal_gpu_active_ns_samples = ns_samples(&gpu_active)?;
    let metal_readback_ns_samples = ns_samples(&metal_readback)?;
    let metal_host_rounds_ns_samples = ns_samples(&metal_host_rounds)?;
    let metal_unattributed_ns_samples = ns_samples(&metal_unattributed)?;
    let cpu_median = median(&mut cpu_samples);
    let cpu_prepare_median = median(&mut cpu_prepare);
    let cpu_host_rounds_median = median(&mut cpu_host_rounds);
    let cpu_unattributed_median = median(&mut cpu_unattributed);
    let metal_median = median(&mut metal_samples);
    let prepare_median = median(&mut metal_prepare);
    let dispatch_median = median(&mut metal_dispatch);
    let readback_median = median(&mut metal_readback);
    let host_rounds_median = median(&mut metal_host_rounds);
    let gpu_active_median = median(&mut gpu_active);
    let metal_unattributed_median = median(&mut metal_unattributed);
    let mut speedups_for_median = paired_speedups.clone();
    let speedup_median = median_f64(&mut speedups_for_median);
    let speedup_mad = mad(&paired_speedups, speedup_median);
    let ratio_of_member_medians = cpu_median.as_secs_f64() / metal_median.as_secs_f64();
    let cpu_selector_row_opportunities_per_second =
        selector_row_opportunities as f64 / cpu_median.as_secs_f64();
    let metal_selector_row_opportunities_per_second =
        selector_row_opportunities as f64 / metal_median.as_secs_f64();
    let cpu_nonzero_recentered_contributions_per_second =
        nonzero_recentered_contributions as f64 / cpu_median.as_secs_f64();
    let metal_nonzero_recentered_contributions_per_second =
        nonzero_recentered_contributions as f64 / metal_median.as_secs_f64();
    let e_in_elements = 1usize << inner_log2;
    let e_out_elements = elements / e_in_elements;
    let trace_cutoff_elements = 1usize << trace_cutoff_log2;
    let selector_bytes = u64::try_from(POLYS * size_of::<[u32; 2]>())?;
    let e_in_bytes = u64::try_from(e_in_elements * size_of::<AkitaField>())?;
    let e_out_bytes = u64::try_from(e_out_elements * size_of::<AkitaField>())?;
    let output_bytes = u64::try_from(POLYS * K * size_of::<AkitaField>())?;
    let expected_partial_bytes =
        u64::try_from(e_out_elements * selectors_per_tile * K * size_of::<AkitaField>())?;
    let hamming_owned_bytes = [
        selector_bytes,
        e_in_bytes,
        e_out_bytes,
        expected_partial_bytes,
        output_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
    .ok_or("Hamming buffer byte count overflowed")?;
    let shape = metal_reference.shape;
    let tile_dynamic_threadgroup_bytes =
        u64::try_from(selectors_per_tile * K * ACCUMULATOR_WORDS * size_of::<u32>())?;
    let finalize_dynamic_threadgroup_bytes =
        u64::try_from(shape.finalize_threads * size_of::<AkitaField>())?;
    let tile_total_threadgroup_bytes = shape
        .tile_limits
        .static_threadgroup_memory_length
        .checked_add(tile_dynamic_threadgroup_bytes)
        .ok_or("tile threadgroup byte count overflowed")?;
    let finalize_total_threadgroup_bytes = shape
        .finalize_limits
        .static_threadgroup_memory_length
        .checked_add(finalize_dynamic_threadgroup_bytes)
        .ok_or("finalize threadgroup byte count overflowed")?;
    let expected_production_specialization = matches!(selectors_per_tile, 3 | 6);
    let (warmup_cpu_unattributed, warmup_cpu_reconciled) = timing_remainder(
        cpu_reference.wall,
        &[cpu_reference.prepare, cpu_reference.host_rounds],
    );
    let (warmup_metal_unattributed, warmup_metal_reconciled) = timing_remainder(
        metal_reference.member.wall,
        &[
            metal_reference.member.prepare,
            metal_reference.dispatch_wall,
            metal_reference.readback,
            metal_reference.member.host_rounds,
        ],
    );
    let warmup_gpu_timestamp_nested = metal_reference.gpu_active > Duration::ZERO
        && metal_reference.gpu_active <= metal_reference.dispatch_wall;
    let final_relations_exact = cpu_reference.trace.final_claim
        == cpu_reference.trace.final_relation
        && metal_reference.member.trace.final_claim == metal_reference.member.trace.final_relation;
    let correctness_exact = reference_mass_lengths_exact
        && timed_mass_lengths_exact
        && timed_samples_exact
        && exact_masses
        && recentered_bucket_zero_exact
        && nonzero_recentered_values_present
        && exact_skipped_q_evals
        && exact_round_polynomials
        && exact_host_fiat_shamir_challenges
        && exact_final_claim
        && final_relations_exact
        && exact_output_claims
        && exact_transcript_state;
    let orders = (0..repeats)
        .map(|repeat| {
            if repeat.is_multiple_of(2) {
                ["optimized", "metal"]
            } else {
                ["metal", "optimized"]
            }
        })
        .collect::<Vec<_>>();
    let info = context.device_info();
    let buffer_lengths_admitted = [
        selector_bytes,
        e_in_bytes,
        e_out_bytes,
        expected_partial_bytes,
        output_bytes,
    ]
    .into_iter()
    .all(|bytes| bytes <= info.max_buffer_length);
    let working_set_admitted = device_before_reference
        .current_allocated_size
        .checked_add(hamming_owned_bytes)
        .is_some_and(|bytes| bytes <= info.recommended_max_working_set_size);
    let sample_cardinality_exact = [
        cpu_ns_samples.len(),
        cpu_prepare_ns_samples.len(),
        cpu_host_rounds_ns_samples.len(),
        cpu_unattributed_ns_samples.len(),
        metal_ns_samples.len(),
        metal_prepare_ns_samples.len(),
        metal_dispatch_ns_samples.len(),
        metal_gpu_active_ns_samples.len(),
        metal_readback_ns_samples.len(),
        metal_host_rounds_ns_samples.len(),
        metal_unattributed_ns_samples.len(),
        paired_speedups.len(),
        orders.len(),
    ]
    .into_iter()
    .all(|length| length == repeats);
    let alternating_orders_exact = orders.iter().enumerate().all(|(index, order)| {
        *order
            == if index.is_multiple_of(2) {
                ["optimized", "metal"]
            } else {
                ["metal", "optimized"]
            }
    });
    let guard_values = [
        ("reference_mass_lengths_exact", reference_mass_lengths_exact),
        ("timed_mass_lengths_exact", timed_mass_lengths_exact),
        ("expected_mass_count_exact", expected_masses == POLYS * K),
        ("exact_masses", exact_masses),
        ("recentered_bucket_zero_exact", recentered_bucket_zero_exact),
        (
            "nonzero_recentered_values_present",
            nonzero_recentered_values_present,
        ),
        (
            "nonzero_contribution_count_admitted",
            nonzero_recentered_contributions > 0
                && nonzero_recentered_contributions <= selector_row_opportunities,
        ),
        ("exact_skipped_q_evals", exact_skipped_q_evals),
        ("exact_round_polynomials", exact_round_polynomials),
        (
            "exact_host_fiat_shamir_challenges",
            exact_host_fiat_shamir_challenges,
        ),
        ("exact_final_claim", exact_final_claim),
        ("exact_output_claims", exact_output_claims),
        (
            "output_claim_count_exact",
            cpu_reference.trace.output_claims.len() == POLYS
                && metal_reference.member.trace.output_claims.len() == POLYS,
        ),
        ("exact_final_relations", final_relations_exact),
        ("exact_transcript_state", exact_transcript_state),
        ("timed_samples_match_reference", timed_samples_exact),
        ("correctness_exact", correctness_exact),
        ("sample_cardinality_exact", sample_cardinality_exact),
        ("alternating_orders_exact", alternating_orders_exact),
        (
            "cpu_component_timings_reconciled",
            cpu_component_timings_reconciled,
        ),
        (
            "metal_component_timings_reconciled",
            metal_component_timings_reconciled,
        ),
        (
            "warmup_cpu_component_timings_reconciled",
            warmup_cpu_reconciled,
        ),
        (
            "warmup_metal_component_timings_reconciled",
            warmup_metal_reconciled,
        ),
        (
            "gpu_active_nested_in_dispatch_wall",
            gpu_timestamps_nested && warmup_gpu_timestamp_nested,
        ),
        (
            "member_durations_positive",
            cpu_ns_samples
                .iter()
                .chain(&metal_ns_samples)
                .all(|value| *value > 0),
        ),
        (
            "speedups_finite_positive",
            paired_speedups
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
        ),
        ("resident_rows_reused", resident_rows_reused),
        (
            "resident_rows_stable_for_stage7_handoff",
            resident_rows.allocation_identity() == resident_identity,
        ),
        ("metal_shape_stable_across_samples", metal_shape_stable),
        ("row_count_exact", shape.rows == elements),
        ("polynomial_count_exact", shape.polys == POLYS),
        (
            "production_selector_schedule_exact",
            production_selector_schedule_is_exact(&selectors),
        ),
        (
            "selector_tile_width_exact",
            shape.selectors_per_tile == selectors_per_tile,
        ),
        (
            "selector_tile_count_exact",
            shape.selector_tiles == POLYS.div_ceil(selectors_per_tile),
        ),
        ("e_in_size_exact", shape.e_in_elements == e_in_elements),
        ("e_out_size_exact", shape.e_out_elements == e_out_elements),
        (
            "output_size_exact",
            shape.output_elements == expected_masses,
        ),
        (
            "partial_size_exact",
            shape.partial_bytes == expected_partial_bytes,
        ),
        (
            "production_specialization_exact",
            shape.production_specialized == expected_production_specialization,
        ),
        (
            "requested_effective_tile_threads_exact",
            shape.tile_threads == tile_threads,
        ),
        (
            "requested_effective_finalize_threads_exact",
            shape.finalize_threads == finalize_threads,
        ),
        (
            "tile_pipeline_simd_width_exact",
            shape.tile_limits.thread_execution_width == SIMD_WIDTH,
        ),
        (
            "finalize_pipeline_simd_width_exact",
            shape.finalize_limits.thread_execution_width == SIMD_WIDTH,
        ),
        (
            "tile_pipeline_thread_limit_admits_dispatch",
            shape.tile_threads <= shape.tile_limits.max_total_threads_per_threadgroup
                && shape
                    .tile_threads
                    .is_multiple_of(shape.tile_limits.thread_execution_width),
        ),
        (
            "finalize_pipeline_thread_limit_admits_dispatch",
            shape.finalize_threads <= shape.finalize_limits.max_total_threads_per_threadgroup
                && shape
                    .finalize_threads
                    .is_multiple_of(shape.finalize_limits.thread_execution_width),
        ),
        (
            "tile_threadgroup_memory_admitted",
            tile_total_threadgroup_bytes <= info.max_threadgroup_memory_length,
        ),
        (
            "finalize_threadgroup_memory_admitted",
            finalize_total_threadgroup_bytes <= info.max_threadgroup_memory_length,
        ),
        (
            "static_device_buffers_stable",
            static_buffers_stable && metal_reference.static_buffers_stable,
        ),
        (
            "static_device_buffers_distinct",
            static_buffers_distinct && metal_reference.static_buffers_distinct,
        ),
        ("buffer_lengths_admitted", buffer_lengths_admitted),
        ("working_set_admitted", working_set_admitted),
        ("solinas_offset_exact", info.offset == 0xffff_a7f7),
        (
            "field_and_row_sizes_exact",
            size_of::<AkitaField>() == 16 && size_of::<BooleanityRow>() == ROW_BYTES,
        ),
        ("one_execute_timed_call_per_member", true),
        ("single_command_completion_contract", true),
        ("single_result_readback_contract", true),
        ("no_per_row_contribution_buffer_contract", true),
        ("host_fiat_shamir", true),
        (
            "production_trace_cutoff_admits_target",
            elements >= trace_cutoff_elements,
        ),
    ];
    let all_exact = guard_values.iter().all(|(_, value)| *value);
    let mut guards = guard_values
        .into_iter()
        .map(|(name, value)| (name.to_owned(), serde_json::Value::Bool(value)))
        .collect::<serde_json::Map<_, _>>();
    let _ = guards.insert("all_exact".to_owned(), serde_json::Value::Bool(all_exact));
    let promotion_scale = log_n >= MIN_PROMOTION_LOG_N;
    let promotion_pair_count = repeats >= MIN_PROMOTION_PAIRS;
    let promotion_local_speedup = speedup_median >= MIN_PROMOTION_SPEEDUP;
    let local_promotion_eligible =
        all_exact && promotion_scale && promotion_pair_count && promotion_local_speedup;

    println!(
        "{}",
        json!({
            "schema": "hamming_weight_claim_reduction_v1",
            "schema_version": 1,
            "kernel": "hamming_weight_claim_reduction",
            "workload": {
                "log_n": log_n,
                "rows": elements,
                "selectors": POLYS,
                "k": K,
                "hamming_address_rounds": CHUNK_BITS,
                "row_bytes": ROW_BYTES,
                "selector_row_opportunities": selector_row_opportunities,
                "nonzero_recentered_contributions": nonzero_recentered_contributions,
                "seed": seed,
                "repeats": repeats,
                "orders": orders,
                "resident_rows_prepared_once_outside_members": true,
                "cpu_row_construction_outside_members": true,
                "resident_row_upload_bytes_inside_metal_member": 0,
                "excluded_warmup_pairs": 1,
                "cpu_member_contract": "optimized shared-row tensor-equality pushforward mirror, bucket-zero recentering, W/baseline construction, and eight host Fiat-Shamir rounds",
                "metal_member_contract": "cycle-equality preparation and upload over resident rows, one command encode/submit/wait, one result readback, bucket-zero recentering, and the same W/baseline and eight host Fiat-Shamir rounds",
                "gpu_active_accounting": "nested in metal dispatch wall; never added to member components",
            },
            "fingerprint": {
                "trace_cutoff_log2": trace_cutoff_log2,
                "trace_cutoff_elements": trace_cutoff_elements,
                "inner_log2": inner_log2,
                "selectors_per_tile": selectors_per_tile,
                "tile_threads": tile_threads,
                "finalize_threads": finalize_threads,
                "effective_selector_tiles": shape.selector_tiles,
                "effective_tile_threads": shape.tile_threads,
                "effective_finalize_threads": shape.finalize_threads,
                "production_specialized": shape.production_specialized,
                "accumulator_words": ACCUMULATOR_WORDS,
                "resident_row_identity": resident_identity,
                "cpu_threads": rayon::current_num_threads(),
                "cpu_control": "standalone parallel optimized TensorEqTable/AkitaAccumulator pushforward mirror",
                "host_round_oracle": "identical deterministic W/baseline and host-round implementation",
            },
            "metrics": {
                "hybrid_speedup": speedup_median,
                "ratio_of_member_medians": ratio_of_member_medians,
                "paired_speedups": paired_speedups,
                "paired_speedup_mad": speedup_mad,
                "cpu_member_ns_samples": cpu_ns_samples,
                "metal_member_ns_samples": metal_ns_samples,
                "selector_row_opportunities": selector_row_opportunities,
                "nonzero_recentered_contributions": nonzero_recentered_contributions,
                "cpu_selector_row_opportunities_per_second": cpu_selector_row_opportunities_per_second,
                "metal_selector_row_opportunities_per_second": metal_selector_row_opportunities_per_second,
                "cpu_nonzero_recentered_contributions_per_second": cpu_nonzero_recentered_contributions_per_second,
                "metal_nonzero_recentered_contributions_per_second": metal_nonzero_recentered_contributions_per_second,
                "minimum_promotion_speedup": MIN_PROMOTION_SPEEDUP,
            },
            "timings": {
                "cpu_member_median_ns": ns(cpu_median)?,
                "cpu_prepare_median_ns": ns(cpu_prepare_median)?,
                "cpu_host_rounds_median_ns": ns(cpu_host_rounds_median)?,
                "cpu_unattributed_median_ns": ns(cpu_unattributed_median)?,
                "metal_member_median_ns": ns(metal_median)?,
                "metal_prepare_median_ns": ns(prepare_median)?,
                "metal_dispatch_wall_median_ns": ns(dispatch_median)?,
                "metal_gpu_active_median_ns": ns(gpu_active_median)?,
                "metal_readback_median_ns": ns(readback_median)?,
                "metal_host_rounds_median_ns": ns(host_rounds_median)?,
                "metal_unattributed_median_ns": ns(metal_unattributed_median)?,
                "cpu_prepare_ns_samples": cpu_prepare_ns_samples,
                "cpu_host_rounds_ns_samples": cpu_host_rounds_ns_samples,
                "cpu_unattributed_ns_samples": cpu_unattributed_ns_samples,
                "metal_prepare_ns_samples": metal_prepare_ns_samples,
                "metal_dispatch_wall_ns_samples": metal_dispatch_ns_samples,
                "metal_gpu_active_ns_samples": metal_gpu_active_ns_samples,
                "metal_readback_ns_samples": metal_readback_ns_samples,
                "metal_host_rounds_ns_samples": metal_host_rounds_ns_samples,
                "metal_unattributed_ns_samples": metal_unattributed_ns_samples,
                "exclusive_component_accounting": ["prepare", "dispatch_wall", "readback", "host_rounds", "unattributed"],
                "excluded_warmup": {
                    "cpu_member_ns": ns(cpu_reference.wall)?,
                    "cpu_prepare_ns": ns(cpu_reference.prepare)?,
                    "cpu_host_rounds_ns": ns(cpu_reference.host_rounds)?,
                    "cpu_unattributed_ns": ns(warmup_cpu_unattributed)?,
                    "metal_member_ns": ns(metal_reference.member.wall)?,
                    "metal_prepare_ns": ns(metal_reference.member.prepare)?,
                    "metal_dispatch_wall_ns": ns(metal_reference.dispatch_wall)?,
                    "metal_gpu_active_ns": ns(metal_reference.gpu_active)?,
                    "metal_readback_ns": ns(metal_reference.readback)?,
                    "metal_host_rounds_ns": ns(metal_reference.member.host_rounds)?,
                    "metal_unattributed_ns": ns(warmup_metal_unattributed)?,
                },
            },
            "guards": guards,
            "all_exact": all_exact,
            "resources": {
                "device": {
                    "name": info.name,
                    "max_buffer_length": info.max_buffer_length,
                    "max_threadgroup_memory_length": info.max_threadgroup_memory_length,
                    "recommended_max_working_set_size": info.recommended_max_working_set_size,
                    "current_allocated_size": info.current_allocated_size,
                    "offset": info.offset,
                },
                "device_allocated_before_reference_bytes": device_before_reference.current_allocated_size,
                "resident_row_bytes": u64::try_from(elements * ROW_BYTES)?,
                "selector_bytes": selector_bytes,
                "e_in_bytes": e_in_bytes,
                "e_out_bytes": e_out_bytes,
                "partial_owned_bytes": shape.partial_bytes,
                "partial_expected_bytes": expected_partial_bytes,
                "hamming_owned_device_bytes": hamming_owned_bytes,
                "result_readback_bytes": output_bytes,
                "static_device_buffer_count": metal_reference.static_buffer_identities.len(),
                "static_device_buffer_identities": metal_reference.static_buffer_identities,
                "gpu_active_total_ns": ns(gpu_active.iter().copied().sum())?,
                "gpu_seconds": (metal_reference.gpu_active + gpu_active.iter().copied().sum::<Duration>()).as_secs_f64(),
            },
            "pipelines": {
                "tile": {
                    "thread_execution_width": shape.tile_limits.thread_execution_width,
                    "max_total_threads_per_threadgroup": shape.tile_limits.max_total_threads_per_threadgroup,
                    "static_threadgroup_bytes": shape.tile_limits.static_threadgroup_memory_length,
                    "dynamic_threadgroup_bytes": tile_dynamic_threadgroup_bytes,
                    "total_threadgroup_bytes": tile_total_threadgroup_bytes,
                    "effective_threads_per_threadgroup": shape.tile_threads,
                },
                "finalize": {
                    "thread_execution_width": shape.finalize_limits.thread_execution_width,
                    "max_total_threads_per_threadgroup": shape.finalize_limits.max_total_threads_per_threadgroup,
                    "static_threadgroup_bytes": shape.finalize_limits.static_threadgroup_memory_length,
                    "dynamic_threadgroup_bytes": finalize_dynamic_threadgroup_bytes,
                    "total_threadgroup_bytes": finalize_total_threadgroup_bytes,
                    "effective_threads_per_threadgroup": shape.finalize_threads,
                },
            },
            "promotion": {
                "minimum_log_n": MIN_PROMOTION_LOG_N,
                "minimum_pairs": MIN_PROMOTION_PAIRS,
                "minimum_speedup": MIN_PROMOTION_SPEEDUP,
                "scale_eligible": promotion_scale,
                "pair_count_eligible": promotion_pair_count,
                "speedup_eligible": promotion_local_speedup,
                "local_eligible": local_promotion_eligible,
                "production_piop_holdout_required": true,
            },
            "oracle_limits": {
                "cpu_denominator_is_production_kernel": false,
                "cpu_denominator_scope": "standalone optimized shared-row pushforward mirror plus identical W/baseline and host rounds",
                "host_rounds_are_independently_implemented": false,
                "mass_oracle_independent_of_metal_shader": true,
                "command_and_readback_counts_are_runtime_counters": false,
                "requires_production_piop_holdout": true,
            },
        })
    );
    Ok(())
}
