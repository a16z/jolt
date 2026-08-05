use std::array;
use std::sync::Arc;

use jolt_field::{
    AdditiveAccumulator, Fr, FrSmallScalarAccumulator, FromPrimitiveInt, Invertible,
    RingAccumulator, SignedScalarAccumulator, WideAccumulator,
};
use jolt_poly::lagrange::interpolate_to_coeffs;
use rayon::prelude::*;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const REGISTERS_ADDRESS_PHASE: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::RegistersAddressPhase(RegistersAddressPhase),
);

const REGISTER_COUNT: usize = 128;
const NO_REGISTER: u8 = u8::MAX;
const MESSAGE_CHUNK_CYCLES: usize = 1 << 14;
const BINARY_EVAL_BASIS: [[i64; 2]; 3] = [[1, 0], [0, 1], [-1, 2]];

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RegistersAddressPhase;

impl Objective for RegistersAddressPhase {
    type Setup = Arc<RegistersAddressPhaseFixture>;

    fn name(&self) -> &str {
        "registers_address_phase"
    }

    fn description(&self) -> String {
        "Full stage-4 address-first RegistersRW phase over trace-shaped CSR and run-list Val state"
            .to_string()
    }

    fn setup(&self) -> Self::Setup {
        thread_local! {
            static FIXTURE: Arc<RegistersAddressPhaseFixture> =
                Arc::new(RegistersAddressPhaseFixture::synthetic(22));
        }
        FIXTURE.with(Arc::clone)
    }

    fn run(&self, setup: Self::Setup) {
        std::hint::black_box(setup.run_radix4());
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AccessRow {
    lanes: [u8; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WriteEvent {
    cycle: u32,
    register: u8,
    post_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    start: u32,
    value: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunList {
    runs: Vec<Run>,
}

impl RunList {
    fn zero() -> Self {
        Self {
            runs: vec![Run {
                start: 0,
                value: Fr::from_u64(0),
            }],
        }
    }

    fn value_at(&self, cycle: usize, cursor: &mut usize) -> Fr {
        while *cursor + 1 < self.runs.len() && self.runs[*cursor + 1].start as usize <= cycle {
            *cursor += 1;
        }
        self.runs[*cursor].value
    }

    fn cursor_at(&self, cycle: usize) -> usize {
        self.runs
            .partition_point(|run| run.start as usize <= cycle)
            .saturating_sub(1)
    }

    fn materialize(&self, cycles: usize) -> Vec<Fr> {
        let mut cursor = 0;
        (0..cycles)
            .map(|cycle| self.value_at(cycle, &mut cursor))
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AddressPhaseOutput {
    pub final_claim: Fr,
    pub message_digest: Fr,
    bound_addresses: [[Fr; REGISTER_COUNT]; 3],
    final_val: RunList,
}

impl AddressPhaseOutput {
    pub fn materialize_final_val(&self, cycles: usize) -> Vec<Fr> {
        self.final_val.materialize(cycles)
    }
}

pub struct RegistersAddressPhaseFixture {
    cycles: usize,
    rows: Vec<AccessRow>,
    eq_cycle: Vec<Fr>,
    rd_inc: Vec<Fr>,
    level_zero_val: Vec<RunList>,
    initial_claim: Fr,
    gamma: Fr,
}

impl RegistersAddressPhaseFixture {
    pub fn synthetic(log_cycles: usize) -> Self {
        assert!(log_cycles < u32::BITS as usize);
        let cycles = 1usize << log_cycles;
        let gamma = Fr::from_u64(0x1_0000_01b3);
        let mut rows = Vec::with_capacity(cycles);
        let mut eq_cycle = Vec::with_capacity(cycles);
        let mut rd_inc = Vec::with_capacity(cycles);
        let mut level_zero_val = (0..REGISTER_COUNT)
            .map(|_| RunList::zero())
            .collect::<Vec<_>>();
        let mut state = [0u64; REGISTER_COUNT];

        for cycle in 0..cycles {
            let hash = mix64(cycle as u64 ^ 0x9e37_79b9_7f4a_7c15);
            let write_register = (cycle & 7 != 0).then(|| synthetic_write_register(hash));
            let rs1 = if cycle & 0x3ff == 0 {
                0
            } else if cycle & 0xff == 1 {
                124
            } else {
                synthetic_read_register(hash.rotate_left(17))
            };
            let rs2 = if cycle & 0xff == 2 {
                127
            } else {
                synthetic_read_register(hash.rotate_left(39))
            };
            rows.push(AccessRow {
                lanes: [write_register.unwrap_or(NO_REGISTER), rs1, rs2],
            });
            eq_cycle.push(Fr::from_u64((mix64(hash) & 0xffff) + 1));

            if let Some(register) = write_register {
                let register_index = register as usize;
                let pre_value = state[register_index];
                let post_value = mix64(hash ^ cycle as u64).wrapping_add(cycle as u64 + 1);
                rd_inc.push(Fr::from_i128(
                    i128::from(post_value) - i128::from(pre_value),
                ));
                state[register_index] = post_value;
                push_post_state_run(
                    &mut level_zero_val[register_index],
                    cycle,
                    cycles,
                    post_value,
                );
            } else {
                rd_inc.push(Fr::from_u64(0));
            }
        }

        let initial_claim =
            compute_initial_claim(&rows, &eq_cycle, &rd_inc, &level_zero_val, gamma);
        Self {
            cycles,
            rows,
            eq_cycle,
            rd_inc,
            level_zero_val,
            initial_claim,
            gamma,
        }
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn run_binary(&self) -> AddressPhaseOutput {
        self.run_binary_with(default_binary_challenges())
    }

    pub fn run_radix4(&self) -> AddressPhaseOutput {
        self.run_radix4_with(default_radix4_challenges())
    }

    fn run_binary_with(&self, challenges: [Fr; 7]) -> AddressPhaseOutput {
        let mut current_nodes: Option<Vec<RunList>> = None;
        let mut bound_addresses = initial_bound_addresses(self.gamma);
        let mut current_claim = self.initial_claim;
        let mut digest = Fr::from_u64(0);
        for (consumed_bits, challenge) in challenges.into_iter().enumerate() {
            let nodes = current_nodes.as_deref().unwrap_or(&self.level_zero_val);
            let evals = compute_message::<2, 3>(
                self,
                nodes,
                &bound_addresses,
                consumed_bits,
                BINARY_EVAL_BASIS,
            );
            let coeffs = interpolate_to_coeffs(0, &evals);
            let round_sum = coeffs[0] + coeffs[0] + coeffs[1] + coeffs[2];
            assert_eq!(round_sum, current_claim, "binary round-sum mismatch");
            current_claim = evaluate_coefficients(&coeffs, challenge);
            digest += digest_coefficients(&coeffs);

            let weights = [Fr::from_u64(1) - challenge, challenge];
            current_nodes = Some(fold_run_lists::<2>(nodes, weights));
            bind_address_weights::<2>(&mut bound_addresses, consumed_bits, weights);
        }

        let final_val = current_nodes.expect("seven binary folds produce one node");
        debug_assert_eq!(final_val.len(), 1);
        AddressPhaseOutput {
            final_claim: current_claim,
            message_digest: digest,
            bound_addresses,
            final_val: final_val.into_iter().next().expect("one final node"),
        }
    }

    fn run_radix4_with(&self, challenges: ([Fr; 3], Fr)) -> AddressPhaseOutput {
        let mut current_nodes: Option<Vec<RunList>> = None;
        let mut bound_addresses = initial_bound_addresses(self.gamma);
        let mut current_claim = self.initial_claim;
        let mut digest = Fr::from_u64(0);
        let mut consumed_bits = 0;
        let radix_basis = radix4_eval_basis();

        for challenge in challenges.0 {
            let nodes = current_nodes.as_deref().unwrap_or(&self.level_zero_val);
            let evals =
                compute_message::<4, 7>(self, nodes, &bound_addresses, consumed_bits, radix_basis);
            let coeffs = interpolate_to_coeffs(-3, &evals);
            let round_sum = d_sum_degree_six(&coeffs);
            assert_eq!(round_sum, current_claim, "radix-4 round-sum mismatch");
            current_claim = evaluate_coefficients(&coeffs, challenge);
            digest += digest_coefficients(&coeffs);

            let weights = radix4_challenge_weights(challenge);
            current_nodes = Some(fold_run_lists::<4>(nodes, weights));
            bind_address_weights::<4>(&mut bound_addresses, consumed_bits, weights);
            consumed_bits += 2;
        }

        let challenge = challenges.1;
        let nodes = current_nodes.as_deref().expect("three radix-4 folds");
        let evals = compute_message::<2, 3>(
            self,
            nodes,
            &bound_addresses,
            consumed_bits,
            BINARY_EVAL_BASIS,
        );
        let coeffs = interpolate_to_coeffs(0, &evals);
        let round_sum = coeffs[0] + coeffs[0] + coeffs[1] + coeffs[2];
        assert_eq!(round_sum, current_claim, "trailing round-sum mismatch");
        current_claim = evaluate_coefficients(&coeffs, challenge);
        digest += digest_coefficients(&coeffs);
        let weights = [Fr::from_u64(1) - challenge, challenge];
        current_nodes = Some(fold_run_lists::<2>(nodes, weights));
        bind_address_weights::<2>(&mut bound_addresses, consumed_bits, weights);

        let final_val = current_nodes.expect("final binary fold");
        debug_assert_eq!(final_val.len(), 1);
        AddressPhaseOutput {
            final_claim: current_claim,
            message_digest: digest,
            bound_addresses,
            final_val: final_val.into_iter().next().expect("one final node"),
        }
    }
}

fn initial_bound_addresses(gamma: Fr) -> [[Fr; REGISTER_COUNT]; 3] {
    let gamma_squared = gamma * gamma;
    [
        [Fr::from_u64(1); REGISTER_COUNT],
        [gamma; REGISTER_COUNT],
        [gamma_squared; REGISTER_COUNT],
    ]
}

fn compute_message<const ARITY: usize, const POINTS: usize>(
    fixture: &RegistersAddressPhaseFixture,
    nodes: &[RunList],
    bound_addresses: &[[Fr; REGISTER_COUNT]; 3],
    consumed_bits: usize,
    basis: [[i64; ARITY]; POINTS],
) -> [Fr; POINTS] {
    let chunks = fixture.rows.len().div_ceil(MESSAGE_CHUNK_CYCLES);
    (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * MESSAGE_CHUNK_CYCLES;
            let end = (start + MESSAGE_CHUNK_CYCLES).min(fixture.rows.len());
            let mut cursors = nodes
                .iter()
                .map(|runs| runs.cursor_at(start))
                .collect::<Vec<_>>();
            let mut sums = [WideAccumulator::default(); POINTS];

            for cycle in start..end {
                let row = fixture.rows[cycle];
                let eq = fixture.eq_cycle[cycle];
                for (lane, &register) in row.lanes.iter().enumerate() {
                    if register == NO_REGISTER {
                        continue;
                    }
                    let node = (register as usize) >> consumed_bits;
                    let group = node / ARITY;
                    let digit = node % ARITY;
                    let values: [Fr; ARITY] = array::from_fn(|child| {
                        let child_node = group * ARITY + child;
                        nodes[child_node].value_at(cycle, &mut cursors[child_node])
                    });
                    let base = eq * bound_addresses[lane][register as usize];

                    for point in 0..POINTS {
                        let lane_basis = basis[point][digit];
                        if lane_basis == 0 {
                            continue;
                        }
                        let mut val = FrSmallScalarAccumulator::default();
                        for child in 0..ARITY {
                            val.fmadd_i64(values[child], basis[point][child]);
                        }
                        let val = val.reduce();
                        let relation_rhs = if lane == 0 {
                            fixture.rd_inc[cycle] + val
                        } else {
                            val
                        };
                        let scaled_rhs = if lane_basis == 1 {
                            relation_rhs
                        } else if lane_basis == -1 {
                            -relation_rhs
                        } else {
                            Fr::from_i64(lane_basis) * relation_rhs
                        };
                        sums[point].fmadd(base, scaled_rhs);
                    }
                }
            }

            sums.map(AdditiveAccumulator::reduce)
        })
        .reduce(
            || [Fr::from_u64(0); POINTS],
            |mut left, right| {
                for point in 0..POINTS {
                    left[point] += right[point];
                }
                left
            },
        )
}

fn fold_run_lists<const ARITY: usize>(children: &[RunList], weights: [Fr; ARITY]) -> Vec<RunList> {
    assert_eq!(children.len() % ARITY, 0);
    (0..children.len() / ARITY)
        .into_par_iter()
        .map(|group| fold_run_group(&children[group * ARITY..(group + 1) * ARITY], weights))
        .collect()
}

fn fold_run_group<const ARITY: usize>(children: &[RunList], weights: [Fr; ARITY]) -> RunList {
    let mut cursors = [0usize; ARITY];
    let mut current = array::from_fn(|child| children[child].runs[0].value);
    let mut runs = Vec::with_capacity(children.iter().map(|child| child.runs.len()).sum());
    runs.push(Run {
        start: 0,
        value: weighted_value(current, weights),
    });

    loop {
        let next_start = (0..ARITY)
            .filter_map(|child| children[child].runs.get(cursors[child] + 1))
            .map(|run| run.start)
            .min();
        let Some(next_start) = next_start else {
            break;
        };
        for child in 0..ARITY {
            if children[child]
                .runs
                .get(cursors[child] + 1)
                .is_some_and(|run| run.start == next_start)
            {
                cursors[child] += 1;
                current[child] = children[child].runs[cursors[child]].value;
            }
        }
        let value = weighted_value(current, weights);
        if runs.last().is_none_or(|run| run.value != value) {
            runs.push(Run {
                start: next_start,
                value,
            });
        }
    }

    RunList { runs }
}

fn weighted_value<const ARITY: usize>(values: [Fr; ARITY], weights: [Fr; ARITY]) -> Fr {
    let mut sum = WideAccumulator::default();
    for child in 0..ARITY {
        sum.fmadd(values[child], weights[child]);
    }
    sum.reduce()
}

fn bind_address_weights<const ARITY: usize>(
    bound_addresses: &mut [[Fr; REGISTER_COUNT]; 3],
    consumed_bits: usize,
    weights: [Fr; ARITY],
) {
    for lane in bound_addresses {
        for (register, value) in lane.iter_mut().enumerate() {
            let digit = (register >> consumed_bits) % ARITY;
            *value *= weights[digit];
        }
    }
}

fn compute_initial_claim(
    rows: &[AccessRow],
    eq_cycle: &[Fr],
    rd_inc: &[Fr],
    level_zero_val: &[RunList],
    gamma: Fr,
) -> Fr {
    let mut cursors = [0usize; REGISTER_COUNT];
    let mut claim = WideAccumulator::default();
    let gamma_squared = gamma * gamma;
    for (cycle, row) in rows.iter().enumerate() {
        for (lane, &register) in row.lanes.iter().enumerate() {
            if register == NO_REGISTER {
                continue;
            }
            let register = register as usize;
            let val = level_zero_val[register].value_at(cycle, &mut cursors[register]);
            let rhs = if lane == 0 { rd_inc[cycle] + val } else { val };
            let coefficient = match lane {
                0 => eq_cycle[cycle],
                1 => eq_cycle[cycle] * gamma,
                2 => eq_cycle[cycle] * gamma_squared,
                _ => unreachable!(),
            };
            claim.fmadd(coefficient, rhs);
        }
    }
    claim.reduce()
}

fn relation_after_address_phase(
    fixture: &RegistersAddressPhaseFixture,
    output: &AddressPhaseOutput,
) -> Fr {
    let mut cursor = 0;
    let mut claim = WideAccumulator::default();
    for cycle in 0..fixture.cycles {
        let val = output.final_val.value_at(cycle, &mut cursor);
        for (lane, &register) in fixture.rows[cycle].lanes.iter().enumerate() {
            if register == NO_REGISTER {
                continue;
            }
            let rhs = if lane == 0 {
                fixture.rd_inc[cycle] + val
            } else {
                val
            };
            claim.fmadd(
                fixture.eq_cycle[cycle] * output.bound_addresses[lane][register as usize],
                rhs,
            );
        }
    }
    claim.reduce()
}

fn push_post_state_run(runs: &mut RunList, cycle: usize, cycles: usize, post_value: u64) {
    let start = cycle + 1;
    if start >= cycles {
        return;
    }
    let value = Fr::from_u64(post_value);
    if runs
        .runs
        .last()
        .is_some_and(|run| run.start as usize == start)
    {
        let last = runs.runs.last_mut().expect("checked nonempty run list");
        last.value = value;
    } else if runs.runs.last().is_none_or(|run| run.value != value) {
        runs.runs.push(Run {
            start: start as u32,
            value,
        });
    }
}

fn build_level_zero_from_history(cycles: usize, writes: &[WriteEvent]) -> Vec<RunList> {
    let mut lists = (0..REGISTER_COUNT)
        .map(|_| RunList::zero())
        .collect::<Vec<_>>();
    for write in writes {
        if write.register == 0 {
            continue;
        }
        push_post_state_run(
            &mut lists[write.register as usize],
            write.cycle as usize,
            cycles,
            write.post_value,
        );
    }
    lists
}

fn default_binary_challenges() -> [Fr; 7] {
    [3, 5, 7, 11, 13, 17, 19].map(Fr::from_u64)
}

fn default_radix4_challenges() -> ([Fr; 3], Fr) {
    ([23, 29, 31].map(Fr::from_u64), Fr::from_u64(37))
}

fn radix4_challenge_weights(challenge: Fr) -> [Fr; 4] {
    let one = Fr::from_u64(1);
    let two = Fr::from_u64(2);
    let inv_two = two.inverse().expect("two is invertible");
    let inv_six = Fr::from_u64(6).inverse().expect("six is invertible");
    [
        -(challenge * (challenge - one) * (challenge - two)) * inv_six,
        (challenge + one) * (challenge - one) * (challenge - two) * inv_two,
        -((challenge + one) * challenge * (challenge - two)) * inv_two,
        (challenge + one) * challenge * (challenge - one) * inv_six,
    ]
}

fn radix4_eval_basis() -> [[i64; 4]; 7] {
    [-3, -2, -1, 0, 1, 2, 3].map(|z| {
        [
            -z * (z - 1) * (z - 2) / 6,
            (z + 1) * (z - 1) * (z - 2) / 2,
            -(z + 1) * z * (z - 2) / 2,
            (z + 1) * z * (z - 1) / 6,
        ]
    })
}

fn d_sum_degree_six(coeffs: &[Fr]) -> Fr {
    const POWER_SUMS: [i64; 7] = [4, 2, 6, 8, 18, 32, 66];
    assert_eq!(coeffs.len(), POWER_SUMS.len());
    let mut sum = FrSmallScalarAccumulator::default();
    for (&coefficient, power_sum) in coeffs.iter().zip(POWER_SUMS) {
        sum.fmadd_i64(coefficient, power_sum);
    }
    sum.reduce()
}

fn evaluate_coefficients(coeffs: &[Fr], point: Fr) -> Fr {
    coeffs
        .iter()
        .rev()
        .fold(Fr::from_u64(0), |value, coefficient| {
            value * point + coefficient
        })
}

fn digest_coefficients(coeffs: &[Fr]) -> Fr {
    coeffs
        .iter()
        .enumerate()
        .fold(Fr::from_u64(0), |digest, (index, coefficient)| {
            digest + *coefficient * Fr::from_u64(index as u64 + 1)
        })
}

fn synthetic_write_register(hash: u64) -> u8 {
    if hash % 10 < 8 {
        1 + ((hash >> 8) % 8) as u8
    } else {
        9 + ((hash >> 16) % 111) as u8
    }
}

fn synthetic_read_register(hash: u64) -> u8 {
    if hash % 10 < 7 {
        1 + ((hash >> 7) % 8) as u8
    } else {
        ((hash >> 15) % REGISTER_COUNT as u64) as u8
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn dense_val_from_history(cycles: usize, writes: &[WriteEvent]) -> Vec<Vec<Fr>> {
    let mut dense = (0..REGISTER_COUNT)
        .map(|_| vec![Fr::from_u64(0); cycles])
        .collect::<Vec<_>>();
    let mut state = [0u64; REGISTER_COUNT];
    let mut write_cursor = 0;
    for cycle in 0..cycles {
        for (register, column) in dense.iter_mut().enumerate() {
            column[cycle] = Fr::from_u64(state[register]);
        }
        while writes
            .get(write_cursor)
            .is_some_and(|write| write.cycle as usize == cycle)
        {
            let write = writes[write_cursor];
            if write.register != 0 {
                state[write.register as usize] = write.post_value;
            }
            write_cursor += 1;
        }
    }
    dense
}

fn dense_fold<const ARITY: usize>(children: &[Vec<Fr>], weights: [Fr; ARITY]) -> Vec<Vec<Fr>> {
    assert_eq!(children.len() % ARITY, 0);
    let cycles = children[0].len();
    (0..children.len() / ARITY)
        .map(|group| {
            (0..cycles)
                .map(|cycle| {
                    let values = array::from_fn(|child| children[group * ARITY + child][cycle]);
                    weighted_value(values, weights)
                })
                .collect()
        })
        .collect()
}

fn dense_address_weights_binary(challenges: [Fr; 7], gamma: Fr) -> [[Fr; REGISTER_COUNT]; 3] {
    let mut factors = [Fr::from_u64(0); REGISTER_COUNT];
    for target in 0..REGISTER_COUNT {
        let mut table = [Fr::from_u64(0); REGISTER_COUNT];
        table[target] = Fr::from_u64(1);
        let mut len = REGISTER_COUNT;
        for challenge in challenges {
            for index in 0..len / 2 {
                let even = table[2 * index];
                table[index] = even + challenge * (table[2 * index + 1] - even);
            }
            len /= 2;
        }
        factors[target] = table[0];
    }
    scale_address_factors(factors, gamma)
}

fn dense_address_weights_radix4(challenges: ([Fr; 3], Fr), gamma: Fr) -> [[Fr; REGISTER_COUNT]; 3] {
    let mut factors = [Fr::from_u64(0); REGISTER_COUNT];
    for target in 0..REGISTER_COUNT {
        let mut table = [Fr::from_u64(0); REGISTER_COUNT];
        table[target] = Fr::from_u64(1);
        let mut len = REGISTER_COUNT;
        for challenge in challenges.0 {
            let weights = radix4_challenge_weights(challenge);
            for index in 0..len / 4 {
                table[index] = weighted_value(
                    [
                        table[4 * index],
                        table[4 * index + 1],
                        table[4 * index + 2],
                        table[4 * index + 3],
                    ],
                    weights,
                );
            }
            len /= 4;
        }
        let even = table[0];
        table[0] = even + challenges.1 * (table[1] - even);
        factors[target] = table[0];
    }
    scale_address_factors(factors, gamma)
}

fn scale_address_factors(factors: [Fr; REGISTER_COUNT], gamma: Fr) -> [[Fr; REGISTER_COUNT]; 3] {
    [
        factors,
        factors.map(|factor| factor * gamma),
        factors.map(|factor| factor * gamma * gamma),
    ]
}

fn dense_final_claim(
    fixture: &RegistersAddressPhaseFixture,
    final_val: &[Fr],
    bound_addresses: &[[Fr; REGISTER_COUNT]; 3],
) -> Fr {
    let mut claim = WideAccumulator::default();
    for (cycle, &val) in final_val.iter().enumerate() {
        for (lane, &register) in fixture.rows[cycle].lanes.iter().enumerate() {
            if register == NO_REGISTER {
                continue;
            }
            let rhs = if lane == 0 {
                fixture.rd_inc[cycle] + val
            } else {
                val
            };
            claim.fmadd(
                fixture.eq_cycle[cycle] * bound_addresses[lane][register as usize],
                rhs,
            );
        }
    }
    claim.reduce()
}

fn boundary_fixture() -> (RegistersAddressPhaseFixture, Vec<WriteEvent>) {
    let cycles = 1 << 12;
    let gamma = Fr::from_u64(0x1_0000_01b3);
    let mut rows = (0..cycles)
        .map(|cycle| AccessRow {
            lanes: [
                if cycle % 5 == 0 {
                    1 + (cycle % 6) as u8
                } else {
                    NO_REGISTER
                },
                if cycle % 31 == 0 {
                    124
                } else {
                    (cycle % REGISTER_COUNT) as u8
                },
                ((cycle * 29 + 65) % REGISTER_COUNT) as u8,
            ],
        })
        .collect::<Vec<_>>();
    rows[17] = AccessRow { lanes: [4, 5, 6] };
    rows[33] = AccessRow {
        lanes: [4, 65, 119],
    };
    rows[34] = AccessRow {
        lanes: [NO_REGISTER, 7, 124],
    };

    let mut writes = Vec::new();
    for cycle in (0..cycles).step_by(5) {
        let register = 1 + (cycle % 6) as u8;
        writes.push(WriteEvent {
            cycle: cycle as u32,
            register,
            post_value: mix64(cycle as u64 + 1),
        });
    }
    writes.extend([
        WriteEvent {
            cycle: 0,
            register: 0,
            post_value: 999,
        },
        WriteEvent {
            cycle: 0,
            register: 1,
            post_value: 11,
        },
        WriteEvent {
            cycle: 17,
            register: 4,
            post_value: 41,
        },
        WriteEvent {
            cycle: 17,
            register: 5,
            post_value: 51,
        },
        WriteEvent {
            cycle: 18,
            register: 6,
            post_value: 61,
        },
        WriteEvent {
            cycle: 19,
            register: 4,
            post_value: 42,
        },
        WriteEvent {
            cycle: (cycles - 1) as u32,
            register: 2,
            post_value: 0xfeed,
        },
    ]);
    writes.sort_by_key(|write| write.cycle);

    let level_zero_val = build_level_zero_from_history(cycles, &writes);
    let eq_cycle = (0..cycles)
        .map(|cycle| Fr::from_u64((mix64(cycle as u64) & 0xff) + 1))
        .collect::<Vec<_>>();
    let mut rd_inc = vec![Fr::from_u64(0); cycles];
    let mut state = [0u64; REGISTER_COUNT];
    let mut cursor = 0;
    for cycle in 0..cycles {
        let primary = rows[cycle].lanes[0];
        while writes
            .get(cursor)
            .is_some_and(|write| write.cycle as usize == cycle)
        {
            let write = writes[cursor];
            if write.register != 0 {
                if write.register == primary {
                    rd_inc[cycle] = Fr::from_i128(
                        i128::from(write.post_value) - i128::from(state[write.register as usize]),
                    );
                }
                state[write.register as usize] = write.post_value;
            }
            cursor += 1;
        }
    }
    let initial_claim = compute_initial_claim(&rows, &eq_cycle, &rd_inc, &level_zero_val, gamma);
    (
        RegistersAddressPhaseFixture {
            cycles,
            rows,
            eq_cycle,
            rd_inc,
            level_zero_val,
            initial_claim,
            gamma,
        },
        writes,
    )
}

fn spot_check_folded_runs(runs: &[RunList], dense: &[Vec<Fr>], round: usize, rng: &mut u64) {
    assert_eq!(runs.len(), dense.len());
    for _ in 0..64 {
        *rng = mix64(*rng);
        let node = *rng as usize % runs.len();
        *rng = mix64(*rng);
        let cycle = *rng as usize % dense[node].len();
        let mut cursor = runs[node].cursor_at(cycle);
        assert_eq!(
            runs[node].value_at(cycle, &mut cursor),
            dense[node][cycle],
            "piecewise-constancy mismatch after fold {round}, node {node}, cycle {cycle}"
        );
    }
}

fn assert_binary_dense_parity(fixture: &RegistersAddressPhaseFixture, writes: &[WriteEvent]) {
    let challenges = default_binary_challenges();
    let output = fixture.run_binary_with(challenges);
    let mut dense = dense_val_from_history(fixture.cycles, writes);
    let mut runs: Option<Vec<RunList>> = None;
    let mut rng = 0x6269_6e61_7279_u64;
    for (round, challenge) in challenges.into_iter().enumerate() {
        let weights = [Fr::from_u64(1) - challenge, challenge];
        dense = dense_fold::<2>(&dense, weights);
        let current = runs.as_deref().unwrap_or(&fixture.level_zero_val);
        runs = Some(fold_run_lists::<2>(current, weights));
        spot_check_folded_runs(
            runs.as_deref().expect("folded runs"),
            &dense,
            round,
            &mut rng,
        );
    }
    assert_eq!(dense.len(), 1);
    let bound_addresses = dense_address_weights_binary(challenges, fixture.gamma);
    assert_eq!(output.bound_addresses, bound_addresses);
    assert_eq!(output.materialize_final_val(fixture.cycles), dense[0]);
    assert_eq!(
        output.final_claim,
        dense_final_claim(fixture, &dense[0], &bound_addresses)
    );
    assert_eq!(
        output.final_claim,
        relation_after_address_phase(fixture, &output)
    );
}

fn assert_radix_dense_parity(fixture: &RegistersAddressPhaseFixture, writes: &[WriteEvent]) {
    let challenges = default_radix4_challenges();
    let output = fixture.run_radix4_with(challenges);
    let mut dense = dense_val_from_history(fixture.cycles, writes);
    let mut runs: Option<Vec<RunList>> = None;
    let mut rng = 0x7261_6469_7834_u64;
    for (round, challenge) in challenges.0.into_iter().enumerate() {
        let weights = radix4_challenge_weights(challenge);
        dense = dense_fold::<4>(&dense, weights);
        let current = runs.as_deref().unwrap_or(&fixture.level_zero_val);
        runs = Some(fold_run_lists::<4>(current, weights));
        spot_check_folded_runs(
            runs.as_deref().expect("folded runs"),
            &dense,
            round,
            &mut rng,
        );
    }
    let weights = [Fr::from_u64(1) - challenges.1, challenges.1];
    dense = dense_fold::<2>(&dense, weights);
    let current = runs.as_deref().expect("three radix folds");
    runs = Some(fold_run_lists::<2>(current, weights));
    spot_check_folded_runs(runs.as_deref().expect("folded runs"), &dense, 3, &mut rng);
    assert_eq!(dense.len(), 1);
    let bound_addresses = dense_address_weights_radix4(challenges, fixture.gamma);
    assert_eq!(output.bound_addresses, bound_addresses);
    assert_eq!(output.materialize_final_val(fixture.cycles), dense[0]);
    assert_eq!(
        output.final_claim,
        dense_final_claim(fixture, &dense[0], &bound_addresses)
    );
    assert_eq!(
        output.final_claim,
        relation_after_address_phase(fixture, &output)
    );
}

fn assert_cross_arm_domain_point_parity(fixture: &RegistersAddressPhaseFixture) {
    let packed_digits = [0usize, 3, 1];
    let packed = packed_digits.map(|digit| Fr::from_i64(digit as i64 - 1));
    let single = Fr::from_u64(1);
    let binary = [
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(0),
        single,
    ];
    let binary_output = fixture.run_binary_with(binary);
    let radix_output = fixture.run_radix4_with((packed, single));
    assert_eq!(binary_output.bound_addresses, radix_output.bound_addresses);
    assert_eq!(binary_output.final_val, radix_output.final_val);
    assert_eq!(binary_output.final_claim, radix_output.final_claim);
}

pub fn assert_small_scale_parity() {
    let (fixture, writes) = boundary_fixture();
    let dense = dense_val_from_history(fixture.cycles, &writes);

    assert_eq!(dense[0], vec![Fr::from_u64(0); fixture.cycles]);
    assert_eq!(dense[1][0], Fr::from_u64(0));
    assert_eq!(dense[1][1], Fr::from_u64(11));
    assert_eq!(dense[4][17], dense[4][16]);
    assert_eq!(dense[4][18], Fr::from_u64(41));
    assert_eq!(dense[5][18], Fr::from_u64(51));
    assert_eq!(dense[2][fixture.cycles - 1], dense[2][fixture.cycles - 2]);
    assert_eq!(dense[7], vec![Fr::from_u64(0); fixture.cycles]);

    let coincident = fold_run_group::<4>(
        &fixture.level_zero_val[4..8],
        [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
    );
    assert_eq!(
        coincident.runs.iter().filter(|run| run.start == 18).count(),
        1,
        "coincident child writes must create one atomic parent run"
    );

    assert_binary_dense_parity(&fixture, &writes);
    assert_radix_dense_parity(&fixture, &writes);
    assert_cross_arm_domain_point_parity(&fixture);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_address_phase_matches_dense_temporal_reference() {
        assert_small_scale_parity();
    }

    #[test]
    fn radix4_d_sum_functional_matches_direct_domain_sum() {
        let coeffs = [2, 3, 5, 7, 11, 13, 17].map(Fr::from_u64);
        let direct = [-1, 0, 1, 2]
            .map(|point| evaluate_coefficients(&coeffs, Fr::from_i64(point)))
            .into_iter()
            .sum::<Fr>();
        assert_eq!(d_sum_degree_six(&coeffs), direct);
    }
}
