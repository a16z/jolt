use jolt_field::Field;
use thiserror::Error;

use super::super::registers::{
    CertifiedRegisterOwner, REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS,
};
use super::model::{
    RegisterClaimComponents, RegisterFamilyCarrier, RegisterFamilyModelError, RegisterValuePoint,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuadraticSamples<F> {
    pub at_0: F,
    pub at_1: F,
    pub at_2: F,
}

impl<F: Field> QuadraticSamples<F> {
    pub fn claim_identity(self) -> F {
        self.at_0 + self.at_1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CubicSamples<F> {
    pub at_0: F,
    pub at_1: F,
    pub at_2: F,
    pub at_3: F,
}

impl<F: Field> CubicSamples<F> {
    pub fn claim_identity(self) -> F {
        self.at_0 + self.at_1
    }

    pub fn evaluate(self, point: F) -> Option<F> {
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);
        let inverse_two = two.inverse()?;
        let inverse_six = F::from_u64(6).inverse()?;
        let l0 = (F::zero() - (point - one) * (point - two) * (point - three)) * inverse_six;
        let l1 = point * (point - two) * (point - three) * inverse_two;
        let l2 = (F::zero() - point * (point - one) * (point - three)) * inverse_two;
        let l3 = point * (point - one) * (point - two) * inverse_six;
        Some(self.at_0 * l0 + self.at_1 * l1 + self.at_2 * l2 + self.at_3 * l3)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimOutputValues<F> {
    pub rd_write_value: F,
    pub rs1_value: F,
    pub rs2_value: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimOracleOutput<F> {
    pub input_claim: F,
    pub messages: Vec<QuadraticSamples<F>>,
    pub outputs: ClaimOutputValues<F>,
    pub opening_point: Vec<F>,
    pub terminal_claim: F,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueBoundRow<F> {
    pub rd_inc: F,
    pub rd_wa: F,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueFirstMessage<F> {
    pub samples: CubicSamples<F>,
    pub relation_claim: F,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueFirstTransition<F> {
    pub next_message: Option<CubicSamples<F>>,
    pub bound_claim: F,
    pub rows_emitted: usize,
}

/// Builds the three prefix Q tables by replaying CSR events in register-state
/// order. The only allocation proportional to the trace is `O(sqrt(T))`.
pub fn claim_components_from_owner<F: Field>(
    owner: &RegisterFamilyCarrier,
    tau: &[F],
) -> Result<RegisterClaimComponents<F>, OwnerOracleError> {
    let geometry = owner.geometry();
    if tau.len() != geometry.log_t() {
        return Err(RegisterFamilyModelError::PointLength {
            point: "claim tau",
            expected: geometry.log_t(),
            got: tau.len(),
        }
        .into());
    }
    let tau_hi = &tau[..geometry.suffix_bits()];
    let suffix_eq = eq_evaluations(tau_hi);
    let mut components = core::array::from_fn(|_| vec![F::zero(); geometry.prefix_elements()]);
    let low_mask = geometry.prefix_elements() - 1;

    for_each_owner_event(owner.owner(), |event| {
        let (column, cycle, value) = match event {
            OwnerEvent::Rs1 { cycle, value } => (1, cycle, value),
            OwnerEvent::Rs2 { cycle, value } => (2, cycle, value),
            OwnerEvent::Rd { cycle, post_value } => (0, cycle, post_value),
        };
        let x_lo = cycle & low_mask;
        let x_hi = cycle >> geometry.prefix_bits();
        components[column][x_lo] += suffix_eq[x_hi] * F::from_u64(value);
    });

    Ok(RegisterClaimComponents::new(
        owner,
        tau.to_vec(),
        components,
    )?)
}

/// Runs all claim-reduction rounds using only `O(sqrt(T))` field tables.
pub fn claim_sumcheck_oracle<F: Field>(
    owner: &RegisterFamilyCarrier,
    components: &RegisterClaimComponents<F>,
    gamma: F,
    challenges: &[F],
) -> Result<ClaimOracleOutput<F>, OwnerOracleError> {
    components.validate_owner(owner)?;
    let geometry = owner.geometry();
    if challenges.len() != geometry.log_t() {
        return Err(RegisterFamilyModelError::PointLength {
            point: "claim challenges",
            expected: geometry.log_t(),
            got: challenges.len(),
        }
        .into());
    }

    let tau = components.tau();
    let tau_hi = &tau[..geometry.suffix_bits()];
    let tau_lo = &tau[geometry.suffix_bits()..];
    let mut p = eq_evaluations(tau_lo);
    let opening_values = components
        .components()
        .each_ref()
        .map(|component| dot(&p, component));
    let gamma_sq = gamma * gamma;
    let input_claim = opening_values[0] + gamma * opening_values[1] + gamma_sq * opening_values[2];
    let mut q = components.combined_q(gamma);
    let mut current_claim = input_claim;
    let mut messages = Vec::with_capacity(geometry.log_t());

    for (round, &challenge) in challenges[..geometry.prefix_bits()].iter().enumerate() {
        let message = quadratic_product_message(&p, &q);
        require_claim_identity(round, current_claim, message.claim_identity())?;
        messages.push(message);
        bind_low(&mut p, challenge);
        bind_low(&mut q, challenge);
        current_claim = dot(&p, &q);
    }

    let mut columns = claim_midpoint_columns(owner, &challenges[..geometry.prefix_bits()]);
    let mut equality = eq_evaluations(tau_hi);
    for value in &mut equality {
        *value *= p[0];
    }
    let midpoint_claim = dense_claim(&equality, &columns, gamma, gamma_sq);
    if midpoint_claim != current_claim {
        return Err(OwnerOracleError::MidpointClaimMismatch);
    }

    for (offset, &challenge) in challenges[geometry.prefix_bits()..].iter().enumerate() {
        let round = geometry.prefix_bits() + offset;
        let message = quadratic_claim_message(&equality, &columns, gamma, gamma_sq);
        require_claim_identity(round, current_claim, message.claim_identity())?;
        messages.push(message);
        bind_low(&mut equality, challenge);
        for column in &mut columns {
            bind_low(column, challenge);
        }
        current_claim = dense_claim(&equality, &columns, gamma, gamma_sq);
    }

    let outputs = ClaimOutputValues {
        rd_write_value: columns[0][0],
        rs1_value: columns[1][0],
        rs2_value: columns[2][0],
    };
    let opening_point = components.opening_point(challenges)?;
    let terminal_claim = eq_mle(&opening_point, tau)
        * (outputs.rd_write_value + gamma * outputs.rs1_value + gamma_sq * outputs.rs2_value);
    if terminal_claim != current_claim {
        return Err(OwnerOracleError::TerminalClaimMismatch);
    }

    Ok(ClaimOracleOutput {
        input_claim,
        messages,
        outputs,
        opening_point,
        terminal_claim,
    })
}

/// Computes value-evaluation message zero directly from the rd CSR view.
pub fn value_first_message_oracle<F: Field>(
    owner: &RegisterFamilyCarrier,
    point: &RegisterValuePoint<F>,
) -> Result<ValueFirstMessage<F>, OwnerOracleError> {
    point.validate_owner(owner)?;
    let address_eq = eq_evaluations(point.address());
    let mut sums = [F::zero(); 4];
    for_each_rd_block(
        owner.owner(),
        |block_start, block_len, increments, registers| {
            for local_pair in 0..block_len / 2 {
                let local = 2 * local_pair;
                let cycle = block_start + local;
                let inc = [
                    F::from_i128(increments[local]),
                    F::from_i128(increments[local + 1]),
                ];
                let wa = [
                    address_value(registers[local], &address_eq),
                    address_value(registers[local + 1], &address_eq),
                ];
                let lt = [
                    lt_at_boolean_index(cycle, point.cycle()),
                    lt_at_boolean_index(cycle + 1, point.cycle()),
                ];
                accumulate_cubic_pair(&mut sums, inc, wa, lt);
            }
        },
    );
    let samples = cubic_samples(sums);
    Ok(ValueFirstMessage {
        relation_claim: samples.claim_identity(),
        samples,
    })
}

/// Applies the first host challenge and emits the exact `T/2` protocol state
/// through `visit`. No event, index, or increment table is retained.
pub fn value_first_transition_oracle<F: Field>(
    owner: &RegisterFamilyCarrier,
    point: &RegisterValuePoint<F>,
    first_message: CubicSamples<F>,
    challenge: F,
    mut visit: impl FnMut(usize, ValueBoundRow<F>),
) -> Result<ValueFirstTransition<F>, OwnerOracleError> {
    point.validate_owner(owner)?;
    let address_eq = eq_evaluations(point.address());
    let mut bound_claim = F::zero();
    let mut next_sums = [F::zero(); 4];
    let mut pending: Option<(ValueBoundRow<F>, F)> = None;
    let mut rows_emitted = 0usize;

    for_each_rd_block(
        owner.owner(),
        |block_start, block_len, increments, registers| {
            for local_pair in 0..block_len / 2 {
                let local = 2 * local_pair;
                let cycle = block_start + local;
                let inc = bind_pair(
                    F::from_i128(increments[local]),
                    F::from_i128(increments[local + 1]),
                    challenge,
                );
                let wa = bind_pair(
                    address_value(registers[local], &address_eq),
                    address_value(registers[local + 1], &address_eq),
                    challenge,
                );
                let lt = bind_pair(
                    lt_at_boolean_index(cycle, point.cycle()),
                    lt_at_boolean_index(cycle + 1, point.cycle()),
                    challenge,
                );
                let row = ValueBoundRow {
                    rd_inc: inc,
                    rd_wa: wa,
                };
                let output_index = block_start / 2 + local_pair;
                visit(output_index, row);
                rows_emitted += 1;
                bound_claim += inc * wa * lt;

                if let Some((low, low_lt)) = pending.take() {
                    accumulate_cubic_pair(
                        &mut next_sums,
                        [low.rd_inc, row.rd_inc],
                        [low.rd_wa, row.rd_wa],
                        [low_lt, lt],
                    );
                } else {
                    pending = Some((row, lt));
                }
            }
        },
    );

    let interpolated = first_message
        .evaluate(challenge)
        .ok_or(OwnerOracleError::NoninvertibleInterpolationConstant)?;
    if interpolated != bound_claim {
        return Err(OwnerOracleError::FirstBindClaimMismatch);
    }
    let next_message = if rows_emitted == 1 {
        None
    } else {
        if pending.is_some() {
            return Err(OwnerOracleError::OddBoundState { rows_emitted });
        }
        let message = cubic_samples(next_sums);
        if message.claim_identity() != bound_claim {
            return Err(OwnerOracleError::SecondMessageClaimMismatch);
        }
        Some(message)
    };

    Ok(ValueFirstTransition {
        next_message,
        bound_claim,
        rows_emitted,
    })
}

#[derive(Clone, Copy)]
enum OwnerEvent {
    Rs1 { cycle: usize, value: u64 },
    Rs2 { cycle: usize, value: u64 },
    Rd { cycle: usize, post_value: u64 },
}

fn for_each_owner_event(owner: &CertifiedRegisterOwner, mut visit: impl FnMut(OwnerEvent)) {
    let csr = owner.csr();
    let parts = csr.parts();
    for block in 0..csr.block_count() {
        let block_start = block * REGISTER_CSR_BLOCK_CYCLES;
        for register in 0..REGISTER_CSR_COLUMNS {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let rs1 = event_range(&parts.rs1_offsets, header);
            let rs2 = event_range(&parts.rs2_offsets, header);
            let rd = event_range(&parts.rd_offsets, header);
            let rs1_positions = &parts.rs1_positions[rs1];
            let rs2_positions = &parts.rs2_positions[rs2];
            let rd_positions = &parts.rd_positions[rd.clone()];
            let rd_posts = &parts.rd_post_values[rd];
            let mut rs1_index = 0usize;
            let mut rs2_index = 0usize;
            let mut rd_index = 0usize;
            let mut state = parts.start_values[header];

            loop {
                let next = [
                    rs1_positions.get(rs1_index).copied(),
                    rs2_positions.get(rs2_index).copied(),
                    rd_positions.get(rd_index).copied(),
                ]
                .into_iter()
                .flatten()
                .min();
                let Some(position) = next else {
                    break;
                };
                let cycle = block_start + usize::from(position);
                if rs1_positions.get(rs1_index) == Some(&position) {
                    visit(OwnerEvent::Rs1 {
                        cycle,
                        value: state,
                    });
                    rs1_index += 1;
                }
                if rs2_positions.get(rs2_index) == Some(&position) {
                    visit(OwnerEvent::Rs2 {
                        cycle,
                        value: state,
                    });
                    rs2_index += 1;
                }
                if rd_positions.get(rd_index) == Some(&position) {
                    let post_value = rd_posts[rd_index];
                    visit(OwnerEvent::Rd { cycle, post_value });
                    state = post_value;
                    rd_index += 1;
                }
            }
        }
    }
}

fn claim_midpoint_columns<F: Field>(
    owner: &RegisterFamilyCarrier,
    prefix_challenges: &[F],
) -> [Vec<F>; 3] {
    let geometry = owner.geometry();
    debug_assert_eq!(prefix_challenges.len(), geometry.prefix_bits());
    let prefix_point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
    let weights = eq_evaluations(&prefix_point);
    let low_mask = geometry.prefix_elements() - 1;
    let mut columns = core::array::from_fn(|_| vec![F::zero(); geometry.suffix_elements()]);
    for_each_owner_event(owner.owner(), |event| {
        let (column, cycle, value) = match event {
            OwnerEvent::Rs1 { cycle, value } => (1, cycle, value),
            OwnerEvent::Rs2 { cycle, value } => (2, cycle, value),
            OwnerEvent::Rd { cycle, post_value } => (0, cycle, post_value),
        };
        let x_lo = cycle & low_mask;
        let x_hi = cycle >> geometry.prefix_bits();
        columns[column][x_hi] += weights[x_lo] * F::from_u64(value);
    });
    columns
}

fn for_each_rd_block(
    owner: &CertifiedRegisterOwner,
    mut visit: impl FnMut(
        usize,
        usize,
        &[i128; REGISTER_CSR_BLOCK_CYCLES],
        &[u8; REGISTER_CSR_BLOCK_CYCLES],
    ),
) {
    let csr = owner.csr();
    let parts = csr.parts();
    for block in 0..csr.block_count() {
        let block_start = block * REGISTER_CSR_BLOCK_CYCLES;
        let block_len = (csr.cycles() - block_start).min(REGISTER_CSR_BLOCK_CYCLES);
        let mut increments = [0i128; REGISTER_CSR_BLOCK_CYCLES];
        let mut registers = [u8::MAX; REGISTER_CSR_BLOCK_CYCLES];
        for register in 0..REGISTER_CSR_COLUMNS {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let rd = event_range(&parts.rd_offsets, header);
            let positions = &parts.rd_positions[rd.clone()];
            let posts = &parts.rd_post_values[rd];
            let mut previous = parts.start_values[header];
            for (&position, &post) in positions.iter().zip(posts) {
                let position = usize::from(position);
                debug_assert_eq!(registers[position], u8::MAX);
                increments[position] = i128::from(post) - i128::from(previous);
                registers[position] = register as u8;
                previous = post;
            }
        }
        visit(block_start, block_len, &increments, &registers);
    }
}

fn quadratic_product_message<F: Field>(left: &[F], right: &[F]) -> QuadraticSamples<F> {
    debug_assert_eq!(left.len(), right.len());
    let mut sums = [F::zero(); 3];
    for index in 0..left.len() / 2 {
        let row = 2 * index;
        let left_samples = quadratic_linear_samples(left[row], left[row + 1]);
        let right_samples = quadratic_linear_samples(right[row], right[row + 1]);
        for sample in 0..3 {
            sums[sample] += left_samples[sample] * right_samples[sample];
        }
    }
    QuadraticSamples {
        at_0: sums[0],
        at_1: sums[1],
        at_2: sums[2],
    }
}

fn quadratic_claim_message<F: Field>(
    equality: &[F],
    columns: &[Vec<F>; 3],
    gamma: F,
    gamma_sq: F,
) -> QuadraticSamples<F> {
    let mut sums = [F::zero(); 3];
    for index in 0..equality.len() / 2 {
        let row = 2 * index;
        let eq = quadratic_linear_samples(equality[row], equality[row + 1]);
        let rd = quadratic_linear_samples(columns[0][row], columns[0][row + 1]);
        let rs1 = quadratic_linear_samples(columns[1][row], columns[1][row + 1]);
        let rs2 = quadratic_linear_samples(columns[2][row], columns[2][row + 1]);
        for sample in 0..3 {
            sums[sample] +=
                eq[sample] * (rd[sample] + gamma * rs1[sample] + gamma_sq * rs2[sample]);
        }
    }
    QuadraticSamples {
        at_0: sums[0],
        at_1: sums[1],
        at_2: sums[2],
    }
}

fn dense_claim<F: Field>(equality: &[F], columns: &[Vec<F>; 3], gamma: F, gamma_sq: F) -> F {
    (0..equality.len()).fold(F::zero(), |sum, index| {
        sum + equality[index]
            * (columns[0][index] + gamma * columns[1][index] + gamma_sq * columns[2][index])
    })
}

fn require_claim_identity<F: Field>(
    round: usize,
    expected: F,
    got: F,
) -> Result<(), OwnerOracleError> {
    if expected == got {
        Ok(())
    } else {
        Err(OwnerOracleError::RoundClaimMismatch { round })
    }
}

fn eq_evaluations<F: Field>(point: &[F]) -> Vec<F> {
    let mut values = vec![F::one()];
    for &coordinate in point {
        let mut next = Vec::with_capacity(2 * values.len());
        for value in values {
            next.push(value * (F::one() - coordinate));
            next.push(value * coordinate);
        }
        values = next;
    }
    values
}

fn eq_mle<F: Field>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).fold(F::one(), |product, (&x, &y)| {
        product * (x * y + (F::one() - x) * (F::one() - y))
    })
}

fn dot<F: Field>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(F::zero(), |sum, (&x, &y)| sum + x * y)
}

fn bind_low<F: Field>(values: &mut Vec<F>, challenge: F) {
    let half = values.len() / 2;
    for index in 0..half {
        let low = values[2 * index];
        values[index] = bind_pair(low, values[2 * index + 1], challenge);
    }
    values.truncate(half);
}

fn bind_pair<F: Field>(low: F, high: F, challenge: F) -> F {
    low + challenge * (high - low)
}

fn quadratic_linear_samples<F: Field>(low: F, high: F) -> [F; 3] {
    [low, high, high + high - low]
}

fn cubic_linear_samples<F: Field>(low: F, high: F) -> [F; 4] {
    let delta = high - low;
    let at_2 = high + delta;
    [low, high, at_2, at_2 + delta]
}

fn accumulate_cubic_pair<F: Field>(sums: &mut [F; 4], inc: [F; 2], wa: [F; 2], lt: [F; 2]) {
    let inc = cubic_linear_samples(inc[0], inc[1]);
    let wa = cubic_linear_samples(wa[0], wa[1]);
    let lt = cubic_linear_samples(lt[0], lt[1]);
    for sample in 0..4 {
        sums[sample] += inc[sample] * wa[sample] * lt[sample];
    }
}

fn cubic_samples<F: Copy>(sums: [F; 4]) -> CubicSamples<F> {
    CubicSamples {
        at_0: sums[0],
        at_1: sums[1],
        at_2: sums[2],
        at_3: sums[3],
    }
}

fn address_value<F: Field>(register: u8, address_eq: &[F]) -> F {
    if register == u8::MAX {
        F::zero()
    } else {
        address_eq[usize::from(register)]
    }
}

fn lt_at_boolean_index<F: Field>(index: usize, point: &[F]) -> F {
    let mut lt = F::zero();
    let mut eq_prefix = F::one();
    for (position, &coordinate) in point.iter().enumerate() {
        let bit = (index >> (point.len() - 1 - position)) & 1;
        if bit == 0 {
            lt += coordinate * eq_prefix;
            eq_prefix *= F::one() - coordinate;
        } else {
            eq_prefix *= coordinate;
        }
    }
    lt
}

fn event_range(offsets: &[u32], header: usize) -> core::ops::Range<usize> {
    offsets[header] as usize..offsets[header + 1] as usize
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OwnerOracleError {
    #[error(transparent)]
    Model(#[from] RegisterFamilyModelError),
    #[error("claim round {round} does not preserve the running claim")]
    RoundClaimMismatch { round: usize },
    #[error("claim midpoint does not match the bound prefix claim")]
    MidpointClaimMismatch,
    #[error("claim terminal expression does not match the fully bound claim")]
    TerminalClaimMismatch,
    #[error("field constants 2 and 6 must be invertible")]
    NoninvertibleInterpolationConstant,
    #[error("value first bind does not evaluate the first round polynomial")]
    FirstBindClaimMismatch,
    #[error("value first transition emitted an odd state of {rows_emitted} rows")]
    OddBoundState { rows_emitted: usize },
    #[error("value second message does not preserve the first bound claim")]
    SecondMessageClaimMismatch,
}

const _: () = assert!(REGISTER_CSR_COLUMNS <= u8::MAX as usize);
