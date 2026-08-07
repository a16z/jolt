use jolt_field::Field;
use thiserror::Error;

use super::owner::{BytecodeReadRafOwner, OwnerError};
use super::relation::{
    address_summand, canonical_opening_point, cycle_summand, resolve_stage_value, AddressOutput,
    AddressRoundMessage, CycleOutput, CycleRoundMessage, RelationError, RelationWeights,
    COMMITTED_CHUNK_BITS, RAW_VALUE_TABLES, RA_FACTORS, STAGES, STAGE_VALUE_SOURCES,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafInputs<F> {
    stage_points: [Vec<F>; STAGES],
    raw_value_tables: [Vec<F>; RAW_VALUE_TABLES],
    gamma: F,
    entry_index: usize,
}

impl<F> BytecodeReadRafInputs<F> {
    pub const fn new(
        stage_points: [Vec<F>; STAGES],
        raw_value_tables: [Vec<F>; RAW_VALUE_TABLES],
        gamma: F,
        entry_index: usize,
    ) -> Self {
        Self {
            stage_points,
            raw_value_tables,
            gamma,
            entry_index,
        }
    }

    pub const fn stage_points(&self) -> &[Vec<F>; STAGES] {
        &self.stage_points
    }

    pub const fn raw_value_tables(&self) -> &[Vec<F>; RAW_VALUE_TABLES] {
        &self.raw_value_tables
    }

    pub const fn gamma(&self) -> &F {
        &self.gamma
    }

    pub const fn entry_index(&self) -> usize {
        self.entry_index
    }
}

pub struct DenseAddressOracle<F> {
    pushforwards: [Vec<F>; STAGES],
    raw_values: [Vec<F>; RAW_VALUE_TABLES],
    identity: Vec<F>,
    entry_trace: Vec<F>,
    entry_expected: Vec<F>,
    weights: RelationWeights<F>,
    rounds: usize,
    round: usize,
    initial_claim: F,
    current_claim: F,
    pending: Option<AddressRoundMessage<F>>,
    challenges: Vec<F>,
}

impl<F: Field> DenseAddressOracle<F> {
    /// Builds every address table by a direct cycle-order scan.
    pub fn new(
        owner: &BytecodeReadRafOwner,
        inputs: &BytecodeReadRafInputs<F>,
    ) -> Result<Self, OracleError> {
        owner.verify_integrity()?;
        validate_inputs(owner, inputs)?;
        let receipt = owner.receipt();
        let stage_eq = inputs
            .stage_points
            .iter()
            .map(|point| dense_eq_table(point))
            .collect::<Result<Vec<_>, _>>()?;
        let mut pushforwards: [Vec<F>; STAGES] =
            core::array::from_fn(|_| vec![F::zero(); receipt.addresses()]);
        for (cycle, row) in owner.rows().iter().copied().enumerate() {
            for stage in 0..STAGES {
                let mut contribution = stage_eq[stage][cycle];
                if stage >= super::relation::BASE_STAGES {
                    contribution *= row.fused_increment().field::<F>();
                }
                pushforwards[stage][row.push_pc()] += contribution;
            }
        }

        let mut entry_trace = vec![F::zero(); receipt.addresses()];
        let first_pc = owner
            .rows()
            .first()
            .copied()
            .ok_or(OracleError::EmptyOwner)?
            .push_pc();
        entry_trace[first_pc] = F::one();
        let mut entry_expected = vec![F::zero(); receipt.addresses()];
        entry_expected[inputs.entry_index] = F::one();
        let identity = (0..receipt.addresses())
            .map(|index| F::from_u64(index as u64))
            .collect::<Vec<_>>();
        let weights = RelationWeights::new(inputs.gamma);
        let mut oracle = Self {
            pushforwards,
            raw_values: inputs.raw_value_tables.clone(),
            identity,
            entry_trace,
            entry_expected,
            weights,
            rounds: receipt.log_k(),
            round: 0,
            initial_claim: F::zero(),
            current_claim: F::zero(),
            pending: None,
            challenges: Vec::with_capacity(receipt.log_k()),
        };
        let claim = oracle.sum_current_domain()?;
        oracle.initial_claim = claim;
        oracle.current_claim = claim;
        Ok(oracle)
    }

    pub const fn num_rounds(&self) -> usize {
        self.rounds
    }

    pub const fn round(&self) -> usize {
        self.round
    }

    pub const fn initial_claim(&self) -> F {
        self.initial_claim
    }

    pub const fn current_claim(&self) -> F {
        self.current_claim
    }

    pub fn current_len(&self) -> usize {
        self.identity.len()
    }

    pub fn pushforward(&self, stage: usize) -> Result<&[F], OracleError> {
        self.pushforwards
            .get(stage)
            .map(Vec::as_slice)
            .ok_or(OracleError::InvalidStage(stage))
    }

    pub fn message(&mut self) -> Result<AddressRoundMessage<F>, OracleError> {
        if self.round >= self.rounds {
            return Err(OracleError::AlreadyFullyBound);
        }
        if self.pending.is_some() {
            return Err(OracleError::MessageAlreadyPending);
        }
        self.validate_address_state()?;
        let pairs = self.identity.len() / 2;
        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        for pair in 0..pairs {
            at_zero += self.address_pair_summand(pair, F::zero())?;
            at_two += self.address_pair_summand(pair, F::from_u64(2))?;
        }
        let message = AddressRoundMessage::new(at_zero, at_two);
        self.pending = Some(message);
        Ok(message)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), OracleError> {
        if self.round >= self.rounds {
            return Err(OracleError::AlreadyFullyBound);
        }
        let message = self.pending.take().ok_or(OracleError::MessageRequired)?;
        let next_claim = message.evaluate(self.current_claim, challenge)?;
        for table in &mut self.pushforwards {
            bind_dense(table, challenge)?;
        }
        for table in &mut self.raw_values {
            bind_dense(table, challenge)?;
        }
        for table in [
            &mut self.identity,
            &mut self.entry_trace,
            &mut self.entry_expected,
        ] {
            bind_dense(table, challenge)?;
        }
        self.round = self.round.checked_add(1).ok_or(OracleError::Overflow)?;
        self.current_claim = next_claim;
        self.challenges.push(challenge);
        self.validate_address_state()?;
        Ok(())
    }

    pub fn output(&self) -> Result<AddressOutput<F>, OracleError> {
        if self.round != self.rounds {
            return Err(OracleError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        if self.pending.is_some() {
            return Err(OracleError::MessageAlreadyPending);
        }
        self.validate_address_state()?;
        let intermediate = self.address_value(0)?;
        if intermediate != self.current_claim {
            return Err(OracleError::TerminalClaimMismatch);
        }
        Ok(AddressOutput {
            intermediate,
            raw_values: core::array::from_fn(|index| self.raw_values[index][0]),
            r_address: canonical_opening_point(&self.challenges),
        })
    }

    fn address_pair_summand(&self, pair: usize, point: F) -> Result<F, OracleError> {
        let pushforwards =
            core::array::from_fn(|stage| interpolate_pair(&self.pushforwards[stage], pair, point));
        let raw_values =
            core::array::from_fn(|table| interpolate_pair(&self.raw_values[table], pair, point));
        address_summand(
            &pushforwards,
            &raw_values,
            interpolate_pair(&self.identity, pair, point),
            interpolate_pair(&self.entry_trace, pair, point),
            interpolate_pair(&self.entry_expected, pair, point),
            &self.weights,
        )
        .map_err(OracleError::from)
    }

    fn address_value(&self, index: usize) -> Result<F, OracleError> {
        let pushforwards = core::array::from_fn(|stage| self.pushforwards[stage][index]);
        let raw_values = core::array::from_fn(|table| self.raw_values[table][index]);
        address_summand(
            &pushforwards,
            &raw_values,
            self.identity[index],
            self.entry_trace[index],
            self.entry_expected[index],
            &self.weights,
        )
        .map_err(OracleError::from)
    }

    fn sum_current_domain(&self) -> Result<F, OracleError> {
        (0..self.identity.len()).try_fold(F::zero(), |claim, index| {
            Ok(claim + self.address_value(index)?)
        })
    }

    fn validate_address_state(&self) -> Result<(), OracleError> {
        let expected = remaining_domain(self.rounds, self.round)?;
        if expected == 0
            || self.identity.len() != expected
            || self.entry_trace.len() != expected
            || self.entry_expected.len() != expected
            || self
                .pushforwards
                .iter()
                .any(|table| table.len() != expected)
            || self.raw_values.iter().any(|table| table.len() != expected)
        {
            return Err(OracleError::InvalidDenseState);
        }
        Ok(())
    }
}

pub struct DenseCycleOracle<F> {
    base_coefficient: Vec<F>,
    fused_coefficient: Vec<F>,
    fused_increment: Vec<F>,
    bytecode_ra: [Vec<F>; RA_FACTORS],
    rounds: usize,
    round: usize,
    initial_claim: F,
    current_claim: F,
    pending: Option<CycleRoundMessage<F>>,
    challenges: Vec<F>,
}

impl<F: Field> DenseCycleOracle<F> {
    /// Reconstructs all five cycle columns directly from cycle-order rows.
    pub fn new(
        owner: &BytecodeReadRafOwner,
        inputs: &BytecodeReadRafInputs<F>,
        address: &AddressOutput<F>,
    ) -> Result<Self, OracleError> {
        owner.verify_integrity()?;
        validate_inputs(owner, inputs)?;
        let receipt = owner.receipt();
        if address.r_address.len() != receipt.log_k() {
            return Err(OracleError::AddressPointLength {
                expected: receipt.log_k(),
                got: address.r_address.len(),
            });
        }
        for (table, expected) in inputs
            .raw_value_tables
            .iter()
            .zip(address.raw_values.iter().copied())
        {
            if evaluate_dense(table, &address.r_address)? != expected {
                return Err(OracleError::RawValueHandoffMismatch);
            }
        }

        let stage_eq = inputs
            .stage_points
            .iter()
            .map(|point| dense_eq_table(point))
            .collect::<Result<Vec<_>, _>>()?;
        let chunk_points = committed_chunks(&address.r_address)?;
        let chunk_eq = [
            dense_eq_table(&chunk_points[0])?,
            dense_eq_table(&chunk_points[1])?,
        ];
        let weights = RelationWeights::new(inputs.gamma);
        let identity = identity_mle(&address.r_address);
        let entry_scalar = eq_index(&address.r_address, inputs.entry_index)?;
        let mut base_coefficient = vec![F::zero(); receipt.cycles()];
        let mut fused_coefficient = vec![F::zero(); receipt.cycles()];
        let mut fused_increment = vec![F::zero(); receipt.cycles()];
        let mut bytecode_ra: [Vec<F>; RA_FACTORS] =
            core::array::from_fn(|_| vec![F::zero(); receipt.cycles()]);

        for (cycle, row) in owner.rows().iter().copied().enumerate() {
            let mut base = F::zero();
            let mut fused = F::zero();
            for stage in 0..STAGES {
                let raw = resolve_stage_value(&address.raw_values, STAGE_VALUE_SOURCES[stage])?;
                let coefficient = weights.stage()[stage]
                    * (raw + weights.within_stage_raf()[stage] * identity)
                    * stage_eq[stage][cycle];
                if stage < super::relation::BASE_STAGES {
                    base += coefficient;
                } else {
                    fused += coefficient;
                }
            }
            if cycle == 0 {
                base += weights.entry() * entry_scalar;
            }
            base_coefficient[cycle] = base;
            fused_coefficient[cycle] = fused;
            fused_increment[cycle] = row.fused_increment().field::<F>();
            let push_pc = row.push_pc();
            bytecode_ra[0][cycle] = chunk_eq[0][(push_pc >> COMMITTED_CHUNK_BITS) & 0xff];
            bytecode_ra[1][cycle] = chunk_eq[1][push_pc & 0xff];
        }

        let mut oracle = Self {
            base_coefficient,
            fused_coefficient,
            fused_increment,
            bytecode_ra,
            rounds: receipt.log_t(),
            round: 0,
            initial_claim: F::zero(),
            current_claim: F::zero(),
            pending: None,
            challenges: Vec::with_capacity(receipt.log_t()),
        };
        let claim = oracle.sum_current_domain();
        if claim != address.intermediate {
            return Err(OracleError::PhaseHandoffMismatch);
        }
        oracle.initial_claim = claim;
        oracle.current_claim = claim;
        Ok(oracle)
    }

    pub const fn num_rounds(&self) -> usize {
        self.rounds
    }

    pub const fn round(&self) -> usize {
        self.round
    }

    pub const fn initial_claim(&self) -> F {
        self.initial_claim
    }

    pub const fn current_claim(&self) -> F {
        self.current_claim
    }

    pub fn current_len(&self) -> usize {
        self.base_coefficient.len()
    }

    pub fn initial_ra(&self, factor: usize, cycle: usize) -> Result<F, OracleError> {
        if self.round != 0 {
            return Err(OracleError::InitialStateUnavailable);
        }
        self.bytecode_ra
            .get(factor)
            .and_then(|table| table.get(cycle))
            .copied()
            .ok_or(OracleError::InvalidDenseState)
    }

    pub fn message(&mut self) -> Result<CycleRoundMessage<F>, OracleError> {
        if self.round >= self.rounds {
            return Err(OracleError::AlreadyFullyBound);
        }
        if self.pending.is_some() {
            return Err(OracleError::MessageAlreadyPending);
        }
        self.validate_cycle_state()?;
        let mut samples = [F::zero(); 4];
        let points = [0u64, 2, 3, 4];
        for pair in 0..self.base_coefficient.len() / 2 {
            for (sample, point) in samples.iter_mut().zip(points) {
                *sample += self.cycle_pair_summand(pair, F::from_u64(point));
            }
        }
        let message = CycleRoundMessage::new(samples[0], samples[1], samples[2], samples[3]);
        self.pending = Some(message);
        Ok(message)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), OracleError> {
        if self.round >= self.rounds {
            return Err(OracleError::AlreadyFullyBound);
        }
        let message = self.pending.take().ok_or(OracleError::MessageRequired)?;
        let next_claim = message.evaluate(self.current_claim, challenge)?;
        for table in [
            &mut self.base_coefficient,
            &mut self.fused_coefficient,
            &mut self.fused_increment,
        ] {
            bind_dense(table, challenge)?;
        }
        for table in &mut self.bytecode_ra {
            bind_dense(table, challenge)?;
        }
        self.round = self.round.checked_add(1).ok_or(OracleError::Overflow)?;
        self.current_claim = next_claim;
        self.challenges.push(challenge);
        self.validate_cycle_state()?;
        Ok(())
    }

    pub fn output(&self) -> Result<CycleOutput<F>, OracleError> {
        if self.round != self.rounds {
            return Err(OracleError::NotFullyBound {
                remaining: self.rounds - self.round,
            });
        }
        if self.pending.is_some() {
            return Err(OracleError::MessageAlreadyPending);
        }
        self.validate_cycle_state()?;
        let bytecode_ra = [self.bytecode_ra[0][0], self.bytecode_ra[1][0]];
        let final_claim = cycle_summand(
            bytecode_ra,
            self.base_coefficient[0],
            self.fused_increment[0],
            self.fused_coefficient[0],
        );
        if final_claim != self.current_claim {
            return Err(OracleError::TerminalClaimMismatch);
        }
        Ok(CycleOutput {
            final_claim,
            bytecode_ra,
            fused_increment: self.fused_increment[0],
            r_cycle: canonical_opening_point(&self.challenges),
        })
    }

    fn cycle_pair_summand(&self, pair: usize, point: F) -> F {
        cycle_summand(
            core::array::from_fn(|factor| interpolate_pair(&self.bytecode_ra[factor], pair, point)),
            interpolate_pair(&self.base_coefficient, pair, point),
            interpolate_pair(&self.fused_increment, pair, point),
            interpolate_pair(&self.fused_coefficient, pair, point),
        )
    }

    fn sum_current_domain(&self) -> F {
        (0..self.base_coefficient.len()).fold(F::zero(), |claim, index| {
            claim
                + cycle_summand(
                    [self.bytecode_ra[0][index], self.bytecode_ra[1][index]],
                    self.base_coefficient[index],
                    self.fused_increment[index],
                    self.fused_coefficient[index],
                )
        })
    }

    fn validate_cycle_state(&self) -> Result<(), OracleError> {
        let expected = remaining_domain(self.rounds, self.round)?;
        if expected == 0
            || self.base_coefficient.len() != expected
            || self.fused_coefficient.len() != expected
            || self.fused_increment.len() != expected
            || self.bytecode_ra.iter().any(|table| table.len() != expected)
        {
            return Err(OracleError::InvalidDenseState);
        }
        Ok(())
    }
}

fn validate_inputs<F>(
    owner: &BytecodeReadRafOwner,
    inputs: &BytecodeReadRafInputs<F>,
) -> Result<(), OracleError> {
    let receipt = owner.receipt();
    for (stage, point) in inputs.stage_points.iter().enumerate() {
        if point.len() != receipt.log_t() {
            return Err(OracleError::StagePointLength {
                stage,
                expected: receipt.log_t(),
                got: point.len(),
            });
        }
    }
    for (table, values) in inputs.raw_value_tables.iter().enumerate() {
        if values.len() != receipt.addresses() {
            return Err(OracleError::RawValueTableLength {
                table,
                expected: receipt.addresses(),
                got: values.len(),
            });
        }
    }
    if inputs.entry_index >= receipt.addresses() {
        return Err(OracleError::EntryOutsideDomain {
            entry: inputs.entry_index,
            addresses: receipt.addresses(),
        });
    }
    Ok(())
}

fn dense_eq_table<F: Field>(point: &[F]) -> Result<Vec<F>, OracleError> {
    let expected = domain_size(point.len())?;
    let mut table = vec![F::one()];
    for &challenge in point {
        let capacity = table.len().checked_mul(2).ok_or(OracleError::Overflow)?;
        let mut next = Vec::with_capacity(capacity);
        for value in table {
            next.push(value * (F::one() - challenge));
            next.push(value * challenge);
        }
        table = next;
    }
    if table.len() != expected {
        return Err(OracleError::InvalidDenseState);
    }
    Ok(table)
}

fn evaluate_dense<F: Field>(table: &[F], point: &[F]) -> Result<F, OracleError> {
    if table.len() != domain_size(point.len())? {
        return Err(OracleError::InvalidDenseState);
    }
    let equality = dense_eq_table(point)?;
    Ok(table
        .iter()
        .copied()
        .zip(equality)
        .fold(F::zero(), |value, (table, eq)| value + table * eq))
}

fn eq_index<F: Field>(point: &[F], index: usize) -> Result<F, OracleError> {
    if index >= domain_size(point.len())? {
        return Err(OracleError::InvalidDenseState);
    }
    let mut value = F::one();
    for (position, &challenge) in point.iter().enumerate() {
        let shift = point
            .len()
            .checked_sub(position + 1)
            .ok_or(OracleError::Overflow)?;
        let bit = index
            .checked_shr(u32::try_from(shift).map_err(|_| OracleError::Overflow)?)
            .ok_or(OracleError::Overflow)?
            & 1;
        value *= if bit == 0 {
            F::one() - challenge
        } else {
            challenge
        };
    }
    Ok(value)
}

fn identity_mle<F: Field>(point: &[F]) -> F {
    point
        .iter()
        .copied()
        .fold(F::zero(), |value, challenge| value + value + challenge)
}

fn committed_chunks<F: Field>(r_address: &[F]) -> Result<[Vec<F>; RA_FACTORS], OracleError> {
    let padded_len = RA_FACTORS
        .checked_mul(COMMITTED_CHUNK_BITS)
        .ok_or(OracleError::Overflow)?;
    if r_address.len() > padded_len {
        return Err(OracleError::AddressPointLength {
            expected: padded_len,
            got: r_address.len(),
        });
    }
    let padding = padded_len - r_address.len();
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend((0..padding).map(|_| F::zero()));
    padded.extend_from_slice(r_address);
    Ok([
        padded[..COMMITTED_CHUNK_BITS].to_vec(),
        padded[COMMITTED_CHUNK_BITS..].to_vec(),
    ])
}

fn interpolate_pair<F: Field>(table: &[F], pair: usize, point: F) -> F {
    let low = table[2 * pair];
    let high = table[2 * pair + 1];
    low + point * (high - low)
}

fn bind_dense<F: Field>(table: &mut Vec<F>, challenge: F) -> Result<(), OracleError> {
    if table.len() < 2 || !table.len().is_multiple_of(2) {
        return Err(OracleError::InvalidDenseState);
    }
    let half = table.len() / 2;
    for pair in 0..half {
        let value = interpolate_pair(table, pair, challenge);
        table[pair] = value;
    }
    table.truncate(half);
    Ok(())
}

fn remaining_domain(rounds: usize, round: usize) -> Result<usize, OracleError> {
    let remaining = rounds.checked_sub(round).ok_or(OracleError::Overflow)?;
    domain_size(remaining)
}

fn domain_size(log_size: usize) -> Result<usize, OracleError> {
    let shift = u32::try_from(log_size).map_err(|_| OracleError::Overflow)?;
    1usize.checked_shl(shift).ok_or(OracleError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum OracleError {
    #[error(transparent)]
    Owner(#[from] OwnerError),
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error("bytecode read/RAF owner is empty")]
    EmptyOwner,
    #[error("bytecode stage {stage} point has {got} coordinates, expected {expected}")]
    StagePointLength {
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error("bytecode raw value table {table} has {got} entries, expected {expected}")]
    RawValueTableLength {
        table: usize,
        expected: usize,
        got: usize,
    },
    #[error("bytecode entry {entry} is outside {addresses} addresses")]
    EntryOutsideDomain { entry: usize, addresses: usize },
    #[error("bytecode address point has {got} coordinates, expected {expected}")]
    AddressPointLength { expected: usize, got: usize },
    #[error("bytecode address handoff raw value does not match its dense table")]
    RawValueHandoffMismatch,
    #[error("bytecode address and cycle phase claims do not match")]
    PhaseHandoffMismatch,
    #[error("bytecode dense oracle is already fully bound")]
    AlreadyFullyBound,
    #[error("bytecode dense oracle already has an unbound round message")]
    MessageAlreadyPending,
    #[error("bytecode dense oracle needs a round message before binding")]
    MessageRequired,
    #[error("bytecode dense oracle still has {remaining} rounds")]
    NotFullyBound { remaining: usize },
    #[error("bytecode dense oracle stage {0} does not exist")]
    InvalidStage(usize),
    #[error("bytecode dense oracle initial state is no longer available")]
    InitialStateUnavailable,
    #[error("bytecode dense oracle state is malformed")]
    InvalidDenseState,
    #[error("bytecode dense oracle terminal relation does not match the round claim")]
    TerminalClaimMismatch,
    #[error("bytecode dense oracle arithmetic overflowed")]
    Overflow,
}
