//! Independent scalar oracle for the 16 address phases and 128 quadratics.
//!
//! The oracle scans cycle-order facts and never consumes the grouped
//! permutation or six-lane map. A candidate with a bad topology or lane map
//! therefore cannot make its own expected output.

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::Field;
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::SuffixEval;
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use thiserror::Error;

use super::carrier::{
    decode_claim, CarrierError, InstructionFactsCarrier, ADDRESS_BITS, ADDRESS_PHASES,
    LOOKUP_TABLES, PHASES_PER_RA_FACTOR, PHASE_BINS, PHASE_BITS, VIRTUAL_RA_FACTORS,
};
use super::model::{PhaseWork, TOTAL_DECLARED_SUFFIXES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum RafOutput {
    InterleavedShift = 0,
    Left = 1,
    Right = 2,
    IdentityShift = 3,
    Identity = 4,
    CanonicalUpperAllOnes = 5,
}

pub const RAF_OUTPUTS: usize = 6;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressPhaseOutput<F> {
    phase: usize,
    suffix_len: usize,
    raf: Vec<F>,
    suffixes: Vec<F>,
    suffix_offsets: [usize; LOOKUP_TABLES + 1],
    pub work: PhaseWork,
}

impl<F> AddressPhaseOutput<F> {
    pub const fn phase(&self) -> usize {
        self.phase
    }

    pub const fn suffix_len(&self) -> usize {
        self.suffix_len
    }

    pub fn raf(&self, output: RafOutput) -> &[F] {
        let start = output as usize * PHASE_BINS;
        &self.raf[start..start + PHASE_BINS]
    }

    pub fn raf_flat(&self) -> &[F] {
        &self.raf
    }

    pub fn suffix(&self, table: usize, slot: usize) -> Result<&[F], OracleError> {
        if table >= LOOKUP_TABLES {
            return Err(OracleError::InvalidTable(table));
        }
        let start_slot = self.suffix_offsets[table];
        let end_slot = self.suffix_offsets[table + 1];
        if start_slot + slot >= end_slot {
            return Err(OracleError::InvalidSuffixSlot { table, slot });
        }
        let start = (start_slot + slot) * PHASE_BINS;
        Ok(&self.suffixes[start..start + PHASE_BINS])
    }

    pub fn suffixes_flat(&self) -> &[F] {
        &self.suffixes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressRound<F> {
    pub round: usize,
    pub phase: usize,
    pub round_in_phase: usize,
    pub previous_claim: F,
    pub evaluations: [F; 3],
}

impl<F: Field> AddressRound<F> {
    pub fn coefficients(&self) -> Result<[F; 3], OracleError> {
        let two_inverse = F::from_u64(2)
            .inverse()
            .ok_or(OracleError::NonInvertibleTwo)?;
        let quadratic = (self.evaluations[0] - self.evaluations[1] - self.evaluations[1]
            + self.evaluations[2])
            * two_inverse;
        let linear = self.evaluations[1] - self.evaluations[0] - quadratic;
        Ok([self.evaluations[0], linear, quadratic])
    }

    pub fn evaluate(&self, point: F) -> Result<F, OracleError> {
        evaluate_quadratic(self.evaluations, point)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressHandoff<F> {
    terminal_address_claim: F,
    address_challenges: Vec<F>,
    table_values: Vec<F>,
    raf_interleaved: F,
    raf_identity: F,
    phase_eq_tables: Vec<Vec<F>>,
}

impl<F: Field> AddressHandoff<F> {
    pub const fn terminal_address_claim(&self) -> F {
        self.terminal_address_claim
    }

    pub fn address_challenges(&self) -> &[F] {
        &self.address_challenges
    }

    pub fn table_values(&self) -> &[F] {
        &self.table_values
    }

    pub const fn raf_interleaved(&self) -> F {
        self.raf_interleaved
    }

    pub const fn raf_identity(&self) -> F {
        self.raf_identity
    }

    pub fn ra_base(&self, factor: usize, raw_lookup: u128) -> Result<F, OracleError> {
        if factor >= VIRTUAL_RA_FACTORS {
            return Err(OracleError::InvalidRaFactor(factor));
        }
        let mut value = F::one();
        for local_phase in 0..PHASES_PER_RA_FACTOR {
            let phase = factor * PHASES_PER_RA_FACTOR + local_phase;
            let suffix_len = ADDRESS_BITS - (phase + 1) * PHASE_BITS;
            let chunk = ((raw_lookup >> suffix_len) as usize) & (PHASE_BINS - 1);
            value *= self.phase_eq_tables[phase][chunk];
        }
        Ok(value)
    }

    pub fn combined_value(&self, packed_claim: u8) -> Result<F, OracleError> {
        let claim = decode_claim(packed_claim)?;
        let table_value = claim
            .table_index()
            .map_or_else(F::zero, |table| self.table_values[table]);
        Ok(table_value
            + if claim.raf_flag() {
                self.raf_identity
            } else {
                self.raf_interleaved
            })
    }

    pub fn cycle_factors(
        &self,
        raw_lookup: u128,
        packed_claim: u8,
    ) -> Result<[F; VIRTUAL_RA_FACTORS + 1], OracleError> {
        let mut factors = [F::zero(); VIRTUAL_RA_FACTORS + 1];
        factors[0] = self.combined_value(packed_claim)?;
        for factor in 0..VIRTUAL_RA_FACTORS {
            factors[factor + 1] = self.ra_base(factor, raw_lookup)?;
        }
        Ok(factors)
    }

    /// Evaluates the exact cycle-side claim produced by this address handoff.
    pub fn terminal_cycle_claim(
        &self,
        facts: InstructionFactsCarrier<'_>,
        r_reduction: &[F],
    ) -> Result<F, OracleError> {
        let expected_log_t = facts.rows().ilog2() as usize;
        if r_reduction.len() != expected_log_t {
            return Err(OracleError::ReductionPointLength {
                expected: expected_log_t,
                got: r_reduction.len(),
            });
        }
        let weights = eq_table(r_reduction);
        facts
            .claims_cycle_order()
            .iter()
            .zip(facts.lookups_cycle_order())
            .zip(weights)
            .try_fold(F::zero(), |claim, ((&packed, &lookup), weight)| {
                let factors = self.cycle_factors(lookup, packed)?;
                let product = factors
                    .into_iter()
                    .fold(F::one(), |product, factor| product * factor);
                Ok(claim + weight * product)
            })
    }

    pub fn validate_terminal_claim(
        &self,
        facts: InstructionFactsCarrier<'_>,
        r_reduction: &[F],
    ) -> Result<(), OracleError> {
        if self.terminal_cycle_claim(facts, r_reduction)? != self.terminal_address_claim {
            return Err(OracleError::TerminalClaimMismatch);
        }
        Ok(())
    }

    pub fn phase_eq_tables(&self) -> &[Vec<F>] {
        &self.phase_eq_tables
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressOracleTrace<F> {
    pub input_claim: F,
    pub phases: Vec<AddressPhaseOutput<F>>,
    pub rounds: Vec<AddressRound<F>>,
    pub handoff: AddressHandoff<F>,
}

impl<F: Field> AddressOracleTrace<F> {
    pub fn generate(
        facts: InstructionFactsCarrier<'_>,
        r_reduction: &[F],
        gamma: F,
        address_challenges: &[F],
        expected_input_claim: F,
    ) -> Result<Self, OracleError> {
        if address_challenges.len() != ADDRESS_BITS {
            return Err(OracleError::AddressChallengeCount {
                expected: ADDRESS_BITS,
                got: address_challenges.len(),
            });
        }
        let mut state = OracleState::new(facts, r_reduction, gamma)?;
        let mut phases = vec![state.phase.output.clone()];
        let first_evaluations = state.message();
        let input_claim = first_evaluations[0] + first_evaluations[1];
        if expected_input_claim != input_claim {
            return Err(OracleError::InputClaimMismatch);
        }

        let mut rounds = Vec::with_capacity(ADDRESS_BITS);
        let mut previous_claim = input_claim;
        let mut terminal_claim = None;
        for (round, &challenge) in address_challenges.iter().enumerate() {
            let evaluations = if round == 0 {
                first_evaluations
            } else {
                state.message()
            };
            if evaluations[0] + evaluations[1] != previous_claim {
                return Err(OracleError::RoundClaimMismatch { round });
            }
            let polynomial = AddressRound {
                round,
                phase: round / PHASE_BITS,
                round_in_phase: round % PHASE_BITS,
                previous_claim,
                evaluations,
            };
            previous_claim = polynomial.evaluate(challenge)?;
            rounds.push(polynomial);
            if let Some(next_phase) = state.bind(challenge)? {
                phases.push(next_phase);
            }
            if round + 1 == ADDRESS_BITS {
                terminal_claim = Some(previous_claim);
            }
        }

        let terminal_address_claim = terminal_claim.ok_or(OracleError::MissingHandoff)?;
        let handoff = state.handoff(terminal_address_claim, address_challenges.to_vec())?;
        handoff.validate_terminal_claim(state.facts, r_reduction)?;
        Ok(Self {
            input_claim,
            phases,
            rounds,
            handoff,
        })
    }
}

#[derive(Clone)]
struct RafState<F> {
    prefix: Vec<F>,
    q_shift: Vec<F>,
    q_value: Vec<F>,
    checkpoint: F,
}

impl<F: Field> RafState<F> {
    fn zero() -> Self {
        Self {
            prefix: Vec::new(),
            q_shift: Vec::new(),
            q_value: Vec::new(),
            checkpoint: F::zero(),
        }
    }

    fn product() -> Self {
        Self {
            checkpoint: F::one(),
            ..Self::zero()
        }
    }

    fn message_eval(&self, index: usize, half: usize, sample: AddressSample) -> F {
        extension(&self.prefix, index, half, sample) * extension(&self.q_shift, index, half, sample)
            + extension(&self.q_value, index, half, sample)
    }

    fn bind(&mut self, challenge: F) -> Result<(), OracleError> {
        bind_high_to_low(&mut self.prefix, challenge)?;
        bind_high_to_low(&mut self.q_shift, challenge)?;
        bind_high_to_low(&mut self.q_value, challenge)
    }
}

struct PhaseState<F> {
    output: AddressPhaseOutput<F>,
    prefix_tables: Vec<Vec<F>>,
    suffix_tables: Vec<Vec<F>>,
}

struct OracleState<'a, F> {
    facts: InstructionFactsCarrier<'a>,
    gamma: F,
    weights: Vec<F>,
    prefix_checkpoints: Vec<PrefixEval<F>>,
    raf_left: RafState<F>,
    raf_right: RafState<F>,
    raf_identity: RafState<F>,
    raf_upper: RafState<F>,
    phase_challenges: Vec<F>,
    phase_eq_tables: Vec<Vec<F>>,
    phase: PhaseState<F>,
    rounds_bound: usize,
}

impl<'a, F: Field> OracleState<'a, F> {
    fn new(
        facts: InstructionFactsCarrier<'a>,
        r_reduction: &[F],
        gamma: F,
    ) -> Result<Self, OracleError> {
        let expected_log_t = facts.rows().ilog2() as usize;
        if r_reduction.len() != expected_log_t {
            return Err(OracleError::ReductionPointLength {
                expected: expected_log_t,
                got: r_reduction.len(),
            });
        }
        let weights = eq_table(r_reduction);
        let mut state = Self {
            facts,
            gamma,
            weights,
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            raf_left: RafState::zero(),
            raf_right: RafState::zero(),
            raf_identity: RafState::zero(),
            raf_upper: RafState::product(),
            phase_challenges: Vec::with_capacity(PHASE_BITS),
            phase_eq_tables: Vec::with_capacity(ADDRESS_PHASES),
            phase: empty_phase_state(),
            rounds_bound: 0,
        };
        state.phase = state.materialize_phase(0)?;
        Ok(state)
    }

    fn materialize_phase(&mut self, phase: usize) -> Result<PhaseState<F>, OracleError> {
        let suffix_len = ADDRESS_BITS - (phase + 1) * PHASE_BITS;
        let suffix_mask = if suffix_len == 0 {
            0
        } else {
            (1u128 << suffix_len) - 1
        };
        let offsets = suffix_offsets()?;
        let mut raf = vec![F::zero(); RAF_OUTPUTS * PHASE_BINS];
        let mut suffixes = vec![F::zero(); TOTAL_DECLARED_SUFFIXES * PHASE_BINS];
        let mut work = PhaseWork {
            rows_scanned: self.facts.rows() as u64,
            equality_products: if phase == 0 {
                self.facts.rows() as u64
            } else {
                0
            },
            condensation_products: if phase == 0 {
                0
            } else {
                self.facts.rows() as u64
            },
            ..PhaseWork::default()
        };
        let tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let upper_suffix_bits = suffix_len.saturating_sub(RISCV_XLEN);

        for cycle in 0..self.facts.rows() {
            let fact = self.facts.cycle_fact(cycle)?;
            let weight = self.weights[cycle];
            let lookup = fact.lookup();
            let chunk = ((lookup >> suffix_len) as usize) & (PHASE_BINS - 1);
            let suffix_bits = lookup & suffix_mask;
            if !fact.raf_flag() {
                add(
                    &mut raf,
                    RafOutput::InterleavedShift as usize,
                    chunk,
                    weight,
                );
                work.accumulated_terms += 1;
                let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                for (output, scalar) in [
                    (RafOutput::Left, u64::from(left)),
                    (RafOutput::Right, u64::from(right)),
                ] {
                    if scalar != 0 {
                        add(
                            &mut raf,
                            output as usize,
                            chunk,
                            weight * F::from_u64(scalar),
                        );
                        work.raf_scalar_products += 1;
                        work.accumulated_terms += 1;
                    }
                }
            } else {
                add(&mut raf, RafOutput::IdentityShift as usize, chunk, weight);
                work.accumulated_terms += 1;
                if suffix_bits != 0 {
                    add(
                        &mut raf,
                        RafOutput::Identity as usize,
                        chunk,
                        weight * F::from_u128(suffix_bits),
                    );
                    work.raf_scalar_products += 1;
                    work.accumulated_terms += 1;
                }
                if CANONICAL_INSTRUCTION_ADDRESS
                    && (upper_suffix_bits == 0
                        || suffix_bits >> (suffix_len - upper_suffix_bits)
                            == (1u128 << upper_suffix_bits) - 1)
                {
                    add(
                        &mut raf,
                        RafOutput::CanonicalUpperAllOnes as usize,
                        chunk,
                        weight,
                    );
                    work.accumulated_terms += 1;
                }
            }

            let Some(table_index) = fact.table_index() else {
                continue;
            };
            let bits = LookupBits::new(suffix_bits, suffix_len);
            for (slot, suffix) in tables[table_index].suffixes().iter().enumerate() {
                let scalar = suffix.suffix_mle(bits);
                if scalar == 0 {
                    continue;
                }
                let value = if scalar == 1 {
                    weight
                } else {
                    work.suffix_scalar_products += 1;
                    weight * F::from_u64(scalar)
                };
                let output = offsets[table_index] + slot;
                suffixes[output * PHASE_BINS + chunk] += value;
                if *suffix != jolt_lookup_tables::tables::suffixes::Suffixes::One {
                    work.accumulated_terms += 1;
                }
            }
        }

        let output = AddressPhaseOutput {
            phase,
            suffix_len,
            raf,
            suffixes,
            suffix_offsets: offsets,
            work,
        };
        self.install_raf_state(phase, &output);
        let prefix_tables = ALL_PREFIXES
            .iter()
            .map(|prefix| {
                (0..PHASE_BINS)
                    .map(|chunk| {
                        prefix
                            .evaluate::<F>(
                                &self.prefix_checkpoints,
                                LookupBits::new(chunk as u128, PHASE_BITS),
                                suffix_len,
                            )
                            .value()
                    })
                    .collect()
            })
            .collect();
        let suffix_tables = (0..TOTAL_DECLARED_SUFFIXES)
            .map(|suffix| output.suffixes[suffix * PHASE_BINS..(suffix + 1) * PHASE_BINS].to_vec())
            .collect();
        Ok(PhaseState {
            output,
            prefix_tables,
            suffix_tables,
        })
    }

    fn install_raf_state(&mut self, phase: usize, output: &AddressPhaseOutput<F>) {
        let suffix_len = output.suffix_len;
        let q_shift_half = output
            .raf(RafOutput::InterleavedShift)
            .iter()
            .map(|value| value.mul_pow_2(suffix_len / 2))
            .collect::<Vec<_>>();
        let q_shift_full = output
            .raf(RafOutput::IdentityShift)
            .iter()
            .map(|value| value.mul_pow_2(suffix_len))
            .collect::<Vec<_>>();
        let identity_prefix = (0..PHASE_BINS)
            .map(|chunk| {
                self.raf_identity.checkpoint.mul_pow_2(PHASE_BITS) + F::from_u64(chunk as u64)
            })
            .collect();
        let (left_prefix, right_prefix) = (0..PHASE_BINS)
            .map(|chunk| {
                let (left, right) = LookupBits::new(chunk as u128, PHASE_BITS).uninterleave();
                (
                    self.raf_left.checkpoint.mul_pow_2(PHASE_BITS / 2)
                        + F::from_u64(u64::from(left)),
                    self.raf_right.checkpoint.mul_pow_2(PHASE_BITS / 2)
                        + F::from_u64(u64::from(right)),
                )
            })
            .unzip();
        self.raf_left.prefix = left_prefix;
        self.raf_left.q_shift.clone_from(&q_shift_half);
        self.raf_left.q_value = output.raf(RafOutput::Left).to_vec();
        self.raf_right.prefix = right_prefix;
        self.raf_right.q_shift = q_shift_half;
        self.raf_right.q_value = output.raf(RafOutput::Right).to_vec();
        self.raf_identity.prefix = identity_prefix;
        self.raf_identity.q_shift = q_shift_full;
        self.raf_identity.q_value = output.raf(RafOutput::Identity).to_vec();

        let chunk_upper_bits = RISCV_XLEN
            .saturating_sub(phase * PHASE_BITS)
            .min(PHASE_BITS);
        self.raf_upper.prefix = (0..PHASE_BINS)
            .map(|chunk| {
                if chunk_upper_bits == 0
                    || chunk >> (PHASE_BITS - chunk_upper_bits) == (1 << chunk_upper_bits) - 1
                {
                    self.raf_upper.checkpoint
                } else {
                    F::zero()
                }
            })
            .collect();
        self.raf_upper.q_shift = output.raf(RafOutput::CanonicalUpperAllOnes).to_vec();
        self.raf_upper.q_value = vec![F::zero(); PHASE_BINS];
    }

    fn message(&self) -> [F; 3] {
        let gamma_squared = self.gamma * self.gamma;
        let half = self.phase.prefix_tables[0].len() / 2;
        let tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
        let offsets = self.phase.output.suffix_offsets;
        let mut evaluations = [F::zero(); 3];
        for (evaluation, sample) in evaluations.iter_mut().zip(ADDRESS_SAMPLES) {
            let mut read = F::zero();
            let mut left = F::zero();
            let mut right = F::zero();
            let mut identity = F::zero();
            let mut upper = F::zero();
            for index in 0..half {
                let prefix_evals: Vec<_> = self
                    .phase
                    .prefix_tables
                    .iter()
                    .map(|table| PrefixEval::from(extension(table, index, half, sample)))
                    .collect();
                for table in &tables {
                    let table_index = table.index();
                    let suffix_evals: Vec<_> = self.phase.suffix_tables
                        [offsets[table_index]..offsets[table_index + 1]]
                        .iter()
                        .map(|suffix| SuffixEval::from(extension(suffix, index, half, sample)))
                        .collect();
                    read += table.combine(&prefix_evals, &suffix_evals);
                }
                left += self.raf_left.message_eval(index, half, sample);
                right += self.raf_right.message_eval(index, half, sample);
                identity += self.raf_identity.message_eval(index, half, sample);
                if CANONICAL_INSTRUCTION_ADDRESS {
                    upper += self.raf_upper.message_eval(index, half, sample);
                }
            }
            let mut value = read + self.gamma * left + gamma_squared * (right + identity);
            if CANONICAL_INSTRUCTION_ADDRESS {
                value += gamma_squared * self.gamma * upper;
            }
            *evaluation = value;
        }
        evaluations
    }

    fn bind(&mut self, challenge: F) -> Result<Option<AddressPhaseOutput<F>>, OracleError> {
        for table in &mut self.phase.prefix_tables {
            bind_high_to_low(table, challenge)?;
        }
        for suffix in &mut self.phase.suffix_tables {
            bind_high_to_low(suffix, challenge)?;
        }
        self.raf_left.bind(challenge)?;
        self.raf_right.bind(challenge)?;
        self.raf_identity.bind(challenge)?;
        self.raf_upper.bind(challenge)?;
        self.phase_challenges.push(challenge);
        self.rounds_bound += 1;

        if !self.rounds_bound.is_multiple_of(PHASE_BITS) {
            return Ok(None);
        }
        let completed_phase = self.rounds_bound / PHASE_BITS - 1;
        let phase_eq = eq_table(&self.phase_challenges);
        self.phase_eq_tables.push(phase_eq.clone());
        for (checkpoint, table) in self
            .prefix_checkpoints
            .iter_mut()
            .zip(&self.phase.prefix_tables)
        {
            *checkpoint = PrefixEval::from(table[0]);
        }
        self.raf_left.checkpoint = self.raf_left.prefix[0];
        self.raf_right.checkpoint = self.raf_right.prefix[0];
        self.raf_identity.checkpoint = self.raf_identity.prefix[0];
        self.raf_upper.checkpoint = self.raf_upper.prefix[0];
        self.phase_challenges.clear();
        if completed_phase + 1 == ADDRESS_PHASES {
            return Ok(None);
        }

        let suffix_len = ADDRESS_BITS - (completed_phase + 1) * PHASE_BITS;
        for cycle in 0..self.facts.rows() {
            let lookup = self.facts.lookups_cycle_order()[cycle];
            let chunk = ((lookup >> suffix_len) as usize) & (PHASE_BINS - 1);
            self.weights[cycle] *= phase_eq[chunk];
        }
        self.phase = self.materialize_phase(completed_phase + 1)?;
        Ok(Some(self.phase.output.clone()))
    }

    fn handoff(
        &self,
        terminal_address_claim: F,
        address_challenges: Vec<F>,
    ) -> Result<AddressHandoff<F>, OracleError> {
        if self.rounds_bound != ADDRESS_BITS
            || self.phase_eq_tables.len() != ADDRESS_PHASES
            || address_challenges.len() != ADDRESS_BITS
        {
            return Err(OracleError::MissingHandoff);
        }
        let empty = LookupBits::new(0, 0);
        let table_values = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffix_evals: Vec<_> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty))))
                    .collect();
                table.combine(&self.prefix_checkpoints, &suffix_evals)
            })
            .collect();
        let gamma_squared = self.gamma * self.gamma;
        let raf_interleaved =
            self.gamma * self.raf_left.checkpoint + gamma_squared * self.raf_right.checkpoint;
        let mut raf_identity = gamma_squared * self.raf_identity.checkpoint;
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity += gamma_squared * self.gamma * self.raf_upper.checkpoint;
        }
        Ok(AddressHandoff {
            terminal_address_claim,
            address_challenges,
            table_values,
            raf_interleaved,
            raf_identity,
            phase_eq_tables: self.phase_eq_tables.clone(),
        })
    }
}

fn empty_phase_state<F: Field>() -> PhaseState<F> {
    PhaseState {
        output: AddressPhaseOutput {
            phase: 0,
            suffix_len: 0,
            raf: Vec::new(),
            suffixes: Vec::new(),
            suffix_offsets: [0; LOOKUP_TABLES + 1],
            work: PhaseWork::default(),
        },
        prefix_tables: Vec::new(),
        suffix_tables: Vec::new(),
    }
}

fn suffix_offsets() -> Result<[usize; LOOKUP_TABLES + 1], OracleError> {
    let mut offsets = [0usize; LOOKUP_TABLES + 1];
    let mut total = 0usize;
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        total += table.suffixes().len();
        offsets[table.index() + 1] = total;
    }
    if total != TOTAL_DECLARED_SUFFIXES {
        return Err(OracleError::SuffixCount {
            expected: TOTAL_DECLARED_SUFFIXES,
            got: total,
        });
    }
    Ok(offsets)
}

fn add<F: Field>(output: &mut [F], lane: usize, chunk: usize, value: F) {
    output[lane * PHASE_BINS + chunk] += value;
}

#[derive(Clone, Copy)]
enum AddressSample {
    Zero,
    One,
    Two,
}

const ADDRESS_SAMPLES: [AddressSample; 3] =
    [AddressSample::Zero, AddressSample::One, AddressSample::Two];

fn extension<F: Field>(table: &[F], index: usize, half: usize, sample: AddressSample) -> F {
    let low = table[index];
    let high = table[index + half];
    match sample {
        AddressSample::Zero => low,
        AddressSample::One => high,
        AddressSample::Two => high + high - low,
    }
}

fn bind_high_to_low<F: Field>(table: &mut Vec<F>, challenge: F) -> Result<(), OracleError> {
    if table.len() < 2 || !table.len().is_power_of_two() {
        return Err(OracleError::InvalidDenseLength(table.len()));
    }
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[index];
        table[index] = low + challenge * (table[index + half] - low);
    }
    table.truncate(half);
    Ok(())
}

fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    let mut table = vec![F::one()];
    for &coordinate in point {
        let mut next = Vec::with_capacity(table.len() * 2);
        for value in table {
            next.push(value * (F::one() - coordinate));
            next.push(value * coordinate);
        }
        table = next;
    }
    table
}

fn evaluate_quadratic<F: Field>(evaluations: [F; 3], point: F) -> Result<F, OracleError> {
    let two_inverse = F::from_u64(2)
        .inverse()
        .ok_or(OracleError::NonInvertibleTwo)?;
    let quadratic =
        (evaluations[0] - evaluations[1] - evaluations[1] + evaluations[2]) * two_inverse;
    let linear = evaluations[1] - evaluations[0] - quadratic;
    Ok(evaluations[0] + point * (linear + point * quadratic))
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum OracleError {
    #[error(transparent)]
    Carrier(#[from] CarrierError),
    #[error("InstructionReadRaf reduction point has {got} coordinates, expected {expected}")]
    ReductionPointLength { expected: usize, got: usize },
    #[error("InstructionReadRaf address challenge tape has {got} values, expected {expected}")]
    AddressChallengeCount { expected: usize, got: usize },
    #[error("InstructionReadRaf oracle input claim differs from the supplied control")]
    InputClaimMismatch,
    #[error("InstructionReadRaf address round {round} does not sum to its previous claim")]
    RoundClaimMismatch { round: usize },
    #[error("InstructionReadRaf address handoff occurred before all 128 binds")]
    MissingHandoff,
    #[error("InstructionReadRaf terminal address claim differs from its cycle-factor handoff")]
    TerminalClaimMismatch,
    #[error("InstructionReadRaf table {0} is outside the 40-table specialization")]
    InvalidTable(usize),
    #[error("InstructionReadRaf table {table} has no suffix slot {slot}")]
    InvalidSuffixSlot { table: usize, slot: usize },
    #[error("InstructionReadRaf suffix registry has {got} outputs, expected {expected}")]
    SuffixCount { expected: usize, got: usize },
    #[error("InstructionReadRaf dense oracle table has invalid length {0}")]
    InvalidDenseLength(usize),
    #[error("InstructionReadRaf field does not invert two")]
    NonInvertibleTwo,
    #[error("InstructionReadRaf virtual RA factor {0} is outside 0..4")]
    InvalidRaFactor(usize),
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests use fixed valid protocol fixtures"
)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::{AkitaField, MulPow2};
    use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_sumcheck::ProveRounds;

    use super::*;
    use crate::metal::solinas::instruction_read_raf_v2::carrier::{
        pack_claim, CycleOrderPlane, GroupedAddressTopology, PlaneReceipt, ProducerIdentity,
    };
    use crate::metal::solinas::instruction_read_raf_v2::model::GroupedAddressCensus;
    use crate::optimized::instruction_read_raf::{
        InstructionCycleRow, OptimizedInstructionReadRafKernel,
    };

    type F = AkitaField;

    fn f(value: u64) -> F {
        F::from_u64(value)
    }

    fn challenge(round: usize) -> F {
        f(0x9e37_79b9_7f4a_7c15 ^ (round as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
    }

    fn fixture(rows: usize) -> (Vec<u128>, Vec<u8>, Vec<InstructionCycleRow>) {
        let modulus = u128::MAX - u128::from(0xffff_a7f7u32) + 1;
        let mut lookups = Vec::with_capacity(rows);
        let mut claims = Vec::with_capacity(rows);
        let mut cpu_rows = Vec::with_capacity(rows);
        for cycle in 0..rows {
            let table = if cycle < 2 || cycle % 43 == 0 {
                None
            } else {
                Some(((cycle - 2) / 2) % LOOKUP_TABLES)
            };
            let lookup = match cycle {
                0 => 0,
                1 => modulus,
                2 => u128::MAX,
                3 => (u128::from(u64::MAX) << 64) | 0x0123_4567_89ab_cdef,
                _ if table.is_some() => 0,
                _ => {
                    let low = (cycle as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left(17);
                    let high = (cycle as u64)
                        .wrapping_mul(0xbf58_476d_1ce4_e5b9)
                        .rotate_right(11);
                    (u128::from(high) << 64) | u128::from(low)
                }
            };
            let raf = cycle < 2 || cycle % 2 == 0;
            lookups.push(lookup);
            claims.push(pack_claim(table, raf).unwrap());
            cpu_rows.push(InstructionCycleRow::new(
                lookup,
                table,
                raf,
                None,
                None,
                #[cfg(feature = "akita")]
                jolt_witness::witnesses::FusedInc::default(),
            ));
        }
        (lookups, claims, cpu_rows)
    }

    fn explicit_input_claim(lookups: &[u128], claims: &[u8], r_reduction: &[F], gamma: F) -> F {
        let tables = LookupTableKind::<RISCV_XLEN>::iter().collect::<Vec<_>>();
        let gamma_squared = gamma * gamma;
        eq_table(r_reduction)
            .into_iter()
            .zip(lookups)
            .zip(claims)
            .map(|((weight, &lookup), &packed)| {
                let decoded = decode_claim(packed).unwrap();
                let table_value = decoded.table_index().map_or_else(F::zero, |table| {
                    F::from_u64(tables[table].materialize_entry(lookup))
                });
                let raf_value = if decoded.raf_flag() {
                    let mut value = gamma_squared
                        * (F::from_u64(lookup as u64)
                            + F::from_u64((lookup >> 64) as u64).mul_pow_2(64));
                    if CANONICAL_INSTRUCTION_ADDRESS && lookup >> RISCV_XLEN == u64::MAX as u128 {
                        value += gamma_squared * gamma;
                    }
                    value
                } else {
                    let (left, right) = LookupBits::new(lookup, ADDRESS_BITS).uninterleave();
                    gamma * F::from_u64(u64::from(left))
                        + gamma_squared * F::from_u64(u64::from(right))
                };
                weight * (table_value + raf_value)
            })
            .sum()
    }

    fn direct_eq_index(point: &[F], index: u64) -> F {
        point
            .iter()
            .enumerate()
            .fold(F::one(), |value, (coordinate, &challenge)| {
                let bit = point.len() - coordinate - 1;
                if index >> bit & 1 == 0 {
                    value * (F::one() - challenge)
                } else {
                    value * challenge
                }
            })
    }

    #[test]
    fn address_trace_matches_optimized_cpu_and_terminal_relation() {
        let rows = 1 << 8;
        let (lookups, claims, cpu_rows) = fixture(rows);
        let producer = ProducerIdentity::new(11, 0x1000, 12, rows).unwrap();
        let lookup_plane = CycleOrderPlane::new(
            &lookups,
            PlaneReceipt::new(producer, 0x2000, "lookup plane").unwrap(),
            "lookup plane",
        )
        .unwrap();
        let claim_plane = CycleOrderPlane::new(
            &claims,
            PlaneReceipt::new(producer, 0x3000, "claim plane").unwrap(),
            "claim plane",
        )
        .unwrap();
        let topology = GroupedAddressTopology::stable_from_claims(claim_plane, 0x4000).unwrap();
        let facts =
            InstructionFactsCarrier::attach(11, lookup_plane, claim_plane, &topology).unwrap();
        let r_reduction = (0..8).map(|index| f(1000 + 37 * index)).collect::<Vec<_>>();
        assert_eq!(
            eq_table(&r_reduction),
            jolt_poly::EqPolynomial::<F>::evals(&r_reduction, None),
            "independent eq-table ordering diverges"
        );
        let tables = LookupTableKind::<RISCV_XLEN>::iter().collect::<Vec<_>>();
        for cycle in 0..rows {
            let point = (0..8)
                .map(|coordinate| F::from_u64(((cycle >> (7 - coordinate)) & 1) as u64))
                .collect::<Vec<_>>();
            let state = OracleState::new(facts, &point, F::zero()).unwrap();
            let evaluations = state.message();
            let expected = decode_claim(claims[cycle])
                .unwrap()
                .table_index()
                .map_or_else(F::zero, |table| {
                    F::from_u64(tables[table].materialize_entry(lookups[cycle]))
                });
            assert_eq!(
                evaluations[0] + evaluations[1],
                expected,
                "table decomposition diverges at cycle {cycle}"
            );
        }
        let gamma = f(0xace1_57ef);
        let address_challenges = (0..ADDRESS_BITS).map(challenge).collect::<Vec<_>>();
        let expected_input = explicit_input_claim(&lookups, &claims, &r_reduction, gamma);
        let read_state = OracleState::new(facts, &r_reduction, F::zero()).unwrap();
        let read_evaluations = read_state.message();
        assert_eq!(
            read_evaluations[0] + read_evaluations[1],
            explicit_input_claim(&lookups, &claims, &r_reduction, F::zero()),
            "explicit table contribution diverges"
        );
        let input_state = OracleState::new(facts, &r_reduction, gamma).unwrap();
        let input_evaluations = input_state.message();
        assert_eq!(
            input_evaluations[0] + input_evaluations[1],
            expected_input,
            "explicit input-claim control diverges before trace generation"
        );
        let trace = AddressOracleTrace::generate(
            facts,
            &r_reduction,
            gamma,
            &address_challenges,
            expected_input,
        )
        .unwrap();

        assert_eq!(trace.input_claim, expected_input);
        assert_eq!(trace.phases.len(), ADDRESS_PHASES);
        assert_eq!(trace.rounds.len(), ADDRESS_BITS);
        for (phase, output) in trace.phases.iter().enumerate() {
            assert_eq!(output.phase(), phase);
            assert_eq!(output.suffix_len(), ADDRESS_BITS - (phase + 1) * PHASE_BITS);
            assert_eq!(output.raf_flat().len(), RAF_OUTPUTS * PHASE_BINS);
            assert_eq!(
                output.suffixes_flat().len(),
                TOTAL_DECLARED_SUFFIXES * PHASE_BINS
            );
        }
        let census = GroupedAddressCensus::from_carrier(facts, 64).unwrap();
        assert!(census.segment_rows().iter().all(|&rows| rows != 0));
        for (phase, output) in trace.phases.iter().enumerate() {
            assert_eq!(
                census.phases()[phase].work(),
                output.work,
                "derived phase-{phase} work disagrees with the independent oracle"
            );
        }

        let dimensions = InstructionReadRafDimensions::new(
            8,
            ADDRESS_BITS,
            NonZeroUsize::new(VIRTUAL_RA_FACTORS).unwrap(),
        );
        let mut optimized = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            Arc::new(cpu_rows),
            gamma,
        )
        .unwrap();
        let mut claim = expected_input;
        for round in 0..ADDRESS_BITS {
            let bind = round
                .checked_sub(1)
                .map(|previous| address_challenges[previous]);
            let polynomial = optimized.prove_round(bind, round, claim).unwrap();
            let expected_coefficients = trace.rounds[round].coefficients().unwrap();
            assert_eq!(
                polynomial.coefficients(),
                expected_coefficients.as_slice(),
                "address round {round}"
            );
            claim = polynomial.evaluate(address_challenges[round]);
        }
        assert_eq!(claim, trace.handoff.terminal_address_claim);
        let first_cycle = optimized
            .prove_round(
                Some(address_challenges[ADDRESS_BITS - 1]),
                ADDRESS_BITS,
                claim,
            )
            .unwrap();
        assert_eq!(
            first_cycle.evaluate(F::zero()) + first_cycle.evaluate(F::one()),
            claim
        );
        assert_eq!(
            trace
                .handoff
                .terminal_cycle_claim(facts, &r_reduction)
                .unwrap(),
            claim
        );

        for &lookup in lookups.iter().take(4) {
            for factor in 0..VIRTUAL_RA_FACTORS {
                let coordinate_start = factor * (ADDRESS_BITS / VIRTUAL_RA_FACTORS);
                let coordinate_end = coordinate_start + ADDRESS_BITS / VIRTUAL_RA_FACTORS;
                let shift = ADDRESS_BITS - coordinate_end;
                let word = (lookup >> shift) as u32;
                assert_eq!(
                    trace.handoff.ra_base(factor, lookup).unwrap(),
                    direct_eq_index(
                        &address_challenges[coordinate_start..coordinate_end],
                        u64::from(word),
                    )
                );
            }
        }

        let mut corrupted = trace.handoff.clone();
        corrupted.terminal_address_claim += F::one();
        assert_eq!(
            corrupted.validate_terminal_claim(facts, &r_reduction),
            Err(OracleError::TerminalClaimMismatch)
        );
        assert_eq!(
            AddressOracleTrace::generate(
                facts,
                &r_reduction,
                gamma,
                &address_challenges,
                expected_input + F::one(),
            ),
            Err(OracleError::InputClaimMismatch)
        );
    }
}
