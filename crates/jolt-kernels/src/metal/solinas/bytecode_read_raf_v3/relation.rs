use jolt_field::Field;
use thiserror::Error;

pub const BASE_STAGES: usize = 5;
pub const FUSED_STAGES: usize = 4;
pub const STAGES: usize = BASE_STAGES + FUSED_STAGES;
pub const RAW_VALUE_TABLES: usize = 6;
pub const RA_FACTORS: usize = 2;
pub const COMMITTED_CHUNK_BITS: usize = 8;

/// The address summand is quadratic even though the symbolic relation admits
/// a zero cubic coefficient for the two committed RA factors.
pub const ADDRESS_ACTUAL_DEGREE: usize = 2;
pub const ADDRESS_DECLARED_MAX_DEGREE: usize = RA_FACTORS + 1;
pub const CYCLE_DEGREE: usize = RA_FACTORS + 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageValueSource {
    Table(usize),
    Complement(usize),
}

pub const STAGE_VALUE_SOURCES: [StageValueSource; STAGES] = [
    StageValueSource::Table(0),
    StageValueSource::Table(1),
    StageValueSource::Table(2),
    StageValueSource::Table(3),
    StageValueSource::Table(4),
    StageValueSource::Table(5),
    StageValueSource::Table(5),
    StageValueSource::Complement(5),
    StageValueSource::Complement(5),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationWeights<F> {
    stage: [F; STAGES],
    within_stage_raf: [F; STAGES],
    entry: F,
}

impl<F: Field> RelationWeights<F> {
    pub fn new(gamma: F) -> Self {
        let mut powers = [F::one(); STAGES + 3];
        for index in 1..powers.len() {
            powers[index] = powers[index - 1] * gamma;
        }
        let mut within_stage_raf = [F::zero(); STAGES];
        within_stage_raf[0] = powers[STAGES];
        within_stage_raf[2] = powers[STAGES - 1];
        Self {
            stage: core::array::from_fn(|index| powers[index]),
            within_stage_raf,
            entry: powers[STAGES + 2],
        }
    }

    pub const fn stage(&self) -> &[F; STAGES] {
        &self.stage
    }

    pub const fn within_stage_raf(&self) -> &[F; STAGES] {
        &self.within_stage_raf
    }

    pub const fn entry(&self) -> F {
        self.entry
    }
}

/// Exact stage-6a summand at one address-domain point.
pub fn address_summand<F: Field>(
    pushforwards: &[F; STAGES],
    raw_values: &[F; RAW_VALUE_TABLES],
    identity: F,
    entry_trace: F,
    entry_expected: F,
    weights: &RelationWeights<F>,
) -> Result<F, RelationError> {
    let mut value = weights.entry * entry_trace * entry_expected;
    for stage in 0..STAGES {
        let stage_value = resolve_stage_value(raw_values, STAGE_VALUE_SOURCES[stage])?;
        value += weights.stage[stage]
            * pushforwards[stage]
            * (stage_value + weights.within_stage_raf[stage] * identity);
    }
    Ok(value)
}

/// Exact Akita stage-6b summand at one cycle-domain point.
pub fn cycle_summand<F: Field>(
    ra: [F; RA_FACTORS],
    base_coefficient: F,
    fused_increment: F,
    fused_coefficient: F,
) -> F {
    ra[0] * ra[1] * (base_coefficient + fused_increment * fused_coefficient)
}

pub(crate) fn resolve_stage_value<F: Field>(
    raw_values: &[F; RAW_VALUE_TABLES],
    source: StageValueSource,
) -> Result<F, RelationError> {
    let (index, complement) = match source {
        StageValueSource::Table(index) => (index, false),
        StageValueSource::Complement(index) => (index, true),
    };
    let value = raw_values
        .get(index)
        .copied()
        .ok_or(RelationError::InvalidStageValueSource(index))?;
    Ok(if complement { F::one() - value } else { value })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressRoundMessage<F> {
    at_zero: F,
    at_two: F,
}

impl<F: Field> AddressRoundMessage<F> {
    pub const fn new(at_zero: F, at_two: F) -> Self {
        Self { at_zero, at_two }
    }

    pub const fn at_zero(self) -> F {
        self.at_zero
    }

    pub const fn at_two(self) -> F {
        self.at_two
    }

    pub fn evaluations_with_hint(self, previous_claim: F) -> [F; 3] {
        [self.at_zero, previous_claim - self.at_zero, self.at_two]
    }

    pub fn evaluate(self, previous_claim: F, point: F) -> Result<F, RelationError> {
        interpolate_consecutive(&self.evaluations_with_hint(previous_claim), point)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the field names are the protocol evaluation points"
)]
pub struct CycleRoundMessage<F> {
    at_zero: F,
    at_two: F,
    at_three: F,
    at_four: F,
}

impl<F: Field> CycleRoundMessage<F> {
    pub const fn new(at_zero: F, at_two: F, at_three: F, at_four: F) -> Self {
        Self {
            at_zero,
            at_two,
            at_three,
            at_four,
        }
    }

    pub const fn at_zero(self) -> F {
        self.at_zero
    }

    pub const fn at_two(self) -> F {
        self.at_two
    }

    pub const fn at_three(self) -> F {
        self.at_three
    }

    pub const fn at_four(self) -> F {
        self.at_four
    }

    pub fn evaluations_with_hint(self, previous_claim: F) -> [F; 5] {
        [
            self.at_zero,
            previous_claim - self.at_zero,
            self.at_two,
            self.at_three,
            self.at_four,
        ]
    }

    pub fn evaluate(self, previous_claim: F, point: F) -> Result<F, RelationError> {
        interpolate_consecutive(&self.evaluations_with_hint(previous_claim), point)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressOutput<F> {
    pub(crate) intermediate: F,
    pub(crate) raw_values: [F; RAW_VALUE_TABLES],
    pub(crate) r_address: Vec<F>,
}

impl<F: Copy> AddressOutput<F> {
    pub const fn intermediate(&self) -> F {
        self.intermediate
    }

    pub const fn raw_values(&self) -> &[F; RAW_VALUE_TABLES] {
        &self.raw_values
    }

    pub fn r_address(&self) -> &[F] {
        &self.r_address
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CycleOutput<F> {
    pub(crate) final_claim: F,
    pub(crate) bytecode_ra: [F; RA_FACTORS],
    pub(crate) fused_increment: F,
    pub(crate) r_cycle: Vec<F>,
}

impl<F: Copy> CycleOutput<F> {
    pub const fn final_claim(&self) -> F {
        self.final_claim
    }

    pub const fn bytecode_ra(&self) -> &[F; RA_FACTORS] {
        &self.bytecode_ra
    }

    pub const fn fused_increment(&self) -> F {
        self.fused_increment
    }

    pub fn r_cycle(&self) -> &[F] {
        &self.r_cycle
    }
}

/// Low-to-high binding encounters the least-significant variable first.
pub fn canonical_opening_point<F: Copy>(encountered_challenges: &[F]) -> Vec<F> {
    encountered_challenges.iter().rev().copied().collect()
}

fn interpolate_consecutive<F: Field>(values: &[F], point: F) -> Result<F, RelationError> {
    let mut result = F::zero();
    for (index, &value) in values.iter().enumerate() {
        let x_i = F::from_u64(index as u64);
        let mut numerator = F::one();
        let mut denominator = F::one();
        for other in 0..values.len() {
            if other == index {
                continue;
            }
            let x_j = F::from_u64(other as u64);
            numerator *= point - x_j;
            denominator *= x_i - x_j;
        }
        let inverse = denominator
            .inverse()
            .ok_or(RelationError::NonInvertibleInterpolation)?;
        result += value * numerator * inverse;
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RelationError {
    #[error("bytecode read/RAF stage value source {0} is outside the six raw tables")]
    InvalidStageValueSource(usize),
    #[error("bytecode read/RAF interpolation denominator is not invertible")]
    NonInvertibleInterpolation,
}
