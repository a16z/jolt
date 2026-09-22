//! Exact finite-capacity D64 candidate sampling from an authenticated local tape.
use akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG;
use jolt_field::{CanonicalEncoding, Fr, Ring};
use jolt_r1cs::bn254_bits::{BitsError, ByteVar};
use jolt_r1cs::integer_bn254::{IntegerError, SignedVar};
use jolt_r1cs::{LinearCombination, R1csBuilder};
use thiserror::Error;

type Expression = LinearCombination<Fr>;

/// Invalid public shape, byte handles or an insufficient assignment capacity.
#[derive(Debug, Error)]
pub enum CandidateError {
    /// Counts, trial budget or tape length do not describe the supported shape.
    #[error("invalid D64 candidate shape")]
    Shape,
    /// A position did not accept within its public trial budget.
    #[error("D64 candidate capacity exceeded")]
    CapacityExceeded,
    #[error(transparent)]
    Bits(#[from] BitsError),
    #[error(transparent)]
    Integer(#[from] IntegerError),
}

/// Public single-candidate counts and per-position trial capacity.
/// Counts are not authenticated policy: the wrapper must bind the selected profile.
#[derive(Clone, Copy, Debug)]
pub struct D64CandidateProfile {
    count_pm1: usize,
    weight: usize,
    trials: usize,
    tape_len: usize,
}

impl D64CandidateProfile {
    /// Describe the native D64 byte sampler with explicit outer-circuit capacity.
    pub fn new(count_pm1: usize, count_pm2: usize, trials: usize) -> Result<Self, CandidateError> {
        let weight = count_pm1
            .checked_add(count_pm2)
            .ok_or(CandidateError::Shape)?;
        if weight == 0 || weight > 64 || trials == 0 {
            return Err(CandidateError::Shape);
        }
        let positions = weight - usize::from(weight == 64);
        let tape_len = positions
            .checked_mul(trials)
            .and_then(|n| n.checked_add(weight))
            .filter(|&n| n < usize::MAX)
            .ok_or(CandidateError::Shape)?;
        Ok(Self {
            count_pm1,
            weight,
            trials,
            tape_len,
        })
    }

    /// Use the native selective-L2 counts; does not apply its norm predicate.
    pub fn selective_l2(trials: usize) -> Result<Self, CandidateError> {
        let cfg = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
        Self::new(cfg.count_pm1, cfg.count_pm2, trials)
    }

    /// Fixed authenticated tape capacity, including sign bytes and unused padding.
    pub fn tape_len(&self) -> usize {
        self.tape_len
    }

    /// Constrain one candidate starting at local tape offset zero.
    ///
    /// ONE must be fixed externally and every handle must use this builder.
    /// Tape authentication, variable global offsets, norm rejection and retry
    /// composition are caller obligations. On error the builder may contain
    /// partial rows and must be discarded. Unknown assignments emit the same
    /// shape; completion constraints exclude capacity-exhausted witnesses.
    pub fn sample(
        &self,
        builder: &mut R1csBuilder<Fr>,
        tape: &[ByteVar],
    ) -> Result<D64CandidateVar, CandidateError> {
        if tape.len() != self.tape_len {
            return Err(CandidateError::Shape);
        }
        let cursor = (0..=tape.len())
            .map(|i| Expression::constant(Fr::from_u64(u64::from(i == 0))))
            .collect();
        self.sample_at(builder, tape, cursor, Expression::one())
    }

    // Internal composition: caller establishes one-hot cursor and Boolean activity.
    // Returned cursor retains that invariant, and inactive candidates consume nothing.
    pub(super) fn sample_at(
        &self,
        builder: &mut R1csBuilder<Fr>,
        tape: &[ByteVar],
        cursor: Vec<Expression>,
        activity: Expression,
    ) -> Result<D64CandidateVar, CandidateError> {
        if tape.len() < self.tape_len || cursor.len().checked_sub(1) != Some(tape.len()) {
            return Err(CandidateError::Shape);
        }
        for byte in tape {
            byte.validate_indices(builder)?;
        }
        let mut machine = CandidateMachine {
            builder,
            tape: tape.iter().map(ByteVar::bit_expressions).collect(),
            cursor,
        };
        let mut permutation: Vec<_> = (0..64)
            .map(|i| Expression::constant(Fr::from_u64(i)))
            .collect();
        let mut positions = Vec::with_capacity(self.weight);
        for i in 0..self.weight {
            let n = 64 - i;
            let bits = (usize::BITS - (n - 1).leading_zeros()) as usize;
            let mut selected_bits = vec![Expression::zero(); bits];
            if n > 1 {
                let mut active = activity.clone();
                for _ in 0..self.trials {
                    let byte = machine.read(active.clone());
                    let trial_bits: Vec<_> = byte.into_iter().take(bits).collect();
                    let valid = machine.less_than(&trial_bits, n);
                    let accept = machine.builder.multiply(active.clone(), valid);
                    for (selected, bit) in selected_bits.iter_mut().zip(trial_bits) {
                        *selected =
                            selected.clone() + machine.builder.multiply(accept.clone(), bit);
                    }
                    active = machine.materialize(active - accept);
                }
                machine.builder.assert_zero(active.clone());
                if machine
                    .builder
                    .evaluate(&active)
                    .is_ok_and(|v| v != Fr::from_u64(0))
                {
                    return Err(CandidateError::CapacityExceeded);
                }
            }
            let selectors: Vec<_> = (0..n)
                .map(|v| machine.equal_bits(&selected_bits, v))
                .collect();
            let left = permutation.get(i).ok_or(CandidateError::Shape)?.clone();
            let mut right = Expression::zero();
            for (selector, entry) in selectors.iter().zip(permutation.iter().skip(i)) {
                right = right + machine.builder.multiply(selector.clone(), entry.clone());
            }
            let right = machine.materialize(right);
            for (slot, selector) in permutation
                .iter_mut()
                .skip(i + 1)
                .zip(selectors.iter().skip(1))
            {
                let delta = machine
                    .builder
                    .multiply(selector.clone(), left.clone() - slot.clone());
                *slot = machine.materialize(slot.clone() + delta);
            }
            *permutation.get_mut(i).ok_or(CandidateError::Shape)? = right.clone();
            let value = machine
                .builder
                .evaluate(&right)
                .ok()
                .and_then(|v| v.to_u64_checked())
                .and_then(|v| u8::try_from(v).ok());
            let position = ByteVar::allocate(machine.builder, value);
            machine.builder.assert_equal(position.expression(), right);
            for high in position.bit_expressions().into_iter().skip(6) {
                machine.builder.assert_zero(high);
            }
            positions.push(position);
        }
        let mut coefficients = Vec::with_capacity(self.weight);
        let mut dense = vec![Expression::zero(); 64];
        for (i, position) in positions.iter().enumerate() {
            let magnitude = if i < self.count_pm1 { 1 } else { 2 };
            let sign = machine
                .read(activity.clone())
                .into_iter()
                .next()
                .ok_or(CandidateError::Shape)?;
            let coefficient =
                (Expression::one() - sign.scale(Fr::from_u64(2))).scale(Fr::from_u64(magnitude));
            let value = machine.signed(coefficient.clone(), magnitude as u128)?;
            coefficients.push(value);
            let bits: Vec<_> = position.bit_expressions().into_iter().take(6).collect();
            for (k, value) in dense.iter_mut().enumerate() {
                let selector = machine.equal_bits(&bits, k);
                *value = value.clone() + machine.builder.multiply(selector, coefficient.clone());
            }
        }
        let dense = dense
            .into_iter()
            .map(|value| machine.signed(value, 2))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| CandidateError::Shape)?;
        let consumed = machine
            .cursor
            .iter()
            .enumerate()
            .fold(Expression::zero(), |sum, (i, selector)| {
                sum + selector.clone().scale(Fr::from_u128(i as u128))
            });
        Ok(D64CandidateVar {
            positions,
            coefficients,
            dense,
            consumed,
            end_cursor: machine.cursor,
        })
    }
}

/// Ordered candidate and dense coefficients, with private local consumption.
/// The result is not a norm-accepted challenge or an authenticated stream offset.
pub struct D64CandidateVar {
    positions: Vec<ByteVar>,
    coefficients: Vec<SignedVar>,
    dense: [SignedVar; 64],
    consumed: Expression,
    end_cursor: Vec<Expression>,
}
impl D64CandidateVar {
    /// Native Fisher–Yates output order, paired with coefficients().
    pub fn positions(&self) -> &[ByteVar] {
        &self.positions
    }
    /// Ordered ±1/±2 coefficients, bound to their corresponding sign bytes.
    pub fn coefficients(&self) -> &[SignedVar] {
        &self.coefficients
    }
    /// Exact scatter of ordered coefficients into D64 coordinates.
    pub fn dense(&self) -> &[SignedVar; 64] {
        &self.dense
    }
    /// Private byte count in [0,tape_len], constrained by one-hot cursor transitions.
    pub fn consumed_bytes(&self) -> Expression {
        self.consumed.clone()
    }
    /// Private one-hot ending offset; not yet linked to another candidate's input.
    pub fn end_cursor(&self) -> &[Expression] {
        &self.end_cursor
    }
}

struct CandidateMachine<'a> {
    builder: &'a mut R1csBuilder<Fr>,
    tape: Vec<[Expression; 8]>,
    cursor: Vec<Expression>,
}
impl CandidateMachine<'_> {
    fn materialize(&mut self, expression: Expression) -> Expression {
        let witness = self.builder.evaluate(&expression).ok();
        let variable = self.builder.alloc_witness(witness);
        self.builder.assert_equal(variable, expression);
        Expression::variable(variable)
    }

    fn read(&mut self, active: Expression) -> [Expression; 8] {
        if let Some(end) = self.cursor.last() {
            self.builder
                .assert_product(active.clone(), end.clone(), Expression::zero());
        }
        let byte = std::array::from_fn(|bit| {
            let mut sum = Expression::zero();
            for (selector, byte) in self.cursor.iter().zip(&self.tape) {
                if let Some(input) = byte.get(bit) {
                    sum = sum + self.builder.multiply(selector.clone(), input.clone());
                }
            }
            self.materialize(sum)
        });
        let old = std::mem::take(&mut self.cursor);
        let previous = std::iter::once(Expression::zero()).chain(old.iter().cloned());
        for (current, previous) in old.iter().zip(previous) {
            let stay = self
                .builder
                .multiply(Expression::one() - active.clone(), current.clone());
            let shift = self.builder.multiply(active.clone(), previous);
            self.cursor.push(stay + shift);
        }
        byte
    }

    fn equal_bits(&mut self, bits: &[Expression], value: usize) -> Expression {
        bits.iter()
            .enumerate()
            .fold(Expression::one(), |equal, (i, bit)| {
                let same = if value & (1 << i) == 0 {
                    Expression::one() - bit.clone()
                } else {
                    bit.clone()
                };
                self.builder.multiply(equal, same)
            })
    }

    fn less_than(&mut self, bits: &[Expression], bound: usize) -> Expression {
        if bound == 1 << bits.len() {
            return Expression::one();
        }
        let mut equal = Expression::one();
        let mut less = Expression::zero();
        for (i, bit) in bits.iter().enumerate().rev() {
            if bound & (1 << i) != 0 {
                less = less
                    + self
                        .builder
                        .multiply(equal.clone(), Expression::one() - bit.clone());
                equal = self.builder.multiply(equal, bit.clone());
            } else {
                equal = self
                    .builder
                    .multiply(equal, Expression::one() - bit.clone());
            }
        }
        self.materialize(less)
    }

    fn signed(&mut self, expression: Expression, bound: u128) -> Result<SignedVar, CandidateError> {
        let hint = self
            .builder
            .evaluate(&expression)
            .ok()
            .map(|value| {
                (-2i128..=2)
                    .find(|&v| {
                        let magnitude = Fr::from_u128(v.unsigned_abs());
                        value == if v < 0 { -magnitude } else { magnitude }
                    })
                    .ok_or(CandidateError::Shape)
            })
            .transpose()?;
        let result = SignedVar::allocate(self.builder, bound, hint)?;
        self.builder.assert_equal(result.variable(), expression);
        Ok(result)
    }
}
