use jolt_field::JoltField;

use crate::{LinearCombination, R1csBuilder, Variable};

/// A boolean value backed by either a constant or an R1CS variable.
#[derive(Clone, Copy, Debug)]
pub struct Bit {
    variable: Option<Variable>,
    negated: bool,
    value: bool,
}

impl Bit {
    /// Allocates a boolean-constrained witness variable.
    pub fn allocate<F: JoltField>(builder: &mut R1csBuilder<F>, value: bool) -> Self {
        let variable = builder.alloc(F::from_u64(u64::from(value)));
        let bit = LinearCombination::variable(variable);
        builder.assert_product(
            bit.clone(),
            LinearCombination::one() - bit,
            LinearCombination::zero(),
        );
        Self {
            variable: Some(variable),
            negated: false,
            value,
        }
    }

    /// Creates a constant boolean without allocating a variable.
    pub const fn constant(value: bool) -> Self {
        Self {
            variable: None,
            negated: false,
            value,
        }
    }

    /// Returns the assigned boolean value.
    pub const fn value(self) -> bool {
        self.value
    }

    fn allocate_unchecked<F: JoltField>(builder: &mut R1csBuilder<F>, value: bool) -> Self {
        Self {
            variable: Some(builder.alloc(F::from_u64(u64::from(value)))),
            negated: false,
            value,
        }
    }

    fn not(mut self) -> Self {
        self.value = !self.value;
        if self.variable.is_some() {
            self.negated = !self.negated;
        }
        self
    }

    fn linear_combination<F: JoltField>(self) -> LinearCombination<F> {
        match (self.variable, self.negated) {
            (None, _) => LinearCombination::constant(F::from_u64(u64::from(self.value))),
            (Some(variable), false) => LinearCombination::variable(variable),
            (Some(variable), true) => {
                LinearCombination::one() - LinearCombination::variable(variable)
            }
        }
    }

    pub(super) fn add_scaled<F: JoltField>(self, target: &mut LinearCombination<F>, scale: F) {
        match (self.variable, self.negated, self.value) {
            (None, _, true) => target.terms.push((Variable::ONE, scale)),
            (None, _, false) => {}
            (Some(variable), false, _) => target.terms.push((variable, scale)),
            (Some(variable), true, _) => {
                target.terms.push((Variable::ONE, scale));
                target.terms.push((variable, -scale));
            }
        }
    }
}

fn xor<F: JoltField>(builder: &mut R1csBuilder<F>, lhs: Bit, rhs: Bit) -> Bit {
    match (lhs.variable, rhs.variable) {
        (None, None) => Bit::constant(lhs.value ^ rhs.value),
        (None, Some(_)) => {
            if lhs.value {
                rhs.not()
            } else {
                rhs
            }
        }
        (Some(_), None) => {
            if rhs.value {
                lhs.not()
            } else {
                lhs
            }
        }
        (Some(lhs_variable), Some(rhs_variable)) if lhs_variable == rhs_variable => {
            Bit::constant(lhs.negated ^ rhs.negated)
        }
        (Some(_), Some(_)) => {
            let output_bit = Bit::allocate_unchecked(builder, lhs.value ^ rhs.value);
            let lhs = lhs.linear_combination();
            let rhs = rhs.linear_combination();
            let output = output_bit.linear_combination();
            builder.assert_product(
                lhs.clone().scale(F::from_u64(2)),
                rhs.clone(),
                lhs + rhs - output,
            );
            output_bit
        }
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "word construction visits exactly the indices of both inputs"
)]
pub(super) fn xor_word<F: JoltField, const BITS: usize>(
    builder: &mut R1csBuilder<F>,
    lhs: &[Bit; BITS],
    rhs: &[Bit; BITS],
) -> [Bit; BITS] {
    std::array::from_fn(|bit| xor(builder, lhs[bit], rhs[bit]))
}
