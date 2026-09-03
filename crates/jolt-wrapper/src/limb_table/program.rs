//! The row program: every row is `z ≡ Σ_s κ_s · Z(x_s) · Z(y_s) (mod q)` over
//! earlier rows (or itself, for input, constant and witness rows whose value
//! is not computed), optionally pinned to a public value. Operands are
//! public-coefficient linear combinations of rows that are never
//! materialized: a product of two combinations distributes into slots.

use std::collections::HashMap;
use std::ops::Range;

use ark_bn254::{Fq, Fq2};
use ark_ff::{AdditiveGroup, Field, PrimeField, Zero};

use super::layout::{CELL_ROWS, ROWS};
use super::tower::{fq12_coords, fq12_from_coords};

pub type RowId = u32;

/// One product term of a row: `kappa · Z(x) · Z(y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub x: RowId,
    pub y: RowId,
    /// Public coefficient (the `X` operand column holds `kappa·Z(x)`).
    pub kappa: i32,
    /// Sign carried by the `Y` operand (`±1`: a conjugated digit-selected
    /// coordinate); the slot's total coefficient is `kappa·y_sign`.
    pub y_sign: i32,
}

impl Slot {
    pub const fn coefficient(&self) -> i32 {
        self.kappa * self.y_sign
    }
}

/// A public small-integer linear combination of rows; empty is zero.
pub type Lin = Vec<(RowId, i32)>;

pub fn lin(row: RowId) -> Lin {
    vec![(row, 1)]
}

pub fn lin_neg(a: &Lin) -> Lin {
    a.iter().map(|&(row, k)| (row, -k)).collect()
}

pub fn lin_scale(a: &Lin, factor: i32) -> Lin {
    a.iter().map(|&(row, k)| (row, k * factor)).collect()
}

pub fn lin_add(a: &Lin, b: &Lin) -> Lin {
    let mut out = a.clone();
    for &(row, k) in b {
        if let Some(slot) = out.iter_mut().find(|(r, _)| *r == row) {
            slot.1 += k;
        } else {
            out.push((row, k));
        }
    }
    out.retain(|(_, k)| *k != 0);
    out
}

pub fn lin_sub(a: &Lin, b: &Lin) -> Lin {
    lin_add(a, &lin_neg(b))
}

/// Slots of `kappa · a · b`.
pub fn mul_terms(a: &Lin, b: &Lin, kappa: i32) -> Vec<Slot> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for &(x, kx) in a {
        for &(y, ky) in b {
            out.push(Slot {
                x,
                y,
                kappa: kappa * kx * ky,
                y_sign: 1,
            });
        }
    }
    out
}

/// Where a row's value comes from; `Compute` rows evaluate their slots, the
/// others hold a value and reference themselves (`z = z · 1`) so one relation
/// covers every row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Compute,
    /// Coordinate `index` of the committed input vector.
    Input(usize),
    /// Public value, pinned by the verifier.
    Constant(Fq),
    /// Prover witness `num / den` (a curve slope; sums of products), bound
    /// by a pinned row.
    Quotient {
        num: Vec<Slot>,
        den: Vec<Slot>,
    },
    /// Prover witness: coordinate `coord` of the `Fq2` quotient `num / den`.
    QuotientFq2 {
        num: [Vec<Slot>; 2],
        den: [Vec<Slot>; 2],
        coord: u8,
    },
    /// Prover witness: coordinate `coord` of the `Fq12` inverse of `coords`.
    InverseFq12 {
        coords: Box<[Lin; 12]>,
        coord: u8,
    },
    /// Prover witness `den⁻¹`, or zero when `den = 0`.
    InverseOrZero {
        den: Vec<Slot>,
    },
    /// `Σ κ·x·y = z` over the integers (`k = 0`; the `exact` VK column).
    Exact,
    /// Exact `Σ κ·x·y + (1 − flag)·2^256 = z` with the committed sign flag
    /// `flag = [of ≥ (q+1)/2]` of row `of` (`of > −of`).
    Sign {
        of: RowId,
    },
}

impl Source {
    /// Rows whose limb identity is exact (`k = 0`).
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact | Self::Sign { .. })
    }
}

/// `(q + 1)/2`: the smallest `y` with `y > −y`.
pub fn half_plus_one() -> Fq {
    Fq::from(2u64)
        .inverse()
        .unwrap_or_else(|| unreachable!("2 is invertible"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSpec {
    pub slots: Vec<Slot>,
    pub source: Source,
    /// Public value the verifier pins the row to.
    pub pin: Option<Fq>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvaluationError {
    #[error("row {row}: inverse of zero (exceptional case)")]
    NonInvertible { row: usize },
    #[error("row {row}: pinned to {expected} but evaluates to {actual}")]
    PinViolated {
        row: usize,
        expected: Fq,
        actual: Fq,
    },
}

/// The fixed row program of one profile: `ROWS` rows at fixed positions
/// (`layout`), filled through a cursor; rows never written are padding
/// (`z = 0`). Emission order is the evaluation order, so a row may read
/// rows placed after it.
#[derive(Clone, Debug)]
pub struct Program {
    pub rows: Vec<RowSpec>,
    pub one: RowId,
    pub zero: RowId,
    pub input_rows: Vec<RowId>,
    /// Rows in emission (evaluation) order.
    pub order: Vec<RowId>,
    constants: HashMap<[u64; 4], RowId>,
    cursor: RowId,
    constant_cursor: RowId,
    constant_end: RowId,
}

impl Program {
    /// An all-padding program whose public constants are allocated from the
    /// rows `constants` (the `one` and `zero` rows come first).
    pub fn new(constants: Range<RowId>) -> Self {
        let padding = RowSpec {
            slots: Vec::new(),
            source: Source::Compute,
            pin: None,
        };
        let mut program = Self {
            rows: vec![padding; ROWS],
            one: 0,
            zero: 0,
            input_rows: Vec::new(),
            order: Vec::new(),
            constants: HashMap::new(),
            cursor: 0,
            constant_cursor: constants.start,
            constant_end: constants.end,
        };
        program.one = program.constant(Fq::ONE);
        program.zero = program.constant(Fq::ZERO);
        program
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Moves the emission cursor to `row`.
    pub fn at(&mut self, row: RowId) {
        self.cursor = row;
    }

    /// The first row of `cell`.
    pub const fn cell_row(cell: u32) -> RowId {
        cell * CELL_ROWS
    }

    pub fn cursor(&self) -> RowId {
        self.cursor
    }

    /// Number of rows written so far.
    pub fn emitted(&self) -> usize {
        self.order.len()
    }

    /// Writes `spec` at `id` (a padding row so far) and appends it to the
    /// evaluation order.
    pub fn write_row(&mut self, id: RowId, spec: RowSpec) {
        self.write(id, spec);
    }

    fn write(&mut self, id: RowId, spec: RowSpec) {
        let slot = &mut self.rows[id as usize];
        assert!(
            slot.slots.is_empty() && slot.source == Source::Compute && slot.pin.is_none(),
            "row {id} written twice"
        );
        *slot = spec;
        self.order.push(id);
    }

    fn next(&mut self) -> RowId {
        let id = self.cursor;
        self.cursor += 1;
        id
    }

    /// A row whose value is not computed from operands (an input or a
    /// public constant): no slots, exempt from the limb identity (`free`).
    fn self_row_at(&mut self, id: RowId, source: Source, pin: Option<Fq>) {
        self.write(
            id,
            RowSpec {
                slots: Vec::new(),
                source,
                pin,
            },
        );
    }

    /// Rows with an exact limb identity (`k = 0`).
    pub fn exact_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row, spec)| spec.source.is_exact().then_some(row))
    }

    /// Rows exempt from the limb identity: inputs and public constants.
    pub fn free_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows.iter().enumerate().filter_map(|(row, spec)| {
            matches!(spec.source, Source::Input(_) | Source::Constant(_)).then_some(row)
        })
    }

    /// A public constant row (deduplicated by value) in the constants region.
    pub fn constant(&mut self, value: Fq) -> RowId {
        let key = value.into_bigint().0;
        if let Some(&row) = self.constants.get(&key) {
            return row;
        }
        assert!(
            self.constant_cursor < self.constant_end,
            "constants region full"
        );
        let row = self.constant_cursor;
        self.constant_cursor += 1;
        self.self_row_at(row, Source::Constant(value), Some(value));
        let _ = self.constants.insert(key, row);
        row
    }

    /// A pinned public constant at a fixed row (setup elements, line
    /// coefficients) outside the constants region.
    pub fn pinned_constant_at(&mut self, row: RowId, value: Fq) {
        self.self_row_at(row, Source::Constant(value), Some(value));
    }

    /// Committed input coordinate `index` at a fixed row.
    pub fn input_at(&mut self, row: RowId, index: usize) {
        assert_eq!(index, self.input_rows.len(), "inputs are declared in order");
        self.self_row_at(row, Source::Input(index), None);
        self.input_rows.push(row);
    }

    pub fn constant_lin(&mut self, value: Fq) -> Lin {
        lin(self.constant(value))
    }

    /// The row at the cursor holding committed input coordinate `index`.
    pub fn input(&mut self, index: usize) -> RowId {
        assert_eq!(index, self.input_rows.len(), "inputs are declared in order");
        let row = self.next();
        self.self_row_at(row, Source::Input(index), None);
        self.input_rows.push(row);
        row
    }

    pub fn witness(&mut self, source: Source) -> RowId {
        let row = self.next();
        self.self_row_at(row, source, None);
        row
    }

    /// A computed row `z = Σ slots` at the cursor; slot positions are kept
    /// (they are the wiring's slot index), including zero-coefficient
    /// placeholders.
    pub fn compute(&mut self, slots: Vec<Slot>) -> RowId {
        let id = self.next();
        self.write(
            id,
            RowSpec {
                slots,
                source: Source::Compute,
                pin: None,
            },
        );
        id
    }

    /// A computed row the verifier pins to `value`.
    pub fn pinned(&mut self, slots: Vec<Slot>, value: Fq) -> RowId {
        let id = self.compute(slots);
        self.rows[id as usize].pin = Some(value);
        id
    }

    pub fn max_slots(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.slots.len())
            .max()
            .unwrap_or(0)
    }

    /// Largest `Σ_s |κ_s|` over rows: bounds the integer size of a row's sum.
    pub fn max_kappa_sum(&self) -> u64 {
        self.rows
            .iter()
            .map(|row| {
                row.slots
                    .iter()
                    .map(|slot| u64::from(slot.kappa.unsigned_abs()))
                    .sum()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn pinned_rows(&self) -> impl Iterator<Item = (usize, Fq)> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row, spec)| spec.pin.map(|value| (row, value)))
    }

    /// Evaluates every row in emission order (padding rows are zero);
    /// `inputs` are the committed coordinates in
    /// [`super::dory::input_elements`] order.
    pub fn evaluate(&self, inputs: &[Fq]) -> Result<Vec<Fq>, EvaluationError> {
        let mut values = vec![Fq::ZERO; self.rows.len()];
        let eval_lin = |values: &[Fq], l: &Lin| {
            l.iter().fold(Fq::ZERO, |acc, &(row, k)| {
                acc + values[row as usize] * signed(k)
            })
        };
        let eval_slots = |values: &[Fq], slots: &[Slot]| {
            slots.iter().fold(Fq::ZERO, |acc, slot| {
                acc + values[slot.x as usize] * values[slot.y as usize] * signed(slot.coefficient())
            })
        };
        for &id in &self.order {
            let row = id as usize;
            let spec = &self.rows[row];
            let value = match &spec.source {
                Source::Compute => eval_slots(&values, &spec.slots),
                Source::Input(index) => inputs[*index],
                Source::Constant(value) => *value,
                // A zero denominator is an exceptional affine add; the slope
                // is set to zero and the gadget's pinned slope row fails, so
                // the case surfaces as a pin violation, not an evaluation error.
                Source::Quotient { num, den } => {
                    eval_slots(&values, num)
                        * eval_slots(&values, den).inverse().unwrap_or(Fq::ZERO)
                }
                Source::QuotientFq2 { num, den, coord } => {
                    let num = Fq2::new(eval_slots(&values, &num[0]), eval_slots(&values, &num[1]));
                    let den = Fq2::new(eval_slots(&values, &den[0]), eval_slots(&values, &den[1]));
                    let quotient = den
                        .inverse()
                        .map_or(Fq2::new(Fq::ZERO, Fq::ZERO), |inverse| num * inverse);
                    if *coord == 0 {
                        quotient.c0
                    } else {
                        quotient.c1
                    }
                }
                Source::InverseFq12 { coords, coord } => {
                    let element: [Fq; 12] = std::array::from_fn(|c| eval_lin(&values, &coords[c]));
                    let inverse = fq12_from_coords(&element)
                        .inverse()
                        .ok_or(EvaluationError::NonInvertible { row })?;
                    fq12_coords(&inverse)[*coord as usize]
                }
                Source::InverseOrZero { den } => {
                    eval_slots(&values, den).inverse().unwrap_or(Fq::ZERO)
                }
                Source::Exact => eval_slots(&values, &spec.slots),
                Source::Sign { of } => {
                    let flag = sign_flag(values[*of as usize]);
                    let offset = if flag { Fq::ZERO } else { pow_256() };
                    eval_slots(&values, &spec.slots) + offset
                }
            };
            values[row] = value;
        }
        Ok(values)
    }

    /// Checks every pinned row against its public value.
    pub fn check_pins(&self, values: &[Fq]) -> Result<(), EvaluationError> {
        for (row, expected) in self.pinned_rows() {
            if values[row] != expected {
                return Err(EvaluationError::PinViolated {
                    row,
                    expected,
                    actual: values[row],
                });
            }
        }
        Ok(())
    }
}

/// The canonical sign of `y`: `y > −y`, i.e. `y ≥ (q + 1)/2`.
pub fn sign_flag(y: Fq) -> bool {
    y >= half_plus_one()
}

/// `2^256 mod q`.
fn pow_256() -> Fq {
    Fq::from(2u64).pow([256u64])
}

pub fn signed(k: i32) -> Fq {
    let magnitude = Fq::from(u64::from(k.unsigned_abs()));
    if k < 0 {
        -magnitude
    } else {
        magnitude
    }
}

impl Program {
    pub fn lin_value(values: &[Fq], l: &Lin) -> Fq {
        l.iter().fold(Fq::zero(), |acc, &(row, k)| {
            acc + values[row as usize] * signed(k)
        })
    }
}
