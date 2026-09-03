//! The row program: every row is `z ≡ Σ_s κ_s · Z(x_s) · Z(y_s) (mod q)` over
//! earlier rows (or itself, for input, constant and witness rows whose value
//! is not computed), optionally pinned to a public value. Operands are
//! public-coefficient linear combinations of rows that are never
//! materialized: a product of two combinations distributes into slots.

use std::collections::HashMap;
use std::ops::Range;

use ark_bn254::{Fq, Fq2};
use ark_ff::{AdditiveGroup, Field, PrimeField, Zero};

use super::tower::{fq12_coords, fq12_from_coords};

pub type RowId = u32;

/// One product term of a row: `kappa · Z(x) · Z(y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub x: RowId,
    pub y: RowId,
    pub kappa: i32,
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
    /// Prover witness `(Σ κ Z(row))⁻¹`, bound by a pinned product row.
    Inverse(Lin),
    /// Prover witness: coordinate `coord` of `(re + im·u)⁻¹ ∈ Fq2`.
    InverseFq2 {
        re: Lin,
        im: Lin,
        coord: u8,
    },
    /// Prover witness: coordinate `coord` of the `Fq12` inverse of `coords`.
    InverseFq12 {
        coords: Box<[Lin; 12]>,
        coord: u8,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSpec {
    pub slots: Vec<Slot>,
    pub source: Source,
    /// Public value the verifier pins the row to.
    pub pin: Option<Fq>,
}

#[derive(Clone, Debug)]
pub struct Section {
    pub name: &'static str,
    pub rows: Range<usize>,
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

/// The fixed row program of one profile; the schedule (sources, slots, pins)
/// is public and instance-dependent only through public digits and constants.
#[derive(Clone, Debug)]
pub struct Program {
    pub rows: Vec<RowSpec>,
    pub one: RowId,
    pub zero: RowId,
    pub sections: Vec<Section>,
    pub input_rows: Vec<RowId>,
    constants: HashMap<[u64; 4], RowId>,
    section_start: usize,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    pub fn new() -> Self {
        let mut program = Self {
            rows: Vec::new(),
            one: 0,
            zero: 0,
            sections: Vec::new(),
            input_rows: Vec::new(),
            constants: HashMap::new(),
            section_start: 0,
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

    /// Closes the current section under `name`.
    pub fn end_section(&mut self, name: &'static str) {
        self.sections.push(Section {
            name,
            rows: self.section_start..self.rows.len(),
        });
        self.section_start = self.rows.len();
    }

    fn self_row(&mut self, source: Source, pin: Option<Fq>) -> RowId {
        let id = self.rows.len() as RowId;
        let one = if self.rows.is_empty() { id } else { self.one };
        self.rows.push(RowSpec {
            slots: vec![Slot {
                x: id,
                y: one,
                kappa: 1,
            }],
            source,
            pin,
        });
        id
    }

    /// A public constant row (deduplicated by value).
    pub fn constant(&mut self, value: Fq) -> RowId {
        let key = value.into_bigint().0;
        if let Some(&row) = self.constants.get(&key) {
            return row;
        }
        let row = self.self_row(Source::Constant(value), Some(value));
        let _ = self.constants.insert(key, row);
        row
    }

    pub fn constant_lin(&mut self, value: Fq) -> Lin {
        lin(self.constant(value))
    }

    /// The row holding committed input coordinate `index`.
    pub fn input(&mut self, index: usize) -> RowId {
        assert_eq!(index, self.input_rows.len(), "inputs are declared in order");
        let row = self.self_row(Source::Input(index), None);
        self.input_rows.push(row);
        row
    }

    pub fn witness(&mut self, source: Source) -> RowId {
        self.self_row(source, None)
    }

    /// Merges slots with equal unordered operand pairs and drops zero terms.
    fn normalize(mut slots: Vec<Slot>) -> Vec<Slot> {
        for slot in &mut slots {
            if slot.x > slot.y {
                std::mem::swap(&mut slot.x, &mut slot.y);
            }
        }
        slots.sort_unstable_by_key(|slot| (slot.x, slot.y));
        let mut out: Vec<Slot> = Vec::with_capacity(slots.len());
        for slot in slots {
            match out.last_mut() {
                Some(last) if last.x == slot.x && last.y == slot.y => last.kappa += slot.kappa,
                _ => out.push(slot),
            }
        }
        out.retain(|slot| slot.kappa != 0);
        out
    }

    /// A computed row `z = Σ slots`.
    pub fn compute(&mut self, slots: Vec<Slot>) -> RowId {
        let id = self.rows.len() as RowId;
        self.rows.push(RowSpec {
            slots: Self::normalize(slots),
            source: Source::Compute,
            pin: None,
        });
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

    /// Evaluates every row in order; `inputs` are the committed coordinates
    /// in [`super::dory::input_elements`] order.
    pub fn evaluate(&self, inputs: &[Fq]) -> Result<Vec<Fq>, EvaluationError> {
        let mut values = Vec::with_capacity(self.rows.len());
        let eval_lin = |values: &[Fq], l: &Lin| {
            l.iter().fold(Fq::ZERO, |acc, &(row, k)| {
                acc + values[row as usize] * signed(k)
            })
        };
        for (row, spec) in self.rows.iter().enumerate() {
            let value = match &spec.source {
                Source::Compute => spec.slots.iter().fold(Fq::ZERO, |acc, slot| {
                    acc + values[slot.x as usize] * values[slot.y as usize] * signed(slot.kappa)
                }),
                Source::Input(index) => inputs[*index],
                Source::Constant(value) => *value,
                Source::Inverse(l) => eval_lin(&values, l)
                    .inverse()
                    .ok_or(EvaluationError::NonInvertible { row })?,
                Source::InverseFq2 { re, im, coord } => {
                    let inverse = Fq2::new(eval_lin(&values, re), eval_lin(&values, im))
                        .inverse()
                        .ok_or(EvaluationError::NonInvertible { row })?;
                    if *coord == 0 {
                        inverse.c0
                    } else {
                        inverse.c1
                    }
                }
                Source::InverseFq12 { coords, coord } => {
                    let element: [Fq; 12] = std::array::from_fn(|c| eval_lin(&values, &coords[c]));
                    let inverse = fq12_from_coords(&element)
                        .inverse()
                        .ok_or(EvaluationError::NonInvertible { row })?;
                    fq12_coords(&inverse)[*coord as usize]
                }
            };
            values.push(value);
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
