//! The two-mode emission context: one builder walk, either laying out
//! unknown wires (build) or assigning them from the verifier replay (witness).

use std::collections::BTreeMap;

use jolt_field::{CanonicalEncoding, Field, Fr, One, Zero};
use jolt_r1cs::{ConstraintMatrices, LinearCombination, R1csBuilder, Variable};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};

use super::replay::{Event, SqueezeKind};
use super::{RelationError, RowSpan, ScheduleEntry};

pub(crate) type Lc = LinearCombination<Fr>;

pub(crate) fn lc_var(variable: Variable) -> Lc {
    Lc::variable(variable)
}

pub(crate) fn lc_const(value: Fr) -> Lc {
    Lc::constant(value)
}

/// `Some(c)` when the combination has no non-constant term.
pub(crate) fn lc_constant(lc: &Lc) -> Option<Fr> {
    let mut value = Fr::zero();
    for &(variable, coefficient) in &lc.terms {
        if coefficient.is_zero() {
            continue;
        }
        if variable != Variable::ONE {
            return None;
        }
        value += coefficient;
    }
    Some(value)
}

/// Sparse accumulator for long linear forms (the builder's `Add` concatenates).
#[derive(Default)]
pub(crate) struct Accum(BTreeMap<usize, Fr>);

impl Accum {
    pub(crate) fn add(&mut self, lc: &Lc, scale: Fr) {
        if scale.is_zero() {
            return;
        }
        for &(variable, coefficient) in &lc.terms {
            *self.0.entry(variable.index()).or_insert_with(Fr::zero) += coefficient * scale;
        }
    }

    pub(crate) fn finish(self) -> Lc {
        Lc {
            terms: self
                .0
                .into_iter()
                .filter(|(_, coefficient)| !coefficient.is_zero())
                .map(|(index, coefficient)| (Variable::new(index), coefficient))
                .collect(),
        }
    }
}

/// Byte-exact encoder for labels: routes through the transcript crate's
/// `AppendToTranscript` impls so the relation never re-spells an encoding.
#[derive(Default)]
struct ByteSink(Vec<u8>);

impl Transcript for ByteSink {
    type Challenge = Fr;

    fn new(_label: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn challenge(&mut self) -> Fr {
        Fr::zero()
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

pub(crate) fn encode<A: AppendToTranscript>(value: &A) -> Vec<u8> {
    let mut sink = ByteSink::default();
    value.append_to_transcript(&mut sink);
    sink.0
}

struct Cursor {
    events: Vec<Event>,
    position: usize,
}

impl Cursor {
    fn next(&mut self, expected: impl Fn() -> String) -> Result<&Event, RelationError> {
        let index = self.position;
        let event = self
            .events
            .get(index)
            .ok_or_else(|| RelationError::Schedule {
                index,
                expected: expected(),
                recorded: "end of replay".to_string(),
            })?;
        self.position += 1;
        Ok(event)
    }
}

pub(crate) struct Ctx {
    builder: R1csBuilder<Fr>,
    values: Vec<Option<Fr>>,
    replay: Option<Cursor>,
    schedule: Vec<ScheduleEntry>,
    rows: Vec<RowSpan>,
    section: (String, usize),
    constraints: usize,
}

impl Ctx {
    pub(crate) fn new(events: Option<Vec<Event>>) -> Self {
        Self {
            builder: R1csBuilder::new(),
            values: vec![Some(Fr::one())],
            replay: events.map(|events| Cursor {
                events,
                position: 0,
            }),
            schedule: Vec::new(),
            rows: Vec::new(),
            section: ("public".to_string(), 0),
            constraints: 0,
        }
    }

    /// Starts a labeled row span; the previous span closes at the current row.
    pub(crate) fn section(&mut self, label: impl Into<String>) {
        let (previous, start) =
            std::mem::replace(&mut self.section, (label.into(), self.constraints));
        if start < self.constraints {
            self.rows.push(RowSpan {
                label: previous,
                start,
                end: self.constraints,
            });
        }
    }

    pub(crate) fn alloc(&mut self, value: Option<Fr>) -> Variable {
        let variable = self.builder.alloc_witness(value);
        self.values.push(value);
        variable
    }

    pub(crate) fn value(&self, lc: &Lc) -> Option<Fr> {
        lc.evaluate(&self.values).ok()
    }

    pub(crate) fn variable_value(&self, variable: Variable) -> Option<Fr> {
        self.values.get(variable.index()).copied().flatten()
    }

    pub(crate) fn assign(&mut self, variable: Variable, value: Fr) -> Result<(), RelationError> {
        self.builder
            .assign(variable, value)
            .map_err(|_| RelationError::Witness(variable.index()))?;
        self.values[variable.index()] = Some(value);
        Ok(())
    }

    fn row(&mut self) {
        self.constraints += 1;
    }

    pub(crate) fn assert_product(&mut self, lhs: &Lc, rhs: &Lc, output: &Lc) {
        self.builder
            .assert_product(lhs.clone(), rhs.clone(), output.clone());
        self.row();
    }

    pub(crate) fn assert_eq(&mut self, lhs: &Lc, rhs: &Lc) {
        self.builder.assert_equal(lhs.clone(), rhs.clone());
        self.row();
    }

    /// `lhs · rhs`, folding constants (no row when either side is constant).
    pub(crate) fn mul(&mut self, lhs: &Lc, rhs: &Lc) -> Lc {
        if let Some(constant) = lc_constant(lhs) {
            return rhs.clone().scale(constant);
        }
        if let Some(constant) = lc_constant(rhs) {
            return lhs.clone().scale(constant);
        }
        let value = self.value(lhs).zip(self.value(rhs)).map(|(a, b)| a * b);
        let output = lc_var(self.alloc(value));
        self.assert_product(lhs, rhs, &output);
        output
    }

    /// `1 / x` as a hinted witness pinned by `x · inv = 1`.
    pub(crate) fn inverse(&mut self, x: &Lc) -> Lc {
        if let Some(constant) = lc_constant(x) {
            return lc_const(constant.inverse().unwrap_or_else(Fr::zero));
        }
        let value = self.value(x).and_then(|value| value.inverse());
        let inverse = lc_var(self.alloc(value));
        self.assert_product(x, &inverse, &Lc::one());
        inverse
    }

    /// A single variable equal to `lc`: the variable itself when `lc` already
    /// is one, else a fresh copy pinned by one equality row.
    pub(crate) fn materialize(&mut self, lc: &Lc) -> Variable {
        if let [(variable, coefficient)] = lc.terms.as_slice() {
            if *variable != Variable::ONE && coefficient.is_one() {
                return *variable;
            }
        }
        let variable = self.alloc(self.value(lc));
        self.assert_eq(&lc_var(variable), lc);
        variable
    }

    fn next_append(
        &mut self,
        expected: impl Fn() -> String,
    ) -> Result<Option<Vec<u8>>, RelationError> {
        let Some(cursor) = self.replay.as_mut() else {
            return Ok(None);
        };
        let index = cursor.position;
        match cursor.next(&expected)? {
            Event::Append(bytes) => Ok(Some(bytes.clone())),
            Event::Squeeze { kind, .. } => Err(RelationError::Schedule {
                index,
                expected: expected(),
                recorded: format!("squeeze {kind:?}"),
            }),
        }
    }

    /// Absorbs constant bytes (labels, counts, the empty domain separator).
    pub(crate) fn absorb_bytes(&mut self, bytes: &[u8]) -> Result<(), RelationError> {
        if let Some(recorded) = self.next_append(|| format!("bytes {bytes:02x?}"))? {
            if recorded != bytes {
                return Err(RelationError::Schedule {
                    index: self.replay.as_ref().map_or(0, |cursor| cursor.position - 1),
                    expected: format!("bytes {bytes:02x?}"),
                    recorded: format!("bytes {recorded:02x?}"),
                });
            }
        }
        self.schedule.push(ScheduleEntry::Bytes(bytes.to_vec()));
        Ok(())
    }

    pub(crate) fn absorb_label(&mut self, label: &'static [u8]) -> Result<(), RelationError> {
        self.absorb_bytes(&encode(&Label(label)))
    }

    pub(crate) fn absorb_label_count(
        &mut self,
        label: &'static [u8],
        count: usize,
    ) -> Result<(), RelationError> {
        self.absorb_bytes(&encode(&LabelWithCount(label, count as u64)))
    }

    fn decode_fr(bytes: &[u8], index: usize) -> Result<Fr, RelationError> {
        let mut le = bytes.to_vec();
        le.reverse();
        Fr::from_bytes_le_checked(&le).ok_or_else(|| RelationError::Schedule {
            index,
            expected: "field element".to_string(),
            recorded: format!("{} bytes {bytes:02x?}", bytes.len()),
        })
    }

    /// A prover-supplied field element absorbed at this point of the schedule.
    pub(crate) fn proof_fr(&mut self) -> Result<Lc, RelationError> {
        let value = match self.next_append(|| "proof field element".to_string())? {
            Some(bytes) => {
                let index = self.replay.as_ref().map_or(0, |cursor| cursor.position - 1);
                Some(Self::decode_fr(&bytes, index)?)
            }
            None => None,
        };
        let variable = self.alloc(value);
        self.schedule.push(ScheduleEntry::Fr(variable));
        Ok(lc_var(variable))
    }

    /// A verifier-computed field element absorbed at this point of the
    /// schedule; in witness mode the replayed value must match the wire.
    pub(crate) fn absorb_computed(&mut self, lc: &Lc) -> Result<Variable, RelationError> {
        let variable = self.materialize(lc);
        if let Some(bytes) = self.next_append(|| "computed field element".to_string())? {
            let index = self.replay.as_ref().map_or(0, |cursor| cursor.position - 1);
            let recorded = Self::decode_fr(&bytes, index)?;
            match self.variable_value(variable) {
                Some(value) if value == recorded => {}
                Some(value) => {
                    return Err(RelationError::Schedule {
                        index,
                        expected: format!("field element {value:?}"),
                        recorded: format!("field element {recorded:?}"),
                    })
                }
                None => self.assign(variable, recorded)?,
            }
        }
        self.schedule.push(ScheduleEntry::Fr(variable));
        Ok(variable)
    }

    /// `absorb_computed` for callers that keep no handle on the wire.
    pub(crate) fn absorb_value(&mut self, lc: &Lc) -> Result<(), RelationError> {
        self.absorb_computed(lc).map(|_| ())
    }

    /// Prover bytes the relation never interprets (Dory group elements).
    pub(crate) fn absorb_opaque(&mut self, len: usize) -> Result<(), RelationError> {
        if let Some(recorded) = self.next_append(|| format!("{len} opaque bytes"))? {
            if recorded.len() != len {
                return Err(RelationError::Schedule {
                    index: self.replay.as_ref().map_or(0, |cursor| cursor.position - 1),
                    expected: format!("{len} opaque bytes"),
                    recorded: format!("{} bytes", recorded.len()),
                });
            }
        }
        self.schedule.push(ScheduleEntry::Opaque { len });
        Ok(())
    }

    pub(crate) fn squeeze(&mut self, kind: SqueezeKind) -> Result<Lc, RelationError> {
        let value = match self.replay.as_mut() {
            None => None,
            Some(cursor) => {
                let index = cursor.position;
                match cursor.next(|| format!("squeeze {kind:?}"))? {
                    Event::Squeeze {
                        kind: recorded,
                        value,
                    } if *recorded == kind => Some(*value),
                    Event::Squeeze { kind: recorded, .. } => {
                        return Err(RelationError::Schedule {
                            index,
                            expected: format!("squeeze {kind:?}"),
                            recorded: format!("squeeze {recorded:?}"),
                        })
                    }
                    Event::Append(bytes) => {
                        return Err(RelationError::Schedule {
                            index,
                            expected: format!("squeeze {kind:?}"),
                            recorded: format!("{} bytes", bytes.len()),
                        })
                    }
                }
            }
        };
        let variable = self.alloc(value);
        self.schedule.push(ScheduleEntry::Squeeze {
            kind,
            var: variable,
        });
        Ok(lc_var(variable))
    }

    pub(crate) fn squeeze_vector(
        &mut self,
        kind: SqueezeKind,
        len: usize,
    ) -> Result<Vec<Lc>, RelationError> {
        (0..len).map(|_| self.squeeze(kind)).collect()
    }

    pub(crate) fn expect_replay_consumed(&self) -> Result<(), RelationError> {
        if let Some(cursor) = &self.replay {
            if cursor.position != cursor.events.len() {
                return Err(RelationError::Schedule {
                    index: cursor.position,
                    expected: "end of schedule".to_string(),
                    recorded: format!("{} more events", cursor.events.len() - cursor.position),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
    ) -> (
        ConstraintMatrices<Fr>,
        Vec<Option<Fr>>,
        Vec<ScheduleEntry>,
        Vec<RowSpan>,
    ) {
        self.section("end");
        (
            self.builder.into_matrices(),
            self.values,
            self.schedule,
            self.rows,
        )
    }
}
