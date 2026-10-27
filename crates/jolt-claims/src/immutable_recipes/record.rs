//! Native-only observation of the canonical verifier's existing expression values.

use super::{KeySink, Phase, Record, IMMUTABLE_RECIPE_MODULUS};
use crate::protocols::composed::ComposedOpeningId;
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId,
};
use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use crate::{Expr, Source};
use jolt_field::Ring;
use std::collections::BTreeSet;
use std::fmt::{self, Write};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, ThreadId};

const MAX_KEYS: usize = 4096;
const MAX_FACTORS: usize = 1_000_000;
const MAX_BYTES: usize = 32 * 1024 * 1024;

// The reviewed23-owner coverage plus actual Expr output evaluators. The
// factored Outer output stays canonical and is not recorded. Bytecode anchor21
// contributes input only; Full output is20.
const REQUIRED: &[(u64, Phase)] = &[
    (1, Phase::Support),
    (2, Phase::Input),
    (2, Phase::Output),
    (2, Phase::Support),
    (3, Phase::Input),
    (3, Phase::Output),
    (3, Phase::Support),
    (4, Phase::Input),
    (4, Phase::Output),
    (4, Phase::Support),
    (5, Phase::Input),
    (5, Phase::Output),
    (5, Phase::Support),
    (6, Phase::Input),
    (6, Phase::Output),
    (6, Phase::Support),
    (7, Phase::Input),
    (7, Phase::Output),
    (7, Phase::Support),
    (8, Phase::Input),
    (8, Phase::Output),
    (8, Phase::Support),
    (9, Phase::Input),
    (9, Phase::Output),
    (9, Phase::Support),
    (10, Phase::Input),
    (10, Phase::Output),
    (11, Phase::Input),
    (11, Phase::Output),
    (11, Phase::Support),
    (12, Phase::Input),
    (12, Phase::Output),
    (12, Phase::Support),
    (13, Phase::Input),
    (13, Phase::Output),
    (13, Phase::Support),
    (14, Phase::Output),
    (14, Phase::Support),
    (15, Phase::Output),
    (15, Phase::Support),
    (16, Phase::Output),
    (16, Phase::Support),
    (17, Phase::Output),
    (17, Phase::Support),
    (18, Phase::Input),
    (18, Phase::Output),
    (19, Phase::Input),
    (19, Phase::Output),
    (20, Phase::Output),
    (21, Phase::Input),
    (22, Phase::Input),
    (22, Phase::Output),
    (23, Phase::Input),
    (23, Phase::Output),
    (24, Phase::Input),
    (24, Phase::Output),
    (25, Phase::Input),
    (25, Phase::Output),
    (26, Phase::Input),
    (26, Phase::Output),
    (27, Phase::Input),
    (27, Phase::Output),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CaptureError {
    #[error("immutable recipe recorder already active")]
    Nested,
    #[error("immutable recipe recorder thread or session mismatch")]
    Ownership,
    #[error("immutable recipe recorder lock poisoned")]
    Poisoned,
    #[error("immutable recipe recorder capacity exceeded")]
    Capacity,
    /// Names the failing check and one detail value (a count or an index; 0 when
    /// the check has none), so a generation failure identifies its cause.
    #[error("canonical expression cannot be represented by the closed recipe: {0} (detail {1})")]
    Representation(&'static str, u64),
    #[error("one complete recipe key produced different canonical values")]
    Conflict,
    #[error("canonical capture did not cover the required relation phases")]
    Coverage,
    #[error("canonical recipe source serialization failed")]
    Serialization,
}

/// Saturating conversion of a count or index into an error detail value.
fn detail(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Implemented by exhaustive identifier matches, never by Debug output.
pub trait RenderId {
    fn render(&self, writer: &mut dyn Write) -> fmt::Result;
}

impl RenderId for usize {
    fn render(&self, writer: &mut dyn Write) -> fmt::Result {
        write!(writer, "{self}usize")
    }
}
impl RenderId for () {
    fn render(&self, writer: &mut dyn Write) -> fmt::Result {
        writer.write_str("()")
    }
}

#[derive(PartialEq, Eq)]
pub(crate) struct NativeRecord<O, D, C> {
    key: Vec<u64>,
    phase: Phase,
    headers: Vec<u64>,
    factors: Vec<u64>,
    coefficients: Vec<u128>,
    sources: Vec<Source<O, D, C>>,
    support: Vec<O>,
}

pub(crate) enum Captured {
    Jolt(NativeRecord<JoltOpeningId, JoltDerivedId, JoltChallengeId>),
    Composed(NativeRecord<ComposedOpeningId, JoltDerivedId, JoltChallengeId>),
    Field(NativeRecord<FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId>),
}

impl Captured {
    fn identity(&self) -> (&[u64], Phase) {
        match self {
            Self::Jolt(record) => (&record.key, record.phase),
            Self::Composed(record) => (&record.key, record.phase),
            Self::Field(record) => (&record.key, record.phase),
        }
    }
    fn same_value(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Jolt(left), Self::Jolt(right)) => left == right,
            (Self::Composed(left), Self::Composed(right)) => left == right,
            (Self::Field(left), Self::Field(right)) => left == right,
            _ => false,
        }
    }
}

struct Observed {
    record: Captured,
    occurrences: u64,
}

struct State {
    owner: ThreadId,
    id: u64,
    failure: Option<CaptureError>,
    records: Vec<Observed>,
    logical_bytes: usize,
    factor_events: usize,
}
struct Global {
    next: u64,
    active: Option<State>,
}
static GLOBAL: Mutex<Global> = Mutex::new(Global {
    next: 0,
    active: None,
});

fn global() -> Result<MutexGuard<'static, Global>, CaptureError> {
    GLOBAL.lock().map_err(|_| CaptureError::Poisoned)
}

/// A native owner must keep this guard around the existing canonical verification
/// and call finish only after every intended proof has been accepted. Dropping it
/// discards all capture. The Rc marker prevents transfer to another thread.
pub struct Session {
    id: u64,
    marker: PhantomData<Rc<()>>,
}
impl Session {
    pub fn begin() -> Result<Self, CaptureError> {
        let mut global = global()?;
        if let Some(active) = &mut global.active {
            active.failure = Some(CaptureError::Nested);
            return Err(CaptureError::Nested);
        }
        global.next = global.next.checked_add(1).ok_or(CaptureError::Capacity)?;
        let id = global.next;
        global.active = Some(State {
            owner: thread::current().id(),
            id,
            failure: None,
            records: Vec::new(),
            logical_bytes: 0,
            factor_events: 0,
        });
        Ok(Self {
            id,
            marker: PhantomData,
        })
    }

    pub fn finish(self) -> Result<Capture, CaptureError> {
        let mut global = global()?;
        let state = global.active.as_ref().ok_or(CaptureError::Ownership)?;
        if state.id != self.id || state.owner != thread::current().id() {
            return Err(CaptureError::Ownership);
        }
        let state = global.active.take().ok_or(CaptureError::Ownership)?;
        drop(global);
        if let Some(error) = state.failure {
            return Err(error);
        }
        if REQUIRED.iter().any(|(tag, phase)| {
            !state.records.iter().any(|record| {
                let (key, actual_phase) = record.record.identity();
                key.get(1) == Some(tag) && actual_phase == *phase
            })
        }) {
            return Err(CaptureError::Coverage);
        }
        Ok(Capture {
            records: state.records,
            logical_bytes: state.logical_bytes,
            factor_events: state.factor_events,
        })
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if let Ok(mut global) = global() {
            if global
                .active
                .as_ref()
                .is_some_and(|state| state.id == self.id)
            {
                global.active = None;
            }
        }
    }
}

pub struct Capture {
    records: Vec<Observed>,
    pub logical_bytes: usize,
    pub factor_events: usize,
}

struct Budget {
    bytes: usize,
    factors: usize,
}
impl Budget {
    fn charge<T>(&mut self, count: usize) -> Result<(), CaptureError> {
        self.bytes = self
            .bytes
            .checked_add(
                count
                    .checked_mul(std::mem::size_of::<T>())
                    .ok_or(CaptureError::Capacity)?,
            )
            .ok_or(CaptureError::Capacity)?;
        if self.bytes > MAX_BYTES {
            return Err(CaptureError::Capacity);
        }
        Ok(())
    }
    fn push<T>(&mut self, values: &mut Vec<T>, value: T) -> Result<(), CaptureError> {
        self.charge::<T>(1)?;
        values.try_reserve(1).map_err(|_| CaptureError::Capacity)?;
        values.push(value);
        Ok(())
    }
}
struct KeyBuffer<'a> {
    words: Vec<u64>,
    budget: &'a mut Budget,
    failed: bool,
}
impl KeySink for KeyBuffer<'_> {
    fn word(&mut self, value: u64) -> bool {
        if self.failed {
            return false;
        }
        if self.budget.push(&mut self.words, value).is_err() {
            self.failed = true;
        }
        !self.failed
    }
}

impl<O: Copy + Eq, D: Copy + Eq, C: Copy + Eq> NativeRecord<O, D, C> {
    fn new(
        phase: Phase,
        visit: impl FnOnce(&mut dyn KeySink) -> bool,
        budget: &mut Budget,
    ) -> Result<Self, CaptureError> {
        let mut key = KeyBuffer {
            words: Vec::new(),
            budget,
            failed: false,
        };
        if !visit(&mut key) || key.failed {
            return Err(CaptureError::Capacity);
        }
        Ok(Self {
            key: key.words,
            phase,
            headers: Vec::new(),
            factors: Vec::new(),
            coefficients: Vec::new(),
            sources: Vec::new(),
            support: Vec::new(),
        })
    }
    fn expression<F: Ring>(
        &mut self,
        expression: &Expr<F, O, D, C>,
        budget: &mut Budget,
    ) -> Result<(), CaptureError> {
        if self.phase == Phase::Support {
            return Err(CaptureError::Representation(
                "expression observed in the support phase",
                0,
            ));
        }
        for term in &expression.terms {
            if term.factors.len() > 12 {
                return Err(CaptureError::Representation(
                    "term has more than 12 factors (detail: tag << 32 | phase << 16 | factors)",
                    detail(term.factors.len())
                        | (u64::from(self.phase.code()) << 16)
                        | (self.key.get(1).copied().unwrap_or(0) << 32),
                ));
            }
            budget.factors = budget
                .factors
                .checked_add(term.factors.len())
                .ok_or(CaptureError::Capacity)?;
            if budget.factors > MAX_FACTORS {
                return Err(CaptureError::Capacity);
            }
            let coefficient =
                term.coefficient
                    .immutable_recipe_export()
                    .ok_or(CaptureError::Representation(
                        "coefficient has no canonical export",
                        0,
                    ))?;
            if coefficient >= IMMUTABLE_RECIPE_MODULUS {
                return Err(CaptureError::Representation(
                    "coefficient is not below the modulus",
                    0,
                ));
            }
            let coefficient_index = if let Some(index) = self
                .coefficients
                .iter()
                .position(|value| *value == coefficient)
            {
                index
            } else {
                let index = self.coefficients.len();
                budget.push(&mut self.coefficients, coefficient)?;
                index
            };
            let coefficient_index = u16::try_from(coefficient_index).map_err(|_| {
                CaptureError::Representation(
                    "coefficient index exceeds 16 bits",
                    detail(coefficient_index),
                )
            })?;
            let count = u64::try_from(term.factors.len())
                .map_err(|_| CaptureError::Representation("factor count exceeds 64 bits", 0))?;
            budget.push(
                &mut self.headers,
                (u64::from(coefficient_index) << 4) | count,
            )?;
            let mut word = 0;
            for (index, source) in term.factors.iter().enumerate() {
                let source_index =
                    if let Some(index) = self.sources.iter().position(|value| value == source) {
                        index
                    } else {
                        let index = self.sources.len();
                        budget.push(&mut self.sources, *source)?;
                        index
                    };
                let source_index = u16::try_from(source_index).map_err(|_| {
                    CaptureError::Representation(
                        "source index exceeds 16 bits",
                        detail(source_index),
                    )
                })?;
                word |= u64::from(source_index) << ((index % 4) * 16);
                if index % 4 == 3 || index + 1 == term.factors.len() {
                    budget.push(&mut self.factors, word)?;
                    word = 0;
                }
            }
        }
        self.validate()
    }
    fn validate(&self) -> Result<(), CaptureError> {
        Record::new(
            &self.key,
            self.phase,
            &self.headers,
            &self.factors,
            &self.coefficients,
            &self.sources,
            &self.support,
        )
        .map(|_| ())
        .map_err(|_| {
            CaptureError::Representation(
                "record failed closed-recipe validation",
                detail(self.key.len()),
            )
        })
    }
}

fn capture(make: impl FnOnce(&mut Budget) -> Result<Captured, CaptureError>) {
    let Ok(mut global) = global() else {
        return;
    };
    let Some(state) = &mut global.active else {
        return;
    };
    if state.failure.is_some() {
        return;
    }
    if state.owner != thread::current().id() {
        state.failure = Some(CaptureError::Ownership);
        return;
    }
    let mut budget = Budget {
        bytes: state.logical_bytes,
        factors: state.factor_events,
    };
    let result = make(&mut budget).and_then(|record| {
        if let Some(previous) = state
            .records
            .iter_mut()
            .find(|previous| previous.record.identity() == record.identity())
        {
            if !previous.record.same_value(&record) {
                return Err(CaptureError::Conflict);
            }
            previous.occurrences = previous
                .occurrences
                .checked_add(1)
                .ok_or(CaptureError::Capacity)?;
        } else {
            if state.records.len() == MAX_KEYS {
                return Err(CaptureError::Capacity);
            }
            budget.push(
                &mut state.records,
                Observed {
                    record,
                    occurrences: 1,
                },
            )?;
        }
        // Count every observation, including duplicate keys, against the finite
        // capture budget. Repeated verifier use cannot escape the event ceiling.
        state.logical_bytes = budget.bytes;
        state.factor_events = budget.factors;
        Ok(())
    });
    if let Err(error) = result {
        state.failure = Some(error);
    }
}

pub(crate) fn expression<F: Ring, O: Copy + Eq, D: Copy + Eq, C: Copy + Eq>(
    phase: Phase,
    visit: impl FnOnce(&mut dyn KeySink) -> bool,
    expression: &Expr<F, O, D, C>,
    wrap: impl FnOnce(NativeRecord<O, D, C>) -> Captured,
) {
    let Some(_field) = F::immutable_recipe_native_field() else {
        return;
    };
    capture(|budget| {
        let mut record = NativeRecord::new(phase, visit, budget)?;
        record.expression(expression, budget)?;
        Ok(wrap(record))
    });
}

pub(crate) fn support<F: Ring, O: Copy + Ord, D: Copy + Eq, C: Copy + Eq>(
    visit: impl FnOnce(&mut dyn KeySink) -> bool,
    openings: &BTreeSet<O>,
    wrap: impl FnOnce(NativeRecord<O, D, C>) -> Captured,
) {
    let Some(_field) = F::immutable_recipe_native_field() else {
        return;
    };
    capture(|budget| {
        let mut record = NativeRecord::new(Phase::Support, visit, budget)?;
        for opening in openings {
            budget.push(&mut record.support, *opening)?;
        }
        record.validate()?;
        Ok(wrap(record))
    });
}

struct SourceText {
    value: String,
}
impl Write for SourceText {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let length = self
            .value
            .len()
            .checked_add(value.len())
            .ok_or(fmt::Error)?;
        if length > MAX_BYTES {
            return Err(fmt::Error);
        }
        self.value
            .try_reserve(value.len())
            .map_err(|_| fmt::Error)?;
        self.value.push_str(value);
        Ok(())
    }
}

impl<O: RenderId, D: RenderId, C: RenderId> NativeRecord<O, D, C> {
    fn render(&self, writer: &mut dyn Write) -> fmt::Result {
        writer.write_str("match Record::new(&[")?;
        for word in &self.key {
            write!(writer, "{word}u64,")?;
        }
        let phase = match self.phase {
            Phase::Input => "Input",
            Phase::Output => "Output",
            Phase::Support => "Support",
        };
        write!(writer, "],Phase::{phase},&[")?;
        for word in &self.headers {
            write!(writer, "{word}u64,")?;
        }
        writer.write_str("],&[")?;
        for word in &self.factors {
            write!(writer, "{word}u64,")?;
        }
        writer.write_str("],&[")?;
        for coefficient in &self.coefficients {
            write!(writer, "{coefficient}u128,")?;
        }
        writer.write_str("],&[")?;
        for source in &self.sources {
            match source {
                Source::Opening(id) => {
                    writer.write_str("Source::Opening(")?;
                    id.render(writer)?;
                }
                Source::Challenge(id) => {
                    writer.write_str("Source::Challenge(")?;
                    id.render(writer)?;
                }
                Source::Derived(id) => {
                    writer.write_str("Source::Derived(")?;
                    id.render(writer)?;
                }
            }
            writer.write_str("),")?;
        }
        writer.write_str("],&[")?;
        for opening in &self.support {
            opening.render(writer)?;
            writer.write_char(',')?;
        }
        writer.write_str("]){Ok(record)=>record,Err(_)=>panic!(\"invalid closed recipe\")},")
    }
}

impl Capture {
    /// Emit only compiler-typed, const-validated records. Identifier spelling is
    /// produced by exhaustive source-owned matches, not parsed external text.
    pub fn source(&mut self) -> Result<String, CaptureError> {
        self.records.sort_unstable_by(|left, right| {
            let (left_key, left_phase) = left.record.identity();
            let (right_key, right_phase) = right.record.identity();
            left_phase
                .cmp(&right_phase)
                .then_with(|| left_key.cmp(right_key))
        });
        let mut writer = SourceText {
            value: String::new(),
        };
        self.render_source(&mut writer)
            .map_err(|_| CaptureError::Serialization)?;
        Ok(writer.value)
    }

    fn render_source(&self, writer: &mut dyn Write) -> fmt::Result {
        writer.write_str("use crate::Source;\n")?;
        for family in ["JOLT", "COMPOSED", "FIELD"] {
            let table = match family {
                "JOLT" => "JoltTable",
                "COMPOSED" => "ComposedTable",
                _ => "FieldTable",
            };
            write!(writer, "pub static {family}:{table}=match {table}::new(&[")?;
            for record in &self.records {
                match &record.record {
                    Captured::Jolt(record) if family == "JOLT" => record.render(writer)?,
                    Captured::Composed(record) if family == "COMPOSED" => record.render(writer)?,
                    Captured::Field(record) if family == "FIELD" => record.render(writer)?,
                    _ => {}
                }
            }
            writer.write_str("]){Ok(table)=>table,Err(_)=>panic!(\"invalid closed table\")};\n")?;
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
pub struct RecordIdentity<'a> {
    pub key: &'a [u64],
    pub phase: &'static str,
    pub observations: u64,
    pub headers: &'a [u64],
    pub factors: &'a [u64],
    pub coefficients: &'a [u128],
    pub sources: Vec<String>,
    pub support: Vec<String>,
}

impl<O: RenderId, D: RenderId, C: RenderId> NativeRecord<O, D, C> {
    fn identity(
        &self,
        observations: u64,
        budget: &mut Budget,
    ) -> Result<RecordIdentity<'_>, CaptureError> {
        let mut sources = Vec::new();
        let mut support = Vec::new();
        budget.charge::<String>(self.sources.len())?;
        budget.charge::<String>(self.support.len())?;
        sources
            .try_reserve(self.sources.len())
            .map_err(|_| CaptureError::Capacity)?;
        support
            .try_reserve(self.support.len())
            .map_err(|_| CaptureError::Capacity)?;
        for source in &self.sources {
            let mut writer = SourceText {
                value: String::new(),
            };
            let result = match source {
                Source::Opening(id) => writer
                    .write_str("Opening:")
                    .and_then(|()| id.render(&mut writer)),
                Source::Challenge(id) => writer
                    .write_str("Challenge:")
                    .and_then(|()| id.render(&mut writer)),
                Source::Derived(id) => writer
                    .write_str("Derived:")
                    .and_then(|()| id.render(&mut writer)),
            };
            result.map_err(|_| CaptureError::Serialization)?;
            budget.charge::<u8>(writer.value.len())?;
            sources.push(writer.value);
        }
        for opening in &self.support {
            let mut writer = SourceText {
                value: String::new(),
            };
            opening
                .render(&mut writer)
                .map_err(|_| CaptureError::Serialization)?;
            budget.charge::<u8>(writer.value.len())?;
            support.push(writer.value);
        }
        let phase = match self.phase {
            Phase::Input => "input",
            Phase::Output => "output",
            Phase::Support => "support",
        };
        Ok(RecordIdentity {
            key: &self.key,
            phase,
            observations,
            headers: &self.headers,
            factors: &self.factors,
            coefficients: &self.coefficients,
            sources,
            support,
        })
    }
}
impl Capture {
    pub fn identities(&self) -> Result<Vec<RecordIdentity<'_>>, CaptureError> {
        let mut identities = Vec::new();
        let mut budget = Budget {
            bytes: 0,
            factors: 0,
        };
        budget.charge::<RecordIdentity<'_>>(self.records.len())?;
        identities
            .try_reserve(self.records.len())
            .map_err(|_| CaptureError::Capacity)?;
        for observed in &self.records {
            identities.push(match &observed.record {
                Captured::Jolt(record) => record.identity(observed.occurrences, &mut budget)?,
                Captured::Composed(record) => record.identity(observed.occurrences, &mut budget)?,
                Captured::Field(record) => record.identity(observed.occurrences, &mut budget)?,
            });
        }
        Ok(identities)
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "bounded literal native recorder fixtures"
    )]
    use super::*;
    use crate::protocols::jolt::JoltRelationId;
    use crate::Term;
    use jolt_field::Prime128OffsetA7F7 as F;

    fn observe(coefficient: u64) {
        let expression = Expr::<F, JoltOpeningId, JoltDerivedId, JoltChallengeId> {
            terms: vec![Term {
                coefficient: F::from_u64(coefficient),
                factors: vec![Source::Opening(JoltOpeningId::TrustedAdvice {
                    relation: JoltRelationId::SpartanOuter,
                })],
            }],
        };
        super::expression(
            Phase::Output,
            |sink| sink.word(1) && sink.word(1),
            &expression,
            Captured::Jolt,
        );
    }

    #[test]
    fn session_conflicts_ownership_and_incomplete_coverage_never_emit() {
        let session = Session::begin().unwrap();
        observe(1);
        observe(2);
        assert!(matches!(session.finish(), Err(CaptureError::Conflict)));

        let session = Session::begin().unwrap();
        assert!(matches!(Session::begin(), Err(CaptureError::Nested)));
        assert!(matches!(session.finish(), Err(CaptureError::Nested)));

        let session = Session::begin().unwrap();
        thread::spawn(|| observe(1)).join().unwrap();
        assert!(matches!(session.finish(), Err(CaptureError::Ownership)));

        let session = Session::begin().unwrap();
        observe(1);
        assert!(matches!(session.finish(), Err(CaptureError::Coverage)));
    }

    #[test]
    fn reservation_limits_are_checked_before_capture_growth() {
        let mut budget = Budget {
            bytes: MAX_BYTES,
            factors: 0,
        };
        let mut words = Vec::new();
        assert_eq!(budget.push(&mut words, 7u64), Err(CaptureError::Capacity));
        assert!(words.is_empty());
        assert_eq!(words.capacity(), 0);
    }

    #[test]
    fn typed_identifier_source_has_a_fixed_constructor_spelling() {
        let source = JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::SpartanOuter,
        };
        let mut writer = SourceText {
            value: String::new(),
        };
        source.render(&mut writer).unwrap();
        assert_eq!(writer.value,
            "crate::protocols::jolt::JoltOpeningId::TrustedAdvice{relation:crate::protocols::jolt::JoltRelationId::SpartanOuter,}");
    }
}
