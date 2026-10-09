#[cfg(feature = "field-inline")]
use std::sync::Arc;

use jolt_riscv::JoltTraceRow;
#[cfg(feature = "serialization")]
use serde::{ser::SerializeSeq, Serialize, Serializer};

#[cfg(feature = "field-inline")]
use crate::execution::TraceError;
#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineTraceData;

/// One emitted row and its associated execution payload. Retained traces
/// separate these into compact core rows and sparse field events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub row: JoltTraceRow,
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<Arc<FieldInlineTraceData>>,
}

impl From<JoltTraceRow> for TraceEvent {
    fn from(row: JoltTraceRow) -> Self {
        Self {
            row,
            #[cfg(feature = "field-inline")]
            field_inline: None,
        }
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug)]
pub struct FieldEvent {
    pub cycle: usize,
    pub data: Arc<FieldInlineTraceData>,
}

/// Shared execution storage. The proof view omits canonical trailing padding
/// without reallocating or changing the execution row count.
#[derive(Default, Debug)]
pub struct TraceData {
    rows: Vec<JoltTraceRow>,
    proof_len: usize,
    #[cfg(feature = "field-inline")]
    field_events: Vec<FieldEvent>,
}

impl TraceData {
    pub fn new(rows: Vec<JoltTraceRow>) -> Self {
        let proof_len = rows
            .iter()
            .rposition(|row| *row != JoltTraceRow::default())
            .map_or(0, |index| index + 1);
        Self {
            rows,
            proof_len,
            #[cfg(feature = "field-inline")]
            field_events: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Transfers indexed parallel producer buffers without copying their rows.
    #[cfg(feature = "field-inline")]
    pub fn from_parts(
        rows: Vec<JoltTraceRow>,
        field_events: Vec<FieldEvent>,
    ) -> Result<Self, TraceError> {
        let mut previous = None;
        for event in &field_events {
            if event.cycle >= rows.len() || previous.is_some_and(|cycle| cycle >= event.cycle) {
                return Err(TraceError::InvalidFieldEvent { cycle: event.cycle });
            }
            previous = Some(event.cycle);
        }
        let mut data = Self::new(rows);
        if let Some(last) = field_events.last() {
            data.proof_len = data.proof_len.max(last.cycle + 1);
        }
        data.field_events = field_events;
        Ok(data)
    }

    pub fn push(&mut self, event: TraceEvent) {
        let cycle = self.rows.len();
        if event.row != JoltTraceRow::default() {
            self.proof_len = cycle + 1;
        }
        self.rows.push(event.row);
        #[cfg(feature = "field-inline")]
        if let Some(data) = event.field_inline {
            self.field_events.push(FieldEvent { cycle, data });
            self.proof_len = cycle + 1;
        }
    }

    pub fn rows(&self) -> &[JoltTraceRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn proof_len(&self) -> usize {
        self.proof_len
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "private proof_len is bounded by rows at construction and append"
    )]
    #[inline]
    pub fn proof_rows(&self) -> &[JoltTraceRow] {
        &self.rows[..self.proof_len]
    }

    #[cfg(feature = "field-inline")]
    pub fn field_events(&self) -> &[FieldEvent] {
        &self.field_events
    }

    #[cfg(feature = "field-inline")]
    pub fn field_inline(&self, cycle: usize) -> Option<&FieldInlineTraceData> {
        self.field_events
            .binary_search_by_key(&cycle, |event| event.cycle)
            .ok()
            .and_then(|index| self.field_events.get(index))
            .map(|event| event.data.as_ref())
    }
}

impl FromIterator<TraceEvent> for TraceData {
    fn from_iter<T: IntoIterator<Item = TraceEvent>>(iter: T) -> Self {
        let mut data = Self::default();
        for event in iter {
            data.push(event);
        }
        data
    }
}

#[cfg(feature = "serialization")]
impl Serialize for TraceData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        #[cfg(feature = "field-inline")]
        let mut events = self.field_events.iter().peekable();
        for (cycle, row) in self.rows.iter().enumerate() {
            #[cfg(not(feature = "field-inline"))]
            let _ = cycle;
            #[derive(Serialize)]
            struct EventRef<'a> {
                row: &'a JoltTraceRow,
                #[cfg(feature = "field-inline")]
                field_inline: Option<&'a FieldInlineTraceData>,
            }
            let event = EventRef {
                row,
                #[cfg(feature = "field-inline")]
                field_inline: events
                    .next_if(|event| event.cycle == cycle)
                    .map(|event| event.data.as_ref()),
            };
            sequence.serialize_element(&event)?;
        }
        sequence.end()
    }
}
