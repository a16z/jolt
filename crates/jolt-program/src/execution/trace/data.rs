use jolt_riscv::JoltTraceRow;
#[cfg(feature = "serialization")]
use serde::{ser::SerializeSeq, Serialize, Serializer};

#[cfg(feature = "field-inline")]
use crate::execution::TraceError;
#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineTraceData;

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug)]
pub struct FieldEvent {
    pub cycle: usize,
    pub data: FieldInlineTraceData,
}

/// Shared execution storage. The proof view omits canonical trailing padding
/// without reallocating or changing the execution row count.
#[derive(Debug)]
pub struct TraceData {
    rows: Vec<JoltTraceRow>,
    proof_len: usize,
    #[cfg(feature = "field-inline")]
    field_events: Vec<FieldEvent>,
}

impl TraceData {
    /// Takes ownership of collected rows. The proof prefix ends at the last
    /// row that is not canonical padding.
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

    /// [`Self::new`] plus sparse field events, which must be strictly
    /// increasing in cycle and index existing rows. The proof prefix also
    /// covers the last event.
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
        reason = "private proof_len is bounded by rows at construction"
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
            .map(|event| &event.data)
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
            struct CycleRecord<'a> {
                row: &'a JoltTraceRow,
                #[cfg(feature = "field-inline")]
                field_inline: Option<&'a FieldInlineTraceData>,
            }
            let record = CycleRecord {
                row,
                #[cfg(feature = "field-inline")]
                field_inline: events
                    .next_if(|event| event.cycle == cycle)
                    .map(|event| &event.data),
            };
            sequence.serialize_element(&record)?;
        }
        sequence.end()
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn event(cycle: usize) -> FieldEvent {
        FieldEvent {
            cycle,
            data: FieldInlineTraceData {
                rs1: None,
                rs2: None,
                rd: None,
            },
        }
    }

    #[test]
    fn from_parts_rejects_misplaced_events_and_proves_through_the_last_event() {
        let padding = || vec![JoltTraceRow::default(); 4];
        for (events, rejected) in [
            (vec![event(2), event(1)], 1),
            (vec![event(1), event(1)], 1),
            (vec![event(4)], 4),
        ] {
            assert!(matches!(
                TraceData::from_parts(padding(), events),
                Err(TraceError::InvalidFieldEvent { cycle }) if cycle == rejected
            ));
        }

        let data = TraceData::from_parts(padding(), vec![event(0), event(2)]).unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(data.proof_len(), 3);
        assert!(data.field_inline(1).is_none());
        assert!(data.field_inline(2).is_some());
    }
}
