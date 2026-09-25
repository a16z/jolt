//! Borrowed cycle windows, independent of the owned trace representation.

#[cfg(feature = "field-inline")]
use jolt_program::{execution::TraceRow as CapturedTraceRow, field_inline::FieldInlineTraceData};
use jolt_riscv::JoltTraceRow;

#[cfg(feature = "field-inline")]
use crate::{field_inline::FIELD_INLINE_LABEL, WitnessError};

/// One execution cycle, including any validated instruction-specific payload.
/// The cycle index stays absolute when the source visits a subrange.
#[derive(Clone, Copy)]
pub struct WitnessRow<'a> {
    pub cycle: usize,
    pub row: &'a JoltTraceRow,
    #[cfg(feature = "field-inline")]
    field_inline: FieldInlineRow<'a>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy)]
enum FieldInlineRow<'a> {
    Unavailable,
    Inactive,
    Active(&'a FieldInlineTraceData),
}

impl<'a> WitnessRow<'a> {
    /// Creates a row view without field-inline witness capability.
    pub fn new(cycle: usize, row: &'a JoltTraceRow) -> Self {
        Self {
            cycle,
            row,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineRow::Unavailable,
        }
    }

    /// An ordinary or padding cycle in a validated field source returns `None`;
    /// a source that has not enabled and validated field witnesses is an error.
    #[cfg(feature = "field-inline")]
    pub fn field_inline(self) -> Result<Option<&'a FieldInlineTraceData>, WitnessError> {
        match self.field_inline {
            FieldInlineRow::Unavailable => Err(WitnessError::UnavailableView {
                label: FIELD_INLINE_LABEL,
            }),
            FieldInlineRow::Inactive => Ok(None),
            FieldInlineRow::Active(data) => Ok(Some(data)),
        }
    }

    #[cfg(feature = "field-inline")]
    pub(crate) fn with_field_inline(mut self, data: Option<&'a FieldInlineTraceData>) -> Self {
        self.field_inline = match data {
            Some(data) => FieldInlineRow::Active(data),
            None => FieldInlineRow::Inactive,
        };
        self
    }
}

/// A borrowed chunk with absolute cycle positions and lookahead beyond its end.
/// Payload storage is borrowed independently of the compact rows so replacing
/// rich captured rows with the side table planned in #1839 does not affect bundles.
#[derive(Clone, Copy)]
pub struct WitnessChunk<'a> {
    pub(crate) start: usize,
    pub(crate) rows: &'a [JoltTraceRow],
    pub(crate) next_after: Option<WitnessRow<'a>>,
    #[cfg(feature = "field-inline")]
    field_rows: Option<&'a [CapturedTraceRow]>,
}

impl<'a> WitnessChunk<'a> {
    pub fn new(start: usize, rows: &'a [JoltTraceRow], next_after: Option<WitnessRow<'a>>) -> Self {
        Self {
            start,
            rows,
            next_after,
            #[cfg(feature = "field-inline")]
            field_rows: None,
        }
    }

    pub fn row(&self, index: usize) -> Option<WitnessRow<'a>> {
        self.rows.get(index).map(|row| self.view(index, row))
    }

    pub(crate) fn view(&self, index: usize, row: &'a JoltTraceRow) -> WitnessRow<'a> {
        let view = WitnessRow::new(self.start + index, row);
        #[cfg(feature = "field-inline")]
        let view = match self.field_rows {
            Some(rows) => {
                view.with_field_inline(rows.get(index).and_then(|row| row.field_inline.as_deref()))
            }
            None => view,
        };
        view
    }

    #[cfg(feature = "field-inline")]
    pub(crate) fn with_field_rows(mut self, rows: &'a [CapturedTraceRow]) -> Self {
        self.field_rows = Some(rows);
        self
    }
}
