use crate::protocols::field_inline::FieldInlineOpeningId;
use crate::protocols::jolt::JoltOpeningId;

/// An opening id from either protocol family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComposedOpeningId {
    Jolt(JoltOpeningId),
    FieldInline(FieldInlineOpeningId),
}

impl From<JoltOpeningId> for ComposedOpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineOpeningId> for ComposedOpeningId {
    fn from(id: FieldInlineOpeningId) -> Self {
        Self::FieldInline(id)
    }
}

/// Downcast from the composite [`ComposedOpeningId`] to one protocol family's
/// opening id: returns the original id on a family mismatch. The inverse
/// of the `From` embeddings above, letting family-typed resolvers (each batch
/// member's claim struct speaks its own id family) participate in one
/// composite-keyed lookup.
impl TryFrom<ComposedOpeningId> for JoltOpeningId {
    type Error = ComposedOpeningId;

    fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
        match id {
            ComposedOpeningId::Jolt(id) => Ok(id),
            ComposedOpeningId::FieldInline(_) => Err(id),
        }
    }
}

impl TryFrom<ComposedOpeningId> for FieldInlineOpeningId {
    type Error = ComposedOpeningId;

    fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
        match id {
            ComposedOpeningId::FieldInline(id) => Ok(id),
            ComposedOpeningId::Jolt(_) => Err(id),
        }
    }
}
