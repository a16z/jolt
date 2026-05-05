use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;

use super::super::result_count::LoweredResultCount;
use super::notation::{
    TRANSCRIPT_ABSORB_BYTES_ATTRS, TRANSCRIPT_INIT_ATTRS, TRANSCRIPT_SQUEEZE_ATTRS,
};
use super::TranscriptDialect;

#[derive(Clone, Copy)]
pub(in crate::pass) enum TranscriptOpFamily {
    Init,
    AbsorbBytes,
    Squeeze,
}

pub(super) enum TranscriptResultTypes {
    State(&'static [&'static str]),
    Squeeze([&'static str; 2]),
}

impl TranscriptResultTypes {
    pub(super) fn as_slice(&self) -> &[&'static str] {
        match self {
            Self::State(result_types) => result_types,
            Self::Squeeze(result_types) => result_types,
        }
    }
}

impl TranscriptOpFamily {
    pub(super) fn attrs(self) -> &'static [&'static str] {
        match self {
            Self::Init => TRANSCRIPT_INIT_ATTRS,
            Self::AbsorbBytes => TRANSCRIPT_ABSORB_BYTES_ATTRS,
            Self::Squeeze => TRANSCRIPT_SQUEEZE_ATTRS,
        }
    }

    pub(super) fn result_count(self) -> LoweredResultCount {
        match self {
            Self::Init | Self::AbsorbBytes => LoweredResultCount::One,
            Self::Squeeze => LoweredResultCount::Two,
        }
    }

    pub(super) fn result_types<D: TranscriptDialect>(
        self,
        op: OperationRef<'_, '_>,
    ) -> Result<TranscriptResultTypes, MlirError> {
        match self {
            Self::Init | Self::AbsorbBytes => {
                Ok(TranscriptResultTypes::State(D::STATE_RESULT_TYPES))
            }
            Self::Squeeze => Ok(TranscriptResultTypes::Squeeze(D::squeeze_result_types(op)?)),
        }
    }
}
