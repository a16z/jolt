use super::super::result_count::LoweredResultCount;
use super::notation::{BATCH_ATTRS, BATCH_OPENING_ATTRS, CLAIM_ATTRS};
use super::PcsDialect;

#[derive(Clone, Copy)]
pub(in crate::pass) enum PcsOpFamily {
    Claim,
    Batch,
    BatchOpening,
}

impl PcsOpFamily {
    pub(super) fn attrs(self) -> &'static [&'static str] {
        match self {
            Self::Claim => CLAIM_ATTRS,
            Self::Batch => BATCH_ATTRS,
            Self::BatchOpening => BATCH_OPENING_ATTRS,
        }
    }

    pub(super) fn result_types<D: PcsDialect>(self) -> &'static [&'static str] {
        match self {
            Self::Claim => D::CLAIM_RESULT_TYPES,
            Self::Batch => D::BATCH_RESULT_TYPES,
            Self::BatchOpening => D::BATCH_OPENING_RESULT_TYPES,
        }
    }

    pub(super) fn result_count(self) -> LoweredResultCount {
        match self {
            Self::Claim | Self::Batch => LoweredResultCount::One,
            Self::BatchOpening => LoweredResultCount::Two,
        }
    }
}
