use super::super::result_count::LoweredResultCount;
use super::notation::{BATCH_ATTRS, CLAIM_ATTRS, CLAIM_EQUAL_ATTRS};
use super::OpeningDialect;

#[derive(Clone, Copy)]
pub(in crate::pass) enum OpeningOpFamily {
    Claim,
    ClaimEqual,
    Batch,
}

impl OpeningOpFamily {
    pub(super) fn attrs(self) -> &'static [&'static str] {
        match self {
            Self::Claim => CLAIM_ATTRS,
            Self::ClaimEqual => CLAIM_EQUAL_ATTRS,
            Self::Batch => BATCH_ATTRS,
        }
    }

    pub(super) fn result_types<D: OpeningDialect>(self) -> &'static [&'static str] {
        match self {
            Self::Claim => D::CLAIM_RESULT_TYPES,
            Self::ClaimEqual => &[],
            Self::Batch => D::BATCH_RESULT_TYPES,
        }
    }

    pub(super) fn result_count(self) -> LoweredResultCount {
        match self {
            Self::Claim | Self::Batch => LoweredResultCount::One,
            Self::ClaimEqual => LoweredResultCount::None,
        }
    }
}
