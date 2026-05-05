use super::super::result_count::LoweredResultCount;
use super::notation::{EVAL_ATTRS, INSTANCE_RESULT_ATTRS};
use super::SumcheckValueDialect;

#[derive(Clone, Copy)]
pub(in crate::pass) enum SumcheckValueFamily {
    Eval,
    InstanceResult,
}

impl SumcheckValueFamily {
    pub(super) fn attrs(self) -> &'static [&'static str] {
        match self {
            Self::Eval => EVAL_ATTRS,
            Self::InstanceResult => INSTANCE_RESULT_ATTRS,
        }
    }

    pub(super) fn result_types<D: SumcheckValueDialect>(self) -> &'static [&'static str] {
        match self {
            Self::Eval => D::EVAL_RESULT_TYPES,
            Self::InstanceResult => D::INSTANCE_RESULT_TYPES,
        }
    }

    pub(super) fn result_count(self) -> LoweredResultCount {
        match self {
            Self::Eval => LoweredResultCount::One,
            Self::InstanceResult => LoweredResultCount::Two,
        }
    }
}
