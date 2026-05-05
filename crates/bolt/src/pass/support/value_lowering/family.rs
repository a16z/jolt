use super::super::result_count::LoweredResultCount;
use super::notation::{
    FIELD_CONST_ATTRS, FIELD_UNIT_ATTRS, OPENING_INPUT_ATTRS, POINT_CONCAT_ATTRS,
    POINT_SLICE_ATTRS, POINT_ZERO_ATTRS,
};
use super::ValueDialect;

#[derive(Clone, Copy)]
pub(in crate::pass) enum ValueOpFamily {
    OpeningInput,
    PointSlice,
    PointZero,
    PointConcat,
    FieldConst,
    FieldUnit,
    FieldExpression,
}

pub(super) struct ValueResultShape {
    pub(super) attrs: &'static [&'static str],
    pub(super) result_types: &'static [&'static str],
    pub(super) result_count: LoweredResultCount,
}

impl ValueOpFamily {
    pub(super) fn fixed_shape<D: ValueDialect>(self) -> Option<ValueResultShape> {
        let shape = match self {
            Self::OpeningInput => ValueResultShape {
                attrs: OPENING_INPUT_ATTRS,
                result_types: D::OPENING_INPUT_RESULT_TYPES,
                result_count: LoweredResultCount::Three,
            },
            Self::PointSlice => ValueResultShape {
                attrs: POINT_SLICE_ATTRS,
                result_types: D::POINT_RESULT_TYPES,
                result_count: LoweredResultCount::One,
            },
            Self::PointZero => ValueResultShape {
                attrs: POINT_ZERO_ATTRS,
                result_types: D::POINT_RESULT_TYPES,
                result_count: LoweredResultCount::One,
            },
            Self::PointConcat => ValueResultShape {
                attrs: POINT_CONCAT_ATTRS,
                result_types: D::POINT_RESULT_TYPES,
                result_count: LoweredResultCount::One,
            },
            Self::FieldConst => ValueResultShape {
                attrs: FIELD_CONST_ATTRS,
                result_types: D::FIELD_RESULT_TYPES,
                result_count: LoweredResultCount::One,
            },
            Self::FieldUnit => ValueResultShape {
                attrs: FIELD_UNIT_ATTRS,
                result_types: D::FIELD_RESULT_TYPES,
                result_count: LoweredResultCount::One,
            },
            Self::FieldExpression => return None,
        };
        Some(shape)
    }
}
