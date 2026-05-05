mod descriptors;

pub(in crate::schema::ops) use descriptors::{CountedOperandShape, ExactOpShape, MinOpShape};

pub(in crate::schema::ops) const ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS: CountedOperandShape =
    CountedOperandShape {
        min_operands: 0,
        results: 1,
        fixed_operands: 0,
        ordered_attr: "ordered_claims",
    };

pub(in crate::schema::ops) const ORDERED_CLAIMS_WITH_ONE_FIXED_OPERAND: CountedOperandShape =
    CountedOperandShape {
        min_operands: 1,
        results: 1,
        fixed_operands: 1,
        ordered_attr: "ordered_claims",
    };

pub(in crate::schema::ops) const NO_OPERANDS_ONE_RESULT: ExactOpShape = ExactOpShape {
    operands: 0,
    results: 1,
};

pub(in crate::schema::ops) const NO_OPERANDS_THREE_RESULTS: ExactOpShape = ExactOpShape {
    operands: 0,
    results: 3,
};

pub(in crate::schema::ops) const ONE_OPERAND_NO_RESULTS: ExactOpShape = ExactOpShape {
    operands: 1,
    results: 0,
};

pub(in crate::schema::ops) const ONE_OPERAND_ONE_RESULT: ExactOpShape = ExactOpShape {
    operands: 1,
    results: 1,
};

pub(in crate::schema::ops) const ONE_OPERAND_TWO_RESULTS: ExactOpShape = ExactOpShape {
    operands: 1,
    results: 2,
};

pub(in crate::schema::ops) const TWO_OPERANDS_NO_RESULTS: ExactOpShape = ExactOpShape {
    operands: 2,
    results: 0,
};

pub(in crate::schema::ops) const TWO_OPERANDS_ONE_RESULT: ExactOpShape = ExactOpShape {
    operands: 2,
    results: 1,
};

pub(in crate::schema::ops) const TWO_OPERANDS_TWO_RESULTS: ExactOpShape = ExactOpShape {
    operands: 2,
    results: 2,
};

pub(in crate::schema::ops) const TWO_OPERANDS_FOUR_RESULTS: ExactOpShape = ExactOpShape {
    operands: 2,
    results: 4,
};

pub(in crate::schema::ops) const AT_LEAST_ONE_OPERAND_ONE_RESULT: MinOpShape = MinOpShape {
    min_operands: 1,
    results: 1,
};
