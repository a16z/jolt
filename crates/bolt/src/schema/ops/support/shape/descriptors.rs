#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::schema::ops) struct ExactOpShape {
    pub(in crate::schema::ops::support) operands: usize,
    pub(in crate::schema::ops::support) results: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::schema::ops) struct MinOpShape {
    pub(in crate::schema::ops::support) min_operands: usize,
    pub(in crate::schema::ops::support) results: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::schema::ops) struct CountedOperandShape {
    pub(in crate::schema::ops::support) min_operands: usize,
    pub(in crate::schema::ops::support) results: usize,
    pub(in crate::schema::ops::support) fixed_operands: usize,
    pub(in crate::schema::ops::support) ordered_attr: &'static str,
}
