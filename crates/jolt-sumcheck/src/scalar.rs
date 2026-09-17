use jolt_field::Field;

/// Algebraic scalar capabilities shared by clear sumcheck proving and verification.
pub trait SumcheckScalar: Field {}

impl<F: Field> SumcheckScalar for F {}
