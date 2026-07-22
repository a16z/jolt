use crate::{CanonicalRepr, FieldCore, FromPrimitiveInt, WithAccumulator};

/// Prime field element abstraction used throughout Jolt.
///
/// This trait provides a backend-agnostic interface over a prime-order scalar
/// field.
///
/// All arithmetic is modular over the field's prime order. Elements are `Copy`,
/// thread-safe, and cheaply serializable. Negative integers are mapped via
/// their canonical representative modulo `p`.
pub trait Field: FieldCore + FromPrimitiveInt + CanonicalRepr + WithAccumulator {}
