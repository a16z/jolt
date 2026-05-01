use crate::{Invertible, RingCore};

/// Algebraic field marker: ring arithmetic plus explicit inversion.
pub trait FieldCore: RingCore + Invertible {}
