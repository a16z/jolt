//! Composition of ordinary Jolt and field-inline protocol algebra.
//!
//! Protocol families retain independent opening namespaces; composed relations
//! resolve both without depending on prover or verifier orchestration.

pub mod geometry;

mod ids;
pub use ids::ComposedOpeningId;

#[cfg(feature = "field-inline")]
mod claims;
#[cfg(feature = "field-inline")]
mod relations;

#[cfg(feature = "field-inline")]
pub use claims::{
    ComposedClaims, ComposedExpr, EmptyClaims, FieldInlineBytecodeReadRafInputs,
    FieldProductUniskipInputs, OuterInputs, OuterOutputs, ProductInputs, ProductOutputs,
    UniskipInputs, UniskipOutputs,
};
#[cfg(feature = "field-inline")]
pub use relations::{OuterRemainder, ProductRemainder, ProductUniskip, ReadRafAddressPhase};
