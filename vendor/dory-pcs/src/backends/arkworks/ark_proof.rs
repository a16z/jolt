//! Arkworks backend proof type
//!
//! Type alias for Dory proofs using arkworks group types.
//! Serialization implementations are in ark_serde.rs.

use super::{ArkG1, ArkG2, ArkGT};
use crate::proof::DoryProof;

/// Arkworks-specific Dory proof type
///
/// This is a type alias for `DoryProof` specialized to arkworks group types.
/// Serialization support via `CanonicalSerialize` and `CanonicalDeserialize`
/// is implemented in the `ark_serde` module.
pub type ArkDoryProof = DoryProof<ArkG1, ArkG2, ArkGT>;
