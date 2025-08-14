//! # vmv
//! In order to turn the Dory argument into a full PCS, we require the ability to
//! commit to something that can be used as an argument for an IP instance, which is exactly what we provide in this submodule.
//! This grants us the ability to provide evaluation proofs (and openings) of polynomials.

pub mod commit;
pub mod evaluate;
pub mod vmv_state;

pub use commit::*;
pub use evaluate::*;
pub use vmv_state::*;
