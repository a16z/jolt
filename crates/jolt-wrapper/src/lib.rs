//! HyperKZG wrapper of the Jolt/Dory verifier.
//!
//! The verifier's field algebra is proved by Spartan, while a Blake3 transcript
//! table and a non-native limb table bind its witness to the original proof.
//! One batched stream and one HyperKZG opening verify the tables and links.

mod carry;
pub mod hash_table;
pub mod limb_table;
pub mod links;
pub mod profile;
pub mod relation;
mod spartan;
pub use spartan::SpartanError;
pub mod stream;
pub mod wrap;
