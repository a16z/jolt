//! # Primitives
//! This submodule defines basic arithmetic, polynomial ops, and fiat-shamir related things
//! used throughout the codebase, and are considered rather generic tools w.r.t. Dory
pub mod arithmetic;
pub mod poly;
pub mod toy_transcript;
pub mod transcript;

pub use poly::*;
