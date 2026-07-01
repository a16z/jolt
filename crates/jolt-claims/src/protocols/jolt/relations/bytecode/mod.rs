//! Bytecode read-RAF symbolic sumcheck relations.

use serde::{Deserialize, Serialize};

use crate::{InputClaims, OutputClaims};

/// The cycle-phase produced openings: the per-chunk committed `BytecodeRa` claims,
/// all sharing the `r_address ++ r_cycle` opening point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafOutputClaims<C> {
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
}

/// The `BytecodeReadRafAddrClaim` intermediate consumed from the address phase.
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReadRafInputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim, from = BytecodeReadRaf)]
    pub address_phase: C,
}

mod read_raf;
mod read_raf_address_phase;
mod read_raf_cycle_phase;
mod read_raf_cycle_phase_committed;

pub use read_raf::*;
pub use read_raf_address_phase::*;
pub use read_raf_cycle_phase::*;
pub use read_raf_cycle_phase_committed::*;
