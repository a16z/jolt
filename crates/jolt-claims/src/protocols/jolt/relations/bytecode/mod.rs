//! Bytecode read-RAF symbolic sumcheck relations.

use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
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
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReadRafInputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim, from = BytecodeReadRaf)]
    pub address_phase: C,
}

/// The cycle-phase shape: the read-raf dimensions plus the staged-val count
/// the output fold spans — five in base mode, six in lattice mode (the
/// store stage, with the RAF and entry publics shifted past it).
pub type BytecodeReadRafCycleShape = (BytecodeReadRafDimensions, usize);

mod read_raf;
mod read_raf_address_phase;
mod read_raf_cycle_phase;
mod read_raf_cycle_phase_committed;

pub use read_raf::*;
pub use read_raf_address_phase::*;
pub use read_raf_cycle_phase::*;
pub use read_raf_cycle_phase_committed::*;
