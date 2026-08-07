//! First-principles Akita bytecode read/RAF design for a Metal backend.
//!
//! This package is not wired into the production backend. It fixes the
//! relation and backend boundary before another shader is admitted:
//!
//! - [`owner`] publishes one immutable row allocation plus a per-outer,
//!   address-grouped carrier during the producer's single witness traversal;
//! - [`oracle`] constructs both phases densely from cycle-order rows and never
//!   consumes the grouped carrier, so a topology bug cannot bless itself;
//! - [`model`] charges useful products and both requested and unavoidable
//!   traffic, and chooses an occupancy-safe Metal prefix;
//! - [`relation`] is the authoritative address/cycle algebra and round-message
//!   contract.
//!
//! The address worker target is one compact occurrence pass with nine local
//! accumulators. For each nonempty `(outer, address)` cell it performs four
//! signed increment products while visiting each occurrence and nine full
//! outer-root products after the cell reduction. The cycle worker reads the
//! shared 40-byte stage-5 row allocation for round zero, reads it again only
//! after the host supplies the first Fiat--Shamir challenge, and then writes
//! five bound dense planes. Materializing those five planes before the first
//! challenge costs more traffic than the required second row read.
//!
//! A legal adjacent fusion is a multi-output stage-6b row dispatch for
//! bytecode read/RAF, bytecode booleanity, and increment reduction: their
//! member challenges are known and their round challenge is shared before the
//! dispatch. Stage 6a and stage 6b cannot be fused across their handoff because
//! `r_address` and the next batch challenge do not exist until the host absorbs
//! stage 6a. Fiat--Shamir therefore remains entirely host-owned.

mod model;
mod oracle;
mod owner;
mod relation;

pub use model::{
    address_accounting, cycle_round_accounting, select_cycle_cutoff, AccountingError,
    AddressAccounting, CycleCutoffPlan, CycleRoundAccounting, CycleRoundKind, ExecutionProfile,
    FamilyShape, FixedCosts, OccupancyFloor, Roof, RoofRates, SelectionError, SpeedupTarget,
};
pub use oracle::{BytecodeReadRafInputs, DenseAddressOracle, DenseCycleOracle, OracleError};
pub use owner::{
    BytecodeReadRafOwner, BytecodeReadRafOwnerBuilder, BytecodeReadRafReceipt, BytecodeWitnessRow,
    OwnerConfig, OwnerError, PackedCell, PackedInnerSign, ProducerIdentity,
    ResidentPlaneIdentities, SignedMagnitude, BYTECODE_READ_RAF_SCHEMA_VERSION,
};
pub use relation::{
    address_summand, canonical_opening_point, cycle_summand, AddressOutput, AddressRoundMessage,
    CycleOutput, CycleRoundMessage, RelationError, RelationWeights, StageValueSource,
    ADDRESS_ACTUAL_DEGREE, ADDRESS_DECLARED_MAX_DEGREE, BASE_STAGES, COMMITTED_CHUNK_BITS,
    CYCLE_DEGREE, FUSED_STAGES, RAW_VALUE_TABLES, RA_FACTORS, STAGES, STAGE_VALUE_SOURCES,
};
#[cfg(test)]
mod tests;
