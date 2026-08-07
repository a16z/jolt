//! First-principles foundation for a resident InstructionReadRaf Metal owner.
//!
//! This module is intentionally not wired into the backend.  It fixes four
//! things before a shader is admitted:
//!
//! - [`oracle`] states the relation and every round equation independently of
//!   the optimized CPU prefix/suffix implementation;
//! - [`abi`] makes reuse of the producer-owned instruction rows, claim bytes,
//!   equality factors, and optional address-atom topology explicit;
//! - [`model`] counts useful field products and every logical byte requested by
//!   the proposed address and cycle sequences;
//! - the tests require dense/atom parity, the sumcheck invariant at every
//!   round, a final-expression match, and fail-closed receipt provenance.
//!
//! For cycle `j`, raw 128-bit lookup address `k_j`, optional table `t_j`, RAF
//! flag `f_j`, reduction weight `u_j = eq(r_reduction, j)`, and address point
//! `x`, define
//!
//! ```text
//! C_j(x) = 1[t_j=t] table_t(x)
//!        + 1[!f_j] (gamma left(x) + gamma^2 right(x))
//!        + 1[f_j]  (gamma^2 identity(x) + gamma^3 upper64_all_ones(x)).
//! ```
//!
//! The relation is
//!
//! ```text
//! sum_j sum_x u_j * eq(x, k_j) * C_j(x).
//! ```
//!
//! The `gamma^3` term is load-bearing for the fp128 canonical-address check.
//! Address variables bind most-significant first.  At address round `a`, with
//! bound prefix `rho`, the exact quadratic is
//!
//! ```text
//! s_a(c) = sum_j u_j * eq(rho, prefix_a(k_j))
//!                    * chi_{bit_a(k_j)}(c)
//!                    * C_j(rho || c || suffix_{a+1}(k_j)).
//! ```
//!
//! After the address point is fixed, each virtual-RA chunk and `C_j` is a
//! cycle table.  The remaining low-to-high cycle round is the ordinary product
//! of `eq(r_reduction, j)`, `C_j`, and every RA chunk factor.  Its degree is
//! `num_virtual_ra + 2`.  [`oracle::DenseReadRafOracle`] evaluates that direct
//! product; it does not share the CPU kernel's Gruen or prefix/suffix code.
//!
//! The intended device sequence never runs Fiat--Shamir.  Each small member
//! polynomial returns to the host, the host absorbs the batched polynomial and
//! derives the next challenge, and only then may the next transition dispatch.

#![expect(
    dead_code,
    unused_imports,
    reason = "the v3 foundation remains hidden until a runtime owner is wired"
)]

mod abi;
mod model;
mod oracle;
mod runtime;
mod shader_abi;
mod topology;

#[cfg(feature = "test-utils")]
pub use runtime::{run_address_atom_probe, AddressAtomProbeResult};

pub(crate) use runtime::{
    AddressAtomPhaseOutput, AddressAtomRuntimeConfig, AddressAtomRuntimeError, AddressAtomSequence,
};
pub(crate) use topology::{AddressAtomTopology, AddressAtomTopologyConfig};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub(crate) const ATOM_MASS_PHASE_PIPELINE: &str = "solinas_instruction_read_raf_v3_atom_mass_phase";
pub(crate) const ATOM_MASS_FINALIZE_PIPELINE: &str =
    "solinas_instruction_read_raf_v3_atom_mass_finalize";
pub(crate) const ATOM_PHASE_PIPELINE: &str = "solinas_instruction_read_raf_v3_atom_phase";
pub(crate) const FINALIZE_RAF_PIPELINE: &str = "solinas_instruction_read_raf_v3_finalize_raf";
pub(crate) const FINALIZE_SUFFIX_PIPELINE: &str = "solinas_instruction_read_raf_v3_finalize_suffix";
pub(crate) const OPEN_FLAGS_PIPELINE: &str = "solinas_instruction_read_raf_v3_open_flags";
pub(crate) const REDUCE_PIPELINE: &str = "solinas_instruction_read_raf_v3_reduce";

pub(crate) use abi::{
    AddressAtomTopologyReceipt, AddressStateReceipt, CycleFactorReceipt, HostRoundBoundary,
    InstructionReadRafGeometry, PlaneDescriptor, ProducerIdentity, ReductionEqReceipt,
    ResidentInstructionFacts, ResidentPlane, ResidentReadRafInputs, StageOutputReceipt,
};
pub(crate) use model::{
    AddressCensus, CutoffDecision, ExecutionModel, GateReport, RoofRates, SequenceWork,
};
pub(crate) use oracle::{
    aggregate_address_atoms, atom_address_message, AddressAtom, DenseReadRafOracle,
    InstructionReadRafRow, ReadRafOutputClaims, RoundMessage,
};

use thiserror::Error;

pub(crate) const ADDRESS_BITS: usize = 128;
pub(crate) const ADDRESS_PHASE_BITS: usize = 8;
pub(crate) const ADDRESS_PHASES: usize = ADDRESS_BITS / ADDRESS_PHASE_BITS;
pub(crate) const ADDRESS_BINS: usize = 1 << ADDRESS_PHASE_BITS;
pub(crate) const PRODUCTION_VIRTUAL_RA: usize = 4;
pub(crate) const PRODUCTION_CYCLE_FACTORS: usize = PRODUCTION_VIRTUAL_RA + 1;
pub(crate) const FP128_BYTES: usize = 16;
pub(crate) const INSTRUCTION_ROW_BYTES: usize = 40;

const _: () = assert!(ADDRESS_PHASES == 16);
const _: () = assert!(PRODUCTION_CYCLE_FACTORS == 5);

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum InstructionReadRafV3Error {
    #[error("InstructionReadRaf cycle count {0} must be a nonzero power of two below 2^32")]
    InvalidCycles(usize),
    #[error("InstructionReadRaf virtual-RA factor count {0} must divide 128 into 8-bit phases")]
    InvalidVirtualRa(usize),
    #[error("InstructionReadRaf {name} identity must be nonzero")]
    MissingIdentity { name: &'static str },
    #[error("InstructionReadRaf {plane} has {got} elements, expected {expected}")]
    PlaneElements {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("InstructionReadRaf {plane} has {got} bytes, expected {expected}")]
    PlaneBytes {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("InstructionReadRaf {plane} belongs to a different producer")]
    ProducerMismatch { plane: &'static str },
    #[error(
        "InstructionReadRaf source allocation identity is {got}, expected authoritative allocation {expected}"
    )]
    SourceAllocationMismatch { expected: usize, got: usize },
    #[error("InstructionReadRaf {plane} initialization generation is {got}, expected {expected}")]
    GenerationMismatch {
        plane: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("InstructionReadRaf {plane} command serial {got} precedes producer serial {minimum}")]
    IncompletePlane {
        plane: &'static str,
        minimum: u64,
        got: u64,
    },
    #[error("InstructionReadRaf allocation identity {identity} is aliased")]
    AliasedAllocation { identity: usize },
    #[error("InstructionReadRaf table index {0} is outside the production table family")]
    InvalidTable(usize),
    #[error("InstructionReadRaf reduction point has {got} coordinates, expected {expected}")]
    ReductionPointLength { expected: usize, got: usize },
    #[error("InstructionReadRaf address oracle is already in the cycle phase")]
    AddressPhaseComplete,
    #[error("InstructionReadRaf cycle oracle was requested before 128 address binds")]
    CyclePhaseNotReady,
    #[error("InstructionReadRaf cycle oracle is already fully bound")]
    CyclePhaseComplete,
    #[error("InstructionReadRaf output was requested with {remaining} rounds remaining")]
    RoundsRemaining { remaining: usize },
    #[error("InstructionReadRaf round message requires at least two evaluations")]
    EmptyRoundMessage,
    #[error("InstructionReadRaf address atom set does not match the dense source generation")]
    AtomSourceMismatch,
    #[error("InstructionReadRaf atom topology has {atoms} atoms for {rows} rows")]
    InvalidAtomCount { rows: usize, atoms: usize },
    #[error(
        "InstructionReadRaf atom topology has an invalid {name} length: {got}, expected {expected}"
    )]
    AtomTopologyLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("InstructionReadRaf receipt transition expected phase {expected}, got {got}")]
    PhaseMismatch { expected: usize, got: usize },
    #[error("InstructionReadRaf receipt transition expected width {expected}, got {got}")]
    WidthMismatch { expected: usize, got: usize },
    #[error("InstructionReadRaf host transcript boundary is invalid at member call {0}")]
    InvalidMemberCall(usize),
    #[error("InstructionReadRaf size arithmetic overflowed while computing {0}")]
    SizeOverflow(&'static str),
    #[error("InstructionReadRaf analytical parameter {0} must be finite and positive")]
    InvalidModelParameter(&'static str),
    #[error("InstructionReadRaf address census is inconsistent: {0}")]
    InvalidCensus(&'static str),
    #[error("InstructionReadRaf shader ABI is invalid: {0}")]
    InvalidShaderAbi(&'static str),
    #[error("InstructionReadRaf topology parameter {name} has invalid value {value}")]
    InvalidTopologyConfig { name: &'static str, value: usize },
    #[error("InstructionReadRaf sorted topology has {got} cycles, expected {expected}")]
    TopologyCycleLength { expected: usize, got: usize },
    #[error(
        "InstructionReadRaf sorted topology cycle {cycle} at position {position} is outside {rows} rows"
    )]
    TopologyCycleOutOfRange {
        position: usize,
        cycle: usize,
        rows: usize,
    },
    #[error("InstructionReadRaf sorted topology repeats cycle {cycle}")]
    DuplicateTopologyCycle { cycle: usize },
    #[error("InstructionReadRaf sorted topology key decreases at position {position}")]
    NonMonotoneTopologyKey { position: usize },
    #[error("InstructionReadRaf topology is invalid: {0}")]
    InvalidTopology(&'static str),
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixed pure-Rust fixtures use direct assertions and unwraps"
)]
mod tests;
