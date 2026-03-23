//! Protocol graph: claim-level IR for SNARK structure.
//!
//! The protocol graph captures the full claim dependency DAG of the Jolt SNARK:
//! which polynomials exist, which claims are made about them, and how claims
//! flow between sumcheck instances until they're discharged by PCS openings.
//!
//! The graph separates **invariant structure** (the claim graph — what must be
//! proven) from **scheduling choices** (staging, batching, commitment grouping).
//! Both prover and verifier derive from the same [`ProtocolGraph`].
//!
//! See `crates/jolt-ir/PROTOCOL_GRAPH.md` for the full design document.

mod build;
mod symbolic;
mod types;
mod validate;

pub use build::{build_jolt_protocol, ProtocolConfig};
pub use symbolic::{Symbol, SymbolicExpr};
pub use types::*;
pub use validate::{
    ChallengeSpecError, ClaimFlowError, CommitmentError, EvalOrderingError, GraphError,
    StagingError,
};
