//! Optimized kernels: legacy-monolith prover techniques ported behind the
//! [`PrepareKernel`](crate::PrepareKernel) seam, byte-parity-equivalent to the
//! [`reference`](crate::reference) tier (identical round polynomials and
//! output claims from identical `ProverInputs`, all rounds, same challenges).
//!
//! The shared playbook, per kernel:
//! - **Sparse one-hot access**: per-cycle hot indices off typed witness
//!   bundles ([`jolt_witness::collect_bundles`]) replace `oracle_table` walks
//!   over materialized `K x T` one-hot grids — `O(T)` per polynomial instead
//!   of `O(K·T)`.
//! - **Split-eq pushforwards**: `Σ_{j: idx(j)=k} eq(r, j)` accumulated as
//!   `E_hi[j_hi] · (Σ_{j_lo} E_lo[j_lo])` — inner sums are additions only,
//!   one multiplication per touched slot per outer block.
//! - **Linear-leaf fusion**: eq/selector leaves that enter the summand
//!   linearly are folded into one combined coefficient table (or a constant),
//!   shrinking per-round bind and extension work; exactness of multilinear
//!   extension under scalar-weighted sums keeps the round messages
//!   byte-identical.
//! - **Eval-at-1 recovery**: round messages sample the summand at
//!   `t ∈ {0, 2, .., degree}` and recover `s(1) = previous_claim − s(0)`,
//!   the same trade the legacy prover makes (a dishonest input claim
//!   surfaces at the driver's final-claim check instead of the round check).
//! - **Rayon cycle walks** with per-thread partial accumulators.

mod support;

pub mod bytecode_read_raf;
pub mod hamming_weight_claim_reduction;
pub mod inc_claim_reduction;

pub use bytecode_read_raf::{OptimizedBytecodeReadRafAddress, OptimizedBytecodeReadRafCycle};
pub use hamming_weight_claim_reduction::OptimizedHammingWeightClaimReduction;
pub use inc_claim_reduction::OptimizedIncClaimReduction;

#[cfg(test)]
pub(crate) mod harness;
