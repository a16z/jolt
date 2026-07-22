// The `not(akita)` modules tamper the base (dory) proof shape, which does not
// exist under the akita feature — one compiled verifier runs exactly one
// protocol. The packed pipeline gets its own typed tamper suite (`akita`:
// clear-claim wire sweep, commitment-byte sweeps, proof-shape and presence
// tampers); only the shape-agnostic `manifest` checks run under both.
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
pub mod akita;
#[cfg(not(feature = "akita"))]
pub mod commitments;
#[cfg(not(feature = "akita"))]
pub mod configs;
pub mod manifest;
#[cfg(not(feature = "akita"))]
pub mod openings;
#[cfg(not(feature = "akita"))]
pub mod preamble;
#[cfg(not(feature = "akita"))]
pub mod proof_shape;
#[cfg(not(feature = "akita"))]
pub mod sumcheck;
#[cfg(not(feature = "akita"))]
pub mod zk;
