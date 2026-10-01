//! Soundness over the argument string: every prover message is a byte range
//! of the proof's NARG, so the suites alter bytes and the statement rather
//! than typed proof fields, and pin where each alteration must be caught
//! (`support::narg::Region::deadline`).
//!
//! Each backend runs the same checks (`checks`) over its own fixtures. The
//! event log that attributes a rejection to a protocol region needs the
//! `logging` feature.

#[cfg(all(feature = "prover-fixtures", feature = "logging"))]
mod checks;

#[cfg(all(feature = "prover-fixtures", feature = "logging", feature = "akita"))]
mod akita;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    not(feature = "akita"),
    not(feature = "zk")
))]
mod dory;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    not(feature = "akita"),
    not(feature = "field-inline"),
    feature = "zk"
))]
mod zk;
