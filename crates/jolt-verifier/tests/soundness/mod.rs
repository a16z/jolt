pub mod tampering;

// Whiteout's soundness suite: the census, the message sweep and the per-class
// tampers over the argument string, whose regions the `logging` event log
// attributes (`support::narg`). The clear and BlindFold tamper sweeps are
// main's `tampering` suite.
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    feature = "akita",
    feature = "zk"
))]
mod checks;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    feature = "akita",
    feature = "zk"
))]
mod whiteout;
