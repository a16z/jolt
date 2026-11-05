pub mod tampering;

// Whiteout's soundness suite: the census, the message sweep and the per-class
// tampers over the masked tape and the argument string, whose Akita regions
// the `logging` record attributes. The clear and BlindFold tamper sweeps are
// main's `tampering` suite.
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    feature = "akita",
    feature = "zk"
))]
mod whiteout;
