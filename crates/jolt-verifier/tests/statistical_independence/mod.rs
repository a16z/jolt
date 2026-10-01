// ZK fixtures exist only with field-inline disabled; the message layout comes
// from the `logging` event log.
#[cfg(all(
    feature = "prover-fixtures",
    feature = "logging",
    feature = "zk",
    not(feature = "field-inline")
))]
pub mod zk;
