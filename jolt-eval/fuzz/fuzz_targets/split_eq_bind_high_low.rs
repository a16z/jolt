#![no_main]
use jolt_eval::invariant::split_eq_bind::SplitEqBindHighLowInvariant;
jolt_eval::fuzz_invariant!(SplitEqBindHighLowInvariant::default());
