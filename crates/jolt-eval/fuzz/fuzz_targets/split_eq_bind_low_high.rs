#![no_main]
use jolt_eval::invariant::split_eq_bind::SplitEqBindLowHighInvariant;
jolt_eval::fuzz_invariant!(SplitEqBindLowHighInvariant::default());
