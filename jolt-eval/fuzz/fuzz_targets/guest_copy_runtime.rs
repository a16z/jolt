#![no_main]
use jolt_eval::invariant::guest_copy_runtime::GuestCopyRuntimeInvariant;
jolt_eval::fuzz_invariant!(GuestCopyRuntimeInvariant::default());
