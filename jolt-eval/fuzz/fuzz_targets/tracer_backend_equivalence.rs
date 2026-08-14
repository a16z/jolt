#![no_main]
use jolt_eval::invariant::tracer_backend_equivalence::TracerBackendEquivalenceInvariant;
jolt_eval::fuzz_invariant!(TracerBackendEquivalenceInvariant::default());
