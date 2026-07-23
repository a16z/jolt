#![no_main]
use jolt_eval::invariant::transcript_symmetry::TranscriptConsistencyInvariant;
jolt_eval::fuzz_invariant!(TranscriptConsistencyInvariant::default());
