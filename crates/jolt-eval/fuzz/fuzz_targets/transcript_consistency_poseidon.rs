#![no_main]
use jolt_eval::invariant::transcript_symmetry::TranscriptConsistencyPoseidonInvariant;
jolt_eval::fuzz_invariant!(TranscriptConsistencyPoseidonInvariant::default());
