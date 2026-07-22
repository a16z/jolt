#![no_main]
use jolt_eval::invariant::transcript_symmetry::TranscriptConsistencyKeccakInvariant;
jolt_eval::fuzz_invariant!(TranscriptConsistencyKeccakInvariant::default());
