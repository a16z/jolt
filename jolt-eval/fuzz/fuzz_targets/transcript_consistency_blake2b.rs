#![no_main]
use jolt_eval::invariant::transcript_symmetry::TranscriptConsistencyBlake2bInvariant;
jolt_eval::fuzz_invariant!(TranscriptConsistencyBlake2bInvariant::default());
