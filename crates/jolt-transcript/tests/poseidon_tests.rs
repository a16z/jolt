//! Tests for PoseidonTranscript implementation.

#![cfg(feature = "poseidon")]

mod common;

use jolt_field::Fr;
use jolt_transcript::PoseidonTranscript;

type Pos = PoseidonTranscript<Fr>;

transcript_tests!(Pos);
