//! Lane M2/N3 measurement harness for the curve-wrapper limb-relation table;
//! the library exposes the table, the relation prover and the column-packing
//! helpers so the wrapper bench can fold the limb relation into its batched
//! sumcheck stream.

#![expect(clippy::cast_possible_truncation, reason = "measurement harness")]

pub mod pack;
pub mod relation;
pub mod table;
