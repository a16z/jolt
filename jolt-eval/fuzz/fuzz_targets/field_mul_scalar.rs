#![no_main]
use jolt_eval::invariant::field_mul_scalar::FieldMulScalarInvariant;
jolt_eval::fuzz_invariant!(FieldMulScalarInvariant::default());
