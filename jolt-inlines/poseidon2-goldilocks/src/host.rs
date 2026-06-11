// SPDX-License-Identifier: MIT

//! Host-side registration of the Poseidon2-Goldilocks inline with
//! the Jolt prover/tracer.
//!
//! The `register_inlines!` macro generates the dispatcher that maps
//! our `(INLINE_OPCODE, FUNCT3, FUNCT7)` triple to
//! `Poseidon2GoldilocksPermutation::build_sequence`.

use crate::sequence_builder::Poseidon2GoldilocksPermutation;

jolt_inlines_sdk::register_inlines! {
    trace_file: "poseidon2_goldilocks_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Poseidon2Goldilocks,
    ops: [Poseidon2GoldilocksPermutation],
}
