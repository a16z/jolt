//! Makes IO values (inputs, outputs, panic) into symbolic witness variables.
//!
//! Without this module, the generated Groth16 circuit hardcodes the concrete IO
//! values as constants — meaning the circuit can only verify ONE specific execution.
//! With it, IO becomes witness inputs, so the same circuit works for any execution.
//!
//! # How it works
//!
//! Two separate interception points are set up before `verifier.verify()` runs:
//!
//! 1. **Transcript FIFO** (`PENDING_BYTES_OVERRIDES` in `io_replay.rs`):
//!    `fiat_shamir_preamble` hashes IO bytes into the transcript via `append_bytes`.
//!    We pre-load a FIFO with symbolic variables (one per 32-byte chunk of
//!    inputs/outputs). When `PoseidonAstTranscript::raw_append_bytes` runs, it
//!    pops from this FIFO instead of converting the concrete bytes.
//!    Note: panic is NOT in the FIFO — it goes through `append_u64`, not `append_bytes`.
//!
//! 2. **RAM MLE override** (`PENDING_IO_MLE` in `jolt-core/zkvm/ram/mod.rs`):
//!    `eval_io_mle` evaluates IO as a sparse polynomial over the RAM address space.
//!    We set a thread-local with symbolic field elements (one per u64 word of IO +
//!    a symbolic panic value). The early-return path in `eval_io_mle` picks these up.
//!
//! # FIFO alignment
//!
//! The FIFO must match the exact order of `raw_append_bytes` calls in
//! `fiat_shamir_preamble` (jolt-core/src/zkvm/mod.rs):
//!
//! ```text
//! append_u64(max_input_size)   → raw_append_u64  (no FIFO)
//! append_u64(max_output_size)  → raw_append_u64  (no FIFO)
//! append_u64(heap_size)        → raw_append_u64  (no FIFO)
//! append_bytes(inputs)         → raw_append_bytes → CONSUMES input chunk overrides
//! append_bytes(outputs)        → raw_append_bytes → CONSUMES output chunk overrides
//! append_u64(panic)            → raw_append_u64  (no FIFO)
//! append_u64(ram_K)            → raw_append_u64  (no FIFO)
//! append_u64(trace_length)     → raw_append_u64  (no FIFO)
//! ```
//!
//! After preamble the FIFO must be empty. If not, a stale override would corrupt
//! the next `raw_append_bytes` call.

use ark_bn254::Fr;
use ark_ff::PrimeField;
use common::jolt_device::JoltDevice;
use jolt_core::zkvm::ram::{set_pending_io_mle, PendingIoMleValues};
use zklean_extractor::mle_ast::MleAst;

use crate::symbolic_proof::VarAllocator;
use crate::symbolic_traits::io_replay::push_bytes_override;

/// Allocate symbolic witness variables for all IO values and set up interception
/// points so that `verifier.verify()` uses them instead of concrete constants.
///
/// Returns `(input_words, output_words)` at u64-word granularity — these are
/// passed to `PENDING_INITIAL_RAM` in main.rs so `eval_initial_ram_mle` can
/// also use symbolic inputs.
pub fn symbolize_io_device(
    io_device: &JoltDevice,
    var_alloc: &mut VarAllocator,
) -> (Vec<MleAst>, Vec<MleAst>) {
    // --- Transcript FIFO: symbolic overrides for fiat_shamir_preamble ---
    //
    // One symbolic variable per 32-byte chunk. Consumed in order by
    // PoseidonAstTranscript::raw_append_bytes when fiat_shamir_preamble
    // calls append_bytes(b"inputs", ...) and append_bytes(b"outputs", ...).
    //
    // Panic is NOT pushed into this FIFO because fiat_shamir_preamble sends
    // it via append_u64 → raw_append_u64 (a different code path that doesn't
    // read from the byte-chunk FIFO). Panic IS still fully symbolic in the
    // circuit — it enters through two paths:
    //   1. Transcript: raw_append_u64 hashes MleAst::from_u64(panic) as a
    //      concrete constant into the Poseidon state (correct for Fiat-Shamir).
    //   2. RAM MLE: allocated as witness variable "io_panic_val" below, used
    //      by eval_io_mle for panic_contribution and termination checks.

    let _input_chunk_vars =
        push_byte_chunk_overrides(&io_device.inputs, "io_input_chunk", var_alloc);

    let _output_chunk_vars =
        push_byte_chunk_overrides(&io_device.outputs, "io_output_chunk", var_alloc);

    // --- RAM MLE override: symbolic field elements for eval_io_mle ---
    //
    // eval_io_mle evaluates IO as a sparse polynomial: each input/output u64 word
    // sits at a specific RAM address. The symbolic override makes these words
    // witness variables instead of constants derived from concrete bytes.
    //
    // Panic is included here (as a single field element) because eval_io_mle
    // needs it to compute: panic_contribution = panic_val * eq_eval(panic_addr)
    // and termination = (1 - panic_val) * eq_eval(term_addr).

    let eval_input_words = bytes_to_word_vars(&io_device.inputs, "io_input_word", var_alloc);
    let eval_output_words = bytes_to_word_vars(&io_device.outputs, "io_output_word", var_alloc);
    let panic_val = var_alloc.alloc_with_value("io_panic_val", &Fr::from(io_device.panic as u64));

    set_pending_io_mle(PendingIoMleValues {
        input_words: eval_input_words.clone(),
        output_words: eval_output_words.clone(),
        panic_val,
    });

    (eval_input_words, eval_output_words)
}

/// Chunk bytes into 32-byte pieces, allocate one symbolic variable per chunk,
/// and push each to the transcript FIFO (`PENDING_BYTES_OVERRIDES`).
fn push_byte_chunk_overrides(
    bytes: &[u8],
    prefix: &str,
    var_alloc: &mut VarAllocator,
) -> Vec<MleAst> {
    bytes
        .chunks(32)
        .enumerate()
        .map(|(i, chunk)| {
            let mut padded = [0u8; 32];
            padded[..chunk.len()].copy_from_slice(chunk);
            let fr_val = Fr::from_le_bytes_mod_order(&padded);
            let var = var_alloc.alloc_with_value(&format!("{prefix}_{i}"), &fr_val);
            push_bytes_override(var);
            var
        })
        .collect()
}

/// Split bytes into u64 words (little-endian) and allocate one symbolic variable per word.
/// Used for the RAM MLE override where IO is addressed at u64 granularity.
fn bytes_to_word_vars(bytes: &[u8], prefix: &str, var_alloc: &mut VarAllocator) -> Vec<MleAst> {
    bytes
        .chunks(8)
        .enumerate()
        .map(|(i, chunk)| {
            let mut word_bytes = [0u8; 8];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            let word = u64::from_le_bytes(word_bytes);
            let fr_val = Fr::from(word);
            var_alloc.alloc_with_value(&format!("{prefix}_{i}"), &fr_val)
        })
        .collect()
}
