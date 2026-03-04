//! Symbolization of IO device for universal circuit generation.
//!
//! Converts concrete IO values (inputs, outputs, panic) into symbolic MleAst variables,
//! enabling the generated circuit to be independent of specific program inputs
//! (same-program universality).

use common::jolt_device::JoltDevice;
use jolt_core::zkvm::ram::{set_pending_io_mle, PendingIoMleValues};
use zklean_extractor::mle_ast::MleAst;

use crate::symbolic_proof::VarAllocator;

/// Symbolize IO device data for universal circuit generation.
///
/// Allocates symbolic variables for inputs, outputs, and panic, then pushes
/// override values to thread-locals consumed during `verify()`:
///
/// 1. **Preamble overrides** (`PENDING_BYTES_OVERRIDES`): Elements for Poseidon hashing
///    in `fiat_shamir_preamble`. Padded to max_input_size/max_output_size.
///
/// 2. **eval_io_mle overrides** (`PENDING_IO_MLE`): u64-word field elements for
///    `eval_io_mle_symbolic`. Padded to max_input_size/8 and max_output_size/8 words.
///
/// All IO is padded to max sizes from MemoryLayout so the circuit structure is
/// fixed regardless of actual IO size. `fiat_shamir_preamble` applies the same
/// padding on the prover side.
///
/// Returns `eval_input_words` for use in `PENDING_INITIAL_RAM`.
pub fn symbolize_io_device(
    io_device: &JoltDevice,
    var_alloc: &mut VarAllocator,
) -> Vec<MleAst> {
    use ark_ff::PrimeField;
    use crate::symbolic_traits::io_replay::push_bytes_override;

    let max_input = io_device.memory_layout.max_input_size as usize;
    let max_output = io_device.memory_layout.max_output_size as usize;

    // Pad inputs and outputs to max sizes (matches fiat_shamir_preamble padding).
    let mut padded_inputs = io_device.inputs.clone();
    padded_inputs.resize(max_input, 0);
    let mut padded_outputs = io_device.outputs.clone();
    padded_outputs.resize(max_output, 0);

    // --- 1. Preamble overrides (32-byte chunk scalars) ---
    // These match what raw_append_bytes produces: bytes → 32-byte padded → Fr scalar.
    let preamble_input_elements: Vec<MleAst> = padded_inputs
        .chunks(32)
        .enumerate()
        .map(|(i, chunk)| {
            let mut buf = [0u8; 32];
            buf[..chunk.len()].copy_from_slice(chunk);
            let concrete = ark_bn254::Fr::from_le_bytes_mod_order(&buf);
            var_alloc.alloc_with_value(&format!("io_preamble_input_{i}"), &concrete)
        })
        .collect();

    let preamble_output_elements: Vec<MleAst> = padded_outputs
        .chunks(32)
        .enumerate()
        .map(|(i, chunk)| {
            let mut buf = [0u8; 32];
            buf[..chunk.len()].copy_from_slice(chunk);
            let concrete = ark_bn254::Fr::from_le_bytes_mod_order(&buf);
            var_alloc.alloc_with_value(&format!("io_preamble_output_{i}"), &concrete)
        })
        .collect();

    // Panic → 8 bytes → 32-byte padded → Fr scalar → symbolic var
    let panic_u64 = io_device.panic as u64;
    let mut panic_padded = [0u8; 32];
    panic_padded[..8].copy_from_slice(&panic_u64.to_le_bytes());
    let panic_concrete = ark_bn254::Fr::from_le_bytes_mod_order(&panic_padded);
    let preamble_panic_element =
        var_alloc.alloc_with_value("io_preamble_panic", &panic_concrete);

    // Push in the exact order fiat_shamir_preamble calls append_bytes:
    // 1. inputs, 2. outputs, 3. panic
    push_bytes_override(preamble_input_elements);
    push_bytes_override(preamble_output_elements);
    push_bytes_override(vec![preamble_panic_element]);

    // --- 2. eval_io_mle overrides (u64-word field elements, padded to max) ---
    let num_input_words = max_input / 8;
    let num_output_words = max_output / 8;

    let eval_input_words: Vec<MleAst> = padded_inputs
        .chunks(8)
        .enumerate()
        .take(num_input_words)
        .map(|(i, chunk)| {
            let val = u64::from_le_bytes(chunk.try_into().unwrap());
            let concrete = ark_bn254::Fr::from(val);
            var_alloc.alloc_with_value(&format!("io_eval_input_{i}"), &concrete)
        })
        .collect();

    let eval_output_words: Vec<MleAst> = padded_outputs
        .chunks(8)
        .enumerate()
        .take(num_output_words)
        .map(|(i, chunk)| {
            let val = u64::from_le_bytes(chunk.try_into().unwrap());
            let concrete = ark_bn254::Fr::from(val);
            var_alloc.alloc_with_value(&format!("io_eval_output_{i}"), &concrete)
        })
        .collect();

    let eval_panic_val = {
        let concrete = ark_bn254::Fr::from(io_device.panic as u64);
        var_alloc.alloc_with_value("io_eval_panic", &concrete)
    };

    // Set pending IO MLE values (consumed by eval_io_mle)
    set_pending_io_mle(PendingIoMleValues {
        input_words: eval_input_words.clone(),
        output_words: eval_output_words,
        panic_val: eval_panic_val,
    });

    println!(
        "  Preamble: {} input chunks + {} output chunks + 1 panic",
        max_input / 32,
        max_output / 32,
    );
    println!(
        "  Eval: {} input words + {} output words + 1 panic",
        num_input_words, num_output_words,
    );

    eval_input_words
}
