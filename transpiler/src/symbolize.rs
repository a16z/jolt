//! Makes IO values (inputs, outputs, panic) into symbolic witness variables for the
//! RAM evaluation math.
//!
//! Under the spongefish/NARG protocol the statement (IO included) is bound into the
//! transcript via the **instance digest** (computed off-circuit, baked as a sponge-seed
//! constant — see `pipeline.rs` / TDEV-7), so there is no per-IO-byte transcript absorb
//! to intercept. The single remaining interception point is the RAM MLE:
//!
//! - **RAM MLE override** (`PENDING_IO_MLE` in `jolt-core/zkvm/ram/mod.rs`):
//!   `eval_io_mle` evaluates IO as a sparse polynomial over the RAM address space. We
//!   set a thread-local with symbolic field elements (one per u64 word of IO + a
//!   symbolic panic value); the early-return path in `eval_io_mle` picks these up.

use ark_bn254::Fr;
use common::jolt_device::JoltDevice;
use jolt_core::zkvm::ram::{set_pending_io_mle, PendingIoMleValues};
use zklean_extractor::mle_ast::MleAst;

use crate::symbolic_proof::VarAllocator;

/// Allocate symbolic witness variables for all IO values and set up the RAM MLE
/// interception point so that `verifier.verify()` uses them instead of concrete
/// constants.
///
/// Returns `(input_words, output_words)` at u64-word granularity — these are
/// passed to `PENDING_INITIAL_RAM` in the pipeline so `eval_initial_ram_mle` can
/// also use symbolic inputs.
pub fn symbolize_io_device(
    io_device: &JoltDevice,
    var_alloc: &mut VarAllocator,
) -> (Vec<MleAst>, Vec<MleAst>) {
    // RAM MLE override: symbolic field elements for eval_io_mle.
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
