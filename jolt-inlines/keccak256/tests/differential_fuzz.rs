//! Differential fuzz of the keccak256 inline against `tiny-keccak`.
//!
//! Two layers:
//! - The expanded inline sequence, executed instruction-by-instruction in the
//!   tracer emulator (`InlineTestHarness`), vs `tiny_keccak::keccakf`. This is
//!   the layer that exercises the emitted `VirtualXORROT*` rows.
//! - The one-shot `Keccak256::digest` sponge (padding, absorption, squeezing)
//!   vs `tiny_keccak::Keccak::v256` over random lengths straddling the rate.
//!
//! Iteration counts scale with `KECCAK_FUZZ_ITERS` (default keeps CI fast);
//! the soundness campaign for the rho-fusion change ran 1M+ permutations.
#![cfg(feature = "host")]

use rand::{RngCore, SeedableRng};
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

fn fuzz_iters(default: usize) -> usize {
    std::env::var("KECCAK_FUZZ_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn fuzz_inline_sequence_vs_tiny_keccak() {
    let _ = jolt_inlines_keccak256::KECCAK256_NAME;
    let iters = fuzz_iters(2_000);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xECCA_C001);

    for i in 0..iters {
        let mut state = [0u64; 25];
        for lane in state.iter_mut() {
            *lane = rng.next_u64();
        }

        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(136, 200));
        harness.setup_registers();
        harness.load_state64(&state);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            jolt_inlines_keccak256::INLINE_OPCODE,
            jolt_inlines_keccak256::KECCAK256_FUNCT3,
            jolt_inlines_keccak256::KECCAK256_FUNCT7,
        ));
        let actual = harness.read_output64(25);

        let mut expected = state;
        tiny_keccak::keccakf(&mut expected);

        assert_eq!(actual.as_slice(), expected.as_slice(), "iteration {i}");
    }
}

#[test]
fn fuzz_digest_vs_tiny_keccak() {
    use tiny_keccak::Hasher;

    let iters = fuzz_iters(2_000);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xD16E57);

    for i in 0..iters {
        // Cover empty inputs, sub-rate, rate-straddling, and multi-block sizes.
        let len = match i % 4 {
            0 => (rng.next_u32() % 16) as usize,
            1 => 120 + (rng.next_u32() % 32) as usize,
            2 => (rng.next_u32() % 600) as usize,
            _ => (rng.next_u32() % 4096) as usize,
        };
        let mut input = vec![0u8; len];
        rng.fill_bytes(&mut input);

        let actual = jolt_inlines_keccak256::Keccak256::digest(&input);

        let mut reference = tiny_keccak::Keccak::v256();
        reference.update(&input);
        let mut expected = [0u8; 32];
        reference.finalize(&mut expected);

        assert_eq!(actual, expected, "iteration {i}, len {len}");
    }
}
