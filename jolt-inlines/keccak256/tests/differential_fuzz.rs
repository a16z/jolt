//! Differential fuzz of the keccak256 inline against `tiny-keccak`.
//!
//! Two layers:
//! - The expanded inline sequence, executed instruction-by-instruction in the
//!   tracer emulator (`InlineTestHarness`), vs `tiny_keccak::keccakf`. This is
//!   the layer that exercises `VirtualXORROTL1` and the in-place rho/pi/chi schedule.
//! - The one-shot `Keccak256::digest` sponge (padding, absorption, squeezing)
//!   vs `tiny_keccak::Keccak::v256` over random lengths straddling the rate.
//!
//! Iteration counts scale with `KECCAK_FUZZ_ITERS` (default keeps CI fast);
//! the soundness gate runs one million iterations for each inline entry point.
#![cfg(feature = "host")]

use jolt_inlines_keccak256::{
    Keccak256, INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_ABSORB_PERMUTE_NAME,
    KECCAK256_FUNCT3, KECCAK256_FUNCT7, KECCAK256_NAME, RATE_IN_BYTES, RATE_IN_U64,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tiny_keccak::Keccak;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

fn fuzz_iters(default: usize) -> usize {
    std::env::var("KECCAK_FUZZ_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn fuzz_inline_sequence_vs_tiny_keccak() {
    let _ = KECCAK256_NAME;
    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xECCA_C001);

    for i in 0..iters {
        let mut state = [0u64; 25];
        for lane in state.iter_mut() {
            *lane = rng.next_u64();
        }

        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(136, 200));
        harness.setup_registers();
        harness.load_state64(&state);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            KECCAK256_FUNCT3,
            KECCAK256_FUNCT7,
        ));
        let actual = harness.read_output64(25);

        let mut expected = state;
        tiny_keccak::keccakf(&mut expected);

        assert_eq!(actual.as_slice(), expected.as_slice(), "iteration {i}");
    }
}

#[test]
fn fuzz_absorb_permute_inline_vs_tiny_keccak() {
    let _ = KECCAK256_ABSORB_PERMUTE_NAME;
    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xAB50_BB01);

    for i in 0..iters {
        let mut state = [0u64; 25];
        let mut block = [0u64; RATE_IN_U64];
        for lane in &mut state {
            *lane = rng.next_u64();
        }
        for lane in &mut block {
            *lane = rng.next_u64();
        }

        let mut harness =
            InlineTestHarness::new(InlineMemoryLayout::single_input(RATE_IN_BYTES, 200));
        harness.setup_registers();
        harness.load_state64(&state);
        harness.load_input64(&block);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            KECCAK256_ABSORB_PERMUTE_FUNCT3,
            KECCAK256_FUNCT7,
        ));
        let actual = harness.read_output64(25);

        for (lane, block_lane) in state.iter_mut().zip(block) {
            *lane ^= block_lane;
        }
        tiny_keccak::keccakf(&mut state);

        assert_eq!(actual.as_slice(), state.as_slice(), "iteration {i}");
    }
}

#[test]
fn fuzz_digest_vs_tiny_keccak() {
    use tiny_keccak::Hasher;

    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xD16E57);

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

        let actual = Keccak256::digest(&input);

        let mut reference = Keccak::v256();
        reference.update(&input);
        let mut expected = [0u8; 32];
        reference.finalize(&mut expected);

        assert_eq!(actual, expected, "iteration {i}, len {len}");
    }
}
