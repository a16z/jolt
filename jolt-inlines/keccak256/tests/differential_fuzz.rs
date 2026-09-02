//! Differential fuzz of the keccak256 inline against `tiny-keccak`.
//!
//! Two layers:
//! - The expanded inline sequence, executed instruction-by-instruction in the
//!   tracer emulator (`InlineTestHarness`), vs `tiny_keccak::keccakf`. This is
//!   the layer that exercises `VirtualXORROTL1` and the in-place rho/pi/chi schedule.
//! - The `Keccak256` sponge (one-shot `digest` and chunked `update`/`finalize`:
//!   buffering, padding, absorption, squeezing) vs `tiny_keccak::Keccak::v256`
//!   over lengths and split points straddling the rate.
//!
//! Iteration counts scale with `KECCAK_FUZZ_ITERS`; CI runs only the 2,000-case
//! default per test. `KECCAK_FUZZ_ITERS=1000000` reproduces the one-off
//! million-case evidence gathered for #1749.
#![cfg(feature = "host")]

use jolt_inlines_keccak256::{
    Keccak256, INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_FUNCT3, KECCAK256_FUNCT7,
    NUM_LANES, RATE_IN_BYTES, RATE_IN_U64,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tiny_keccak::{Hasher, Keccak};
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

const STATE_IN_BYTES: usize = NUM_LANES * size_of::<u64>();

fn fuzz_iters(default: usize) -> usize {
    std::env::var("KECCAK_FUZZ_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn reference_digest(input: &[u8]) -> [u8; 32] {
    let mut reference = Keccak::v256();
    reference.update(input);
    let mut expected = [0u8; 32];
    reference.finalize(&mut expected);
    expected
}

#[test]
fn fuzz_inline_sequence_vs_tiny_keccak() {
    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xECCA_C001);

    for i in 0..iters {
        let mut state = [0u64; NUM_LANES];
        for lane in state.iter_mut() {
            *lane = rng.next_u64();
        }

        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(
            RATE_IN_BYTES,
            STATE_IN_BYTES,
        ));
        harness.setup_registers();
        harness.load_state64(&state);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            KECCAK256_FUNCT3,
            KECCAK256_FUNCT7,
        ));
        let actual = harness.read_output64(NUM_LANES);

        let mut expected = state;
        tiny_keccak::keccakf(&mut expected);

        assert_eq!(actual.as_slice(), expected.as_slice(), "iteration {i}");
    }
}

#[test]
fn fuzz_absorb_permute_inline_vs_tiny_keccak() {
    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xAB50_BB01);

    for i in 0..iters {
        let mut state = [0u64; NUM_LANES];
        let mut block = [0u64; RATE_IN_U64];
        for lane in &mut state {
            *lane = rng.next_u64();
        }
        for lane in &mut block {
            *lane = rng.next_u64();
        }

        let mut harness = InlineTestHarness::new(InlineMemoryLayout::single_input(
            RATE_IN_BYTES,
            STATE_IN_BYTES,
        ));
        harness.setup_registers();
        harness.load_state64(&state);
        harness.load_input64(&block);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            KECCAK256_ABSORB_PERMUTE_FUNCT3,
            KECCAK256_FUNCT7,
        ));
        let actual = harness.read_output64(NUM_LANES);

        for (lane, block_lane) in state.iter_mut().zip(block) {
            *lane ^= block_lane;
        }
        tiny_keccak::keccakf(&mut state);

        assert_eq!(actual.as_slice(), state.as_slice(), "iteration {i}");
    }
}

#[test]
fn fuzz_digest_vs_tiny_keccak() {
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
        // Odd iterations hash from a 1-byte offset into the (8-aligned) heap
        // buffer so the unaligned absorb paths run as often as the aligned ones.
        let offset = i % 2;
        let mut buffer = vec![0u8; len + offset];
        rng.fill_bytes(&mut buffer);
        let input = &buffer[offset..];

        assert_eq!(
            Keccak256::digest(input),
            reference_digest(input),
            "iteration {i}, len {len}, offset {offset}"
        );
    }
}

#[test]
fn fuzz_chunked_update_vs_tiny_keccak() {
    // Split schedules landing exactly on, and one byte either side of, the
    // 136- and 272-byte block boundaries.
    const BOUNDARY_SPLITS: &[&[usize]] = &[
        &[136],
        &[135, 1],
        &[1, 135],
        &[137],
        &[136, 136],
        &[135, 1, 136],
        &[136, 135, 1],
        &[137, 135],
        &[271, 1],
        &[1, 271],
        &[272],
        &[273],
    ];

    let iters = fuzz_iters(2_000);
    let mut rng = StdRng::seed_from_u64(0xC4A2_5E0D);

    for i in 0..iters {
        let chunks: Vec<usize> = if i % 2 == 0 {
            BOUNDARY_SPLITS[(i / 2) % BOUNDARY_SPLITS.len()].to_vec()
        } else {
            (0..1 + rng.next_u32() % 6)
                .map(|_| (rng.next_u32() % 300) as usize)
                .collect()
        };
        // Every fourth iteration finalizes right after a boundary schedule with
        // nothing buffered (or one byte, or the 0x81 combined-padding case).
        let tail = if i % 4 == 0 {
            0
        } else {
            (rng.next_u32() % 200) as usize
        };
        let len = chunks.iter().sum::<usize>() + tail;
        // Odd iterations feed chunks from a 1-byte offset so `update` absorbs
        // full blocks through both its aligned and unaligned paths.
        let offset = i % 2;
        let mut buffer = vec![0u8; len + offset];
        rng.fill_bytes(&mut buffer);
        let input = &buffer[offset..];

        let mut hasher = Keccak256::new();
        let mut consumed = 0;
        for chunk in chunks {
            hasher.update(&input[consumed..consumed + chunk]);
            consumed += chunk;
        }
        hasher.update(&input[consumed..]);

        assert_eq!(
            hasher.finalize(),
            reference_digest(input),
            "iteration {i}, len {len}, offset {offset}"
        );
    }
}
