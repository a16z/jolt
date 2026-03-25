use crate::sequence_builder::{ROTATION_OFFSETS, ROUND_CONSTANTS};
use crate::{Keccak256State, NUM_LANES};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness, INLINE};

#[cfg(all(test, feature = "host"))]
pub(crate) fn execute_keccak256(msg: &[u8]) -> [u8; 32] {
    const RATE_IN_BYTES: usize = 136;

    let mut state = [0u64; NUM_LANES];

    let mut offset = 0;
    while offset + RATE_IN_BYTES <= msg.len() {
        for (i, lane_bytes) in msg[offset..offset + RATE_IN_BYTES]
            .chunks_exact(8)
            .enumerate()
        {
            state[i] ^= u64::from_le_bytes(lane_bytes.try_into().unwrap());
        }

        execute_keccak_f(&mut state);
        offset += RATE_IN_BYTES;
    }

    let mut block = [0u8; RATE_IN_BYTES];
    let remaining = &msg[offset..];
    block[..remaining.len()].copy_from_slice(remaining);

    block[remaining.len()] ^= 0x01;
    block[RATE_IN_BYTES - 1] ^= 0x80;

    for (i, lane_bytes) in block.chunks_exact(8).enumerate() {
        state[i] ^= u64::from_le_bytes(lane_bytes.try_into().unwrap());
    }
    execute_keccak_f(&mut state);

    let mut hash = [0u8; 32];
    for (i, lane) in state.iter().take(4).enumerate() {
        hash[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    hash
}

pub(crate) fn execute_keccak_f(state: &mut Keccak256State) {
    for rc in ROUND_CONSTANTS {
        execute_theta(state);
        execute_rho_and_pi(state);
        execute_chi(state);
        execute_iota(state, rc);
    }
}

pub(crate) fn execute_theta(state: &mut Keccak256State) {
    let mut c = [0u64; 5];
    for x in 0..5 {
        c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    let mut d = [0u64; 5];
    for x in 0..5 {
        d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
    }
    for x in 0..5 {
        for y in 0..5 {
            state[x + 5 * y] ^= d[x];
        }
    }
}

pub(crate) fn execute_rho_and_pi(state: &mut Keccak256State) {
    let mut b = [0u64; NUM_LANES];
    for x in 0..5 {
        for y in 0..5 {
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            b[nx + 5 * ny] = state[x + 5 * y].rotate_left(ROTATION_OFFSETS[x][y]);
        }
    }
    state.copy_from_slice(&b);
}

pub(crate) fn execute_chi(state: &mut Keccak256State) {
    for y in 0..5 {
        let mut row = [0u64; 5];
        for x in 0..5 {
            row[x] = state[x + 5 * y];
        }
        for x in 0..5 {
            state[x + 5 * y] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
        }
    }
}

pub(crate) fn execute_iota(state: &mut Keccak256State, round_constant: u64) {
    state[0] ^= round_constant;
}

pub struct Keccak256PermutationSpec;

impl InlineSpec for Keccak256PermutationSpec {
    type Input = [u64; 25];
    type Output = [u64; 25];

    fn reference(input: &Self::Input) -> Self::Output {
        let mut state = *input;
        execute_keccak_f(&mut state);
        state
    }

    fn create_harness() -> InlineTestHarness {
        // Keccak permutation is in-place: rs1 points to the 200-byte state region
        // which serves as both input and output. The input_size=136 is the rate
        // (unused for the permutation itself); output_size=200 is the full state.
        let layout = InlineMemoryLayout::single_input(136, 200);
        InlineTestHarness::new(layout, Xlen::Bit64)
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            crate::KECCAK256_FUNCT3,
            crate::KECCAK256_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        // In-place: state is loaded into the output region (rs1 target)
        harness.load_state64(input);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let vec = harness.read_output64(25);
        vec.try_into().unwrap()
    }
}
