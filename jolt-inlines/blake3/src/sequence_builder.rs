//! This file contains BLAKE3-specific logic to expand the inline instruction to a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Internal state" = 16-word state array (v[0..15]) used during compression
//!   - "Chaining value" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use crate::{
    CHAINING_VALUE_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START, FLAG_KEYED_HASH, FLAG_ROOT, IV,
    MSG_BLOCK_LEN, MSG_SCHEDULE, NUM_ROUNDS,
};
use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, Kind, NoAdvice,
};
use jolt_inlines_sdk::jolt_asm;

/// Layout: v[0..15] + m[0..15] only (no separate h/counter/flags banks, no temp regs):
/// inputs load directly into their `v` slots and the chaining value is produced
/// in place via `v[i] ^= v[i+8]`.
pub const NEEDED_REGISTERS: usize = 32;

/// Virtual register layout:
/// - vr[0..15]:  Internal state `v`
/// - vr[16..31]: Message block `m`
const INTERNAL_STATE_VR_START: usize = 0;
const MSG_BLOCK_START_VR: usize = 16;

struct Blake3SequenceBuilder {
    asm: InlineExpansionBuilder,
    round: u8,
    vr: [InlineRegister; NEEDED_REGISTERS],
    operands: InlineOperands,
}

impl Blake3SequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let vr = asm.allocate_inline_array::<NEEDED_REGISTERS>()?;
        Ok(Self {
            asm,
            round: 0,
            vr,
            operands,
        })
    }

    fn build_general(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let output_register = self.operands.rs1;
        // Compression mode:
        // - Load chaining value (key) from rs1 directly into v[0..7]
        self.load_data_range_paired_dirty(
            self.operands.rs1,
            0,
            INTERNAL_STATE_VR_START,
            CHAINING_VALUE_LEN,
        );
        // - Load message from rs2 into m[0..15]
        self.load_data_range_paired_dirty(self.operands.rs2, 0, MSG_BLOCK_START_VR, MSG_BLOCK_LEN);
        // - Load counter, block_len, flags from rs2 tail directly into v[12..15]
        self.load_data_range_paired_dirty(
            self.operands.rs2,
            MSG_BLOCK_LEN * 4,
            INTERNAL_STATE_VR_START + 12,
            4,
        );
        self.initialize_internal_state();
        self.compress_and_store(output_register)
    }

    fn build_keyed64(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let output_register = self.operands.rs3;
        // Load key from rs3/rd directly into v[0..7]
        self.load_data_range_paired_dirty(
            self.operands.rs3,
            0,
            INTERNAL_STATE_VR_START,
            CHAINING_VALUE_LEN,
        );
        // Load left (32 bytes) from rs1 as message[0..7]
        self.load_data_range_paired_dirty(
            self.operands.rs1,
            0,
            MSG_BLOCK_START_VR,
            CHAINING_VALUE_LEN,
        );
        // Load right (32 bytes) from rs2 as message[8..15]
        self.load_data_range_paired_dirty(
            self.operands.rs2,
            0,
            MSG_BLOCK_START_VR + CHAINING_VALUE_LEN,
            CHAINING_VALUE_LEN,
        );
        self.initialize_internal_state();

        // v[12..15] = counter, block_len, flags
        // Keyed64: matches blake3::keyed_hash for 64-byte input
        // counter = 0, block_len = 64, flags = CHUNK_START|CHUNK_END|ROOT|KEYED_HASH
        //
        // NOTE: We intentionally omit the two `LUI 0` initializations for v[12], v[13].
        // Inline virtual registers are cleared by `finalize_inline`, so newly allocated
        // inline registers start at 0 across inline calls.
        self.asm
            .emit_u(Kind::LUI, *self.vr[INTERNAL_STATE_VR_START + 14], 64);
        self.asm.emit_u(
            Kind::LUI,
            *self.vr[INTERNAL_STATE_VR_START + 15],
            (FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH) as u64,
        );

        self.compress_and_store(output_register)
    }

    fn initialize_internal_state(&mut self) {
        // v[8..11] = IV[0..3]
        for (i, val) in IV.iter().enumerate().take(4) {
            self.asm
                .emit_u(Kind::LUI, *self.vr[CHAINING_VALUE_LEN + i], *val as u64);
        }
    }

    fn compress_and_store(
        mut self,
        output_register: u8,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        for round in 0..NUM_ROUNDS {
            self.round = round;
            self.blake3_round();
        }

        // Finalize: h[i] = v[i] ^ v[i+8], produced in place in v[0..7]
        for i in 0..CHAINING_VALUE_LEN {
            let vi = *self.vr[INTERNAL_STATE_VR_START + i];
            let vi8 = *self.vr[INTERNAL_STATE_VR_START + i + CHAINING_VALUE_LEN];
            jolt_asm!(self.asm, {
                xor vi, vi, vi8;
            });
        }

        // Store state
        for i in 0..CHAINING_VALUE_LEN / 2 {
            self.asm.store_paired_u32(
                output_register,
                (i * 2) as i64 * 4,
                *self.vr[INTERNAL_STATE_VR_START + i * 2],
                *self.vr[INTERNAL_STATE_VR_START + i * 2 + 1],
            );
        }

        self.asm.release_many(self.vr);
        self.asm.finalize()
    }

    /// Execute one round of BLAKE3 compression
    fn blake3_round(&mut self) {
        let msg_schedule_round = &MSG_SCHEDULE[self.round as usize];

        // Column step: apply G function to columns
        self.g_function(0, 4, 8, 12, msg_schedule_round[0], msg_schedule_round[1]);
        self.g_function(1, 5, 9, 13, msg_schedule_round[2], msg_schedule_round[3]);
        self.g_function(2, 6, 10, 14, msg_schedule_round[4], msg_schedule_round[5]);
        self.g_function(3, 7, 11, 15, msg_schedule_round[6], msg_schedule_round[7]);

        // Diagonal step: apply G function to diagonals
        self.g_function(0, 5, 10, 15, msg_schedule_round[8], msg_schedule_round[9]);
        self.g_function(1, 6, 11, 12, msg_schedule_round[10], msg_schedule_round[11]);
        self.g_function(2, 7, 8, 13, msg_schedule_round[12], msg_schedule_round[13]);
        self.g_function(3, 4, 9, 14, msg_schedule_round[14], msg_schedule_round[15]);
    }

    #[inline]
    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = *self.vr[a];
        let vb = *self.vr[b];
        let vc = *self.vr[c];
        let vd = *self.vr[d];
        let mx = *self.vr[MSG_BLOCK_START_VR + x];
        let my = *self.vr[MSG_BLOCK_START_VR + y];

        jolt_asm!(self.asm, {
            // v[a] = v[a] + v[b] + m[x]
            add va, va, vb;
            add va, va, mx;
            // v[d] = rotr32(v[d] ^ v[a], 16)
            xorrotw16 vd, vd, va;
            // v[c] = v[c] + v[d]
            add vc, vc, vd;
            // v[b] = rotr32(v[b] ^ v[c], 12)
            xorrotw12 vb, vb, vc;
            // v[a] = v[a] + v[b] + m[y]
            add va, va, vb;
            add va, va, my;
            // v[d] = rotr32(v[d] ^ v[a], 8)
            xorrotw8 vd, vd, va;
            // v[c] = v[c] + v[d]
            add vc, vc, vd;
            // v[b] = rotr32(v[b] ^ v[c], 7)
            xorrotw7 vb, vb, vc;
        });
    }

    fn load_data_range_paired_dirty(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        // WARNING: upper bits remain dirty until 32-bit XOR-rotate or paired-store masking.
        self.asm.load_paired_u32_range_dirty(
            base_register,
            memory_offset_start as i64,
            &self.vr[vr_start..vr_start + count],
        );
    }
}

pub struct Blake3Compression;

impl InlineOp for Blake3Compression {
    type Advice = NoAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::BLAKE3_FUNCT3;
    const FUNCT7: u32 = crate::BLAKE3_FUNCT7;
    const NAME: &'static str = crate::BLAKE3_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Blake3SequenceBuilder::new(asm, operands)?.build_general()
    }
}

pub struct Blake3Keyed64Compression;

impl InlineOp for Blake3Keyed64Compression {
    type Advice = NoAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::BLAKE3_KEYED64_FUNCT3;
    const FUNCT7: u32 = crate::BLAKE3_FUNCT7;
    const NAME: &'static str = crate::BLAKE3_KEYED64_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Blake3SequenceBuilder::new(asm, operands)?.build_keyed64()
    }
}

#[cfg(test)]
mod tests {
    use crate::{spec::Blake3Keyed64Input, CHAINING_VALUE_LEN};

    use super::{Blake3Compression, Blake3Keyed64Compression};
    use jolt_inlines_sdk::{
        assert_edge_cases_match_reference, assert_random_cases_match_reference, InlineSpec,
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_trace_result_equals_blake3_compress_reference() {
        assert_edge_cases_match_reference::<Blake3Compression>();
        assert_random_cases_match_reference::<Blake3Compression>(0xB1A3E3, 1000);
    }

    fn words_to_bytes(words: &[u32; CHAINING_VALUE_LEN]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn keyed_hash_words((left, right, key): &Blake3Keyed64Input) -> [u32; CHAINING_VALUE_LEN] {
        let key_bytes = words_to_bytes(key);
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&words_to_bytes(left));
        input[32..].copy_from_slice(&words_to_bytes(right));

        let digest = blake3::keyed_hash(&key_bytes, &input);
        core::array::from_fn(|i| {
            u32::from_le_bytes(digest.as_bytes()[i * 4..i * 4 + 4].try_into().unwrap())
        })
    }

    fn assert_keyed64_trace_matches_blake3_keyed_hash(input: &Blake3Keyed64Input) {
        let expected = keyed_hash_words(input);
        let mut harness = <Blake3Keyed64Compression as InlineSpec>::harness();
        harness.setup_registers();
        <Blake3Keyed64Compression as InlineSpec>::load(&mut harness, input);
        harness.execute_inline(<Blake3Keyed64Compression as InlineSpec>::instruction());
        let actual = <Blake3Keyed64Compression as InlineSpec>::read(&mut harness);

        assert_eq!(
            actual, expected,
            "BLAKE3 keyed64 trace output mismatch against blake3::keyed_hash"
        );
    }

    #[test]
    fn test_trace_keyed64_matches_blake3_keyed_hash() {
        assert_edge_cases_match_reference::<Blake3Keyed64Compression>();
        assert_random_cases_match_reference::<Blake3Keyed64Compression>(0x88888, 100);

        for input in <Blake3Keyed64Compression as InlineSpec>::edge_cases() {
            assert_keyed64_trace_matches_blake3_keyed_hash(&input);
        }

        let mut rng = StdRng::seed_from_u64(0x88888);
        for _ in 0..100 {
            let input = <Blake3Keyed64Compression as InlineSpec>::random(&mut rng);
            assert_keyed64_trace_matches_blake3_keyed_hash(&input);
        }
    }
}
