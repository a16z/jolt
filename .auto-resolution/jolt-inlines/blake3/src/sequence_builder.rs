//! This file contains BLAKE3-specific logic to expand the inline instruction to a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Internal state" = 16-word state array (v[0..15]) used during compression
//!   - "Chaining value" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use core::array;

use crate::{
    CHAINING_VALUE_LEN, COUNTER_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START, FLAG_KEYED_HASH, FLAG_ROOT,
    IV, MSG_BLOCK_LEN, MSG_SCHEDULE, NUM_ROUNDS,
};
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::ld::LD;
use tracer::instruction::lui::LUI;
use tracer::instruction::lw::LW;
use tracer::instruction::or::OR;
use tracer::instruction::sd::SD;
use tracer::instruction::slli::SLLI;
use tracer::instruction::srli::SRLI;
use tracer::instruction::virtual_xor_rotw::{
    VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8,
};
use tracer::instruction::virtual_zero_extend_word::VirtualZeroExtendWord;
use tracer::instruction::Instruction;
use tracer::utils::inline_helpers::{InstrAssembler, Value::Imm, Value::Reg};
use tracer::utils::virtual_registers::VirtualRegisterGuard;

/// Number of virtual registers needed for the general compression builder.
/// Layout: v[0..15] + m[0..15] + h[0..7] + counter[0..1] + block_len + flags + temp1 + temp2
pub const NEEDED_REGISTERS: u8 = 46;

/// Number of virtual registers needed for the keyed64 builder (smaller footprint).
/// Layout: v[0..15] + m[0..15] only (no separate h/counter/flags banks, no temp regs).
pub const NEEDED_REGISTERS_KEYED64: u8 = 32;

/// Apply the BLAKE3 round schedule for a given round index by calling `g` 8 times
/// in the exact order required by the spec.
#[inline]
fn blake3_apply_round_schedule<F>(round: u8, mut g: F)
where
    F: FnMut(usize, usize, usize, usize, usize, usize),
{
    let msg_schedule_round = &MSG_SCHEDULE[round as usize];

    // Column step: apply G function to columns
    g(0, 4, 8, 12, msg_schedule_round[0], msg_schedule_round[1]);
    g(1, 5, 9, 13, msg_schedule_round[2], msg_schedule_round[3]);
    g(2, 6, 10, 14, msg_schedule_round[4], msg_schedule_round[5]);
    g(3, 7, 11, 15, msg_schedule_round[6], msg_schedule_round[7]);

    // Diagonal step: apply G function to diagonals
    g(0, 5, 10, 15, msg_schedule_round[8], msg_schedule_round[9]);
    g(1, 6, 11, 12, msg_schedule_round[10], msg_schedule_round[11]);
    g(2, 7, 8, 13, msg_schedule_round[12], msg_schedule_round[13]);
    g(3, 4, 9, 14, msg_schedule_round[14], msg_schedule_round[15]);
}

/// Virtual register layout:
/// - vr[0..15]:  Internal state `v`
/// - vr[16..31]: Message block `m`
/// - vr[32..39]: Chaining value `h`
/// - vr[40..41]: Counter values
/// - vr[42]:     Input bytes length
/// - vr[43]:     Flags
/// - vr[44]:     Temporary register 1
/// - vr[45]:     Temporary register 2 (for paired store)
const INTERNAL_STATE_VR_START: usize = 0;
const MSG_BLOCK_START_VR: usize = 16;
const CV_START_VR: usize = 32;
const COUNTER_START_VR: usize = 40;
const INPUT_BYTES_VR: usize = 42;
const FLAG_VR: usize = 43;
const TEMP_VR: usize = 44;
const TEMP_VR2: usize = 45;

struct Blake3SequenceBuilder {
    asm: InstrAssembler,
    round: u8,
    vr: [VirtualRegisterGuard; NEEDED_REGISTERS as usize],
    operands: FormatInline,
}

/// Keyed64-only sequence builder with a smaller VR footprint.
///
/// This builder is separate from `Blake3SequenceBuilder` because:
/// 1. It uses only 32 VRs (vs 46), reducing `finalize_inline` zeroing overhead.
/// 2. It loads key directly into v[0..7] (no separate CV bank).
/// 3. Its `g_function` uses in-place adds (no temp register needed).
///
/// Virtual register layout:
/// - vr[0..15]:  Internal state `v`
/// - vr[16..31]: Message block `m` (left||right)
struct Blake3Keyed64SequenceBuilder {
    asm: InstrAssembler,
    round: u8,
    vr: [VirtualRegisterGuard; NEEDED_REGISTERS_KEYED64 as usize],
    operands: FormatInline,
}

impl Blake3SequenceBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = array::from_fn(|_| asm.allocator.allocate_for_inline());
        Blake3SequenceBuilder {
            asm,
            round: 0,
            vr,
            operands,
        }
    }

    fn build(mut self) -> Vec<Instruction> {
        // Compression mode:
        // - Load chaining value (key) from rs1
        // - Load message from rs2
        // - Load counter, block_len, flags from rs2 tail
        self.load_chaining_value();
        self.load_message_blocks();
        self.load_counter();
        self.load_input_len_and_flags();

        self.initialize_internal_state();

        for round in 0..NUM_ROUNDS {
            self.round = round;
            self.blake3_round();
        }

        // Finalize: h[i] = v[i] ^ v[i+8]
        self.finalize_state();

        // Store state
        self.store_state();

        drop(self.vr);
        self.asm.finalize_inline()
    }

    fn initialize_internal_state(&mut self) {
        // v[0..7] = chaining value (loaded from memory via rs1)
        for i in 0..CHAINING_VALUE_LEN {
            self.asm.xor(
                Reg(*self.vr[CV_START_VR + i]),
                Imm(0),
                *self.vr[INTERNAL_STATE_VR_START + i],
            );
        }

        // v[8..11] = IV[0..3]
        for (i, val) in IV.iter().enumerate().take(4) {
            self.asm
                .emit_u::<LUI>(*self.vr[CHAINING_VALUE_LEN + i], *val as u64);
        }

        // v[12..15] = counter, block_len, flags (loaded from memory)
        self.asm.xor(
            Reg(*self.vr[COUNTER_START_VR]),
            Imm(0),
            *self.vr[INTERNAL_STATE_VR_START + 12],
        );
        self.asm.xor(
            Reg(*self.vr[COUNTER_START_VR + 1]),
            Imm(0),
            *self.vr[INTERNAL_STATE_VR_START + 13],
        );
        self.asm.xor(
            Reg(*self.vr[INPUT_BYTES_VR]),
            Imm(0),
            *self.vr[INTERNAL_STATE_VR_START + 14],
        );
        self.asm.xor(
            Reg(*self.vr[FLAG_VR]),
            Imm(0),
            *self.vr[INTERNAL_STATE_VR_START + 15],
        );
    }

    /// Execute one round of BLAKE3 compression
    fn blake3_round(&mut self) {
        let round = self.round;
        blake3_apply_round_schedule(round, |a, b, c, d, x, y| self.g_function(a, b, c, d, x, y));
    }

    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = *self.vr[a];
        let vb = *self.vr[b];
        let vc = *self.vr[c];
        let vd = *self.vr[d];
        let mx = *self.vr[MSG_BLOCK_START_VR + x];
        let my = *self.vr[MSG_BLOCK_START_VR + y];
        let temp1 = *self.vr[TEMP_VR];

        // v[a] = v[a] + v[b] + m[x]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(mx), va);

        // v[d] = rotr32(v[d] ^ v[a], 16)
        self.asm.emit_r::<VirtualXORROTW16>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 12)
        self.asm.emit_r::<VirtualXORROTW12>(vb, vb, vc);

        // v[a] = v[a] + v[b] + m[y]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(my), va);

        // v[d] = rotr32(v[d] ^ v[a], 8)
        self.asm.emit_r::<VirtualXORROTW8>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 7)
        self.asm.emit_r::<VirtualXORROTW7>(vb, vb, vc);
    }

    fn finalize_state(&mut self) {
        for i in 0..CHAINING_VALUE_LEN {
            let hi = *self.vr[CV_START_VR + i];
            let vi = *self.vr[INTERNAL_STATE_VR_START + i];
            let vi8 = *self.vr[INTERNAL_STATE_VR_START + i + 8];
            self.asm.xor(Reg(vi), Reg(vi8), hi);
        }
    }

    /// Update chaining value using paired store (optimized for 8-byte aligned access)
    fn store_state(&mut self) {
        // Store 8 u32 values as 4 paired u64 stores
        for i in 0..CHAINING_VALUE_LEN / 2 {
            self.store_paired_u32(
                self.operands.rs1,
                (i * 2) as i64 * 4,
                *self.vr[CV_START_VR + i * 2],
                *self.vr[CV_START_VR + i * 2 + 1],
            );
        }
    }

    /// Load two u32 values from an 8-byte aligned address using a single LD
    /// This is more efficient than two separate LW instructions in 64-bit mode
    fn load_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        let v_dword = *self.vr[TEMP_VR];

        // Load 64 bits (2 x u32)
        self.asm.emit_ld::<LD>(v_dword, base, offset);

        // Extract low 32 bits: zero-extend word
        self.asm.emit_i::<VirtualZeroExtendWord>(vr_lo, v_dword, 0);

        // Extract high 32 bits: shift right by 32
        self.asm.emit_i::<SRLI>(vr_hi, v_dword, 32);
    }

    /// Store two u32 values to an 8-byte aligned address using a single SD
    fn store_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        let v_dword = *self.vr[TEMP_VR];
        let v_high_shifted = *self.vr[TEMP_VR2];

        // Zero-extend low word to clear upper 32 bits
        self.asm.emit_i::<VirtualZeroExtendWord>(v_dword, vr_lo, 0);

        // Shift high word to upper 32 bits
        self.asm.emit_i::<SLLI>(v_high_shifted, vr_hi, 32);

        // OR them together
        self.asm.emit_r::<OR>(v_dword, v_dword, v_high_shifted);

        // Store 64 bits
        self.asm.emit_s::<SD>(base, v_dword, offset);
    }

    /// Load data from memory using paired access (optimized)
    /// Requires 8-byte alignment
    fn load_data_range_paired(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        debug_assert!(count % 2 == 0, "count must be even for paired loading");
        for i in 0..count / 2 {
            self.load_paired_u32(
                base_register,
                (memory_offset_start + i * 2) as i64 * 4,
                *self.vr[vr_start + i * 2],
                *self.vr[vr_start + i * 2 + 1],
            );
        }
    }

    /// Load data from memory into virtual registers (non-paired, used for counter)
    fn load_data_range(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        (0..count).for_each(|i| {
            self.asm.emit_ld::<LW>(
                *self.vr[vr_start + i],
                base_register,
                (memory_offset_start + i) as i64 * 4,
            );
        });
    }

    fn load_chaining_value(&mut self) {
        // Use paired loading for chaining value (8 u32 = 4 pairs)
        self.load_data_range_paired(self.operands.rs1, 0, CV_START_VR, CHAINING_VALUE_LEN);
    }

    fn load_message_blocks(&mut self) {
        // Use paired loading for message blocks (16 u32 = 8 pairs)
        self.load_data_range_paired(self.operands.rs2, 0, MSG_BLOCK_START_VR, MSG_BLOCK_LEN);
    }

    fn load_counter(&mut self) {
        self.load_data_range(
            self.operands.rs2,
            MSG_BLOCK_LEN,
            COUNTER_START_VR,
            COUNTER_LEN,
        );
    }

    fn load_input_len_and_flags(&mut self) {
        // input length
        self.asm.emit_ld::<LW>(
            *self.vr[INPUT_BYTES_VR],
            self.operands.rs2,
            (MSG_BLOCK_LEN + COUNTER_LEN) as i64 * 4,
        );
        // flag
        self.asm.emit_ld::<LW>(
            *self.vr[FLAG_VR],
            self.operands.rs2,
            (MSG_BLOCK_LEN + COUNTER_LEN + 1) as i64 * 4,
        );
    }
}

impl Blake3Keyed64SequenceBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = array::from_fn(|_| asm.allocator.allocate_for_inline());
        Self {
            asm,
            round: 0,
            vr,
            operands,
        }
    }

    fn build(mut self) -> Vec<Instruction> {
        // Load key from rs3/rd directly into v[0..7]
        self.load_data_range_paired(self.operands.rs3, 0, INTERNAL_STATE_VR_START, 8);
        // Load left (32 bytes) from rs1 as message[0..7]
        self.load_data_range_paired(self.operands.rs1, 0, MSG_BLOCK_START_VR, 8);
        // Load right (32 bytes) from rs2 as message[8..15]
        self.load_data_range_paired(self.operands.rs2, 0, MSG_BLOCK_START_VR + 8, 8);

        self.initialize_internal_state();

        for round in 0..NUM_ROUNDS {
            self.round = round;
            self.blake3_round();
        }

        // Finalize for this inline: v[i] = v[i] ^ v[i+8], store v[0..7] to rs3/rd.
        for i in 0..CHAINING_VALUE_LEN {
            let vi = *self.vr[INTERNAL_STATE_VR_START + i];
            let vi8 = *self.vr[INTERNAL_STATE_VR_START + i + 8];
            self.asm.xor(Reg(vi), Reg(vi8), vi);
        }

        for i in 0..CHAINING_VALUE_LEN / 2 {
            self.store_paired_u32_in_place(
                self.operands.rs3,
                (i * 2) as i64 * 4,
                *self.vr[INTERNAL_STATE_VR_START + i * 2],
                *self.vr[INTERNAL_STATE_VR_START + i * 2 + 1],
            );
        }

        drop(self.vr);
        self.asm.finalize_inline()
    }

    fn initialize_internal_state(&mut self) {
        // v[8..11] = IV[0..3]
        for (i, val) in IV.iter().enumerate().take(4) {
            self.asm
                .emit_u::<LUI>(*self.vr[CHAINING_VALUE_LEN + i], *val as u64);
        }

        // v[12..15] = counter, block_len, flags
        // Keyed64: matches blake3::keyed_hash for 64-byte input
        // counter = 0, block_len = 64, flags = CHUNK_START|CHUNK_END|ROOT|KEYED_HASH
        //
        // NOTE: We intentionally omit the two `LUI 0` initializations for v[12], v[13].
        // Inline virtual registers are cleared by `finalize_inline`, so newly allocated
        // inline registers start at 0 across inline calls.
        self.asm
            .emit_u::<LUI>(*self.vr[INTERNAL_STATE_VR_START + 14], 64);
        self.asm.emit_u::<LUI>(
            *self.vr[INTERNAL_STATE_VR_START + 15],
            (FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | FLAG_KEYED_HASH) as u64,
        );
    }

    /// Execute one round of BLAKE3 compression
    fn blake3_round(&mut self) {
        let round = self.round;
        blake3_apply_round_schedule(round, |a, b, c, d, x, y| self.g_function(a, b, c, d, x, y));
    }

    #[inline]
    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = *self.vr[a];
        let vb = *self.vr[b];
        let vc = *self.vr[c];
        let vd = *self.vr[d];
        let mx = *self.vr[MSG_BLOCK_START_VR + x];
        let my = *self.vr[MSG_BLOCK_START_VR + y];

        // v[a] = v[a] + v[b] + m[x]
        self.asm.add(Reg(va), Reg(vb), va);
        self.asm.add(Reg(va), Reg(mx), va);

        // v[d] = rotr32(v[d] ^ v[a], 16)
        self.asm.emit_r::<VirtualXORROTW16>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 12)
        self.asm.emit_r::<VirtualXORROTW12>(vb, vb, vc);

        // v[a] = v[a] + v[b] + m[y]
        self.asm.add(Reg(va), Reg(vb), va);
        self.asm.add(Reg(va), Reg(my), va);

        // v[d] = rotr32(v[d] ^ v[a], 8)
        self.asm.emit_r::<VirtualXORROTW8>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 7)
        self.asm.emit_r::<VirtualXORROTW7>(vb, vb, vc);
    }

    /// Load two u32 values from an 8-byte aligned address using a single LD.
    /// Uses `vr_hi` as the temporary 64-bit container (no extra scratch register).
    fn load_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        // Load 64 bits (2 x u32) into vr_hi temporarily.
        self.asm.emit_ld::<LD>(vr_hi, base, offset);

        // Extract low 32 bits: zero-extend word.
        self.asm.emit_i::<VirtualZeroExtendWord>(vr_lo, vr_hi, 0);

        // Extract high 32 bits: shift right by 32 (in place).
        self.asm.emit_i::<SRLI>(vr_hi, vr_hi, 32);
    }

    fn load_data_range_paired(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        debug_assert!(count % 2 == 0, "count must be even for paired loading");
        for i in 0..count / 2 {
            self.load_paired_u32(
                base_register,
                (memory_offset_start + i * 2) as i64 * 4,
                *self.vr[vr_start + i * 2],
                *self.vr[vr_start + i * 2 + 1],
            );
        }
    }

    /// Store two u32 values to an 8-byte aligned address using a single SD.
    /// Mutates `vr_lo` and `vr_hi` (safe at end of the keyed64 sequence).
    fn store_paired_u32_in_place(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        // Zero-extend low word to clear upper 32 bits (in place).
        self.asm.emit_i::<VirtualZeroExtendWord>(vr_lo, vr_lo, 0);
        // Shift high word to upper 32 bits (in place).
        self.asm.emit_i::<SLLI>(vr_hi, vr_hi, 32);
        // OR them together into vr_lo.
        self.asm.emit_r::<OR>(vr_lo, vr_lo, vr_hi);
        // Store 64 bits from vr_lo.
        self.asm.emit_s::<SD>(base, vr_lo, offset);
    }
}

pub fn blake3_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Blake3SequenceBuilder::new(asm, operands);
    builder.build()
}

/// Build sequence for keyed64 (`blake3::keyed_hash` for a single 64B block):
/// - **Input**: left 32B from `rs1`, right 32B from `rs2`
/// - **Key**: 32B from `rs3/rd` (also used as the output destination)
/// - **Params**: `counter = 0`, `block_len = 64`,
///   `flags = CHUNK_START | CHUNK_END | ROOT | KEYED_HASH`
/// - **Output**: overwrites `rs3/rd` with the 32B digest
pub fn blake3_keyed64_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Blake3Keyed64SequenceBuilder::new(asm, operands);
    builder.build()
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        create_blake3_harness, create_blake3_keyed64_harness, helpers::*, instruction,
        keyed64_instruction, load_blake3_data, load_blake3_keyed64_data, read_output,
        ChainingValue, MessageBlock,
    };

    fn generate_trace_result(
        chaining_value: &ChainingValue,
        message: &MessageBlock,
        counter: &[u32; 2],
        block_len: u32,
        flags: u32,
    ) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        let mut harness = create_blake3_harness();
        load_blake3_data(
            &mut harness,
            chaining_value,
            message,
            counter,
            block_len,
            flags,
        );
        harness.execute_inline(instruction());
        let words = read_output(&mut harness);

        let mut bytes = [0u8; crate::OUTPUT_SIZE_IN_BYTES];
        for (i, w) in words.iter().enumerate() {
            let le = w.to_le_bytes();
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&le);
        }
        bytes
    }

    #[test]
    fn test_trace_result_equals_blake3_compress_reference() {
        for _ in 0..1000 {
            let message_bytes = generate_random_bytes(crate::MSG_BLOCK_LEN * 4);
            // Convert bytes to message block (u32 words)
            assert_eq!(
                message_bytes.len(),
                crate::MSG_BLOCK_LEN * 4,
                "Message must be exactly {} bytes",
                crate::MSG_BLOCK_LEN * 4
            );
            let words_vec = bytes_to_u32_vec(&message_bytes);
            let mut message_words = [0u32; crate::MSG_BLOCK_LEN];
            message_words.copy_from_slice(&words_vec);

            let expected_hash_bytes = compute_expected_result(&message_bytes);
            let counter = [0u32, 0u32];
            let block_len = 64u32;
            let flags = crate::FLAG_CHUNK_START | crate::FLAG_CHUNK_END | crate::FLAG_ROOT;
            let trace_hash_bytes =
                generate_trace_result(&crate::IV, &message_words, &counter, block_len, flags);
            assert_eq!(
                trace_hash_bytes, expected_hash_bytes,
                "trace hash bytes mismatch"
            );
        }
    }

    #[test]
    fn test_trace_result_equals_blake3_keyed_compress_reference() {
        for _ in 0..1000 {
            // Generate random key
            let key_bytes = generate_random_bytes(crate::CHAINING_VALUE_LEN * 4);
            let mut key = [0u32; crate::CHAINING_VALUE_LEN];
            key.copy_from_slice(&bytes_to_u32_vec(&key_bytes));

            // Generate random message
            let message_bytes = generate_random_bytes(crate::MSG_BLOCK_LEN * 4);
            let words_vec = bytes_to_u32_vec(&message_bytes);
            let mut message_words = [0u32; crate::MSG_BLOCK_LEN];
            message_words.copy_from_slice(&words_vec);

            // Compute expected result using keyed hash
            let expected_hash_bytes = compute_keyed_expected_result(&message_bytes, key);

            // Generate trace result with keyed hash flag
            let counter = [0u32, 0u32];
            let block_len = 64u32;
            let flags = crate::FLAG_CHUNK_START
                | crate::FLAG_CHUNK_END
                | crate::FLAG_ROOT
                | crate::FLAG_KEYED_HASH;
            let trace_hash_bytes =
                generate_trace_result(&key, &message_words, &counter, block_len, flags);

            assert_eq!(
                trace_hash_bytes, expected_hash_bytes,
                "keyed trace hash bytes mismatch"
            );
        }
    }

    #[test]
    fn test_trace_keyed64_matches_blake3_keyed_hash() {
        // Test that sequence builder's Keyed64 mode matches blake3::keyed_hash for 64-byte input
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(88888);

        for _ in 0..100 {
            // Generate random left, right, and key
            let mut left = [0u32; crate::CHAINING_VALUE_LEN];
            let mut right = [0u32; crate::CHAINING_VALUE_LEN];
            let mut key = [0u32; crate::CHAINING_VALUE_LEN];
            for i in 0..crate::CHAINING_VALUE_LEN {
                left[i] = rng.gen();
                right[i] = rng.gen();
                key[i] = rng.gen();
            }

            // Execute sequence builder with key as IV
            let mut harness = create_blake3_keyed64_harness();
            load_blake3_keyed64_data(&mut harness, &left, &right, &key);
            harness.execute_inline(keyed64_instruction());
            let result_words = read_output(&mut harness);

            // Convert result to bytes
            let mut result_bytes = [0u8; 32];
            for (i, w) in result_words.iter().enumerate() {
                let le = w.to_le_bytes();
                result_bytes[i * 4..(i + 1) * 4].copy_from_slice(&le);
            }

            // Convert left/right/key to bytes for blake3 reference
            let mut left_bytes = [0u8; 32];
            let mut right_bytes = [0u8; 32];
            let mut key_bytes = [0u8; 32];
            for (i, w) in left.iter().enumerate() {
                left_bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
            }
            for (i, w) in right.iter().enumerate() {
                right_bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
            }
            for (i, w) in key.iter().enumerate() {
                key_bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
            }

            // Concatenate left || right as 64-byte input
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&left_bytes);
            input[32..].copy_from_slice(&right_bytes);

            // Compute expected using official blake3::keyed_hash
            let expected = blake3::keyed_hash(&key_bytes, &input);

            assert_eq!(
                result_bytes,
                *expected.as_bytes(),
                "Keyed64 sequence builder does not match blake3::keyed_hash"
            );
        }
    }
}
