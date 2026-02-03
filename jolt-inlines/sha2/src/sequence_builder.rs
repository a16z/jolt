use core::array;

use tracer::{
    emulator::cpu::Xlen,
    instruction::{
        andn::ANDN, format::format_inline::FormatInline, ld::LD, lw::LW, or::OR, sd::SD,
        slli::SLLI, srli::SRLI, sw::SW, virtual_zero_extend_word::VirtualZeroExtendWord,
        Instruction,
    },
    utils::{
        inline_helpers::{
            InstrAssembler,
            Value::{self, Imm, Reg},
        },
        virtual_registers::VirtualRegisterGuard,
    },
};

/// SHA-256 initial hash values
pub const BLOCK: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 round constants (K)
pub const K: [u64; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Builds assembly sequence for SHA256 compression
/// Expects A..H to be in RAM at location rs1..rs1+8 (also where output is written)
/// Expects input words to be in RAM at location rs2..rs2+16
/// Output will be written to rs1..rs1+8
struct Sha256SequenceBuilder {
    asm: InstrAssembler,
    /// Round id
    round: i32,
    /// Working state registers A-H
    state: [VirtualRegisterGuard; 8],
    /// Message schedule W[0..15] (16 registers)
    message: [VirtualRegisterGuard; 16],
    /// Initial state values for final addition (8 registers, only used when !initial)
    iv: Vec<VirtualRegisterGuard>,
    /// Operands
    operands: FormatInline,
    /// Whether this is the initial compression (use BLOCK constants)
    initial: bool,
}

impl Sha256SequenceBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline, initial: bool) -> Self {
        let state = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let message = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let iv = if initial {
            vec![]
        } else {
            (0..8)
                .map(|_| asm.allocator.allocate_for_inline())
                .collect()
        };

        Sha256SequenceBuilder {
            asm,
            round: 0,
            state,
            message,
            iv,
            operands,
            initial,
        }
    }

    /// Loads and runs all SHA256 rounds
    fn build(mut self) -> Vec<Instruction> {
        if !self.initial {
            // Load initial hash values from memory when using custom IV
            // Load all A-H into initial_state registers (used both for initial values and final addition)
            if self.asm.xlen == Xlen::Bit32 {
                // if 32-bit, LW is cheap so use it
                (0..8).for_each(|i| {
                    self.asm
                        .emit_ld::<LW>(*self.iv[i], self.operands.rs1, (i * 4) as i64)
                });
            } else {
                // if 64-bit, LW expands into a long sequence of virtual instructions
                // so we use LD to load two registers at a time
                // the sequence is A,B | C,D | E,F | G,H
                (0..4).for_each(|i| {
                    let dst1 = *self.iv[i * 2];
                    let dst2 = *self.iv[i * 2 + 1];
                    // Load both words into dst1 using LD
                    self.asm
                        .emit_ld::<LD>(dst1, self.operands.rs1, (i as i64) * 8);
                    // Extract the upper 32 bits into dst2 via a shift
                    self.asm.emit_i::<SRLI>(dst2, dst1, 32);
                    // Note that dst1 has junk in its upper 32 bits
                    // but that's fine since the lower 32 bits are not affected
                    // by the upper 32 bits in all subsequent operations in this inline
                    // except for the final store, where we are careful to clean them up
                });
            }
        }
        // Load input words into message registers
        if self.asm.xlen == Xlen::Bit32 {
            // if 32-bit, LW is cheap so use it
            (0..16).for_each(|i| {
                self.asm
                    .emit_ld::<LW>(*self.message[i], self.operands.rs2, (i * 4) as i64)
            });
        } else {
            // if 64-bit, LW expands into a long sequence of virtual instructions
            // so we use LD to load two registers at a time
            // the sequence is W0,1 | W2,3 | ... | W14,15
            (0..8).for_each(|i| {
                let dst1 = *self.message[i * 2];
                let dst2 = *self.message[i * 2 + 1];
                // Load both words into dst1 using LD
                self.asm
                    .emit_ld::<LD>(dst1, self.operands.rs2, (i as i64) * 8);
                // Extract the upper 32 bits into dst2 via a shift
                self.asm.emit_i::<SRLI>(dst2, dst1, 32);
                // Note that dst1 has junk in its upper 32 bits
                // but that's fine since the lower 32 bits are not affected
                // by the upper 32 bits in all subsequent operations in this inline
                // except for the final store, where we are careful to clean them up
            });
        }
        // Run 64 rounds
        (0..64).for_each(|_| self.round());
        self.final_add_iv();
        // Store output values to rs1 location
        // Store output A..H in-order using the current VR mapping after all rotations
        if self.asm.xlen == Xlen::Bit32 {
            // if 32-bit, SW is cheap so use it
            let outs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
            for (i, ch) in outs.iter().enumerate() {
                let src = self.vr(*ch);
                self.asm
                    .emit_s::<SW>(self.operands.rs1, src, (i as i64) * 4);
            }
        } else {
            // if 64-bit, SW is expands into a long sequence of virtual instructions
            // so we use SD to store two registers at a time
            // the sequence is A,B | C,D | E,F | G,H
            let outs = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'H')];
            for (i, (ch1, ch2)) in outs.iter().enumerate() {
                let src1 = self.vr(*ch1);
                let src2 = self.vr(*ch2);
                // clear top 32 bits of src1
                self.asm.emit_i::<VirtualZeroExtendWord>(src1, src1, 0);
                // shift src2 to top 32 bits
                self.asm.emit_i::<SLLI>(src2, src2, 32);
                // or them together into src1
                self.asm.emit_r::<OR>(src1, src2, src1);
                // and store pair as 64-bit word
                self.asm
                    .emit_s::<SD>(self.operands.rs1, src1, (i as i64) * 8);
            }
        }
        // Total allocated: 8 (state) + 16 (message) + 8 (initial_state) + 4 (temps per round) = 36
        // The temps are allocated/deallocated per round, but we need to reserve space for them
        drop(self.state);
        drop(self.message);
        drop(self.iv);
        self.asm.finalize_inline()
    }

    /// Adds IV to the final hash value to produce output
    fn final_add_iv(&mut self) {
        if !self.initial {
            // We have all initial values A-H stored in iv registers
            self.asm.add(self.vri('A'), Reg(*self.iv[0]), self.vr('A'));
            self.asm.add(self.vri('B'), Reg(*self.iv[1]), self.vr('B'));
            self.asm.add(self.vri('C'), Reg(*self.iv[2]), self.vr('C'));
            self.asm.add(self.vri('D'), Reg(*self.iv[3]), self.vr('D'));
            self.asm.add(self.vri('E'), Reg(*self.iv[4]), self.vr('E'));
            self.asm.add(self.vri('F'), Reg(*self.iv[5]), self.vr('F'));
            self.asm.add(self.vri('G'), Reg(*self.iv[6]), self.vr('G'));
            self.asm.add(self.vri('H'), Reg(*self.iv[7]), self.vr('H'));
        } else {
            // We are using constants for final addition round
            self.asm.add(self.vri('A'), Imm(BLOCK[0]), self.vr('A'));
            self.asm.add(self.vri('B'), Imm(BLOCK[1]), self.vr('B'));
            self.asm.add(self.vri('C'), Imm(BLOCK[2]), self.vr('C'));
            self.asm.add(self.vri('D'), Imm(BLOCK[3]), self.vr('D'));
            self.asm.add(self.vri('E'), Imm(BLOCK[4]), self.vr('E'));
            self.asm.add(self.vri('F'), Imm(BLOCK[5]), self.vr('F'));
            self.asm.add(self.vri('G'), Imm(BLOCK[6]), self.vr('G'));
            self.asm.add(self.vri('H'), Imm(BLOCK[7]), self.vr('H'));
        }
    }

    /// Performs one round of SHA256 compression
    fn round(&mut self) {
        assert!(self.round < 64);
        let t1 = self.asm.allocator.allocate_for_inline();
        let t2 = self.asm.allocator.allocate_for_inline();
        let ss = self.asm.allocator.allocate_for_inline();
        let ss2 = self.asm.allocator.allocate_for_inline();

        let t1_val = self.compute_t1(*t1, *ss, *ss2);
        let t2_val = self.compute_t2(*t2, *ss, *ss2);
        let old_d = self.vri('D');
        self.apply_round_update(t1_val, t2_val, old_d);
    }

    /// Compute T1 into the provided `t1` register and return it as a Value.
    fn compute_t1(&mut self, t1: u8, ss: u8, ss2: u8) -> Value {
        // Put H + K
        // We do this first because H is going to be Imm the longest of all inputs
        let h_add_k = self.asm.add(Imm(K[self.round as usize]), self.vri('H'), t1);
        // Put Sigma_1(E_0) into register t1
        let sigma_1 = self.sha_sigma_1(self.vri('E'), ss, ss2);
        let add_sigma_1 = self.asm.add(h_add_k, sigma_1, t1);
        // Put Ch(E_0, F_0, G_0) into register t2
        let ch = self.sha_ch(self.vri('E'), self.vri('F'), self.vri('G'), ss, ss2);
        let add_ch = self.asm.add(add_sigma_1, ch, t1);
        self.update_w([ss, ss2]);
        // Add W_(rid)
        self.asm.add(add_ch, Reg(self.w(0)), t1)
    }

    /// Compute T2 into the provided `t2` register and return it as a Value.
    fn compute_t2(&mut self, t2: u8, ss: u8, ss2: u8) -> Value {
        // Put Sigma_0(A_0) into register t2
        let sigma_0 = self.sha_sigma_0(self.vri('A'), t2, ss);
        // Put Maj(A_0, B_0, C_0) into register ss
        let maj = self.sha_maj(self.vri('A'), self.vri('B'), self.vri('C'), ss, ss2);
        // Add Maj to t2
        self.asm.add(sigma_0, maj, t2)
    }

    /// Apply A/E updates for the current round using computed T1/T2 and then advance the round.
    fn apply_round_update(&mut self, t1: Value, t2: Value, old_d: Value) {
        self.round += 1;
        // After incrementing round, the rotation has happened
        // So vr('A') now points to the right place to write the new A
        self.asm.add(t1, t2, self.vr('A'));
        // Overwrite D_0 with D_0 + T_1
        self.asm.add(t1, old_d, self.vr('E'));
    }

    /// Returns either Register or Immediate input for a working variable (A-H)
    /// When initial is true, uses BLOCK constants for the first few rounds
    /// until all values have been computed
    fn vri(&self, shift: char) -> Value {
        // For initial rounds without custom IV, some values haven't been computed yet
        // Round 0: Only A,E are computed (from initial values)
        // Round 1: A,B,E,F are available
        // Round 2: A,B,C,E,F,G are available
        // Round 3+: All values are in registers
        if self.initial
            && (self.round == 0
                || (self.round == 1 && !['A', 'E'].contains(&shift))
                || (self.round == 2 && !['A', 'B', 'E', 'F'].contains(&shift))
                || (self.round == 3 && !['A', 'B', 'C', 'E', 'F', 'G'].contains(&shift)))
        {
            // Our values are getting shifted each round, so we subtract round_id
            // for example in round 1 we have B equal to A from round 0.
            let shift = shift as i32 - 'A' as i32;
            return Imm(BLOCK[(shift - self.round).rem_euclid(8) as usize]);
        }
        Reg(self.vr(shift))
    }

    /// Maps working variable (A-H) to its current register location
    /// Variables rotate through state registers as rounds progress
    /// For custom IV (!initial), values start in iv and gradually move into rotation
    fn vr(&self, shift: char) -> u8 {
        assert!(('A'..='H').contains(&shift));
        // For custom IV: check if this value hasn't been computed yet
        // In each round, we compute new A and new E. After rotation:
        // Round 0: None computed yet, use saved for all
        // Round 1: A,E computed (now at H,D positions), use saved for B,C,D,F,G,H
        // Round 2: A,B,E,F computed (now at G,H,C,D positions), use saved for C,D,G,H
        // Round 3: A,B,C,E,F,G computed (now at F,G,H,B,C,D positions), use saved for D,H
        // Round 4+: All have been computed, use rotation only
        if !self.initial
            && (self.round == 0
                || (self.round == 1 && !['A', 'E'].contains(&shift))
                || (self.round == 2 && !['A', 'B', 'E', 'F'].contains(&shift))
                || (self.round == 3 && !['A', 'B', 'C', 'E', 'F', 'G'].contains(&shift)))
        {
            return *self.iv[(shift as i32 - 'A' as i32 - self.round).rem_euclid(8) as usize];
        }
        let shift = shift as i32 - 'A' as i32;

        // Standard rotation: each round shifts all variables by -1
        *self.state[(-self.round + shift).rem_euclid(8) as usize]
    }

    /// Register number containing W_(rid+shift)
    fn w(&self, shift: i32) -> u8 {
        *self.message[((self.round + shift).rem_euclid(16)) as usize]
    }

    /// Updates message schedule for rounds 16-63
    /// W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]
    fn update_w(&mut self, ss: [u8; 2]) {
        if self.round < 16 {
            return;
        }
        // Calculate σ₀(W[t-15])
        self.sha_word_sigma_0(self.w(-15), ss[0], ss[1]);
        // Add σ₀ to W[t-16]
        self.asm.add(Reg(self.w(-16)), Reg(ss[0]), self.w(-16));
        // Add W[t-7] to W[t-16]
        self.asm.add(Reg(self.w(-7)), Reg(self.w(-16)), self.w(-16));
        // Calculate σ₁(W[t-2])
        self.sha_word_sigma_1(self.w(-2), ss[0], ss[1]);
        // Add σ₁ to W[t-16] to get final W[t]
        self.asm.add(Reg(self.w(-16)), Reg(ss[0]), self.w(-16));
    }

    /// Computes sha256 Ch function
    /// Ch(E, F, G) = (E and F) xor ((not E) and G)
    fn sha_ch(&mut self, rs1: Value, rs2: Value, rs3: Value, rd: u8, ss: u8) -> Value {
        let e_and_f = self.asm.and(rs1, rs2, ss);
        // Use ANDN to compute (not E) and G in one instruction
        // ANDN computes rs1 & !rs2, so andn(G, E) gives G & !E = !E & G
        match (rs1, rs3) {
            (Reg(r1), Reg(r3)) => {
                self.asm.emit_r::<ANDN>(rd, r3, r1);
                let neg_e_and_g = Reg(rd);
                self.asm.xor(e_and_f, neg_e_and_g, rd)
            }
            _ => {
                // Fallback for immediate values (used in first few rounds)
                let neg_e = self.asm.xor(rs1, Imm(u32::MAX as u64), rd);
                let neg_e_and_g = self.asm.and(neg_e, rs3, rd);
                self.asm.xor(e_and_f, neg_e_and_g, rd)
            }
        }
    }

    /// Computes sha256 Maj function: Maj(A, B, C) = (A and B) xor (A and C) xor (B and C)
    fn sha_maj(&mut self, rs1: Value, rs2: Value, rs3: Value, rd: u8, ss: u8) -> Value {
        let b_and_c = self.asm.and(rs2, rs3, ss);
        let b_xor_c = self.asm.xor(rs2, rs3, rd);
        let a_and_b_xor_c = self.asm.and(rs1, b_xor_c, rd);
        self.asm.xor(b_and_c, a_and_b_xor_c, rd)
    }

    /// Sigma_0 function of SHA256 compression function: Σ₀(x) = ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)
    fn sha_sigma_0(&mut self, rs1: Value, rd: u8, ss: u8) -> Value {
        let rotri_xor = self.asm.rotri_xor_rotri32(rs1, 2, 13, rd, ss);
        let rotri_22 = self.asm.rotri32(rs1, 22, ss);
        self.asm.xor(rotri_xor, rotri_22, rd)
    }

    /// Sigma_1 function of SHA256 compression function: Σ₁(x) = ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)
    fn sha_sigma_1(&mut self, rs1: Value, rd: u8, ss: u8) -> Value {
        let rotri_xor = self.asm.rotri_xor_rotri32(rs1, 6, 11, rd, ss);
        let rotri_25 = self.asm.rotri32(rs1, 25, ss);
        self.asm.xor(rotri_xor, rotri_25, rd)
    }

    /// sigma_0 for word computation: σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
    fn sha_word_sigma_0(&mut self, rs1: u8, rd: u8, ss: u8) {
        self.asm.rotri_xor_rotri32(Reg(rs1), 7, 18, rd, ss);
        self.asm.srli(Reg(rs1), 3, ss);
        self.asm.xor(Reg(rd), Reg(ss), rd);
    }

    /// sigma_1 for word computation: σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
    fn sha_word_sigma_1(&mut self, rs1: u8, rd: u8, ss: u8) {
        // We don't need to do Imm shenanigans here since words are always in registers
        self.asm.rotri_xor_rotri32(Reg(rs1), 17, 19, rd, ss);
        self.asm.srli(Reg(rs1), 10, ss);
        self.asm.xor(Reg(rd), Reg(ss), rd);
    }
}

// Virtual instructions builder for sha256
pub fn sha2_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Sha256SequenceBuilder::new(
        asm, operands, false, // not initial - uses custom IV from rs1
    );
    builder.build()
}

// Virtual instructions builder for sha256_init
pub fn sha2_init_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Sha256SequenceBuilder::new(
        asm, operands, true, // initial - uses BLOCK constants
    );
    builder.build()
}
