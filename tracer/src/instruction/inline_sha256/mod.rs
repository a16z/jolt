use crate::inline_helpers::{
    InstrAssembler, Value,
    Value::{Imm, Reg},
};

use crate::instruction::andn::ANDN;
use crate::instruction::lw::LW;
use crate::instruction::sw::SW;
use crate::instruction::RV32IMInstruction;

pub mod sha256;
pub mod sha256init;

#[cfg(test)]
mod test_constants;
#[cfg(test)]
mod test_utils;

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

/// Number of virtual registers needed for SHA256 computation
/// Layout:
/// - 0..7:   Working variables A-H (rotated during rounds)
/// - 8..23:  Message schedule W[0..15]
/// - 24..27: Temporary registers (t1, t2, scratch space)
/// - 28..31: Initial E-H values when using custom IV
pub const NEEDED_REGISTERS: usize = 32;

/// Builds assembly sequence for SHA256 compression
/// Expects input words to be in RAM at location rs1..rs1+16
/// Expects A..H to be in RAM at location rs2..rs2+8
/// Output will be written to rs2..rs2+8
struct Sha256SequenceBuilder {
    asm: InstrAssembler,
    /// Round id
    round: i32,
    /// Virtual registers used by the sequence
    vr: [usize; NEEDED_REGISTERS],
    /// Location input words to the hash function in 16 memory slots
    operand_rs1: usize,
    /// Location of previous hash values A..H (also where output is written)
    operand_rs2: usize,
    /// Whether this is the initial compression (use BLOCK constants)
    initial: bool,
}

impl Sha256SequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        vr: [usize; NEEDED_REGISTERS],
        operand_rs1: usize,
        operand_rs2: usize,
        initial: bool,
    ) -> Self {
        Sha256SequenceBuilder {
            asm: InstrAssembler::new(address, is_compressed),
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
            initial,
        }
    }

    /// Loads and runs all SHA256 rounds
    fn build(mut self) -> Vec<RV32IMInstruction> {
        if !self.initial {
            // Load initial hash values from memory when using custom IV
            // A..D loaded into registers 0..3 (will be used immediately)
            // E..H loaded into registers 28..31 (preserved until needed)
            (0..4).for_each(|i| {
                self.asm
                    .emit_ld::<LW>(self.vr[i as usize], self.operand_rs2, i * 4)
            });
            (0..4).for_each(|i| {
                self.asm
                    .emit_ld::<LW>(self.vr[(i + 28) as usize], self.operand_rs2, (i + 4) * 4)
            });
        }
        // Load input words into registers 8..23
        (0..16).for_each(|i| {
            self.asm
                .emit_ld::<LW>(self.vr[(i + 8) as usize], self.operand_rs1, i * 4)
        });
        // Run 64 rounds
        (0..64).for_each(|_| self.round());
        self.final_add_iv();
        // Store output values to rs2 location
        (0..8).for_each(|i| {
            self.asm
                .emit_s::<SW>(self.operand_rs2, self.vr[i as usize], i * 4)
        });
        self.asm.finalize()
    }

    /// Adds IV to the final hash value to produce output
    fn final_add_iv(&mut self) {
        if !self.initial {
            // We have initial values E, F, G, H stored in the end registers, but we didn't have
            // enough space for A, B, C, D, so we need to load them from memory. We can load them
            // into space that was used for t1, t2, ss, ss2. (technically there's no preference,
            // but it just keeps those in order).
            (0..4).for_each(|i| {
                self.asm
                    .emit_ld::<LW>(self.vr[24 + i as usize], self.operand_rs2, i * 4)
            });
            self.asm.add(self.vri('A'), Reg(self.vr[24]), self.vr('A'));
            self.asm.add(self.vri('B'), Reg(self.vr[25]), self.vr('B'));
            self.asm.add(self.vri('C'), Reg(self.vr[26]), self.vr('C'));
            self.asm.add(self.vri('D'), Reg(self.vr[27]), self.vr('D'));
            self.asm.add(self.vri('E'), Reg(self.vr[28]), self.vr('E'));
            self.asm.add(self.vri('F'), Reg(self.vr[29]), self.vr('F'));
            self.asm.add(self.vri('G'), Reg(self.vr[30]), self.vr('G'));
            self.asm.add(self.vri('H'), Reg(self.vr[31]), self.vr('H'));
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

    /// Assumes for words A-H to be loaded in registers 0..7
    /// Assumes for words W_0..W_15 to be loaded in registers 8..24
    fn round(&mut self) {
        assert!(self.round < 64);
        let t1 = self.vr[24];
        let t2 = self.vr[25];
        // scratch space
        let ss = self.vr[26];
        let ss2 = self.vr[27];
        // Put T_1 into register t1
        // Put H + K
        // We do this first because H is going to be Imm the longest of all inputs
        let h_add_k = self.asm.add(Imm(K[self.round as usize]), self.vri('H'), t1);
        let sigma_1 = self.sha_sigma_1(self.vri('E'), ss, ss2);
        let add_sigma_1 = self.asm.add(h_add_k, sigma_1, t1);
        // Put Ch(E_0, F_0, G_0) into register t2
        let ch = self.sha_ch(self.vri('E'), self.vri('F'), self.vri('G'), ss, ss2);
        let add_ch = self.asm.add(add_sigma_1, ch, t1);
        self.update_w([ss, ss2]);
        // Add W_(rid)
        let t1 = self.asm.add(add_ch, Reg(self.w(0)), t1);
        // Done with T_1

        // Put T_2 into register t2
        // Put Sigma_0(A_0) into register t2
        let sigma_0 = self.sha_sigma_0(self.vri('A'), t2, ss);
        // Put Maj(A_0, B_0, C_0) into register ss
        let maj = self.sha_maj(self.vri('A'), self.vri('B'), self.vri('C'), ss, ss2);
        // Add Maj to t2
        let t2 = self.asm.add(sigma_0, maj, t2);
        // Done with T_2

        let old_d = self.vri('D');
        self.round += 1;
        // Overwrite new A with T_1 + T_2
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
    /// Variables rotate through registers 0-7 as rounds progress
    /// When not initial, E-H start in registers 28-31
    fn vr(&self, shift: char) -> usize {
        assert!(('A'..='H').contains(&shift));
        let shift = shift as i32 - 'A' as i32;

        // Special handling for custom IV: E-H values start in registers 28-31
        // and gradually move into the main rotation (registers 0-7)
        if !self.initial
            && (self.round == 0 && shift >= 4
                || self.round == 1 && shift >= 5
                || self.round == 2 && shift >= 6
                || self.round == 3 && shift >= 7)
        {
            return self.vr[24 - self.round as usize + shift as usize];
        }

        // Standard rotation: each round shifts all variables by -1
        self.vr[(-self.round + shift).rem_euclid(8) as usize]
    }

    /// Register number containing W_(rid+shift)
    fn w(&self, shift: i32) -> usize {
        self.vr[((self.round + shift).rem_euclid(16) + 8) as usize]
    }

    /// Updates message schedule for rounds 16-63
    /// W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]
    fn update_w(&mut self, ss: [usize; 2]) {
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
    fn sha_ch(&mut self, rs1: Value, rs2: Value, rs3: Value, rd: usize, ss: usize) -> Value {
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
    fn sha_maj(&mut self, rs1: Value, rs2: Value, rs3: Value, rd: usize, ss: usize) -> Value {
        let b_and_c = self.asm.and(rs2, rs3, ss);
        let b_xor_c = self.asm.xor(rs2, rs3, rd);
        let a_and_b_xor_c = self.asm.and(rs1, b_xor_c, rd);
        self.asm.xor(b_and_c, a_and_b_xor_c, rd)
    }

    /// Sigma_0 function of SHA256 compression function: Σ₀(x) = ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)
    fn sha_sigma_0(&mut self, rs1: Value, rd: usize, ss: usize) -> Value {
        let rotri_xor = self.asm.rotri_xor_rotri32(rs1, 2, 13, rd, ss);
        let rotri_22 = self.asm.rotri32(rs1, 22, ss);
        self.asm.xor(rotri_xor, rotri_22, rd)
    }

    /// Sigma_1 function of SHA256 compression function: Σ₁(x) = ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)
    fn sha_sigma_1(&mut self, rs1: Value, rd: usize, ss: usize) -> Value {
        let rotri_xor = self.asm.rotri_xor_rotri32(rs1, 6, 11, rd, ss);
        let rotri_25 = self.asm.rotri32(rs1, 25, ss);
        self.asm.xor(rotri_xor, rotri_25, rd)
    }

    /// sigma_0 for word computation: σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
    fn sha_word_sigma_0(&mut self, rs1: usize, rd: usize, ss: usize) {
        self.asm.rotri_xor_rotri32(Reg(rs1), 7, 18, rd, ss);
        self.asm.srli(Reg(rs1), 3, ss);
        self.asm.xor(Reg(rd), Reg(ss), rd);
    }

    /// sigma_1 for word computation: σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
    fn sha_word_sigma_1(&mut self, rs1: usize, rd: usize, ss: usize) {
        // We don't need to do Imm shenanigans here since words are always in registers
        self.asm.rotri_xor_rotri32(Reg(rs1), 17, 19, rd, ss);
        self.asm.srli(Reg(rs1), 10, ss);
        self.asm.xor(Reg(rd), Reg(ss), rd);
    }
}

pub fn execute_sha256_compression_initial(input: [u32; 16]) -> [u32; 8] {
    execute_sha256_compression(BLOCK.map(|x| x as u32), input)
}

pub fn execute_sha256_compression(initial_state: [u32; 8], input: [u32; 16]) -> [u32; 8] {
    let mut a = initial_state[0];
    let mut b = initial_state[1];
    let mut c = initial_state[2];
    let mut d = initial_state[3];
    let mut e = initial_state[4];
    let mut f = initial_state[5];
    let mut g = initial_state[6];
    let mut h = initial_state[7];

    let mut w = [0u32; 64];

    w[..16].copy_from_slice(&input);

    // Calculate word schedule
    for i in 16..64 {
        // σ₁(w[i-2]) + w[i-7] + σ₀(w[i-15]) + w[i-16]
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    // Perform 64 rounds
    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);

        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22); // Σ₀(a)
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25); // Σ₁(e)

        let t1 = h
            .wrapping_add(sigma1)
            .wrapping_add(ch)
            .wrapping_add(K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = sigma0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    // Final IV addition
    [
        initial_state[0].wrapping_add(a),
        initial_state[1].wrapping_add(b),
        initial_state[2].wrapping_add(c),
        initial_state[3].wrapping_add(d),
        initial_state[4].wrapping_add(e),
        initial_state[5].wrapping_add(f),
        initial_state[6].wrapping_add(g),
        initial_state[7].wrapping_add(h),
    ]
}
