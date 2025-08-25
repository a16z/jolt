use tracer::{
    emulator::cpu::Xlen,
    instruction::{
        add::ADD, ld::LD, mul::MUL, mulhu::MULHU, sd::SD, sltu::SLTU, RV32IMInstruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::allocate_virtual_register},
};

use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Number of virtual registers needed for BigInt multiplication
/// Layout:
/// - a0..a3: First operand (4 u64 limbs)
/// - a4..a7: Second operand (4 u64 limbs)
/// - s0..s7: Result accumulator (8 u64 limbs)
/// - t0..t3: Temporary registers for multiplication and carry
pub(crate) const NEEDED_REGISTERS: usize = 20;

/// Builds assembly sequence for 256-bit × 256-bit multiplication
/// Expects first operand (4 u64 words) in RAM at location rs1
/// Expects second operand (4 u64 words) in RAM at location rs2
/// Output (8 u64 words) will be written to rd
struct BigIntMulSequenceBuilder {
    asm: InstrAssembler,
    /// Virtual registers used by the sequence
    vr: [u8; NEEDED_REGISTERS],
    /// Location of first operand in memory
    operand_rs1: u8,
    /// Location of second operand in memory
    operand_rs2: u8,
    /// Location of result in memory
    operand_rd: u8,
}

impl BigIntMulSequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        vr: [u8; NEEDED_REGISTERS],
        operand_rs1: u8,
        operand_rs2: u8,
        operand_rd: u8,
    ) -> Self {
        BigIntMulSequenceBuilder {
            asm: InstrAssembler::new(address, is_compressed, xlen),
            vr,
            operand_rs1,
            operand_rs2,
            operand_rd,
        }
    }

    /// Register indices for operands and temporaries
    // LHS
    fn a(&self, i: usize) -> u8 {
        self.vr[i]
    }
    // RHS
    fn b(&self, i: usize) -> u8 {
        self.vr[INPUT_LIMBS + i]
    }
    // Results
    fn s(&self, i: usize) -> u8 {
        self.vr[INPUT_LIMBS + INPUT_LIMBS + i]
    }
    // Temporaries
    fn t(&self, i: usize) -> u8 {
        self.vr[INPUT_LIMBS + INPUT_LIMBS + OUTPUT_LIMBS + i]
    }

    /// Builds the complete multiplication sequence
    fn build(mut self) -> Vec<RV32IMInstruction> {
        for i in 0..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.a(i), self.operand_rs1, i as i64 * 8);
        }

        for i in 0..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.b(i), self.operand_rs2, i as i64 * 8);
        }

        // Initialize result accumulator registers to zero
        for i in 0..OUTPUT_LIMBS {
            self.asm.emit_r::<ADD>(self.s(i), 0, 0); // s[i] = 0 + 0
        }

        for i in 0..INPUT_LIMBS {
            for j in 0..INPUT_LIMBS {
                self.mul_and_accumulate(i, j); // A[i] × B[j] → R[i+j]
            }
        }

        // Store result (8 u64 words) back to rd
        for i in 0..OUTPUT_LIMBS {
            self.asm
                .emit_s::<SD>(self.operand_rd, self.s(i), i as i64 * 8);
        }

        self.asm.finalize()
    }

    /// Implements the MUL-ACC pattern: A[i] × B[j] → R[k] where k = i+j
    /// This multiplies A[i] by B[j] and accumulates the 128-bit result
    /// into R[k], R[k+1], R[k+2] with carry propagation
    fn mul_and_accumulate(&mut self, i: usize, j: usize) {
        let k = i + j;

        // Get register indices
        let ai = self.a(i);
        let bj = self.b(j);
        let sk = self.s(k);
        let t0 = self.t(0);
        let t1 = self.t(1);
        let t2 = self.t(2);

        // mulhu t1, ai, bj     # High 64 bits of product (do this first)
        self.asm.emit_r::<MULHU>(t1, ai, bj);

        // mul t0, ai, bj       # Low 64 bits of product
        self.asm.emit_r::<MUL>(t0, ai, bj);

        // add sk, sk, t0       # Add low bits to R[k]
        self.asm.emit_r::<ADD>(sk, sk, t0);

        let sk1 = self.s(k + 1);

        // No overflow at this case
        if k == 0 {
            // add sk1, sk1, t1     # Add high to R[k+1]
            self.asm.emit_r::<ADD>(sk1, sk1, t1);
            return;
        }

        // sltu t2, sk, t0      # Check for carry (sk < t0 means overflow)
        self.asm.emit_r::<SLTU>(t2, sk, t0);

        // add t1, t1, t2       # Add carry from low part to high part
        self.asm.emit_r::<ADD>(t1, t1, t2);

        // add sk1, sk1, t1     # Add (high + carry) to R[k+1]
        self.asm.emit_r::<ADD>(sk1, sk1, t1);

        // Propagate carry through higher limbs if needed
        if k + 2 < OUTPUT_LIMBS {
            // sltu t2, sk1, t1     # Check for carry from adding high part into R[k+1]
            self.asm.emit_r::<SLTU>(t2, sk1, t1);

            // Ripple-carry add into R[k+2..]
            for m in (k + 2)..OUTPUT_LIMBS {
                let sm = self.s(m);
                // add s[m], s[m], t2
                self.asm.emit_r::<ADD>(sm, sm, t2);
                // sltu t2, s[m], t2  # carry out
                if m + 1 < OUTPUT_LIMBS {
                    self.asm.emit_r::<SLTU>(t2, sm, t2);
                }
            }
        }
    }
}

/// Virtual instructions builder for bigint multiplication
pub fn bigint_mul_sequence_builder(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    rs1: u8,
    rs2: u8,
    rd: u8,
) -> Vec<RV32IMInstruction> {
    let guards: Vec<_> = (0..NEEDED_REGISTERS)
        .map(|_| allocate_virtual_register())
        .collect();
    let mut vr = [0u8; NEEDED_REGISTERS];
    for (i, guard) in guards.iter().enumerate() {
        vr[i] = **guard;
    }

    let builder = BigIntMulSequenceBuilder::new(address, is_compressed, xlen, vr, rs1, rs2, rd);
    builder.build()
}
