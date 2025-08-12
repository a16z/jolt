use tracer::{
    emulator::cpu::Xlen,
    inline_helpers::{virtual_register_index, InstrAssembler},
    instruction::{
        add::ADD,
        ld::LD,
        mul::MUL,
        mulhu::MULHU,
        sd::SD,
        sltu::SLTU,
        RV32IMInstruction,
    },
};

/// Number of virtual registers needed for BigInt multiplication
/// Layout:
/// - a0..a3: First operand (4 u64 limbs)
/// - a4..a7: Second operand (4 u64 limbs)
/// - s0..s7: Result accumulator (8 u64 limbs)
/// - t0..t3: Temporary registers for multiplication and carry
pub const NEEDED_REGISTERS: u8 = 20;

/// Builds assembly sequence for 256-bit × 256-bit multiplication
/// Expects first operand (4 u64 words) in RAM at location rs1
/// Expects second operand (4 u64 words) in RAM at location rs2
/// Output (8 u64 words) will be written to rd
struct BigIntMulSequenceBuilder {
    asm: InstrAssembler,
    /// Virtual registers used by the sequence
    vr: [u8; NEEDED_REGISTERS as usize],
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
        vr: [u8; NEEDED_REGISTERS as usize],
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
    fn a(&self, i: usize) -> u8 { self.vr[i] }           // a0-a3
    fn b(&self, i: usize) -> u8 { self.vr[4 + i] }       // a4-a7 (mapped to registers 4-7)
    fn s(&self, i: usize) -> u8 { self.vr[8 + i] }       // s0-s7 (result accumulator)
    fn t(&self, i: usize) -> u8 { self.vr[16 + i] }      // t0-t3 (temporaries)

    /// Builds the complete multiplication sequence
    fn build(mut self) -> Vec<RV32IMInstruction> {
        // Load first operand (A) into a0-a3
        for i in 0..4 {
            self.asm.emit_ld::<LD>(self.a(i), self.operand_rs1, i as i64 * 8);
        }

        // Load second operand (B) into a4-a7  
        for i in 0..4 {
            self.asm.emit_ld::<LD>(self.b(i), self.operand_rs2, i as i64 * 8);
        }

        // // Initialize result registers s0-s7 to zero
        // // We'll accumulate the partial products into these
        // for i in 0..8 {
        //     self.asm.emit_r::<ADD>(self.s(i), 0, 0); // ADD s[i], x0, x0 (zero register)
        // }

        // Perform schoolbook multiplication
        // For each A[i] × B[j], accumulate the 128-bit product into R[i+j]
        
        // Row 0: A0 × B*
        self.mul_and_accumulate(0, 0); // A0 × B0 → R0
        self.mul_and_accumulate(0, 1); // A0 × B1 → R1
        self.mul_and_accumulate(0, 2); // A0 × B2 → R2
        self.mul_and_accumulate(0, 3); // A0 × B3 → R3

        // Row 1: A1 × B*
        self.mul_and_accumulate(1, 0); // A1 × B0 → R1
        self.mul_and_accumulate(1, 1); // A1 × B1 → R2
        self.mul_and_accumulate(1, 2); // A1 × B2 → R3
        self.mul_and_accumulate(1, 3); // A1 × B3 → R4

        // Row 2: A2 × B*
        self.mul_and_accumulate(2, 0); // A2 × B0 → R2
        self.mul_and_accumulate(2, 1); // A2 × B1 → R3
        self.mul_and_accumulate(2, 2); // A2 × B2 → R4
        self.mul_and_accumulate(2, 3); // A2 × B3 → R5

        // Row 3: A3 × B*
        self.mul_and_accumulate(3, 0); // A3 × B0 → R3
        self.mul_and_accumulate(3, 1); // A3 × B1 → R4
        self.mul_and_accumulate(3, 2); // A3 × B2 → R5
        self.mul_and_accumulate(3, 3); // A3 × B3 → R6

        // Store result (8 u64 words) back to rd
        for i in 0..8 {
            self.asm.emit_s::<SD>(self.operand_rd, self.s(i), i as i64 * 8);
        }

        self.asm.finalize()
    }

    /// Implements the MUL-ACC pattern: A[i] × B[j] → R[k] where k = i+j
    /// This multiplies A[i] by B[j] and accumulates the 128-bit result
    /// into R[k], R[k+1], R[k+2] with carry propagation
    /// Optimized to use fewer instructions similar to arkworks implementation
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

        // Only propagate to k+2 if it exists
        if k + 2 < 8 {
            // sltu t2, sk1, t1     # Check for final carry
            self.asm.emit_r::<SLTU>(t2, sk1, t1);
            let sk2 = self.s(k + 2);
            
            // add sk2, sk2, t2     # Add final carry to R[k+2]
            self.asm.emit_r::<ADD>(sk2, sk2, t2);
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
    rd: u8
) -> Vec<RV32IMInstruction> {
    // Allocate virtual registers
    let mut vr = [0u8; NEEDED_REGISTERS as usize];
    for i in 0..NEEDED_REGISTERS as usize {
        vr[i] = virtual_register_index(i as u8);
    }

    let builder = BigIntMulSequenceBuilder::new(
        address,
        is_compressed,
        xlen,
        vr,
        rs1,
        rs2,
        rd,
    );
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracer::{
        emulator::{
            cpu::Cpu,
            default_terminal::DefaultTerminal,
        },
        instruction::{
            format::format_r::FormatR,
            inline::INLINE,
        },
        register_inline,
    };
    use tracer::instruction::RISCVTrace;
    pub const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    #[test]
    fn test_bigint_mul_trace() {
        // Try to register the bigint multiplication inline (it may already be registered)
        let _ = register_inline(
            0x0B,
            0x00,
            0x01,
            "BIGINT256_MUL",
            Box::new(crate::exec::bigint_mul_exec),
            Box::new(bigint_mul_sequence_builder),
        );

        // Create a CPU with memory
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Test with the example values from the Python script
        let a_lo: u128 = 0xf3a8_9b7c_4d2e_1f0a_8b6c_5d3e_2f1a_0b9c;
        let a_hi: u128 = 0x9c8b_7a6f_5e4d_3c2b_1a09_8776_5544_3322;
        let b_lo: u128 = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210;
        let b_hi: u128 = 0xa5b6_c7d8_e9fa_0b1c_2d3e_4f50_6172_8394;

        // Convert to u64 limbs (little-endian)
        let a_limbs = [
            (a_lo & 0xFFFFFFFFFFFFFFFF) as u64,
            ((a_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
            (a_hi & 0xFFFFFFFFFFFFFFFF) as u64,
            ((a_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
        ];

        let b_limbs = [
            (b_lo & 0xFFFFFFFFFFFFFFFF) as u64,
            ((b_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
            (b_hi & 0xFFFFFFFFFFFFFFFF) as u64,
            ((b_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64,
        ];

        // Set up memory locations (in valid memory range)
        let rs1_addr = 0x80002000;  // Start of general memory
        let rs2_addr = 0x80002100;  // Offset by 256 bytes
        let rd_addr = 0x80002200;   // Destination for result

        // Store operand A at rs1_addr
        for i in 0..4 {
            cpu.mmu
                .store_doubleword(rs1_addr + (i * 8) as u64, a_limbs[i])
                .expect("Failed to store operand A");
        }

        // Store operand B at rs2_addr
        for i in 0..4 {
            cpu.mmu
                .store_doubleword(rs2_addr + (i * 8) as u64, b_limbs[i])
                .expect("Failed to store operand B");
        }

        // Set register values to point to memory locations
        cpu.x[10] = rs1_addr as i64; // rs1 = x10
        cpu.x[11] = rs2_addr as i64; // rs2 = x11
        cpu.x[12] = rd_addr as i64;  // rd = x12 (destination for result)

        // Create INLINE instruction
        let instr = INLINE {
            opcode: 0x0B,
            funct3: 0x00,
            funct7: 0x01,
            address: 0x0,
            operands: FormatR {
                rs1: 10,
                rs2: 11,
                rd: 12,  // Used for result destination
            },
            inline_sequence_remaining: None,
            is_compressed: false,
        };

        instr.trace(&mut cpu, None);

        // Expected result from Python computation
        let expected = [
            0xc90fab9bbf1531c0_u64,
            0xaad973dd55fab9b8_u64,
            0xbca2ca1b10cfc4cf_u64,
            0xbb1ee1c3c94e6a79_u64,
            0xa6ae39a57f3091a7_u64,
            0x2dae6201791d3cf5_u64,
            0x50221be9fec14f26_u64,
            0x6555ab47e3e48c66_u64,
        ];

        // Read result from memory at rd_addr (where result was stored)
        let mut result = [0u64; 8];
        for i in 0..8 {
            result[i] = cpu.mmu
                .load_doubleword(rd_addr + (i * 8) as u64)
                .expect("Failed to load result")
                .0;
        }

        // Verify the result
        assert_eq!(result, expected, "BigInt multiplication result mismatch");
    }

    #[test]
    fn test_instruction_count_optimization() {
        // Test to verify the optimization reduces instruction count
        let sequence = bigint_mul_sequence_builder(
            0x0,     // address
            false,   // not compressed
            Xlen::Bit64,
            10,      // rs1
            11,      // rs2
            12,
        );
        
        // Count the total instructions generated
        let total_instructions = sequence.len();
        
        // Expected breakdown:
        // - 4 LD instructions to load operand A
        // - 4 LD instructions to load operand B  
        // - 8 ADD instructions to initialize result registers to zero
        // - 16 multiplication operations, each using:
        //   * MULHU + MUL + ADD + SLTU = 4 base instructions
        //   * ADD + ADD + SLTU = 3 for carry propagation to k+1
        //   * ADD = 1 for carry to k+2 (sometimes)
        //   * Average ~7-8 instructions per multiplication
        // - 8 SD instructions to store the result
        
        println!("Total instructions generated: {}", total_instructions);
        
        // With optimization, we expect around:
        // 4 + 4 + 8 + (16 * 7) + 8 = 136 instructions (approximately)
        // Without optimization it would be: 4 + 4 + 8 + (16 * 10) + 8 = 184 instructions
        
        // Verify we're in the optimized range (should be significantly less than 184)
        assert!(total_instructions < 160, 
            "Instruction count {} is too high, optimization may not be working", 
            total_instructions);
        
        println!("✓ Optimization successful: {} instructions (down from ~184)", total_instructions);
    }
}