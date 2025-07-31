use tracer::emulator::cpu::Cpu;
use tracer::instruction::precompile::PRECOMPILE;
use tracer::instruction::RV32IMInstruction;
use tracer::{register_precompile, list_registered_precompiles};

// Example 1: Simple XOR precompile
fn xor_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
    let result = cpu.x[instr.operands.rs1] ^ cpu.x[instr.operands.rs2];
    cpu.x[instr.operands.rd] = result;
    println!("XOR precompile: x{} = x{} ^ x{} = {:#x} (at {:#x})", 
             instr.operands.rd, instr.operands.rs1, instr.operands.rs2, result, instr.address);
}

// Builder for XOR - returns empty sequence as XOR is atomic
fn xor_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // XOR doesn't need a virtual sequence, it's a single atomic operation
    Vec::new()
}

// Example 2: Bit rotation precompile
fn rotate_left_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
    let value = cpu.x[instr.operands.rs1] as u64;
    let shift = (cpu.x[instr.operands.rs2] & 0x3F) as u32; // Use lower 6 bits for shift amount
    let result = value.rotate_left(shift);
    cpu.x[instr.operands.rd] = result as i64;
    println!("ROTL precompile: x{} = rotl(x{}, {}) = {:#x} (at {:#x})", 
             instr.operands.rd, instr.operands.rs1, shift, result, instr.address);
}

// Builder for ROTL - could decompose into shifts and ORs if needed
fn rotl_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // For simplicity, return empty sequence
    // In a real implementation, you might decompose this into:
    // - SLL for left shift
    // - SRL for right shift of (64-shift) positions  
    // - OR to combine results
    Vec::new()
}

// Example 3: Population count (popcount) precompile
fn popcount_precompile(instr: &PRECOMPILE, cpu: &mut Cpu, _ram: &mut ()) {
    let value = cpu.x[instr.operands.rs1] as u64;
    let count = value.count_ones() as i64;
    cpu.x[instr.operands.rd] = count;
    println!("POPCOUNT precompile: x{} = popcount(x{}) = {} (at {:#x})", 
             instr.operands.rd, instr.operands.rs1, count, instr.address);
}

// Builder for POPCOUNT - could decompose into bit manipulation if needed
fn popcount_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // For simplicity, return empty sequence
    // In a real implementation, you might generate a sequence of:
    // - Shift and AND operations to isolate bits
    // - ADD operations to sum them up
    Vec::new()
}

// Initialize and register precompiles
fn init_precompiles() -> Result<(), String> {
    // Register XOR with funct7=0x01
    register_precompile(0x01, "XOR_PRECOMPILE", 
        Box::new(xor_precompile), 
        Box::new(xor_builder))?;
    
    // Register ROTL with funct7=0x02
    register_precompile(0x02, "ROTL_PRECOMPILE", 
        Box::new(rotate_left_precompile),
        Box::new(rotl_builder))?;
    
    // Register POPCOUNT with funct7=0x03
    register_precompile(0x03, "POPCOUNT_PRECOMPILE", 
        Box::new(popcount_precompile),
        Box::new(popcount_builder))?;
    
    Ok(())
}

// Optional: Use ctor for automatic registration when the library loads
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_precompiles() {
        eprintln!("Failed to register precompiles: {}", e);
    }
}

fn main() {
    // Manual registration (if not using ctor)
    // init_precompiles().expect("Failed to register precompiles");
    
    // List all registered precompiles
    println!("Registered precompiles:");
    for (funct7, name) in list_registered_precompiles() {
        println!("  funct7={:#04x}: {}", funct7, name);
    }
    
    // In actual usage, these precompiles would be invoked when the emulator
    // encounters PRECOMPILE instructions with matching funct7 values:
    //
    // Assembly example:
    // .word 0x02A382AB  # PRECOMPILE x5, x7, x10 with funct7=0x01 (XOR)
    //                   # Binary: [0000001][01010][00111][010][00101][0001011]
    //                   # funct7=0x01, rs2=10, rs1=7, funct3=2, rd=5, opcode=0x0B
    
    println!("\nTo use in assembly:");
    println!("  XOR:      .word 0x02A382AB  # funct7=0x01");
    println!("  ROTL:     .word 0x04A382AB  # funct7=0x02");
    println!("  POPCOUNT: .word 0x06A382AB  # funct7=0x03");
    
    println!("\nNote: The builder functions can return virtual instruction sequences");
    println!("that decompose complex operations into simpler RISC-V instructions.");
    println!("This is useful for verification and analysis purposes.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_precompile_registration() {
        // The precompiles might already be registered by the ctor
        let registered_before = list_registered_precompiles();
        
        if registered_before.is_empty() {
            // If not registered yet, register them
            assert!(init_precompiles().is_ok());
        }
        
        // Verify all precompiles are registered
        let registered = list_registered_precompiles();
        assert!(registered.len() >= 3); // At least our 3 precompiles
        
        // Check specific registrations
        let mut found_xor = false;
        let mut found_rotl = false;
        let mut found_popcount = false;
        
        for (funct7, name) in registered {
            match funct7 {
                0x01 => {
                    assert_eq!(name, "XOR_PRECOMPILE");
                    found_xor = true;
                }
                0x02 => {
                    assert_eq!(name, "ROTL_PRECOMPILE");
                    found_rotl = true;
                }
                0x03 => {
                    assert_eq!(name, "POPCOUNT_PRECOMPILE");
                    found_popcount = true;
                }
                _ => {} // Ignore other precompiles that might be registered
            }
        }
        
        assert!(found_xor && found_rotl && found_popcount);
    }
    
    #[test]
    fn test_duplicate_registration() {
        // Try to register the same funct7 twice (this should always fail)
        let result = register_precompile(0x01, "DUPLICATE", Box::new(|_, _, _| {}), Box::new(|_address, _rs1, _rs2| Vec::new()));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already registered"));
    }
} 