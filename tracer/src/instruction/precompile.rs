use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use lazy_static::lazy_static;

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, VirtualInstructionSequence, RV32IMInstruction, RV32IMCycle,
};

// Type alias for the precompile function signature
// Matches the signature of the exec() method exactly
pub type PrecompileFunction = Box<dyn Fn(&PRECOMPILE, &mut Cpu, &mut ()) + Send + Sync>;

// Type alias for the builder function signature
// Takes address, rs1, rs2 and returns a virtual instruction sequence
pub type BuilderFunction = Box<dyn Fn(u64, usize, usize) -> Vec<RV32IMInstruction> + Send + Sync>;

// Global registry that maps funct7 values to precompile implementations
lazy_static! {
    static ref PRECOMPILE_REGISTRY: RwLock<HashMap<u32, (String, PrecompileFunction, BuilderFunction)>> = 
        RwLock::new(HashMap::new());
}

// Public API for registering precompiles
pub fn register_precompile(
    funct7: u32, 
    name: &str, 
    exec_fn: PrecompileFunction,
    builder_fn: BuilderFunction,
) -> Result<(), String> {
    // Validate funct7 is in valid range (0-127)
    if funct7 > 127 {
        return Err(format!("funct7 value {} is out of range (0-127)", funct7));
    }
    
    let mut registry = PRECOMPILE_REGISTRY.write()
        .map_err(|_| "Failed to acquire write lock on precompile registry")?;
    
    if registry.contains_key(&funct7) {
        return Err(format!("Precompile with funct7={} is already registered", funct7));
    }
    
    registry.insert(funct7, (name.to_string(), exec_fn, builder_fn));
    Ok(())
}

// Optional: Function to list registered precompiles
pub fn list_registered_precompiles() -> Vec<(u32, String)> {
    if let Ok(registry) = PRECOMPILE_REGISTRY.read() {
        registry.iter()
            .map(|(&funct7, (name, _, _))| (funct7, name.clone()))
            .collect()
    } else {
        Vec::new()
    }
}

// First declare the basic struct with the macro
declare_riscv_instr!(
    name   = PRECOMPILE,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000200b,  // funct7=0x00, funct3=0x2, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

// Now we need to store funct7 with the instruction
// We'll use a thread-local storage to map instruction addresses to their funct7 values
use std::cell::RefCell;
thread_local! {
    static PRECOMPILE_FUNCT7_MAP: RefCell<HashMap<u64, u32>> = RefCell::new(HashMap::new());
}

impl PRECOMPILE {
    // Override the new method to extract and store funct7
    pub fn new_with_funct7(word: u32, address: u64, validate: bool) -> Self {
        // Extract funct7 from the instruction word
        let funct7 = (word >> 25) & 0x7f;
        
        // Store the funct7 value for this instruction address
        PRECOMPILE_FUNCT7_MAP.with(|map| {
            map.borrow_mut().insert(address, funct7);
        });
        
        // Call the original new method
        <Self as RISCVInstruction>::new(word, address, validate)
    }
    
    fn exec(&self, cpu: &mut Cpu, _: &mut <PRECOMPILE as RISCVInstruction>::RAMAccess) {
        // Retrieve the funct7 value for this instruction
        let funct7 = PRECOMPILE_FUNCT7_MAP.with(|map| {
            map.borrow().get(&self.address).copied().unwrap_or(0)
        });
        
        // Look up the precompile function in the registry
        if let Ok(registry) = PRECOMPILE_REGISTRY.read() {
            if let Some((_name, exec_fn, _)) = registry.get(&funct7) {
                // Execute the registered function
                exec_fn(self, cpu, &mut ());
                return;
            }
        }
        
        // For unregistered funct7 values, we could either:
        // 1. Panic (strict mode)
        // 2. Do nothing (safe mode)
        // 3. Raise an illegal instruction exception
        // Here we'll panic with a descriptive error
        panic!("Unregistered PRECOMPILE with funct7={:#x} at address {:#x}", funct7, self.address);
    }
}

impl RISCVTrace for PRECOMPILE {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}


impl VirtualInstructionSequence for PRECOMPILE {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Retrieve the funct7 value for this instruction
        let funct7 = PRECOMPILE_FUNCT7_MAP.with(|map| {
            map.borrow().get(&self.address).copied().unwrap_or(0)
        });
        
        // Look up the builder function in the registry
        if let Ok(registry) = PRECOMPILE_REGISTRY.read() {
            if let Some((_name, _exec_fn, builder_fn)) = registry.get(&funct7) {
                // Execute the registered builder function
                return builder_fn(self.address, self.operands.rs1, self.operands.rs2);
            }
        }
        
        // If no builder is registered, return an empty sequence
        // This allows precompiles to work without virtual sequences if they don't need them
        Vec::new()
    }
}