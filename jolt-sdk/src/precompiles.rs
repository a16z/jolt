//! Precompile functions for optimized cryptographic operations

/// Calls the SHA256 compression custom instruction
/// 
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `state` - Pointer to 8 u32 words (32 bytes) of initial state
/// 
/// The result will be written to `state + 8` (next 32 bytes after the initial state)
/// 
/// # Safety
/// This function uses inline assembly and requires valid pointers to properly aligned memory
#[cfg(not(feature = "host"))]
pub unsafe fn sha256_compression(input: *const u32, state: *const u32) {
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x00, x0, {}, {}",
        in(reg) input,
        in(reg) state,
        options(nostack)
    );
}

/// Calls the SHA256 compression custom instruction with initial block
/// 
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
/// * `output` - Pointer to 8 u32 words (32 bytes) where result will be written
/// 
/// Uses the SHA256 initial state constants internally
/// 
/// # Safety
/// This function uses inline assembly and requires valid pointers to properly aligned memory
#[cfg(not(feature = "host"))]
pub unsafe fn sha256_compression_initial(input: *const u32, output: *const u32) {
    core::arch::asm!(
        ".insn r 0x0B, 0x1, 0x00, x0, {}, {}",
        in(reg) input,
        in(reg) output,
        options(nostack)
    );
}

/// Safe wrapper for SHA256 compression
/// 
/// # Arguments
/// * `input` - 16 u32 words (64 bytes) of input data
/// * `state` - 8 u32 words (32 bytes) of initial state
/// 
/// # Returns
/// The compressed state as 8 u32 words
#[cfg(not(feature = "host"))]
pub fn sha256_compress(input: &[u32; 16], state: &[u32; 8]) -> [u32; 8] {
    // Allocate memory layout expected by the builder:
    // rs1: [unused: 8 words][input: 16 words][output: 8 words] = 32 words total
    // rs2: [state: 8 words] = 8 words total
    let mut rs1_memory = [0u32; 32];
    let mut rs2_memory = [0u32; 8];
    
    // Copy state to rs2 memory (offsets 0-7)
    rs2_memory.copy_from_slice(state);
    // Copy input to rs1 memory at offsets 8-23 
    rs1_memory[8..24].copy_from_slice(input);
    // Output will be written to rs1 offsets 16-23
    
    unsafe {
        // Pass pointer to rs1 memory and pointer to rs2 memory
        sha256_compression(rs1_memory.as_ptr(), rs2_memory.as_ptr());
    }
    
    // Copy result from output section (rs1 offsets 16-23)
    let mut result = [0u32; 8];
    result.copy_from_slice(&rs1_memory[16..24]);
    result
}

/// Safe wrapper for SHA256 compression with initial block
/// 
/// # Arguments
/// * `input` - 16 u32 words (64 bytes) of input data
/// 
/// # Returns
/// The compressed state as 8 u32 words
#[cfg(not(feature = "host"))]
pub fn sha256_compress_initial(input: &[u32; 16]) -> [u32; 8] {
    // Allocate memory layout expected by the builder:
    // rs1: [unused: 8 words][input: 16 words][output: 8 words] = 32 words total
    // rs2: [output: 8 words] = 8 words total
    let mut rs1_memory = [0u32; 32];
    let mut rs2_memory = [0u32; 8];
    
    // Copy input to rs1 memory at offsets 8-23 
    rs1_memory[8..24].copy_from_slice(input);
    // Output will be written to rs2 offsets 0-7
    
    unsafe {
        // Pass pointer to rs1 memory and pointer to rs2 memory
        sha256_compression_initial(rs1_memory.as_ptr(), rs2_memory.as_ptr());
    }
    
    // Copy result from rs2 memory
    let mut result = [0u32; 8];
    result.copy_from_slice(&rs2_memory);
    result
}

// Host implementations (no-op for now, could call actual SHA256 implementation)
#[cfg(feature = "host")]
pub unsafe fn sha256_compression(_input: *const u32, _state: *const u32) {
    // No-op on host, should use actual SHA256 implementation
}

#[cfg(feature = "host")]
pub unsafe fn sha256_compression_initial(_input: *const u32, _output: *const u32) {
    // No-op on host, should use actual SHA256 implementation
}

#[cfg(feature = "host")]
pub fn sha256_compress(_input: &[u32; 16], _state: &[u32; 8]) -> [u32; 8] {
    // No-op on host, should use actual SHA256 implementation
    [0u32; 8]
}

#[cfg(feature = "host")]
pub fn sha256_compress_initial(_input: &[u32; 16]) -> [u32; 8] {
    // No-op on host, should use actual SHA256 implementation
    [0u32; 8]
}