//! Precompile functions for optimized cryptographic operations

/// Calls the SHA256 compression custom instruction
///
/// # Arguments
/// * `input` - Pointer to 16 u32 words (64 bytes) of input data
///             `input` has to have 24 u32 words preallocted, and output
///             will be written to `input + 16`
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
///             `input` has to have 24 u32 words preallocted, and output
///             will be written to `input + 16`
///
///
/// Uses the SHA256 initial state constants internally
///
/// # Safety
/// This function uses inline assembly and requires valid pointers to properly aligned memory
#[cfg(not(feature = "host"))]
pub unsafe fn sha256_compression_initial(input: *const u32) {
    core::arch::asm!(
        ".insn i 0x0B, 0x1, x0, {}, 0",
        in(reg) input,
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
    // rs1: [input: 16 words][output: 8 words] = 24 words total
    // rs2: [state: 8 words] = 8 words total
    let mut rs1_memory = [0u32; 24];
    let mut rs2_memory = [0u32; 8];

    // Copy state to rs2 memory (offsets 0-7)
    rs2_memory.copy_from_slice(state);
    // Copy input to rs1 memory at offsets 0-15
    rs1_memory[0..16].copy_from_slice(input);
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
    // rs1: [input: 16 words][output: 8 words] = 24 words total
    // rs2: not used (immediate version uses built-in initial values)
    let mut rs1_memory = [0u32; 24];

    // Copy input to rs1 memory at offsets 0-15
    rs1_memory[0..16].copy_from_slice(input);
    // Output will be written to rs1 offsets 16-23

    unsafe {
        // Pass pointer to rs1 memory, rs2 can be null since not used
        sha256_compression_initial(rs1_memory.as_ptr());
    }

    // Copy result from rs1 output section (offsets 16-23)
    let mut result = [0u32; 8];
    result.copy_from_slice(&rs1_memory[16..24]);
    result
}

// Host implementations using actual SHA256 implementation from tracer
#[cfg(feature = "host")]
pub unsafe fn sha256_compression(input: *const u32, state: *const u32) {
    // rs1 points to [input: 16 words][output: 8 words] = 24 words total
    // rs2 points to [state: 8 words]

    // Read input from rs1 (first 16 words)
    let input_slice = std::slice::from_raw_parts(input, 16);
    let mut input_array = [0u32; 16];
    input_array.copy_from_slice(input_slice);

    // Read state from rs2
    let state_slice = std::slice::from_raw_parts(state, 8);
    let mut state_array = [0u32; 8];
    state_array.copy_from_slice(state_slice);

    // Execute SHA256 compression
    let result = tracer::instruction::precompile_sha256::execute_sha256_compression(
        state_array,
        input_array,
    );

    // Write result to rs1+16 (output section after input)
    let output_ptr = (input as *mut u32).add(16);
    let output_slice = std::slice::from_raw_parts_mut(output_ptr, 8);
    output_slice.copy_from_slice(&result);
}

#[cfg(feature = "host")]
pub unsafe fn sha256_compression_initial(input: *const u32, _output: *const u32) {
    // rs1 points to [input: 16 words][output: 8 words] = 24 words total
    // rs2 is not used (immediate version uses built-in initial values)

    // Read input from rs1 (first 16 words)
    let input_slice = std::slice::from_raw_parts(input, 16);
    let mut input_array = [0u32; 16];
    input_array.copy_from_slice(input_slice);

    // Execute SHA256 compression
    let result =
        tracer::instruction::precompile_sha256::execute_sha256_compression_initial(input_array);

    // Write result to rs1+16 (output section after input)
    let output_ptr = (input as *mut u32).add(16);
    let output_slice = std::slice::from_raw_parts_mut(output_ptr, 8);
    output_slice.copy_from_slice(&result);
}

#[cfg(feature = "host")]
pub fn sha256_compress(input: &[u32; 16], state: &[u32; 8]) -> [u32; 8] {
    tracer::instruction::precompile_sha256::execute_sha256_compression(*state, *input)
}

#[cfg(feature = "host")]
pub fn sha256_compress_initial(input: &[u32; 16]) -> [u32; 8] {
    tracer::instruction::precompile_sha256::execute_sha256_compression_initial(*input)
}

