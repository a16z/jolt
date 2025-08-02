use tracer::{emulator::cpu::Cpu, instruction::{inline::INLINE, RISCVInstruction}};

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

pub fn execute_sha256_compression_initial(input: [u32; 16]) -> [u32; 8] {
    execute_sha256_compression(BLOCK.map(|x| x as u32), input)
}

pub fn sha2_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Load 16 input words from memory at rs1
    let mut input = [0u32; 16];
    for (i, word) in input.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load input word")
            .0;
    }

    // Load 8 initial state words from memory at rs2
    let mut iv = [0u32; 8];
    for (i, word) in iv.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load initial state")
            .0;
    }

    // Execute compression and store result at rs2
    let result = execute_sha256_compression(iv, input);
    for (i, &word) in result.iter().enumerate() {
        cpu.mmu
            .store_word(
                cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                word,
            )
            .expect("SHA256: Failed to store result");
    }
}

pub fn sha2_init_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Load 16 input words from memory at rs1
    let mut input = [0u32; 16];
    for (i, word) in input.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256INIT: Failed to load input word")
            .0;
    }

    // Execute compression with default initial state and store result
    let result = execute_sha256_compression_initial(input);
    for (i, &word) in result.iter().enumerate() {
        cpu.mmu
            .store_word(
                cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                word,
            )
            .expect("SHA256INIT: Failed to store result");
    }
}