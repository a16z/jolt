use tracer::{emulator::cpu::Cpu, instruction::{inline::INLINE, RISCVInstruction}};

use crate::trace_generator::{BLOCK, K};

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