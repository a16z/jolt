use crate::test_constants::TestVectors;
use crate::test_utils::{sverify, Sha256CpuHarness};
use tracer::emulator::cpu::Xlen;
use tracer::emulator::test_harness::CpuTestHarness;
use tracer::inline_helpers::virtual_register_index;
use tracer::inline_helpers::{
    InstrAssembler,
    Value::{Imm, Reg},
};
use tracer::instruction::RISCVInstruction;

#[test]
fn test_sha256_direct_execution() {
    // Test against multiple canonical NIST test vectors
    for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
        let mut harness = Sha256CpuHarness::new();
        harness.load_block(&block);
        harness.load_state(&initial_state);
        Sha256CpuHarness::instruction_sha256().execute(&mut harness.harness.cpu, &mut ());
        let result = harness.read_state();

        sverify::assert_states_equal(
            &expected,
            &result,
            &format!("SHA256 direct execution: {desc}"),
        );
    }
}

#[test]
fn test_sha256_exec_trace_equal() {
    // Test exec vs trace equivalence with canonical test vectors
    for (desc, block, initial_state, _expected) in TestVectors::get_standard_test_vectors() {
        sverify::assert_exec_trace_equiv_custom(
            &block,
            &initial_state,
            &format!("SHA256 exec vs trace: {desc}"),
        );
    }
}

#[test]
fn measure_sha256_length() {
    use tracer::instruction::RISCVTrace;
    let instr = Sha256CpuHarness::instruction_sha256();
    for xlen in [Xlen::Bit32, Xlen::Bit64] {
        let sequence = instr.inline_sequence(xlen);
        let bytecode_len = sequence.len();
        println!(
            "SHA256 compression: xlen={:?}, bytecode length {}, {:.2} instructions per byte",
            xlen,
            bytecode_len,
            bytecode_len as f64 / 64.0
        );
    }
}

#[test]
fn test_sha256init_direct_execution() {
    // Test against canonical NIST test vectors with initial IV
    for (desc, block, _initial_state, expected) in TestVectors::get_standard_test_vectors() {
        let mut harness = Sha256CpuHarness::new();
        harness.load_block(&block);
        // Note: SHA256INIT doesn't need to load an initial state (uses default IV from BLOCK constants)
        // but it still needs RS2 set to a valid output address
        harness.setup_output_only();
        Sha256CpuHarness::instruction_sha256init().execute(&mut harness.harness.cpu, &mut ());
        let result = harness.read_state();

        sverify::assert_states_equal(
            &expected,
            &result,
            &format!("SHA256INIT direct execution: {desc}"),
        );
    }
}

#[test]
fn test_sha256init_exec_trace_equal() {
    // Test exec vs trace equivalence with canonical test vectors
    for (desc, block, _initial_state, _expected) in TestVectors::get_standard_test_vectors() {
        sverify::assert_exec_trace_equiv_initial(
            &block,
            &format!("SHA256INIT exec vs trace: {desc}"),
        );
    }
}

fn s0_word(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
fn s1_word(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}
fn s0_big(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
fn s1_big(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

#[test]
fn test_sha256_sigma_word_functions_rv64() {
    // Verify σ0/σ1 implemented via assembler ops on RV64 match pure formulas (low 32 bits)
    let mut harness = CpuTestHarness::new();
    let rs1 = 5u8; // use x5
    let out = virtual_register_index(0);
    let scratch1 = virtual_register_index(1);
    let scratch2 = virtual_register_index(2);

    for seed in [
        0u32,
        1,
        0xFFFFFFFF,
        0x0123_4567,
        0x89AB_CDEF,
        0x8000_0000,
        0x7FFF_FFFF,
    ] {
        harness.cpu.x[rs1 as usize] = seed as i32 as i64; // canonical RV behavior

        // σ0(x) = ROTR7(x) ^ ROTR18(x) ^ SHR3(x)
        // Also check each component.
        // ROTR7
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let _ = asm.rotri32(Reg(rs1), 7, out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let rotr7 = harness.cpu.x[out as usize] as u32;
        assert_eq!(rotr7, seed.rotate_right(7), "rotr7 for {seed:#010x}");
        // ROTR18
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let _ = asm.rotri32(Reg(rs1), 18, out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let rotr18 = harness.cpu.x[out as usize] as u32;
        assert_eq!(rotr18, seed.rotate_right(18), "rotr18 for {seed:#010x}");
        // SHR3
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let _ = asm.srli(Reg(rs1), 3, out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let shr3 = harness.cpu.x[out as usize] as u32;
        assert_eq!(shr3, seed >> 3, "shr3 for {seed:#010x}");

        // Now the combined σ0
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let r = asm.rotri_xor_rotri32(Reg(rs1), 7, 18, out, scratch1);
        let s = asm.srli(Reg(rs1), 3, scratch2);
        let _o = asm.xor(r, s, out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let got = harness.cpu.x[out as usize] as u32;
        assert_eq!(got, s0_word(seed), "σ0 mismatch for {seed:#010x}");

        // σ1(x) = ROTR17(x) ^ ROTR19(x) ^ SHR10(x)
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let r = asm.rotri_xor_rotri32(Reg(rs1), 17, 19, out, scratch1);
        let s = asm.srli(Reg(rs1), 10, scratch2);
        let _o = asm.xor(r, s, out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let got = harness.cpu.x[out as usize] as u32;
        assert_eq!(got, s1_word(seed), "σ1 mismatch for {seed:#010x}");
    }
}

#[test]
fn test_sha256_sigma_compression_functions_rv64() {
    // Verify Σ0/Σ1 via assembler ops on RV64 match pure formulas (low 32 bits)
    let mut harness = CpuTestHarness::new();
    let rs1 = 6u8; // x6
    let out = virtual_register_index(3);
    let scratch1 = virtual_register_index(4);
    let scratch2 = virtual_register_index(5);

    for seed in [0u32, 1, 0xDEAD_BEEF, 0x0123_4567, 0x89AB_CDEF, 0x8000_0001] {
        harness.cpu.x[rs1 as usize] = seed as i32 as i64;

        // Σ0(x) = ROTR2 ^ ROTR13 ^ ROTR22
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let _r = asm.rotri_xor_rotri32(Reg(rs1), 2, 13, out, scratch1);
        let _s = asm.rotri32(Reg(rs1), 22, scratch2);
        let _o = asm.xor(Reg(out), Reg(scratch2), out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let got = harness.cpu.x[out as usize] as u32;
        assert_eq!(got, s0_big(seed), "Σ0 mismatch for {seed:#010x}");

        // Σ1(x) = ROTR6 ^ ROTR11 ^ ROTR25
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        let _r = asm.rotri_xor_rotri32(Reg(rs1), 6, 11, out, scratch1);
        let _s = asm.rotri32(Reg(rs1), 25, scratch2);
        let _o = asm.xor(Reg(out), Reg(scratch2), out);
        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let got = harness.cpu.x[out as usize] as u32;
        assert_eq!(got, s1_big(seed), "Σ1 mismatch for {seed:#010x}");
    }
}

#[test]
fn test_sha256_round_t1_t2_rv64() {
    // Build T1/T2 with assembler ops on RV64 and compare to pure 32-bit reference.
    // T1 = h + Σ1(e) + Ch(e,f,g) + K[t] + W[t]
    // T2 = Σ0(a) + Maj(a,b,c)
    use crate::trace_generator::K;

    fn ch(e: u32, f: u32, g: u32) -> u32 {
        (e & f) ^ ((!e) & g)
    }
    fn maj(a: u32, b: u32, c: u32) -> u32 {
        (a & b) ^ (a & c) ^ (b & c)
    }

    let mut harness = CpuTestHarness::new();
    // Assign real registers for inputs
    let ra = 10u8;
    let rb = 11u8;
    let rc = 12u8;
    let rd = 13u8;
    let re = 14u8;
    let rf = 15u8;
    let rg = 16u8;
    let rh = 17u8;
    let rw = 18u8; // W[t]

    // Use a few deterministic test vectors
    let vectors: &[(u32, u32, u32, u32, u32, u32, u32, u32, u32, usize)] = &[
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 1),
        (
            0x89AB_CDEF,
            0x0123_4567,
            0xDEAD_BEEF,
            0xCAFEBABE,
            0xFEED_FACE,
            0xAAAA_5555,
            0x1357_9BDF,
            0x2468_ACED,
            0x0BAD_F00D,
            2,
        ),
    ];

    for &(a, b, c, d, e, f, g, h, w, t) in vectors {
        // load inputs (sign-extend as ISA would)
        for (reg, val) in [
            (ra, a),
            (rb, b),
            (rc, c),
            (rd, d),
            (re, e),
            (rf, f),
            (rg, g),
            (rh, h),
            (rw, w),
        ] {
            harness.cpu.x[reg as usize] = val as i32 as i64;
        }

        // Assemble T1/T2 into virtual registers
        let v_t1 = virtual_register_index(20);
        let v_t2 = virtual_register_index(21);
        let v_ss1 = virtual_register_index(22);
        let v_ss2 = virtual_register_index(23);

        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
        // Σ1(e)
        let s1e = {
            let r = asm.rotri_xor_rotri32(Reg(re), 6, 11, v_ss1, v_ss2);
            let s = asm.rotri32(Reg(re), 25, v_ss2);
            asm.xor(r, s, v_ss1)
        };
        // Ch(e,f,g)
        let chefg = {
            let e_and_f = asm.and(Reg(re), Reg(rf), v_ss2);
            asm.emit_r::<tracer::instruction::andn::ANDN>(v_ss1, rg, re);
            let neg_e_and_g = Reg(v_ss1);
            asm.xor(e_and_f, neg_e_and_g, v_ss1)
        };
        // h + K[t]
        let hk = asm.add(Imm(K[t] as u64), Reg(rh), v_t1);
        let t1 = asm.add(hk, s1e, v_t1);
        let t1 = asm.add(t1, chefg, v_t1);
        let _t1 = asm.add(t1, Reg(rw), v_t1);

        // Σ0(a)
        let s0a = {
            let r = asm.rotri_xor_rotri32(Reg(ra), 2, 13, v_ss1, v_ss2);
            let s = asm.rotri32(Reg(ra), 22, v_ss2);
            asm.xor(r, s, v_ss1)
        };
        // Maj(a,b,c)
        let majabc = {
            let b_and_c = asm.and(Reg(rb), Reg(rc), v_ss2);
            let b_xor_c = asm.xor(Reg(rb), Reg(rc), v_ss1);
            let a_and_bxor_c = asm.and(Reg(ra), b_xor_c, v_ss1);
            asm.xor(b_and_c, a_and_bxor_c, v_ss1)
        };
        let _t2 = asm.add(s0a, majabc, v_t2);

        let seq = asm.finalize();
        harness.execute_inline_sequence(&seq);
        let got_t1 = harness.cpu.x[v_t1 as usize] as u32;
        let got_t2 = harness.cpu.x[v_t2 as usize] as u32;

        let exp_t1 = h
            .wrapping_add(s1_big(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[t] as u32)
            .wrapping_add(w);
        let exp_t2 = s0_big(a).wrapping_add(maj(a, b, c));

        assert_eq!(got_t1, exp_t1, "T1 mismatch for t={t}");
        assert_eq!(got_t2, exp_t2, "T2 mismatch for t={t}");
    }
}

#[test]
fn test_rotri32_rv64_basic() {
    let mut h = CpuTestHarness::new();
    let rs1 = 7u8;
    let out = virtual_register_index(30);
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    h.cpu.x[rs1 as usize] = 1; // seed = 1
    let _r = asm.rotri32(Reg(rs1), 7, out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got = h.cpu.x[out as usize] as u32;
    assert_eq!(got, 1u32.rotate_right(7));
}

#[test]
fn test_rotri_xor_rotri32_rv64() {
    let mut h = CpuTestHarness::new();
    let rs1 = 8u8;
    let out = virtual_register_index(10);
    let scratch = virtual_register_index(11);
    h.cpu.x[rs1 as usize] = 1; // seed = 1
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let _v = asm.rotri_xor_rotri32(Reg(rs1), 7, 18, out, scratch);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got = h.cpu.x[out as usize] as u32;
    let expected = 1u32.rotate_right(7) ^ 1u32.rotate_right(18);
    assert_eq!(got, expected, "rotri_xor_rotri32 mismatch");
}

#[test]
fn test_sha256_build_up_to_step_t1_t2_rv64() {
    use crate::trace_generator::{sha2_build_up_to_step, K};
    let mut h = CpuTestHarness::new();
    // Memory layout matches harness utils: rs1 points to block, rs2 to state
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    h.cpu.x[rs1 as usize] = block_addr as i64;
    h.cpu.x[rs2 as usize] = state_addr as i64;

    // All zeros block and initial IV
    let block = [0u32; 16];
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    h.set_memory32(block_addr, &block);
    h.set_memory32(state_addr, &iv);

    // Build sequence up to T1/T2 for rounds 0..2 and compare against pure reference computation
    for t in 0..3usize {
        let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
        let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, t, "t1_t2");
        h.execute_inline_sequence(&seq);
        let t1 = h.cpu.x[tracer::inline_helpers::virtual_register_index(24) as usize] as u32;
        let t2 = h.cpu.x[tracer::inline_helpers::virtual_register_index(25) as usize] as u32;

        // Compute expected via exec.rs reference for one round
        // Reconstruct current a..h and w[t]
        // For t < 16 and zeros block, w[t] = 0
        let mut a = iv[0];
        let mut b = iv[1];
        let mut c = iv[2];
        let mut d = iv[3];
        let mut e = iv[4];
        let mut f = iv[5];
        let mut g = iv[6];
        let mut hh = iv[7];
        let mut w = [0u32; 64];
        w[..16].copy_from_slice(&block);
        for i in 0..t {
            let ch = (e & f) ^ ((!e) & g);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let S0a = s0_big(a);
            let S1e = s1_big(e);
            let tt1 = hh
                .wrapping_add(S1e)
                .wrapping_add(ch)
                .wrapping_add(K[i] as u32)
                .wrapping_add(w[i]);
            let tt2 = S0a.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(tt1);
            d = c;
            c = b;
            b = a;
            a = tt1.wrapping_add(tt2);
        }
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let exp_t1 = hh
            .wrapping_add(s1_big(e))
            .wrapping_add(ch)
            .wrapping_add(K[t] as u32)
            .wrapping_add(w[t]);
        let exp_t2 = s0_big(a).wrapping_add(maj);

        assert_eq!(t1, exp_t1, "t1 mismatch at round {t}");
        assert_eq!(t2, exp_t2, "t2 mismatch at round {t}");
    }
}

#[test]
fn test_sha256_build_up_to_step_after_round_rv64() {
    use crate::trace_generator::sha2_build_up_to_step;
    let mut h_exec = CpuTestHarness::new();
    let mut h_trace = CpuTestHarness::new();
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    for h in [&mut h_exec, &mut h_trace] {
        h.cpu.x[rs1 as usize] = block_addr as i64;
        h.cpu.x[rs2 as usize] = state_addr as i64;
        let block = [0u32; 16];
        let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
        h.set_memory32(block_addr, &block);
        h.set_memory32(state_addr, &iv);
    }

    // Helper to compute current VR index for a working var letter after `rounds`
    fn vr_idx_for(vr: &[u8; 32], rounds: usize, letter: char, initial: bool) -> u8 {
        let shift = (letter as i32 - 'A' as i32) as i32;
        if !initial
            && ((rounds == 0 && shift >= 4)
                || (rounds == 1 && shift >= 5)
                || (rounds == 2 && shift >= 6)
                || (rounds == 3 && shift >= 7))
        {
            return vr[(24 - rounds as i32 + shift) as usize];
        }
        vr[(-(rounds as i32) + shift).rem_euclid(8) as usize]
    }

    // Execute helper up to t rounds and compare A/E against pure reference state after t rounds
    for t in 0..2usize {
        let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
        let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, t, "after_round");
        h_trace.execute_inline_sequence(&seq);

        // Read A/E via VR mapping
        let a_idx = vr_idx_for(&vr, t + 1, 'A', false);
        let e_idx = vr_idx_for(&vr, t + 1, 'E', false);
        let mut regs = [0u64; 2];
        h_trace.read_registers(&[a_idx, e_idx], &mut regs);
        let (a_val, e_val) = (regs[0] as u32, regs[1] as u32);

        // Compute expected A/E after t+1 rounds using reference
        let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
        let block = [0u32; 16];
        let mut a = iv[0];
        let mut b = iv[1];
        let mut c = iv[2];
        let mut d = iv[3];
        let mut e = iv[4];
        let mut f = iv[5];
        let mut g = iv[6];
        let mut hh = iv[7];
        let w = block; // zeros
        for i in 0..=t {
            let ch = (e & f) ^ ((!e) & g);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let tt1 = hh
                .wrapping_add(s1_big(e))
                .wrapping_add(ch)
                .wrapping_add(crate::trace_generator::K[i] as u32)
                .wrapping_add(w[i]);
            let tt2 = s0_big(a).wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(tt1);
            d = c;
            c = b;
            b = a;
            a = tt1.wrapping_add(tt2);
        }
        assert_eq!(a_val, a, "A mismatch after round {t}");
        assert_eq!(e_val, e, "E mismatch after round {t}");

        // Reset trace harness for next iteration
        h_trace = CpuTestHarness::new();
        h_trace.cpu.x[rs1 as usize] = block_addr as i64;
        h_trace.cpu.x[rs2 as usize] = state_addr as i64;
        let block = [0u32; 16];
        let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
        h_trace.set_memory32(block_addr, &block);
        h_trace.set_memory32(state_addr, &iv);
    }
}

#[test]
fn test_sha256_t1_components_round0_rv64() {
    use crate::trace_generator::K;
    let mut h = CpuTestHarness::new();
    let _rs1 = tracer::inline_helpers::virtual_register_index(100); // unused
    let ra = 10u8;
    let rb = 11u8;
    let rc = 12u8;
    let rd = 13u8;
    let re = 14u8;
    let rf = 15u8;
    let rg = 16u8;
    let rh = 17u8;
    let v_out = tracer::inline_helpers::virtual_register_index(20);
    let v_s1 = tracer::inline_helpers::virtual_register_index(21);
    let v_tmp = tracer::inline_helpers::virtual_register_index(22);

    // Initial IV values
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    let a = iv[0];
    let b = iv[1];
    let c = iv[2];
    let d = iv[3];
    let e = iv[4];
    let f = iv[5];
    let g = iv[6];
    let hh = iv[7];
    for (reg, val) in [
        (ra, a),
        (rb, b),
        (rc, c),
        (rd, d),
        (re, e),
        (rf, f),
        (rg, g),
        (rh, hh),
    ] {
        h.cpu.x[reg as usize] = val as i32 as i64;
    }

    // h + K[0]
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let _hk = asm.add(Imm(K[0] as u64), Reg(rh), v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_hk = h.cpu.x[v_out as usize] as u32;
    assert_eq!(got_hk, hh.wrapping_add(K[0] as u32), "h+K0 mismatch");

    // Σ1(e)
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let r = asm.rotri_xor_rotri32(Reg(re), 6, 11, v_s1, v_tmp);
    let s = asm.rotri32(Reg(re), 25, v_tmp);
    let _ = asm.xor(r, s, v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_s1e = h.cpu.x[v_out as usize] as u32;
    assert_eq!(got_s1e, s1_big(e), "Σ1(e) mismatch");

    // Ch(e,f,g)
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let e_and_f = asm.and(Reg(re), Reg(rf), v_tmp);
    asm.emit_r::<tracer::instruction::andn::ANDN>(v_out, rg, re);
    let neg_e_and_g = Reg(v_out);
    let _ = asm.xor(e_and_f, neg_e_and_g, v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_chefg = h.cpu.x[v_out as usize] as u32;
    let exp_chefg = (e & f) ^ ((!e) & g);
    assert_eq!(got_chefg, exp_chefg, "Ch(e,f,g) mismatch");
}

#[test]
fn test_sha256_t1_components_round1_rv64() {
    use crate::trace_generator::{sha2_build_up_to_step, K};
    // Set up harness and memory
    let mut h = CpuTestHarness::new();
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    h.cpu.x[rs1 as usize] = block_addr as i64;
    h.cpu.x[rs2 as usize] = state_addr as i64;
    let block = [0u32; 16];
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    h.set_memory32(block_addr, &block);
    h.set_memory32(state_addr, &iv);

    // Execute one full round (t=0)
    let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
    let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, 0, "after_round");
    h.execute_inline_sequence(&seq);

    // Helper mapping for VR indices at the start of round 1
    fn vr_idx_for(vr: &[u8; 32], rounds: usize, letter: char) -> u8 {
        let initial = false;
        let shift = (letter as i32 - 'A' as i32) as i32;
        if !initial
            && ((rounds == 0 && shift >= 4)
                || (rounds == 1 && shift >= 5)
                || (rounds == 2 && shift >= 6)
                || (rounds == 3 && shift >= 7))
        {
            return vr[(24 - rounds as i32 + shift) as usize];
        }
        vr[(-(rounds as i32) + shift).rem_euclid(8) as usize]
    }

    // Read h,e,f,g at start of round 1
    let idx_h = vr_idx_for(&vr, 1, 'H');
    let idx_e = vr_idx_for(&vr, 1, 'E');
    let idx_f = vr_idx_for(&vr, 1, 'F');
    let idx_g = vr_idx_for(&vr, 1, 'G');
    let mut regs = [0u64; 4];
    h.read_registers(&[idx_h, idx_e, idx_f, idx_g], &mut regs);
    let (h1, e1, f1, g1) = (
        regs[0] as u32,
        regs[1] as u32,
        regs[2] as u32,
        regs[3] as u32,
    );

    // Compute components via assembler using spare VRs
    let v_out = tracer::inline_helpers::virtual_register_index(60);
    let v_tmp = tracer::inline_helpers::virtual_register_index(61);
    let v_tmp2 = tracer::inline_helpers::virtual_register_index(62);

    // Place e,f,g,h into some real regs: reuse their current regs directly via Reg(idx)
    // Σ1(e)
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let r = asm.rotri_xor_rotri32(Reg(idx_e), 6, 11, v_tmp, v_tmp2);
    let s = asm.rotri32(Reg(idx_e), 25, v_tmp2);
    let _ = asm.xor(r, s, v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_s1e = h.cpu.x[v_out as usize] as u32;

    // Ch(e,f,g) = (e&f) ^ (!e & g)
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let e_and_f = asm.and(Reg(idx_e), Reg(idx_f), v_tmp);
    asm.emit_r::<tracer::instruction::andn::ANDN>(v_out, idx_g, idx_e);
    let neg_e_and_g = Reg(v_out);
    let _ = asm.xor(e_and_f, neg_e_and_g, v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_chefg = h.cpu.x[v_out as usize] as u32;

    // h + K[1]
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let _ = asm.add(Imm(K[1] as u64), Reg(idx_h), v_out);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);
    let got_hk = h.cpu.x[v_out as usize] as u32;

    // Expected components
    let exp_s1e = s1_big(e1);
    let exp_chefg = (e1 & f1) ^ ((!e1) & g1);
    let exp_hk = h1.wrapping_add(K[1] as u32);

    assert_eq!(got_s1e, exp_s1e, "Σ1(e) at round 1 mismatch");
    assert_eq!(got_chefg, exp_chefg, "Ch at round 1 mismatch");
    assert_eq!(got_hk, exp_hk, "h+K1 mismatch");
}

#[test]
fn test_sha256_t1_terms_round1_rv64() {
    use crate::trace_generator::{sha2_build_up_to_step, K};
    let mut h = CpuTestHarness::new();
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    h.cpu.x[rs1 as usize] = block_addr as i64;
    h.cpu.x[rs2 as usize] = state_addr as i64;
    let block = [0u32; 16];
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    h.set_memory32(block_addr, &block);
    h.set_memory32(state_addr, &iv);

    // Execute one round (t=0) to reach the start of round 1
    let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
    let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, 0, "after_round");
    h.execute_inline_sequence(&seq);

    // Now materialize T1 terms at round 1
    let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, 1, "t1_terms");
    h.execute_inline_sequence(&seq);

    // Terms are placed in vr[24..27]
    let mut out = [0u64; 4];
    h.read_registers(
        &[
            tracer::inline_helpers::virtual_register_index(24),
            tracer::inline_helpers::virtual_register_index(25),
            tracer::inline_helpers::virtual_register_index(26),
            tracer::inline_helpers::virtual_register_index(27),
        ],
        &mut out,
    );
    let (hk, s1e, chefg, wt) = (out[0] as u32, out[1] as u32, out[2] as u32, out[3] as u32);

    // Compute expected e,f,g,h at start of round 1
    let mut a = iv[0];
    let mut b = iv[1];
    let mut c = iv[2];
    let mut d = iv[3];
    let mut e = iv[4];
    let mut f = iv[5];
    let mut g = iv[6];
    let mut hh = iv[7];
    let w = block;
    {
        let ch = (e & f) ^ ((!e) & g);
        let tt1 = hh
            .wrapping_add(s1_big(e))
            .wrapping_add(ch)
            .wrapping_add(K[0] as u32)
            .wrapping_add(w[0]);
        let tt2 = s0_big(a).wrapping_add((a & b) ^ (a & c) ^ (b & c));
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(tt1);
        d = c;
        c = b;
        b = a;
        a = tt1.wrapping_add(tt2);
    }

    let exp_hk = hh.wrapping_add(K[1] as u32);
    let exp_s1e = s1_big(e);
    let exp_chefg = (e & f) ^ ((!e) & g);
    let exp_wt = w[1];

    assert_eq!(hk, exp_hk, "h+K mismatch at round 1");
    assert_eq!(s1e, exp_s1e, "Sigma1(e) mismatch at round 1");
    assert_eq!(chefg, exp_chefg, "Ch(e,f,g) mismatch at round 1");
    assert_eq!(wt, exp_wt, "W[t] mismatch at round 1");
}

#[test]
fn test_sha256_full_final_state_vr_vs_reference_rv64() {
    use crate::trace_generator::sha2_build_up_to_step;
    let mut h = CpuTestHarness::new();
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    h.cpu.x[rs1 as usize] = block_addr as i64;
    h.cpu.x[rs2 as usize] = state_addr as i64;
    let block = [0u32; 16];
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    h.set_memory32(block_addr, &block);
    h.set_memory32(state_addr, &iv);

    let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
    let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, 63, "final_state");
    h.execute_inline_sequence(&seq);

    // After final_add_iv, A..H reside in their current VR slots (rotated). Read them in order A..H.
    fn vr_idx_for(vr: &[u8; 32], rounds: usize, letter: char) -> u8 {
        let shift = (letter as i32 - 'A' as i32) as i32;
        // After 64 rounds, custom IV path: the special E..H mapping is long gone; use rotation only
        vr[(-(rounds as i32) + shift).rem_euclid(8) as usize]
    }
    let rounds = 64usize; // total rounds applied when building
    let mut regs = [0u64; 8];
    let idxs: [u8; 8] = [
        vr_idx_for(&vr, rounds, 'A'),
        vr_idx_for(&vr, rounds, 'B'),
        vr_idx_for(&vr, rounds, 'C'),
        vr_idx_for(&vr, rounds, 'D'),
        vr_idx_for(&vr, rounds, 'E'),
        vr_idx_for(&vr, rounds, 'F'),
        vr_idx_for(&vr, rounds, 'G'),
        vr_idx_for(&vr, rounds, 'H'),
    ];
    h.read_registers(&idxs, &mut regs);
    let got: [u32; 8] = regs.map(|x| x as u32);

    // Compute reference final state
    let ref_final = crate::exec::execute_sha256_compression(iv, block);
    assert_eq!(
        got, ref_final,
        "Final VR state (after IV add) mismatch vs reference"
    );
}

#[test]
fn test_sha256_pre_final_state_vr_vs_reference_rv64() {
    use crate::trace_generator::sha2_build_up_to_step;
    let mut h = CpuTestHarness::new();
    let block_addr = tracer::emulator::mmu::DRAM_BASE;
    let state_addr = block_addr + 64;
    let rs1 = 10u8;
    let rs2 = 11u8;
    h.cpu.x[rs1 as usize] = block_addr as i64;
    h.cpu.x[rs2 as usize] = state_addr as i64;
    let block = [0u32; 16];
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    h.set_memory32(block_addr, &block);
    h.set_memory32(state_addr, &iv);

    let vr = core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
    // After 64 rounds, use after_round with target_round=63 (0-based)
    let seq = sha2_build_up_to_step(0, false, Xlen::Bit64, vr, rs1, rs2, 63, "after_round");
    h.execute_inline_sequence(&seq);

    // Read VRs for A..H pre-IV add
    fn vr_idx_for(vr: &[u8; 32], rounds: usize, letter: char) -> u8 {
        let shift = (letter as i32 - 'A' as i32) as i32;
        vr[(-(rounds as i32) + shift).rem_euclid(8) as usize]
    }
    let rounds = 64usize;
    let idxs: [u8; 8] = [
        vr_idx_for(&vr, rounds, 'A'),
        vr_idx_for(&vr, rounds, 'B'),
        vr_idx_for(&vr, rounds, 'C'),
        vr_idx_for(&vr, rounds, 'D'),
        vr_idx_for(&vr, rounds, 'E'),
        vr_idx_for(&vr, rounds, 'F'),
        vr_idx_for(&vr, rounds, 'G'),
        vr_idx_for(&vr, rounds, 'H'),
    ];
    let mut regs64 = [0u64; 8];
    h.read_registers(&idxs, &mut regs64);
    let got_pre: [u32; 8] = regs64.map(|x| x as u32);

    // Reference pre-final (a..h) via local round loop
    let mut a = iv[0];
    let mut b = iv[1];
    let mut c = iv[2];
    let mut d = iv[3];
    let mut e = iv[4];
    let mut f = iv[5];
    let mut g = iv[6];
    let mut hh = iv[7];
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(&block);
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t1 = hh
            .wrapping_add(s1_big(e))
            .wrapping_add(ch)
            .wrapping_add(crate::trace_generator::K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = s0_big(a).wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    let ref_pre = [a, b, c, d, e, f, g, hh];
    assert_eq!(got_pre, ref_pre, "Pre-final (a..h) mismatch");
}

#[test]
fn test_sha256_final_add_iv_g_isolated_rv64() {
    let mut h = CpuTestHarness::new();
    let vr: [u8; 32] =
        core::array::from_fn(|i| tracer::inline_helpers::virtual_register_index(i as u8));
    let iv = crate::trace_generator::BLOCK.map(|x| x as u32);
    let block = [0u32; 16];
    let pre_final_state = compute_pre_final_state(iv, block);

    // Manually set up registers to match state just before final_add_iv
    let g_final_vr = vr[(-64i32 + 6).rem_euclid(8) as usize];
    let g_iv_vr = vr[30];
    h.cpu.x[g_final_vr as usize] = pre_final_state[6] as i32 as i64; // Pre-final G
    h.cpu.x[g_iv_vr as usize] = iv[6] as i32 as i64; // Initial G

    // Execute only the ADD for G
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    let _ = asm.add(Reg(g_final_vr), Reg(g_iv_vr), g_final_vr);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);

    let final_g = h.cpu.x[g_final_vr as usize] as u32;
    let expected_g = pre_final_state[6].wrapping_add(iv[6]);

    assert_eq!(final_g, expected_g, "Isolated G add operation failed");
}

// Helper to get pre-final state for tests
fn compute_pre_final_state(iv: [u32; 8], block: [u32; 16]) -> [u32; 8] {
    let mut a = iv[0];
    let mut b = iv[1];
    let mut c = iv[2];
    let mut d = iv[3];
    let mut e = iv[4];
    let mut f = iv[5];
    let mut g = iv[6];
    let mut hh = iv[7];
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(&block);
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t1 = hh
            .wrapping_add(s1_big(e))
            .wrapping_add(ch)
            .wrapping_add(crate::trace_generator::K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = s0_big(a).wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    [a, b, c, d, e, f, g, hh]
}

#[test]
fn test_lw_rv64_no_clobber_saved_iv_regs() {
    use tracer::emulator::mmu::DRAM_BASE;
    let mut h = CpuTestHarness::new();
    let base = DRAM_BASE;
    // Prepare memory with 4 words
    let words: [u32; 4] = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
    h.set_memory32(base, &words);

    // Select rs1 (x5) to point at base
    let rs1 = 5u8;
    h.cpu.x[rs1 as usize] = base as i64;

    // Saved IV VRs to protect (like sha2 uses): 28..31
    let vr28 = virtual_register_index(28);
    let vr29 = virtual_register_index(29);
    let vr30 = virtual_register_index(30);
    let vr31 = virtual_register_index(31);

    // Fill with sentinels
    h.cpu.x[vr28 as usize] = 0xA5A5_A5A5u32 as i32 as i64;
    h.cpu.x[vr29 as usize] = 0x5A5A_5A5Au32 as i32 as i64;
    h.cpu.x[vr30 as usize] = 0xDEAD_BEEFu32 as i32 as i64;
    h.cpu.x[vr31 as usize] = 0xFEED_FACEu32 as i32 as i64;

    // Emit 4 LWs to vr24..vr27
    let rd_vrs = [24, 25, 26, 27].map(virtual_register_index);
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    for (i, rd) in rd_vrs.iter().enumerate() {
        asm.emit_ld::<tracer::instruction::lw::LW>(*rd, rs1, (i as i64) * 4);
    }
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);

    // Assert saved IV VRs unchanged
    assert_eq!(h.cpu.x[vr28 as usize] as u32, 0xA5A5_A5A5);
    assert_eq!(h.cpu.x[vr29 as usize] as u32, 0x5A5A_5A5A);
    assert_eq!(h.cpu.x[vr30 as usize] as u32, 0xDEAD_BEEF);
    assert_eq!(h.cpu.x[vr31 as usize] as u32, 0xFEED_FACE);
}

#[test]
fn test_lw_rv64_no_clobber_g_iv_reg_30() {
    use tracer::emulator::mmu::DRAM_BASE;
    let mut h = CpuTestHarness::new();
    let base = DRAM_BASE + 0x100;
    let words: [u32; 2] = [0xABCDEF01, 0x10203040];
    h.set_memory32(base, &words);

    let rs1 = 6u8;
    h.cpu.x[rs1 as usize] = base as i64;

    // Protect vr30 specifically
    let vr30 = virtual_register_index(30);
    h.cpu.x[vr30 as usize] = 0xCAFEBABEu32 as i32 as i64;

    // Perform two LWs to unrelated VRs
    let mut asm = InstrAssembler::new(0, false, Xlen::Bit64);
    asm.emit_ld::<tracer::instruction::lw::LW>(virtual_register_index(20), rs1, 0);
    asm.emit_ld::<tracer::instruction::lw::LW>(virtual_register_index(21), rs1, 4);
    let seq = asm.finalize();
    h.execute_inline_sequence(&seq);

    assert_eq!(
        h.cpu.x[vr30 as usize] as u32, 0xCAFEBABE,
        "vr30 was clobbered by LW inline"
    );
}
