mod helpers {
    pub(super) fn s0_word(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }
    pub(super) fn s1_word(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }
    pub(super) fn s0_big(x: u32) -> u32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }
    pub(super) fn s1_big(x: u32) -> u32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }
}

mod exec {
    use crate::test_constants::TestVectors;
    use crate::test_utils::{sverify, Sha256CpuHarness};
    use tracer::emulator::cpu::Xlen;
    use tracer::instruction::RISCVInstruction;

    #[test]
    fn test_sha256_direct_execution() {
        for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = Sha256CpuHarness::new(xlen);
                harness.load_block(&block);
                harness.load_state(&initial_state);
                Sha256CpuHarness::instruction_sha256().execute(&mut harness.harness.cpu, &mut ());
                let result = harness.read_state();

                sverify::assert_states_equal(
                    &expected,
                    &result,
                    &format!("SHA256 direct execution for {xlen:?}: {desc}"),
                );
            }
        }
    }

    #[test]
    fn test_sha256init_direct_execution() {
        for (desc, block, _initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = Sha256CpuHarness::new(xlen);
                harness.load_block(&block);
                harness.setup_output_only();
                Sha256CpuHarness::instruction_sha256init()
                    .execute(&mut harness.harness.cpu, &mut ());
                let result = harness.read_state();

                sverify::assert_states_equal(
                    &expected,
                    &result,
                    &format!("SHA256INIT direct execution for {xlen:?}: {desc}"),
                );
            }
        }
    }
}

mod exec_trace_equivalence {
    use super::helpers::*;
    use crate::test_constants::TestVectors;
    use crate::test_utils::{sverify, Sha256CpuHarness};

    use tracer::emulator::cpu::Xlen;
    use tracer::emulator::test_harness::CpuTestHarness;
    use tracer::utils::virtual_registers::{allocate_virtual_register, VirtualRegisterGuard};

    #[test]
    fn test_sha256_exec_trace_equal() {
        for (desc, block, initial_state, _expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                sverify::assert_exec_trace_equiv_custom(
                    &block,
                    &initial_state,
                    &format!("SHA256 exec vs trace for {xlen:?}: {desc}"),
                    xlen,
                );
            }
        }
    }

    #[test]
    fn test_sha256init_exec_trace_equal() {
        for (desc, block, _initial_state, _expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                sverify::assert_exec_trace_equiv_initial(
                    &block,
                    &format!("SHA256INIT exec vs trace for {xlen:?}: {desc}"),
                    xlen,
                );
            }
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
    fn test_sha256_exec_trace_intermediate_vr_equal() {
        use crate::trace_generator::sha2_build_up_to_step;
        let vectors = TestVectors::get_standard_test_vectors();
        for (desc, block, initial_state, _expected_final) in vectors {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                for t in 0..4usize {
                    let mut h = if xlen == Xlen::Bit32 {
                        CpuTestHarness::new_32()
                    } else {
                        CpuTestHarness::new()
                    };
                    let block_addr = tracer::emulator::mmu::DRAM_BASE;
                    let state_addr = block_addr + 64;
                    let rs1 = 10u8;
                    let rs2 = 11u8;
                    h.cpu.x[rs1 as usize] = block_addr as i64;
                    h.cpu.x[rs2 as usize] = state_addr as i64;
                    h.set_memory32(block_addr, &block);
                    h.set_memory32(state_addr, &initial_state);

                    let guards: [VirtualRegisterGuard; 32] =
                        core::array::from_fn(|_| allocate_virtual_register());
                    let vr: [u8; 32] = core::array::from_fn(|i| *guards[i]);
                    let seq = sha2_build_up_to_step(0, false, xlen, vr, rs1, rs2, t, "after_round");
                    h.execute_inline_sequence(&seq);

                    fn vr_idx_for(vr: &[u8; 32], rounds: usize, letter: char) -> u8 {
                        let shift = letter as i32 - 'A' as i32;
                        // Match the implementation's special handling for custom IV
                        if rounds == 0 && shift >= 4
                            || rounds == 1 && shift >= 5
                            || rounds == 2 && shift >= 6
                            || rounds == 3 && shift >= 7
                        {
                            return vr[24 - rounds + shift as usize];
                        }
                        vr[(-(rounds as i32) + shift).rem_euclid(8) as usize]
                    }
                    let idxs: [u8; 8] = [
                        vr_idx_for(&vr, t + 1, 'A'),
                        vr_idx_for(&vr, t + 1, 'B'),
                        vr_idx_for(&vr, t + 1, 'C'),
                        vr_idx_for(&vr, t + 1, 'D'),
                        vr_idx_for(&vr, t + 1, 'E'),
                        vr_idx_for(&vr, t + 1, 'F'),
                        vr_idx_for(&vr, t + 1, 'G'),
                        vr_idx_for(&vr, t + 1, 'H'),
                    ];
                    let mut regs64 = [0u64; 8];
                    h.read_registers(&idxs, &mut regs64);
                    let got: [u32; 8] = regs64.map(|x| x as u32);

                    let mut a = initial_state[0];
                    let mut b = initial_state[1];
                    let mut c = initial_state[2];
                    let mut d = initial_state[3];
                    let mut e = initial_state[4];
                    let mut f = initial_state[5];
                    let mut g = initial_state[6];
                    let mut hh = initial_state[7];
                    let mut w = [0u32; 64];
                    w[..16].copy_from_slice(&block);
                    for i in 16..64 {
                        let s0 = w[i - 15].rotate_right(7)
                            ^ w[i - 15].rotate_right(18)
                            ^ (w[i - 15] >> 3);
                        let s1 = w[i - 2].rotate_right(17)
                            ^ w[i - 2].rotate_right(19)
                            ^ (w[i - 2] >> 10);
                        w[i] = w[i - 16]
                            .wrapping_add(s0)
                            .wrapping_add(w[i - 7])
                            .wrapping_add(s1);
                    }
                    for (i, &wi) in w.iter().enumerate().take(t + 1) {
                        let ch = (e & f) ^ ((!e) & g);
                        let maj = (a & b) ^ (a & c) ^ (b & c);
                        let t1 = hh
                            .wrapping_add(s1_big(e))
                            .wrapping_add(ch)
                            .wrapping_add(crate::trace_generator::K[i] as u32)
                            .wrapping_add(wi);
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
                    let exp = [a, b, c, d, e, f, g, hh];
                    assert_eq!(
                        got, exp,
                        "Intermediate VR mismatch: {desc}, {xlen:?}, round {t}"
                    );
                }
            }
        }
    }
}

mod exec_unit {
    use super::helpers::*;

    use tracer::emulator::cpu::Xlen;

    type TestVector = (u32, u32, u32, u32, u32, u32, u32, u32, u32, usize);
    use tracer::emulator::test_harness::CpuTestHarness;
    use tracer::utils::inline_helpers::{
        InstrAssembler,
        Value::{Imm, Reg},
    };
    use tracer::utils::virtual_registers::allocate_virtual_register;

    #[test]
    fn test_sha256_sigma_word_functions() {
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut harness = if xlen == Xlen::Bit32 {
                CpuTestHarness::new_32()
            } else {
                CpuTestHarness::new()
            };
            let rs1 = 5u8;
            let out = allocate_virtual_register();
            let scratch1 = allocate_virtual_register();
            let scratch2 = allocate_virtual_register();

            for seed in [
                0u32,
                1,
                0xFFFF_FFFF,
                0x0123_4567,
                0x89AB_CDEF,
                0x8000_0000,
                0x7FFF_FFFF,
            ] {
                harness.cpu.x[rs1 as usize] = seed as i32 as i64;

                let mut asm = InstrAssembler::new(0, false, xlen);
                let _ = asm.rotri32(Reg(rs1), 7, *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let rotr7 = harness.cpu.x[*out as usize] as u32;
                assert_eq!(rotr7, seed.rotate_right(7), "rotr7 for {seed:#010x}");

                let mut asm = InstrAssembler::new(0, false, xlen);
                let _ = asm.rotri32(Reg(rs1), 18, *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let rotr18 = harness.cpu.x[*out as usize] as u32;
                assert_eq!(rotr18, seed.rotate_right(18), "rotr18 for {seed:#010x}");

                let mut asm = InstrAssembler::new(0, false, xlen);
                let _ = asm.srli(Reg(rs1), 3, *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let shr3 = harness.cpu.x[*out as usize] as u32;
                assert_eq!(shr3, seed >> 3, "shr3 for {seed:#010x}");

                let mut asm = InstrAssembler::new(0, false, xlen);
                let r = asm.rotri_xor_rotri32(Reg(rs1), 7, 18, *out, *scratch1);
                let s = asm.srli(Reg(rs1), 3, *scratch2);
                let _o = asm.xor(r, s, *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got = harness.cpu.x[*out as usize] as u32;
                assert_eq!(
                    got,
                    s0_word(seed),
                    "σ0 mismatch for {seed:#010x} ({xlen:?})"
                );

                let mut asm = InstrAssembler::new(0, false, xlen);
                let r = asm.rotri_xor_rotri32(Reg(rs1), 17, 19, *out, *scratch1);
                let s = asm.srli(Reg(rs1), 10, *scratch2);
                let _o = asm.xor(r, s, *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got = harness.cpu.x[*out as usize] as u32;
                assert_eq!(
                    got,
                    s1_word(seed),
                    "σ1 mismatch for {seed:#010x} ({xlen:?})"
                );
            }
        }
    }

    #[test]
    fn test_sha256_sigma_compression_functions() {
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut harness = if xlen == Xlen::Bit32 {
                CpuTestHarness::new_32()
            } else {
                CpuTestHarness::new()
            };
            let rs1 = 6u8;
            let out = allocate_virtual_register();
            let scratch1 = allocate_virtual_register();
            let scratch2 = allocate_virtual_register();

            for seed in [0u32, 1, 0xDEAD_BEEF, 0x0123_4567, 0x89AB_CDEF, 0x8000_0001] {
                harness.cpu.x[rs1 as usize] = seed as i32 as i64;

                let mut asm = InstrAssembler::new(0, false, xlen);
                let _r = asm.rotri_xor_rotri32(Reg(rs1), 2, 13, *out, *scratch1);
                let _s = asm.rotri32(Reg(rs1), 22, *scratch2);
                let _o = asm.xor(Reg(*out), Reg(*scratch2), *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got = harness.cpu.x[*out as usize] as u32;
                assert_eq!(got, s0_big(seed), "Σ0 mismatch for {seed:#010x} ({xlen:?})");

                let mut asm = InstrAssembler::new(0, false, xlen);
                let _r = asm.rotri_xor_rotri32(Reg(rs1), 6, 11, *out, *scratch1);
                let _s = asm.rotri32(Reg(rs1), 25, *scratch2);
                let _o = asm.xor(Reg(*out), Reg(*scratch2), *out);
                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got = harness.cpu.x[*out as usize] as u32;
                assert_eq!(got, s1_big(seed), "Σ1 mismatch for {seed:#010x} ({xlen:?})");
            }
        }
    }

    #[test]
    fn test_debug_t1_components() {
        // Debug test to isolate T1 components
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut harness = if xlen == Xlen::Bit32 {
                CpuTestHarness::new_32()
            } else {
                CpuTestHarness::new()
            };
            let re = 14u8;
            let rf = 15u8;
            let rg = 16u8;
            let rh = 17u8;
            let rw = 18u8;

            // Test vector t=1: e=5, f=6, g=7, h=8, w=9
            let (e, f, g, h, w, _t) = (5u32, 6u32, 7u32, 8u32, 9u32, 1usize);

            harness.cpu.x[re as usize] = e as i64;
            harness.cpu.x[rf as usize] = f as i64;
            harness.cpu.x[rg as usize] = g as i64;
            harness.cpu.x[rh as usize] = h as i64;
            harness.cpu.x[rw as usize] = w as i64;

            // Test S1(e) computation separately
            {
                let v_s1_out = allocate_virtual_register();
                let v_ss1 = allocate_virtual_register();
                let v_ss2 = allocate_virtual_register();

                let mut asm = InstrAssembler::new(0, false, xlen);
                let r = asm.rotri_xor_rotri32(Reg(re), 6, 11, *v_ss1, *v_ss2);
                let s = asm.rotri32(Reg(re), 25, *v_ss2);
                let _s1 = asm.xor(r, s, *v_s1_out);

                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got_s1 = harness.cpu.x[*v_s1_out as usize] as u32;
                let exp_s1 = s1_big(e);
                println!("S1({e}) = got:{got_s1:#x}, exp:{exp_s1:#x} ({xlen:?})");
                assert_eq!(got_s1, exp_s1, "S1 mismatch ({xlen:?})");
            }

            // Test Ch(e,f,g) computation separately
            {
                let v_ch_out = allocate_virtual_register();
                let v_ss1 = allocate_virtual_register();
                let v_ss2 = allocate_virtual_register();

                let mut asm = InstrAssembler::new(0, false, xlen);
                let e_and_f = asm.and(Reg(re), Reg(rf), *v_ss2);
                asm.emit_r::<tracer::instruction::andn::ANDN>(*v_ss1, rg, re);
                let neg_e_and_g = Reg(*v_ss1);
                let _ch = asm.xor(e_and_f, neg_e_and_g, *v_ch_out);

                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got_ch = harness.cpu.x[*v_ch_out as usize] as u32;
                let exp_ch = (e & f) ^ ((!e) & g);
                println!("Ch({e},{f},{g}) = got:{got_ch:#x}, exp:{exp_ch:#x} ({xlen:?})");
                assert_eq!(got_ch, exp_ch, "Ch mismatch ({xlen:?})");
            }
        }
    }

    #[test]
    fn test_sha256_round_t1_t2() {
        use crate::trace_generator::K;
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut harness = if xlen == Xlen::Bit32 {
                CpuTestHarness::new_32()
            } else {
                CpuTestHarness::new()
            };
            let ra = 10u8;
            let rb = 11u8;
            let rc = 12u8;
            let rd = 13u8;
            let re = 14u8;
            let rf = 15u8;
            let rg = 16u8;
            let rh = 17u8;
            let rw = 18u8;

            let vectors: &[TestVector] = &[
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

                let v_t1 = allocate_virtual_register();
                let v_t2 = allocate_virtual_register();
                let v_ss1 = allocate_virtual_register();
                let v_ss2 = allocate_virtual_register();
                let v_temp1 = allocate_virtual_register();
                let v_temp2 = allocate_virtual_register();

                let mut asm = InstrAssembler::new(0, false, xlen);
                let s1e = {
                    let r = asm.rotri_xor_rotri32(Reg(re), 6, 11, *v_ss1, *v_ss2);
                    let s = asm.rotri32(Reg(re), 25, *v_ss2);
                    asm.xor(r, s, *v_temp1)
                };
                let chefg = {
                    let e_and_f = asm.and(Reg(re), Reg(rf), *v_ss2);
                    asm.emit_r::<tracer::instruction::andn::ANDN>(*v_ss1, rg, re);
                    let neg_e_and_g = Reg(*v_ss1);
                    asm.xor(e_and_f, neg_e_and_g, *v_temp2)
                };
                let hk = asm.add(Imm(K[t]), Reg(rh), *v_t1);
                let t1a = asm.add(hk, s1e, *v_t1);
                let t1b = asm.add(t1a, chefg, *v_t1);
                let _t1 = asm.add(t1b, Reg(rw), *v_t1);
                let s0a = {
                    // Reuse v_temp1 for S0 output since S1 computation is done
                    let r = asm.rotri_xor_rotri32(Reg(ra), 2, 13, *v_ss1, *v_ss2);
                    let s = asm.rotri32(Reg(ra), 22, *v_ss2);
                    asm.xor(r, s, *v_temp1)
                };
                let majabc = {
                    // Reuse v_temp2 for Maj output since Ch computation is done
                    let b_and_c = asm.and(Reg(rb), Reg(rc), *v_ss2);
                    let b_xor_c = asm.xor(Reg(rb), Reg(rc), *v_ss1);
                    let a_and_bxor_c = asm.and(Reg(ra), b_xor_c, *v_ss1);
                    asm.xor(b_and_c, a_and_bxor_c, *v_temp2)
                };
                let _t2 = asm.add(s0a, majabc, *v_t2);

                let seq = asm.finalize();
                harness.execute_inline_sequence(&seq);
                let got_t1 = harness.cpu.x[*v_t1 as usize] as u32;
                let got_t2 = harness.cpu.x[*v_t2 as usize] as u32;

                let ch = |e: u32, f: u32, g: u32| (e & f) ^ ((!e) & g);
                let maj = |a: u32, b: u32, c: u32| (a & b) ^ (a & c) ^ (b & c);
                let exp_t1 = h
                    .wrapping_add(s1_big(e))
                    .wrapping_add(ch(e, f, g))
                    .wrapping_add(K[t] as u32)
                    .wrapping_add(w);
                let exp_t2 = s0_big(a).wrapping_add(maj(a, b, c));

                assert_eq!(got_t1, exp_t1, "T1 mismatch for t={t} ({xlen:?})");
                assert_eq!(got_t2, exp_t2, "T2 mismatch for t={t} ({xlen:?})");
            }
        }
    }
}
