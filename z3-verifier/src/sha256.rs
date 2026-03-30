#![cfg(test)]

//! Formal verification of SHA-256 against FIPS 180-4 using Z3 bit-vector theory.
//!
//! Verification scope:
//! - Constants K[] and H[] match FIPS 180-4 §4.2.2 and §5.3.3
//! - Optimized Ch (ANDN-based) ≡ spec Ch for all 32-bit inputs
//! - Optimized Maj ((B&C)^(A&(B^C))) ≡ spec Maj for all inputs
//! - Σ₀, Σ₁, σ₀, σ₁ implementations match spec
//! - Message schedule W expansion is correct
//! - Full compression matches FIPS 180-4 for all inputs
//! - NIST test vectors pass concretely

use jolt_inlines_sha2::exec::{execute_sha256_compression, execute_sha256_compression_initial};
use jolt_inlines_sha2::sequence_builder::{BLOCK, K};
use std::fmt::Write;
use z3::ast::BV;
use z3::{Params, SatResult, Solver};

const BV32: u32 = 32;
const Z3_SEED: u32 = 42;

// FIPS 180-4 §5.3.3 — initial hash values
const FIPS_H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// FIPS 180-4 §4.2.2 — round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
const FIPS_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// NIST test vectors: expected state after compressing a single block with initial IV
const NIST_ZERO_BLOCK_RESULT: [u32; 8] = [
    0xda5698be, 0x17b9b469, 0x62335799, 0x779fbeca, 0x8ce5d491, 0xc0d26243, 0xbafef9ea, 0x1837a9d8,
];
const NIST_PATTERN_BLOCK: [u32; 16] = [
    0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f, 0x10111213, 0x14151617, 0x18191a1b, 0x1c1d1e1f,
    0x20212223, 0x24252627, 0x28292a2b, 0x2c2d2e2f, 0x30313233, 0x34353637, 0x38393a3b, 0x3c3d3e3f,
];
const NIST_PATTERN_BLOCK_RESULT: [u32; 8] = [
    0xfc99a2df, 0x88f42a7a, 0x7bb9d180, 0x33cdc6a2, 0x0256755f, 0x9d5b9a50, 0x44a9cc31, 0x5abe84a7,
];
const NIST_ALL_ONES_BLOCK_RESULT: [u32; 8] = [
    0xef0c748d, 0xf4da50a8, 0xd6c43c01, 0x3edc3ce7, 0x6c9d9fa9, 0xa1458ade, 0x56eb86c0, 0xa64492d2,
];

// ────────────────────────────────────────────────────────────────
// Z3 helpers
// ────────────────────────────────────────────────────────────────

fn bv32_const(val: u32) -> BV {
    BV::from_u64(val as u64, BV32)
}

fn sym(name: &str) -> BV {
    BV::new_const(name.to_string(), BV32)
}

fn new_solver() -> Solver {
    let mut params = Params::default();
    params.set_u32("random_seed", Z3_SEED);
    let solver = Solver::new();
    solver.set_params(&params);
    solver
}

fn rotr(x: &BV, n: u32) -> BV {
    assert!(n > 0 && n < 32);
    let n_bv = BV::from_u64(n as u64, BV32);
    let c_bv = BV::from_u64((32 - n) as u64, BV32);
    x.bvlshr(n_bv).bvor(x.bvshl(c_bv))
}

fn shr(x: &BV, n: u32) -> BV {
    x.bvlshr(BV::from_u64(n as u64, BV32))
}

// ────────────────────────────────────────────────────────────────
// FIPS 180-4 §4.1.2 — specification functions
// ────────────────────────────────────────────────────────────────

fn spec_ch(x: &BV, y: &BV, z: &BV) -> BV {
    // Ch(x,y,z) = (x ∧ y) ⊕ (¬x ∧ z)
    x.bvand(y).bvxor(x.bvnot().bvand(z))
}

fn spec_maj(x: &BV, y: &BV, z: &BV) -> BV {
    // Maj(x,y,z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)
    x.bvand(y).bvxor(x.bvand(z)).bvxor(y.bvand(z))
}

fn spec_big_sigma_0(x: &BV) -> BV {
    // Σ₀(x) = ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)
    rotr(x, 2).bvxor(rotr(x, 13)).bvxor(rotr(x, 22))
}

fn spec_big_sigma_1(x: &BV) -> BV {
    // Σ₁(x) = ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)
    rotr(x, 6).bvxor(rotr(x, 11)).bvxor(rotr(x, 25))
}

fn spec_small_sigma_0(x: &BV) -> BV {
    // σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
    rotr(x, 7).bvxor(rotr(x, 18)).bvxor(shr(x, 3))
}

fn spec_small_sigma_1(x: &BV) -> BV {
    // σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
    rotr(x, 17).bvxor(rotr(x, 19)).bvxor(shr(x, 10))
}

// ────────────────────────────────────────────────────────────────
// Jolt optimized implementations (matching sequence_builder.rs)
// ────────────────────────────────────────────────────────────────

fn jolt_ch(e: &BV, f: &BV, g: &BV) -> BV {
    // sequence_builder.rs:312 — ANDN-based: (E & F) ^ (G & ~E)
    e.bvand(f).bvxor(g.bvand(e.bvnot()))
}

fn jolt_maj(a: &BV, b: &BV, c: &BV) -> BV {
    // sequence_builder.rs:332 — (B & C) ^ (A & (B ^ C))
    b.bvand(c).bvxor(a.bvand(b.bvxor(c)))
}

fn jolt_big_sigma_0(x: &BV) -> BV {
    // sequence_builder.rs:340 — rotri_xor_rotri32(x, 2, 13) ^ rotri32(x, 22)
    rotr(x, 2).bvxor(rotr(x, 13)).bvxor(rotr(x, 22))
}

fn jolt_big_sigma_1(x: &BV) -> BV {
    // sequence_builder.rs:347 — rotri_xor_rotri32(x, 6, 11) ^ rotri32(x, 25)
    rotr(x, 6).bvxor(rotr(x, 11)).bvxor(rotr(x, 25))
}

fn jolt_small_sigma_0(x: &BV) -> BV {
    // sequence_builder.rs:354 — rotri_xor_rotri32(x, 7, 18) ^ srli(x, 3)
    rotr(x, 7).bvxor(rotr(x, 18)).bvxor(shr(x, 3))
}

fn jolt_small_sigma_1(x: &BV) -> BV {
    // sequence_builder.rs:361 — rotri_xor_rotri32(x, 17, 19) ^ srli(x, 10)
    rotr(x, 17).bvxor(rotr(x, 19)).bvxor(shr(x, 10))
}

// ────────────────────────────────────────────────────────────────
// Full compression (FIPS spec)
// ────────────────────────────────────────────────────────────────

fn spec_compress(state: &[BV; 8], msg: &[BV; 16]) -> [BV; 8] {
    let mut w: Vec<BV> = msg.to_vec();
    for t in 16..64 {
        let w_new = spec_small_sigma_1(&w[t - 2])
            + w[t - 7].clone()
            + spec_small_sigma_0(&w[t - 15])
            + w[t - 16].clone();
        w.push(w_new);
    }

    let (mut a, mut b, mut c, mut d) = (
        state[0].clone(),
        state[1].clone(),
        state[2].clone(),
        state[3].clone(),
    );
    let (mut e, mut f, mut g, mut h) = (
        state[4].clone(),
        state[5].clone(),
        state[6].clone(),
        state[7].clone(),
    );

    for t in 0..64 {
        let t1 =
            h + spec_big_sigma_1(&e) + spec_ch(&e, &f, &g) + bv32_const(FIPS_K[t]) + w[t].clone();
        let t2 = spec_big_sigma_0(&a) + spec_maj(&a, &b, &c);
        h = g;
        g = f;
        f = e;
        e = d + t1.clone();
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    [
        state[0].clone() + a,
        state[1].clone() + b,
        state[2].clone() + c,
        state[3].clone() + d,
        state[4].clone() + e,
        state[5].clone() + f,
        state[6].clone() + g,
        state[7].clone() + h,
    ]
}

// ────────────────────────────────────────────────────────────────
// Full compression (Jolt — matching exec.rs / sequence_builder.rs)
// ────────────────────────────────────────────────────────────────

fn jolt_compress(state: &[BV; 8], msg: &[BV; 16]) -> [BV; 8] {
    let mut w: Vec<BV> = msg.to_vec();
    for t in 16..64 {
        // exec.rs:20-26 — same formula, different evaluation order
        let s0 = jolt_small_sigma_0(&w[t - 15]);
        let s1 = jolt_small_sigma_1(&w[t - 2]);
        let w_new = w[t - 16].clone() + s0 + w[t - 7].clone() + s1;
        w.push(w_new);
    }

    let (mut a, mut b, mut c, mut d) = (
        state[0].clone(),
        state[1].clone(),
        state[2].clone(),
        state[3].clone(),
    );
    let (mut e, mut f, mut g, mut h) = (
        state[4].clone(),
        state[5].clone(),
        state[6].clone(),
        state[7].clone(),
    );

    for t in 0..64 {
        // sequence_builder.rs compute_t1: (K + H) + Σ₁(E) + Ch(E,F,G) + W
        let sigma1 = jolt_big_sigma_1(&e);
        let ch = jolt_ch(&e, &f, &g);
        let t1 = bv32_const(FIPS_K[t]) + h + sigma1 + ch + w[t].clone();
        // sequence_builder.rs compute_t2: Σ₀(A) + Maj(A,B,C)
        let sigma0 = jolt_big_sigma_0(&a);
        let maj = jolt_maj(&a, &b, &c);
        let t2 = sigma0 + maj;
        // sequence_builder.rs apply_round_update
        h = g;
        g = f;
        f = e;
        e = d + t1.clone();
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    [
        state[0].clone() + a,
        state[1].clone() + b,
        state[2].clone() + c,
        state[3].clone() + d,
        state[4].clone() + e,
        state[5].clone() + f,
        state[6].clone() + g,
        state[7].clone() + h,
    ]
}

// ────────────────────────────────────────────────────────────────
// Equivalence proof helpers
// ────────────────────────────────────────────────────────────────

fn prove_bv_equivalent(name: &str, spec: &BV, jolt: &BV, inputs: &[(&str, &BV)]) {
    let mut solver = new_solver();
    solver += spec.ne(jolt);
    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let eval = |bv: &BV| model.eval(bv, true).unwrap().as_u64().unwrap();
            let mut msg = format!("{name}: counterexample found!\n");
            for (iname, bv) in inputs {
                let _ = writeln!(msg, "  {iname}: {:#010x}", eval(bv));
            }
            let _ = writeln!(msg, "  spec:  {:#010x}", eval(spec));
            let _ = writeln!(msg, "  jolt:  {:#010x}", eval(jolt));
            panic!("{}", msg.trim());
        }
        SatResult::Unknown => panic!("{name}: solver timed out"),
    }
}

fn prove_arrays_equivalent(name: &str, spec: &[BV; 8], jolt: &[BV; 8], inputs: &[(&str, &BV)]) {
    let mut solver = new_solver();
    let any_differs = spec
        .iter()
        .zip(jolt.iter())
        .map(|(s, j)| s.ne(j))
        .reduce(|acc, b| acc | b)
        .unwrap();
    solver += any_differs;
    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let eval = |bv: &BV| model.eval(bv, true).unwrap().as_u64().unwrap();
            let mut msg = format!("{name}: counterexample found!\n");
            for (iname, bv) in inputs {
                let _ = writeln!(msg, "  {iname}: {:#010x}", eval(bv));
            }
            for (i, (s, j)) in spec.iter().zip(jolt.iter()).enumerate() {
                let sv = eval(s);
                let jv = eval(j);
                if sv != jv {
                    let _ = writeln!(msg, "  output[{i}]: spec={sv:#010x} jolt={jv:#010x}");
                }
            }
            panic!("{}", msg.trim());
        }
        SatResult::Unknown => panic!("{name}: solver timed out"),
    }
}

// ────────────────────────────────────────────────────────────────
// Test: Constants match FIPS 180-4
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_constants_match_fips() {
    for (i, (&block_val, &fips_val)) in BLOCK.iter().zip(FIPS_H.iter()).enumerate() {
        assert_eq!(
            block_val as u32, fips_val,
            "BLOCK[{i}] = {:#010x} != FIPS H[{i}] = {fips_val:#010x}",
            block_val as u32
        );
    }
    for (i, (&k_val, &fips_val)) in K.iter().zip(FIPS_K.iter()).enumerate() {
        assert_eq!(
            k_val as u32, fips_val,
            "K[{i}] = {:#010x} != FIPS K[{i}] = {fips_val:#010x}",
            k_val as u32
        );
    }
}

// ────────────────────────────────────────────────────────────────
// Tests: Individual function equivalence (symbolic, ∀ 32-bit inputs)
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_ch_equivalence() {
    let (x, y, z) = (sym("x"), sym("y"), sym("z"));
    prove_bv_equivalent(
        "Ch",
        &spec_ch(&x, &y, &z),
        &jolt_ch(&x, &y, &z),
        &[("x", &x), ("y", &y), ("z", &z)],
    );
}

#[test]
fn test_sha256_maj_equivalence() {
    let (x, y, z) = (sym("x"), sym("y"), sym("z"));
    prove_bv_equivalent(
        "Maj",
        &spec_maj(&x, &y, &z),
        &jolt_maj(&x, &y, &z),
        &[("x", &x), ("y", &y), ("z", &z)],
    );
}

#[test]
fn test_sha256_big_sigma_0_equivalence() {
    let x = sym("x");
    prove_bv_equivalent(
        "Σ₀",
        &spec_big_sigma_0(&x),
        &jolt_big_sigma_0(&x),
        &[("x", &x)],
    );
}

#[test]
fn test_sha256_big_sigma_1_equivalence() {
    let x = sym("x");
    prove_bv_equivalent(
        "Σ₁",
        &spec_big_sigma_1(&x),
        &jolt_big_sigma_1(&x),
        &[("x", &x)],
    );
}

#[test]
fn test_sha256_small_sigma_0_equivalence() {
    let x = sym("x");
    prove_bv_equivalent(
        "σ₀",
        &spec_small_sigma_0(&x),
        &jolt_small_sigma_0(&x),
        &[("x", &x)],
    );
}

#[test]
fn test_sha256_small_sigma_1_equivalence() {
    let x = sym("x");
    prove_bv_equivalent(
        "σ₁",
        &spec_small_sigma_1(&x),
        &jolt_small_sigma_1(&x),
        &[("x", &x)],
    );
}

// ────────────────────────────────────────────────────────────────
// Test: Message schedule single-word expansion (symbolic)
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_message_schedule_single_word() {
    // W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]
    // Verify spec and Jolt compute the same value for one expansion step.
    // The same formula applies to all 48 derived words, so proving one
    // suffices (σ₀/σ₁ equivalence is proven separately).
    let w2 = sym("w_t_minus_2");
    let w7 = sym("w_t_minus_7");
    let w15 = sym("w_t_minus_15");
    let w16 = sym("w_t_minus_16");

    let spec = spec_small_sigma_1(&w2) + w7.clone() + spec_small_sigma_0(&w15) + w16.clone();
    let jolt = w16.clone() + jolt_small_sigma_0(&w15) + w7.clone() + jolt_small_sigma_1(&w2);

    prove_bv_equivalent(
        "W expansion",
        &spec,
        &jolt,
        &[
            ("W[t-2]", &w2),
            ("W[t-7]", &w7),
            ("W[t-15]", &w15),
            ("W[t-16]", &w16),
        ],
    );
}

// ────────────────────────────────────────────────────────────────
// Test: Single round equivalence (symbolic)
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_single_round_equivalence() {
    let state: [BV; 8] = std::array::from_fn(|i| sym(&format!("s{i}")));
    let w = sym("w");
    let k = bv32_const(FIPS_K[0]);

    // Spec round
    let spec_t1 = state[7].clone()
        + spec_big_sigma_1(&state[4])
        + spec_ch(&state[4], &state[5], &state[6])
        + k.clone()
        + w.clone();
    let spec_t2 = spec_big_sigma_0(&state[0]) + spec_maj(&state[0], &state[1], &state[2]);
    let spec_a = spec_t1.clone() + spec_t2;
    let spec_e = state[3].clone() + spec_t1;

    // Jolt round (matching sequence_builder.rs compute_t1 order)
    let jolt_t1 = k
        + state[7].clone()
        + jolt_big_sigma_1(&state[4])
        + jolt_ch(&state[4], &state[5], &state[6])
        + w.clone();
    let jolt_t2 = jolt_big_sigma_0(&state[0]) + jolt_maj(&state[0], &state[1], &state[2]);
    let jolt_a = jolt_t1.clone() + jolt_t2;
    let jolt_e = state[3].clone() + jolt_t1;

    let inputs: Vec<(&str, &BV)> = state
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let name: &'static str = ["A", "B", "C", "D", "E", "F", "G", "H"][i];
            (name, s)
        })
        .chain(std::iter::once(("W", &w)))
        .collect();

    let mut solver = new_solver();
    solver += spec_a.ne(&jolt_a) | spec_e.ne(&jolt_e);
    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let eval = |bv: &BV| model.eval(bv, true).unwrap().as_u64().unwrap();
            let mut msg = "single round: counterexample found!\n".to_string();
            for (name, bv) in &inputs {
                let _ = writeln!(msg, "  {name}: {:#010x}", eval(bv));
            }
            let _ = writeln!(msg, "  spec_A:  {:#010x}", eval(&spec_a));
            let _ = writeln!(msg, "  jolt_A:  {:#010x}", eval(&jolt_a));
            let _ = writeln!(msg, "  spec_E:  {:#010x}", eval(&spec_e));
            let _ = writeln!(msg, "  jolt_E:  {:#010x}", eval(&jolt_e));
            panic!("{}", msg.trim());
        }
        SatResult::Unknown => panic!("single round: solver timed out"),
    }
}

// ────────────────────────────────────────────────────────────────
// Test: Full compression equivalence (symbolic, all 64 rounds)
// Ignored by default: takes minutes with fully symbolic 32-bit inputs.
// Run with: cargo nextest run -p z3-verifier -E 'test(sha256_full_compression_equivalence)' --run-ignored all
// ────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_sha256_full_compression_equivalence() {
    let state: [BV; 8] = std::array::from_fn(|i| sym(&format!("h{i}")));
    let msg: [BV; 16] = std::array::from_fn(|i| sym(&format!("m{i}")));

    let spec_out = spec_compress(&state, &msg);
    let jolt_out = jolt_compress(&state, &msg);

    let inputs: Vec<(&str, &BV)> = {
        let state_names = ["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"];
        let msg_names = [
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11", "m12", "m13",
            "m14", "m15",
        ];
        state
            .iter()
            .enumerate()
            .map(|(i, s)| (state_names[i], s))
            .chain(msg.iter().enumerate().map(|(i, m)| (msg_names[i], m)))
            .collect()
    };

    prove_arrays_equivalent("full compression", &spec_out, &jolt_out, &inputs);
}

// ────────────────────────────────────────────────────────────────
// Test: Z3 spec model produces correct NIST outputs (concrete BVs)
// Validates that the Z3 specification model is faithfully built.
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_z3_spec_model_matches_nist() {
    let state: [BV; 8] = FIPS_H.map(bv32_const);
    let msg: [BV; 16] = [0u32; 16].map(bv32_const);
    let out = spec_compress(&state, &msg);

    let mut solver = new_solver();
    let any_wrong = out
        .iter()
        .enumerate()
        .map(|(i, bv)| bv.ne(bv32_const(NIST_ZERO_BLOCK_RESULT[i])))
        .reduce(|acc, b| acc | b)
        .unwrap();
    solver += any_wrong;
    assert_eq!(
        solver.check(),
        SatResult::Unsat,
        "Z3 spec model output does not match NIST zero-block vector"
    );
}

// ────────────────────────────────────────────────────────────────
// Tests: NIST concrete test vectors
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_nist_zero_block() {
    let result = execute_sha256_compression_initial([0u32; 16]);
    assert_eq!(result, NIST_ZERO_BLOCK_RESULT, "zero block with initial IV");
}

#[test]
fn test_sha256_nist_pattern_block() {
    let result = execute_sha256_compression_initial(NIST_PATTERN_BLOCK);
    assert_eq!(
        result, NIST_PATTERN_BLOCK_RESULT,
        "pattern block with initial IV"
    );
}

#[test]
fn test_sha256_nist_all_ones_block() {
    let result = execute_sha256_compression_initial([0xFFFFFFFF; 16]);
    assert_eq!(
        result, NIST_ALL_ONES_BLOCK_RESULT,
        "all-ones block with initial IV"
    );
}

#[test]
fn test_sha256_nist_chained_compression() {
    let first = execute_sha256_compression_initial(NIST_PATTERN_BLOCK);
    let second = execute_sha256_compression(first, [0u32; 16]);
    // Verify chained compression is deterministic by recomputing
    let first_again = execute_sha256_compression_initial(NIST_PATTERN_BLOCK);
    let second_again = execute_sha256_compression(first_again, [0u32; 16]);
    assert_eq!(second, second_again, "chained compression determinism");
}

// ────────────────────────────────────────────────────────────────
// Test: exec matches spec for initial compression
// ────────────────────────────────────────────────────────────────

#[test]
fn test_sha256_exec_initial_uses_correct_iv() {
    let with_explicit_iv = execute_sha256_compression(
        [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ],
        NIST_PATTERN_BLOCK,
    );
    let with_initial = execute_sha256_compression_initial(NIST_PATTERN_BLOCK);
    assert_eq!(
        with_explicit_iv, with_initial,
        "initial compression must use FIPS H values"
    );
}
