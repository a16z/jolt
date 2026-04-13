//! Numeric verification of Stage 2 claim formula expansions.
//!
//! Evaluates the original jolt-core closed-form formulas and the expanded
//! sum-of-products representations from jolt_core_module.rs, asserting
//! numeric equivalence with concrete test values.
//!
//! Usage:
//!   cargo run --example verify_expansions -p jolt-compiler

#![allow(non_snake_case, clippy::print_stderr)]

fn main() {
    verify_ram_rw_output_check();
    verify_product_remainder_output_check();
    verify_instruction_cr_output_check();
    verify_output_check_output();
    verify_stage1_cycle_challenge_indices();
    verify_ram_rw_normalization();
    verify_raf_normalization();
    eprintln!("\nAll expansions verified ✓");
}

// Helpers: Lagrange basis and kernel over small integer domains

/// L_k(r) = ∏_{j≠k} (r - j) / (k - j) over domain {0, 1, ..., N-1}
fn lagrange_basis(r: f64, domain_size: usize, k: usize) -> f64 {
    let mut numer = 1.0;
    let mut denom = 1.0;
    for j in 0..domain_size {
        if j == k {
            continue;
        }
        numer *= r - j as f64;
        denom *= k as f64 - j as f64;
    }
    numer / denom
}

/// L(τ, r) = Σ_k L_k(τ) × L_k(r) over domain {0, ..., N-1}
fn lagrange_kernel(tau: f64, r: f64, domain_size: usize) -> f64 {
    (0..domain_size)
        .map(|k| lagrange_basis(tau, domain_size, k) * lagrange_basis(r, domain_size, k))
        .sum()
}

/// eq(r, s) = ∏_i (r_i * s_i + (1 - r_i) * (1 - s_i))
fn eq_eval(r: &[f64], s: &[f64]) -> f64 {
    assert_eq!(r.len(), s.len());
    r.iter()
        .zip(s)
        .map(|(&ri, &si)| ri * si + (1.0 - ri) * (1.0 - si))
        .product()
}

const EPS: f64 = 1e-9;

fn assert_close(a: f64, b: f64, label: &str) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        diff / scale < EPS,
        "{label}: {a} ≠ {b} (diff={diff}, rel={:.2e})",
        diff / scale
    );
}

// Test 1: RamRW output_check
//
// Original:  eq(r_cycle_s1, r_cycle) * ra * (val + γ*(val + inc))
// Expanded:  eq*ra*val + γ*eq*ra*val + γ*eq*ra*inc   (3 terms)

fn verify_ram_rw_output_check() {
    eprintln!("=== RamRW output_check ===");

    let gamma = 3.7;
    let ra = 2.1;
    let val = 5.3;
    let inc = 1.9;
    // Use short vectors for eq — just need dimensional correctness
    let r_cycle_s1 = vec![0.3, 0.7, 0.9];
    let r_cycle = vec![0.4, 0.6, 0.8];

    let eq_val = eq_eval(&r_cycle_s1, &r_cycle);

    // Original formula
    let original = eq_val * ra * (val + gamma * (val + inc));

    // Expanded: 3 terms
    let term1 = eq_val * ra * val;
    let term2 = gamma * eq_val * ra * val;
    let term3 = gamma * eq_val * ra * inc;
    let expanded = term1 + term2 + term3;

    assert_close(original, expanded, "RamRW output_check");
    eprintln!("  original = {original:.10}");
    eprintln!("  expanded = {expanded:.10}");
    eprintln!("  PASS ✓");
}

// Test 2: ProductRemainder output_check
//
// Original:
//   L(τ_high, r0) * eq(τ_low, r_rev)
//   * (w[0]*l_inst + w[1]*lookup + w[2]*jump)
//   * (w[0]*r_inst + w[1]*branch + w[2]*(1 - noop))
//
// Expanded: 12 terms from distributing fused_left × fused_right
//   with the (1 - noop) split into positive and negative parts.

fn verify_product_remainder_output_check() {
    eprintln!("\n=== ProductRemainder output_check ===");

    let tau_high = 1.3;
    let r0 = 2.1;
    let tau_low = vec![0.2, 0.5, 0.8, 0.1, 0.9];
    let r_rev = vec![0.7, 0.4, 0.3, 0.6, 0.55];

    let l_inst = 3.0;
    let r_inst = 4.0;
    let jump = 0.7;
    let lookup = 2.5;
    let branch = 1.1;
    let noop = 0.3;

    // Lagrange weights over domain {0, 1, 2}
    let w0 = lagrange_basis(r0, 3, 0);
    let w1 = lagrange_basis(r0, 3, 1);
    let w2 = lagrange_basis(r0, 3, 2);
    eprintln!("  w[0]={w0:.6}, w[1]={w1:.6}, w[2]={w2:.6}");

    let lk = lagrange_kernel(tau_high, r0, 3);
    let eq_val = eq_eval(&tau_low, &r_rev);

    let fused_left = w0 * l_inst + w1 * lookup + w2 * jump;
    let fused_right = w0 * r_inst + w1 * branch + w2 * (1.0 - noop);

    // Original formula
    let original = lk * eq_val * fused_left * fused_right;

    // 12-term expansion (matches jolt_core_module.rs ordering)
    let terms = [
        // w[0]*l_inst × w[0]*r_inst
        lk * eq_val * w0 * w0 * l_inst * r_inst,
        // w[0]*l_inst × w[1]*branch
        lk * eq_val * w0 * w1 * l_inst * branch,
        // w[0]*l_inst × w[2]*1
        lk * eq_val * w0 * w2 * l_inst,
        // -w[0]*l_inst × w[2]*noop
        -lk * eq_val * w0 * w2 * l_inst * noop,
        // w[1]*lookup × w[0]*r_inst
        lk * eq_val * w1 * w0 * lookup * r_inst,
        // w[1]*lookup × w[1]*branch
        lk * eq_val * w1 * w1 * lookup * branch,
        // w[1]*lookup × w[2]*1
        lk * eq_val * w1 * w2 * lookup,
        // -w[1]*lookup × w[2]*noop
        -lk * eq_val * w1 * w2 * lookup * noop,
        // w[2]*jump × w[0]*r_inst
        lk * eq_val * w2 * w0 * jump * r_inst,
        // w[2]*jump × w[1]*branch
        lk * eq_val * w2 * w1 * jump * branch,
        // w[2]*jump × w[2]*1
        lk * eq_val * w2 * w2 * jump,
        // -w[2]*jump × w[2]*noop
        -lk * eq_val * w2 * w2 * jump * noop,
    ];
    let expanded: f64 = terms.iter().sum();

    assert_close(original, expanded, "ProductRemainder output_check");
    eprintln!("  original = {original:.10}");
    eprintln!("  expanded = {expanded:.10}");

    // Also verify the 12 terms reconstruct correctly by grouping back
    // Group by left factor: terms 0-3 (l_inst), 4-7 (lookup), 8-11 (jump)
    let group_l: f64 = terms[0..4].iter().sum();
    let group_lookup: f64 = terms[4..8].iter().sum();
    let group_jump: f64 = terms[8..12].iter().sum();
    let regrouped = group_l + group_lookup + group_jump;
    assert_close(original, regrouped, "ProductRemainder regrouped");
    eprintln!("  regrouped = {regrouped:.10}");

    // Verify grouping matches: group_l = lk*eq * w0*l_inst * fused_right
    let group_l_expected = lk * eq_val * w0 * l_inst * fused_right;
    assert_close(group_l, group_l_expected, "group_l");

    let group_lookup_expected = lk * eq_val * w1 * lookup * fused_right;
    assert_close(group_lookup, group_lookup_expected, "group_lookup");

    let group_jump_expected = lk * eq_val * w2 * jump * fused_right;
    assert_close(group_jump, group_jump_expected, "group_jump");

    eprintln!("  PASS ✓ (12 terms, 3 groups, all match)");
}

// Test 3: InstructionCR output_check
//
// Original:
//   eq(r_spartan, r) * (LO + γ*LLOp + γ²*RLOp + γ³*LII + γ⁴*RII)
//
// Expanded: 5 terms, each with eq factor and γ^k weight

fn verify_instruction_cr_output_check() {
    eprintln!("\n=== InstructionCR output_check ===");

    let gamma: f64 = 2.3;
    let r_spartan = vec![0.1, 0.4, 0.7, 0.9];
    let r = vec![0.2, 0.5, 0.6, 0.8];

    let lo = 1.5;
    let l_op = 2.3;
    let r_op = 3.1;
    let l_inst = 4.7;
    let r_inst = 0.9;

    let eq_val = eq_eval(&r_spartan, &r);

    // Original
    let original = eq_val
        * (lo
            + gamma * l_op
            + gamma.powi(2) * r_op
            + gamma.powi(3) * l_inst
            + gamma.powi(4) * r_inst);

    // Expanded: 5 terms
    let expanded = eq_val * lo
        + eq_val * gamma * l_op
        + eq_val * gamma * gamma * r_op
        + eq_val * gamma * gamma * gamma * l_inst
        + eq_val * gamma * gamma * gamma * gamma * r_inst;

    assert_close(original, expanded, "InstructionCR output_check");
    eprintln!("  original = {original:.10}");
    eprintln!("  expanded = {expanded:.10}");
    eprintln!("  PASS ✓");
}

// Test 4: OutputCheck output
//
// Original:
//   eq(r_address, r') * io_mask(r') * (val_final(r') - val_io(r'))
//
// Expanded: 2 terms (positive val_final, negative val_io)

fn verify_output_check_output() {
    eprintln!("\n=== OutputCheck output ===");

    let r_address = vec![0.3, 0.7, 0.5];
    let r_prime = vec![0.4, 0.6, 0.8];

    let io_mask = 0.9;
    let val_final = 5.0;
    let val_io = 2.0;

    let eq_val = eq_eval(&r_address, &r_prime);

    // Original
    let original = eq_val * io_mask * (val_final - val_io);

    // Expanded: 2 terms
    let term1 = eq_val * io_mask * val_final;
    let term2 = -(eq_val * io_mask * val_io);
    let expanded = term1 + term2;

    assert_close(original, expanded, "OutputCheck output");
    eprintln!("  original = {original:.10}");
    eprintln!("  expanded = {expanded:.10}");
    eprintln!("  PASS ✓");
}

// Test 5: Stage 1 cycle challenge indices
//
// Simulates the outer remaining's normalize_opening_point:
//   r_cycle = challenges[1..].reverse()
// and verifies that our challenge index mapping (30..55).rev() produces
// the same values.

fn verify_stage1_cycle_challenge_indices() {
    eprintln!("\n=== Stage 1 cycle challenge indices ===");

    // Simulate: 26 raw sumcheck challenges for outer remaining
    // In the global challenge table, these are at indices 29..54 (outer_r_0 through outer_r_25).
    // We assign distinct values so we can trace them.
    let num_rounds = 26;
    let ch_base = 29; // outer_r_0 starts at global index 29

    // Global challenge table (simplified: only indices 29..54 matter)
    let mut challenge_values = vec![0.0_f64; 55];
    for i in 0..num_rounds {
        challenge_values[ch_base + i] = 100.0 + i as f64; // outer_r_i = 100+i
    }

    // jolt-core's normalize_opening_point: skip challenges[0], reverse rest
    let raw_challenges: Vec<f64> = (0..num_rounds)
        .map(|i| challenge_values[ch_base + i])
        .collect();
    let r_cycle_joltcore: Vec<f64> = raw_challenges[1..].iter().rev().copied().collect();
    eprintln!(
        "  jolt-core r_cycle: first={}, last={}, len={}",
        r_cycle_joltcore[0],
        r_cycle_joltcore[r_cycle_joltcore.len() - 1],
        r_cycle_joltcore.len()
    );

    // Our Module's stage1_cycle_challenges = (30..55).rev()
    let our_indices: Vec<usize> = (30..55).rev().collect();
    let our_r_cycle: Vec<f64> = our_indices.iter().map(|&i| challenge_values[i]).collect();
    eprintln!(
        "  Module r_cycle:    first={}, last={}, len={}",
        our_r_cycle[0],
        our_r_cycle[our_r_cycle.len() - 1],
        our_r_cycle.len()
    );

    assert_eq!(r_cycle_joltcore.len(), our_r_cycle.len(), "length mismatch");
    for (i, (&a, &b)) in r_cycle_joltcore.iter().zip(our_r_cycle.iter()).enumerate() {
        assert_close(a, b, &format!("r_cycle[{i}]"));
    }
    eprintln!("  PASS ✓ (25 elements match)");

    // Also verify the WRONG indices (29..54).rev() would be different
    let wrong_indices: Vec<usize> = (29..54).rev().collect();
    let wrong_r_cycle: Vec<f64> = wrong_indices.iter().map(|&i| challenge_values[i]).collect();
    assert_ne!(
        r_cycle_joltcore[0], wrong_r_cycle[0],
        "wrong indices should differ at [0]"
    );
    assert_ne!(
        r_cycle_joltcore[24], wrong_r_cycle[24],
        "wrong indices should differ at [24]"
    );
    eprintln!("  Confirmed (29..54).rev() would be WRONG ✓");
}

// Test 6: RamRW normalize_opening_point (Segments normalization)
//
// jolt-core's RamReadWriteCheckingParams::normalize_opening_point with
// default config (phase1=25, phase2=20, phase3_cycle=0, phase3_addr=0):
//   r_cycle = rev(phase1)
//   r_address = rev(phase2)
//   result = [r_address, r_cycle]
//
// Our Segments { sizes: [25, 20], output_order: [1, 0] }:
//   segment 0: rev(raw[0..25])  = rev(phase1) = r_cycle
//   segment 1: rev(raw[25..45]) = rev(phase2) = r_address
//   output [1, 0]: [segment_1, segment_0] = [r_address, r_cycle]

fn verify_ram_rw_normalization() {
    eprintln!("\n=== RamRW normalization ===");

    let log_t = 25;
    let log_k_ram = 20;
    let total_rounds = log_t + log_k_ram; // 45

    // Assign distinct values to each challenge
    let raw: Vec<f64> = (0..total_rounds).map(|i| 200.0 + i as f64).collect();

    // jolt-core with default config (phase3_cycle_rounds = 0):
    //   phase1 = raw[0..25], phase2 = raw[25..45]
    //   phase3_cycle = [], phase3_addr = []
    //   r_cycle = rev(phase3_cycle) ++ rev(phase1) = rev(raw[0..25])
    //   r_address = rev(phase3_addr) ++ rev(phase2) = rev(raw[25..45])
    //   result = [r_address, r_cycle]
    let r_cycle: Vec<f64> = raw[..log_t].iter().rev().copied().collect();
    let r_address: Vec<f64> = raw[log_t..].iter().rev().copied().collect();
    let joltcore_point: Vec<f64> = [r_address.clone(), r_cycle.clone()].concat();

    // Our Segments normalization
    let seg0: Vec<f64> = raw[..log_t].iter().rev().copied().collect(); // rev(phase1)
    let seg1: Vec<f64> = raw[log_t..].iter().rev().copied().collect(); // rev(phase2)
                                                                       // output_order [1, 0]: segment 1 first, then segment 0
    let our_point: Vec<f64> = [seg1, seg0].concat();

    assert_eq!(joltcore_point.len(), our_point.len());
    for (i, (&a, &b)) in joltcore_point.iter().zip(our_point.iter()).enumerate() {
        assert_close(a, b, &format!("rw_point[{i}]"));
    }
    eprintln!(
        "  point[0] (addr MSB) = {}, point[19] (addr LSB) = {}",
        our_point[0], our_point[19]
    );
    eprintln!(
        "  point[20] (cycle MSB) = {}, point[44] (cycle LSB) = {}",
        our_point[20], our_point[44]
    );
    eprintln!("  PASS ✓ (45 elements match)");

    // Verify the cycle portion at offset 20 matches r_cycle
    let cycle_at_offset: &[f64] = &our_point[log_k_ram..];
    for (i, (&a, &b)) in r_cycle.iter().zip(cycle_at_offset.iter()).enumerate() {
        assert_close(a, b, &format!("cycle[{i}]"));
    }
    eprintln!("  Cycle portion at offset {log_k_ram} matches ✓");
}

// Test 7: RafEvaluation normalize_opening_point
//
// jolt-core with default config (phase3_cycle_rounds = 0):
//   phase2_addr = 20 challenges
//   gap = 0
//   addr_challenges = challenges[0..20] ++ challenges[20..] = challenges[0..20]
//   result = reverse(addr_challenges)
//
// Our: Reverse on 20 challenges

fn verify_raf_normalization() {
    eprintln!("\n=== RafEval normalization ===");

    let log_k_ram = 20;
    let raw: Vec<f64> = (0..log_k_ram).map(|i| 300.0 + i as f64).collect();

    // jolt-core with phase3_cycle_rounds = 0
    let phase2_addr = log_k_ram;
    let gap = 0;
    let phase3_addr_start = phase2_addr + gap;
    let addr_challenges: Vec<f64> = [&raw[..phase2_addr], &raw[phase3_addr_start..]].concat();
    let joltcore_point: Vec<f64> = addr_challenges.iter().rev().copied().collect();

    // Our: simple Reverse
    let our_point: Vec<f64> = raw.iter().rev().copied().collect();

    assert_eq!(joltcore_point.len(), our_point.len());
    for (i, (&a, &b)) in joltcore_point.iter().zip(our_point.iter()).enumerate() {
        assert_close(a, b, &format!("raf_point[{i}]"));
    }
    eprintln!("  PASS ✓ ({log_k_ram} elements match)");
}
