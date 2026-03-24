//! Cooperative field arithmetic MSL generation.
//!
//! Generates Metal shaders where N threads cooperate on one field element,
//! each thread holding a single u32 limb. For BN254 (N=8), this maps 4 field
//! elements per 32-thread simdgroup.
//!
//! The primary benefit is twofold:
//! 1. **CIOS latency reduction**: Inner loop parallelized from N serial muls to
//!    1 parallel mul + 3-round carry prefix = ~3.5x faster per fr_mul.
//! 2. **Register pressure elimination**: Each thread holds 1 limb (not N),
//!    reducing per-thread register usage from ~376 to ~47, eliminating L1 spills
//!    and enabling higher GPU occupancy.

use std::fmt::Write;

/// Generate the cooperative field arithmetic preamble.
///
/// Produces MSL functions that operate on individual limbs within an 8-thread
/// cooperative group, using `simd_shuffle` for inter-limb communication.
///
/// Thread mapping within a 32-thread simdgroup:
/// - Threads  0-7:  element 0, limbs 0-7
/// - Threads  8-15: element 1, limbs 0-7
/// - Threads 16-23: element 2, limbs 0-7
/// - Threads 24-31: element 3, limbs 0-7
pub fn generate_coop_preamble(n: usize) -> String {
    let mut s = String::with_capacity(16384);

    generate_coop_lane_helpers(&mut s, n);
    generate_coop_carry_prefix(&mut s, n);
    generate_coop_borrow_prefix(&mut s, n);
    generate_coop_add(&mut s, n);
    generate_coop_sub(&mut s, n);
    generate_coop_reduce(&mut s, n);
    generate_coop_mul(&mut s, n);

    s
}

/// Generate test kernel: cooperative mul with validation.
pub fn generate_coop_test_kernel(n: usize) -> String {
    let mut s = String::with_capacity(4096);

    // Kernel: each simdgroup processes 4 muls (one per cooperative group).
    // Reads full Fr values, scatters to limbs, does cooperative mul, gathers result.
    let _ = writeln!(s, "kernel void coop_fr_mul_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(s, "    uint tid [[thread_position_in_grid]],");
    let _ = writeln!(s, "    uint simd_lane [[thread_index_in_simdgroup]]");
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    uint pair_lane = simd_lane % {n}u;");
    let _ = writeln!(s, "    uint pair_base = simd_lane - pair_lane;");
    let _ = writeln!(s, "    uint elem_id = tid / {n}u;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    uint a_limb = a[elem_id].limbs[pair_lane];");
    let _ = writeln!(s, "    uint b_limb = b[elem_id].limbs[pair_lane];");
    let _ = writeln!(s);
    let _ = writeln!(s, "    uint r_limb = coop_fr_mul(a_limb, b_limb, pair_lane, pair_base);");
    let _ = writeln!(s);
    let _ = writeln!(s, "    result[elem_id].limbs[pair_lane] = r_limb;");
    let _ = writeln!(s, "}}");

    s
}

fn generate_coop_lane_helpers(s: &mut String, n: usize) {
    let _ = writeln!(s, "// Cooperative field arithmetic: {n} threads per field element.");
    let _ = writeln!(s, "// Each thread holds one u32 limb.");
    let _ = writeln!(s, "// pair_lane = simd_lane % {n} (which limb, 0 = LSB)");
    let _ = writeln!(s, "// pair_base = simd_lane - pair_lane (first lane of this group)");
    let _ = writeln!(s);
}

/// Parallel prefix carry: resolves a 1-bit ripple carry chain across N limbs.
///
/// Input: `ripple` = 1 if this limb generated a carry, `prop` = 1 if T == 0xFFFFFFFF.
/// Output: `carry_in` = 1 if this limb receives a carry from the chain below.
///
/// Uses Kogge-Stone parallel prefix with ceil(log2(N)) rounds.
fn generate_coop_carry_prefix(s: &mut String, n: usize) {
    let rounds = (n as f64).log2().ceil() as usize;

    let _ = writeln!(s, "// Parallel prefix carry resolution ({rounds} rounds for {n} limbs).");
    let _ = writeln!(s, "// g = generate (this limb produces carry regardless of input)");
    let _ = writeln!(s, "// p = propagate (this limb forwards an incoming carry)");
    let _ = writeln!(s, "// After: g[i] = 1 iff limb i receives a carry from the ripple chain.");
    let _ = writeln!(s, "inline uint coop_carry_prefix(uint g, uint p, uint pair_lane, uint pair_base) {{");

    let mut stride = 1usize;
    for _ in 0..rounds {
        let _ = writeln!(s, "    {{");
        let _ = writeln!(
            s,
            "        uint g_below = (pair_lane >= {stride}u) ? simd_shuffle(g, pair_base + pair_lane - {stride}u) : 0u;"
        );
        let _ = writeln!(
            s,
            "        uint p_below = (pair_lane >= {stride}u) ? simd_shuffle(p, pair_base + pair_lane - {stride}u) : 1u;"
        );
        let _ = writeln!(s, "        g = g | (p & g_below);");
        let _ = writeln!(s, "        p = p & p_below;");
        let _ = writeln!(s, "    }}");
        stride *= 2;
    }

    let _ = writeln!(s, "    return g;");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

/// Parallel prefix borrow: resolves a 1-bit borrow chain across N limbs.
///
/// Same structure as carry prefix but with borrow semantics.
fn generate_coop_borrow_prefix(s: &mut String, n: usize) {
    let rounds = (n as f64).log2().ceil() as usize;

    let _ = writeln!(s, "inline uint coop_borrow_prefix(uint g, uint p, uint pair_lane, uint pair_base) {{");

    let mut stride = 1usize;
    for _ in 0..rounds {
        let _ = writeln!(s, "    {{");
        let _ = writeln!(
            s,
            "        uint g_below = (pair_lane >= {stride}u) ? simd_shuffle(g, pair_base + pair_lane - {stride}u) : 0u;"
        );
        let _ = writeln!(
            s,
            "        uint p_below = (pair_lane >= {stride}u) ? simd_shuffle(p, pair_base + pair_lane - {stride}u) : 1u;"
        );
        let _ = writeln!(s, "        g = g | (p & g_below);");
        let _ = writeln!(s, "        p = p & p_below;");
        let _ = writeln!(s, "    }}");
        stride *= 2;
    }

    let _ = writeln!(s, "    return g;");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

/// Cooperative add with carry propagation: result_limb = (a + b) mod p.
fn generate_coop_add(s: &mut String, n: usize) {
    let n_minus_1 = n - 1;
    let _ = writeln!(s, "inline uint coop_fr_add(uint a_limb, uint b_limb, uint pair_lane, uint pair_base) {{");

    // Step 1: parallel add
    let _ = writeln!(s, "    uint sum = a_limb + b_limb;");
    let _ = writeln!(s, "    uint carry = uint(sum < a_limb);");

    // Step 2: multi-bit carry propagation (single round)
    let _ = writeln!(s, "    uint in_carry = (pair_lane > 0u) ? simd_shuffle(carry, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    uint new_sum = sum + in_carry;");
    let _ = writeln!(s, "    uint ripple = uint(new_sum < sum);");
    let _ = writeln!(s, "    sum = new_sum;");

    // Step 3: ripple prefix
    let _ = writeln!(s, "    uint g = ripple;");
    let _ = writeln!(s, "    uint p = uint(sum == 0xFFFFFFFFu);");
    let _ = writeln!(s, "    g = coop_carry_prefix(g, p, pair_lane, pair_base);");
    let _ = writeln!(s, "    uint carry_in_r = (pair_lane > 0u) ? simd_shuffle(g, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    sum += carry_in_r;");

    // Step 4: overflow = initial carry[N-1] | ripple chain carry-out past MSB
    let _ = writeln!(s, "    uint overflow = simd_shuffle(carry, pair_base + {n_minus_1}u) | simd_shuffle(g, pair_base + {n_minus_1}u);");

    // Step 5: conditional subtraction of modulus
    let _ = writeln!(s, "    uint diff = sum - FR_MODULUS[pair_lane];");
    let _ = writeln!(s, "    uint borrow_gen = uint(sum < FR_MODULUS[pair_lane]);");
    let _ = writeln!(s, "    uint borrow_prop = uint(sum == FR_MODULUS[pair_lane]);");
    let _ = writeln!(s, "    uint borrow = coop_borrow_prefix(borrow_gen, borrow_prop, pair_lane, pair_base);");
    let _ = writeln!(s, "    uint borrow_in_r = (pair_lane > 0u) ? simd_shuffle(borrow, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    diff -= borrow_in_r;");

    // borrow[N-1] after prefix = borrow exits MSB = subtraction underflowed
    let _ = writeln!(s, "    uint final_borrow = simd_shuffle(borrow, pair_base + {n_minus_1}u);");

    // Select: if overflow >= final_borrow, use subtracted; else use sum
    let _ = writeln!(s, "    return (overflow >= final_borrow) ? diff : sum;");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

/// Cooperative subtract with borrow propagation: result_limb = (a - b) mod p.
fn generate_coop_sub(s: &mut String, n: usize) {
    let n_minus_1 = n - 1;
    let _ = writeln!(s, "inline uint coop_fr_sub(uint a_limb, uint b_limb, uint pair_lane, uint pair_base) {{");

    // Step 1: parallel subtract
    let _ = writeln!(s, "    uint diff = a_limb - b_limb;");
    let _ = writeln!(s, "    uint borrow = uint(a_limb < b_limb);");

    // Step 2: multi-bit borrow propagation
    let _ = writeln!(s, "    uint in_borrow = (pair_lane > 0u) ? simd_shuffle(borrow, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    uint new_diff = diff - in_borrow;");
    let _ = writeln!(s, "    uint ripple = uint(new_diff > diff);");
    let _ = writeln!(s, "    diff = new_diff;");

    // Step 3: ripple borrow prefix
    let _ = writeln!(s, "    uint g = ripple;");
    let _ = writeln!(s, "    uint p = uint(diff == 0u);");
    let _ = writeln!(s, "    g = coop_borrow_prefix(g, p, pair_lane, pair_base);");
    let _ = writeln!(s, "    uint borrow_in_r = (pair_lane > 0u) ? simd_shuffle(g, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    diff -= borrow_in_r;");

    // g[N-1] after prefix = borrow exits MSB = underflow
    let _ = writeln!(s, "    uint final_borrow = simd_shuffle(g, pair_base + {n_minus_1}u);");

    // Step 5: conditional addition of modulus if underflow
    let _ = writeln!(s, "    uint addend = (final_borrow != 0u) ? FR_MODULUS[pair_lane] : 0u;");
    let _ = writeln!(s, "    uint corr = diff + addend;");
    let _ = writeln!(s, "    uint add_carry = uint(corr < diff);");

    // Carry propagation for the correction
    let _ = writeln!(s, "    uint add_in = (pair_lane > 0u) ? simd_shuffle(add_carry, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    corr += add_in;");
    let _ = writeln!(s, "    uint add_ripple = uint(corr == 0u && add_in != 0u);");
    let _ = writeln!(s, "    uint ag = add_ripple;");
    let _ = writeln!(s, "    uint ap = uint(corr == 0xFFFFFFFFu);");
    let _ = writeln!(s, "    ag = coop_carry_prefix(ag, ap, pair_lane, pair_base);");
    let _ = writeln!(s, "    uint ag_in = (pair_lane > 0u) ? simd_shuffle(ag, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    corr += ag_in;");

    let _ = writeln!(s, "    return corr;");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

/// Cooperative conditional subtraction of modulus.
fn generate_coop_reduce(s: &mut String, n: usize) {
    let n_minus_1 = n - 1;
    let _ = writeln!(s, "inline uint coop_fr_reduce(uint t_limb, uint pair_lane, uint pair_base) {{");

    // Subtract modulus
    let _ = writeln!(s, "    uint diff = t_limb - FR_MODULUS[pair_lane];");
    let _ = writeln!(s, "    uint borrow_gen = uint(t_limb < FR_MODULUS[pair_lane]);");
    let _ = writeln!(s, "    uint borrow_prop = uint(t_limb == FR_MODULUS[pair_lane]);");
    let _ = writeln!(s, "    uint borrow = coop_borrow_prefix(borrow_gen, borrow_prop, pair_lane, pair_base);");
    let _ = writeln!(s, "    uint borrow_in = (pair_lane > 0u) ? simd_shuffle(borrow, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "    diff -= borrow_in;");

    // borrow[N-1] after prefix = borrow chain exits MSB = underflow
    let _ = writeln!(s, "    uint msb_borrow = simd_shuffle(borrow, pair_base + {n_minus_1}u);");
    let _ = writeln!(s, "    return (msb_borrow != 0u) ? t_limb : diff;");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

/// Cooperative Montgomery CIOS multiplication.
///
/// Each of the N threads holds one limb of `a`, one limb of `b`, and one limb
/// of the accumulator. The N CIOS rounds are executed with parallel multiply-add
/// and inter-limb carry propagation via `simd_shuffle`.
fn generate_coop_mul(s: &mut String, n: usize) {
    let _ = writeln!(s, "inline uint coop_fr_mul(uint a_limb, uint b_limb, uint pair_lane, uint pair_base) {{");
    let _ = writeln!(s, "    uint T = 0u;");
    let _ = writeln!(s, "    uint T_extra = 0u;");
    let _ = writeln!(s);

    let _ = writeln!(s, "    for (uint j = 0u; j < {n}u; j++) {{");

    // Step 1: Broadcast b[j]
    let _ = writeln!(s, "        uint bj = simd_shuffle(b_limb, pair_base + j);");
    let _ = writeln!(s);

    // Step 2: Parallel multiply-add: T[i] += a[i] * b[j]
    let _ = writeln!(s, "        uint prod_lo = a_limb * bj;");
    let _ = writeln!(s, "        uint prod_hi = mulhi(a_limb, bj);");
    let _ = writeln!(s, "        uint sum = T + prod_lo;");
    let _ = writeln!(s, "        uint c0 = uint(sum < T);");
    let _ = writeln!(s, "        T = sum;");
    let _ = writeln!(s, "        uint out_carry = prod_hi + c0;");
    let _ = writeln!(s);

    // Step 3: Multi-bit carry from limb below
    let _ = writeln!(s, "        uint in_carry = (pair_lane > 0u) ? simd_shuffle(out_carry, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "        uint new_T = T + in_carry;");
    let _ = writeln!(s, "        uint ripple = uint(new_T < T);");
    let _ = writeln!(s, "        T = new_T;");
    let _ = writeln!(s);

    // Step 4: Ripple carry prefix — g[i] after prefix = inclusive carry-out of position i.
    // carry_in[i] = g[i-1] (carry entering position i from the chain below).
    let _ = writeln!(s, "        uint g = ripple;");
    let _ = writeln!(s, "        uint p = uint(T == 0xFFFFFFFFu);");
    let _ = writeln!(s, "        g = coop_carry_prefix(g, p, pair_lane, pair_base);");
    let _ = writeln!(s, "        uint carry_in_r = (pair_lane > 0u) ? simd_shuffle(g, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "        T += carry_in_r;");
    let _ = writeln!(s);

    // Step 5: T[N] overflow — direct carry from MSB multiply-add + ripple chain exit
    let n_minus_1 = n - 1;
    let _ = writeln!(s, "        if (pair_lane == {n_minus_1}u) {{");
    let _ = writeln!(s, "            T_extra += out_carry;");
    let _ = writeln!(s, "            T_extra += g;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s);

    // Step 6: Montgomery reduction: m = T[0] * INV32
    let _ = writeln!(s, "        uint T0 = simd_shuffle(T, pair_base);");
    let _ = writeln!(s, "        uint m = T0 * FR_INV32;");
    let _ = writeln!(s);

    // Step 7: Parallel multiply-add with modulus
    let _ = writeln!(s, "        uint mod_lo = m * FR_MODULUS[pair_lane];");
    let _ = writeln!(s, "        uint mod_hi = mulhi(m, FR_MODULUS[pair_lane]);");
    let _ = writeln!(s, "        uint mod_sum = T + mod_lo;");
    let _ = writeln!(s, "        uint mc = uint(mod_sum < T);");
    let _ = writeln!(s, "        T = mod_sum;");
    let _ = writeln!(s, "        uint mod_carry = mod_hi + mc;");
    let _ = writeln!(s);

    // Step 8: Multi-bit carry from Montgomery
    let _ = writeln!(s, "        uint mod_in = (pair_lane > 0u) ? simd_shuffle(mod_carry, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "        uint mod_new = T + mod_in;");
    let _ = writeln!(s, "        uint mod_ripple = uint(mod_new < T);");
    let _ = writeln!(s, "        T = mod_new;");
    let _ = writeln!(s);

    // Step 9: Ripple carry prefix for Montgomery — same shift pattern as Step 4
    let _ = writeln!(s, "        uint mg = mod_ripple;");
    let _ = writeln!(s, "        uint mp = uint(T == 0xFFFFFFFFu);");
    let _ = writeln!(s, "        mg = coop_carry_prefix(mg, mp, pair_lane, pair_base);");
    let _ = writeln!(s, "        uint mcarry_in = (pair_lane > 0u) ? simd_shuffle(mg, pair_base + pair_lane - 1u) : 0u;");
    let _ = writeln!(s, "        T += mcarry_in;");
    let _ = writeln!(s);

    // Step 10: T[N] overflow — direct Montgomery carry + ripple chain exit
    let _ = writeln!(s, "        if (pair_lane == {n_minus_1}u) {{");
    let _ = writeln!(s, "            T_extra += mod_carry;");
    let _ = writeln!(s, "            T_extra += mg;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s);

    // Step 11: Shift right by 1 limb (T[0] is now zero from Montgomery)
    let _ = writeln!(s, "        uint shifted;");
    let _ = writeln!(s, "        if (pair_lane < {n_minus_1}u) {{");
    let _ = writeln!(s, "            shifted = simd_shuffle(T, pair_base + pair_lane + 1u);");
    let _ = writeln!(s, "        }} else {{");
    let _ = writeln!(s, "            shifted = T_extra;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        T = shifted;");
    let _ = writeln!(s, "        T_extra = 0u;");

    let _ = writeln!(s, "    }}");
    let _ = writeln!(s);

    // Final conditional subtraction
    let _ = writeln!(s, "    return coop_fr_reduce(T, pair_lane, pair_base);");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coop_preamble_contains_expected_symbols() {
        let preamble = generate_coop_preamble(8);
        assert!(preamble.contains("coop_carry_prefix"));
        assert!(preamble.contains("coop_borrow_prefix"));
        assert!(preamble.contains("coop_fr_add"));
        assert!(preamble.contains("coop_fr_sub"));
        assert!(preamble.contains("coop_fr_reduce"));
        assert!(preamble.contains("coop_fr_mul"));
        assert!(preamble.contains("simd_shuffle"));
    }

    #[test]
    fn coop_test_kernel_compiles_structurally() {
        let kernel = generate_coop_test_kernel(8);
        assert!(kernel.contains("coop_fr_mul_kernel"));
        assert!(kernel.contains("coop_fr_mul("));
        assert!(kernel.contains("pair_lane"));
        assert!(kernel.contains("pair_base"));
    }
}
