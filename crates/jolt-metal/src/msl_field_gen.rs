//! Dynamic MSL source generation for arbitrary Montgomery-form prime fields.
//!
//! Generates Metal Shading Language code for field arithmetic, wide accumulators,
//! and test kernels parameterized by [`MontgomeryConstants`]. The generated MSL is
//! functionally identical to the former static `.metal` shader files when called
//! with N=8 and BN254 constants.

use std::fmt::Write;

use jolt_field::MontgomeryConstants;

/// Generate the complete MSL preamble: Fr struct, constants, arithmetic, WideAcc.
///
/// This replaces `bn254_fr.metal`, `wide_accumulator.metal`, and `test_kernels.metal`.
/// The returned string is suitable for prepending to kernel-specific MSL bodies.
pub fn generate_full_preamble<F: MontgomeryConstants>() -> String {
    let n = F::NUM_U32_LIMBS;
    let acc_n = F::ACC_U32_LIMBS;
    let modulus = F::modulus_u32();
    let inv32 = F::inv32();
    let r2 = F::r2_u32();
    let one = F::one_u32();

    let mut s = String::with_capacity(16384);
    // FR_FUNC_ATTR: inline by default, noinline for fast compilation
    let _ = writeln!(s, "#ifdef FR_NOINLINE");
    let _ = writeln!(s, "#define FR_FUNC_ATTR __attribute__((noinline))");
    let _ = writeln!(s, "#else");
    let _ = writeln!(s, "#define FR_FUNC_ATTR inline");
    let _ = writeln!(s, "#endif");
    s.push('\n');
    generate_fr_struct(&mut s, n);
    generate_field_constants(&mut s, n, modulus, inv32, r2, one);
    generate_fr_arithmetic(&mut s, n);
    generate_cios_macro(&mut s, n);
    generate_fr_mul(&mut s, n);
    generate_wide_acc(&mut s, n, acc_n);
    generate_acc_fmadd(&mut s, n, acc_n);
    s
}

/// Generate MSL for the test/benchmark kernels (mul, add, sub, sqr, neg, fmadd, from_u64).
pub fn generate_test_kernels<F: MontgomeryConstants>() -> String {
    let n = F::NUM_U32_LIMBS;
    let mut s = String::with_capacity(4096);

    let _ = writeln!(s, "kernel void fr_mul_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    result[tid] = fr_mul(a[tid], b[tid]);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    let _ = writeln!(s, "kernel void fr_add_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    result[tid] = fr_add(a[tid], b[tid]);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    let _ = writeln!(s, "kernel void fr_sub_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    result[tid] = fr_sub(a[tid], b[tid]);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    let _ = writeln!(s, "kernel void fr_sqr_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(1)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    result[tid] = fr_sqr(a[tid]);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    let _ = writeln!(s, "kernel void fr_neg_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(1)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    result[tid] = fr_neg(a[tid]);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    let _ = writeln!(s, "constant uint N_FMADD = 256;");
    s.push('\n');
    let _ = writeln!(s, "kernel void fr_fmadd_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(s, "    device const uint* params [[buffer(3)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    uint stride = params[0];");
    let _ = writeln!(s, "    WideAcc acc = acc_zero();");
    let _ = writeln!(s, "    uint base = tid * N_FMADD;");
    let _ = writeln!(s, "    for (uint i = 0; i < N_FMADD; i++) {{");
    let _ = writeln!(s, "        uint idx = (base + i) % stride;");
    let _ = writeln!(s, "        acc_fmadd(acc, a[idx], b[idx]);");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    result[tid] = acc_reduce(acc);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // Parameterized fmadd: reads n_fmadd from params[1] instead of constant.
    // Used by benchmarks to sweep accumulation depth.
    let _ = writeln!(s, "kernel void fr_fmadd_param_kernel(");
    let _ = writeln!(s, "    device const Fr* a       [[buffer(0)]],");
    let _ = writeln!(s, "    device const Fr* b       [[buffer(1)]],");
    let _ = writeln!(s, "    device Fr*       result  [[buffer(2)]],");
    let _ = writeln!(s, "    device const uint* params [[buffer(3)]],");
    let _ = writeln!(
        s,
        "    uint tid                 [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    uint stride = params[0];");
    let _ = writeln!(s, "    uint n_fmadd = params[1];");
    let _ = writeln!(s, "    WideAcc acc = acc_zero();");
    let _ = writeln!(s, "    uint base = tid * n_fmadd;");
    let _ = writeln!(s, "    for (uint i = 0; i < n_fmadd; i++) {{");
    let _ = writeln!(s, "        uint idx = (base + i) % stride;");
    let _ = writeln!(s, "        acc_fmadd(acc, a[idx], b[idx]);");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    result[tid] = acc_reduce(acc);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_from_u64: only meaningful for N >= 2 (which is always the case)
    let _ = writeln!(s, "kernel void fr_from_u64_kernel(");
    let _ = writeln!(s, "    device const ulong* vals  [[buffer(0)]],");
    let _ = writeln!(s, "    device Fr*          result [[buffer(1)]],");
    let _ = writeln!(
        s,
        "    uint tid                   [[thread_position_in_grid]]"
    );
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    Fr a;");
    let _ = writeln!(s, "    a.limbs[0] = uint(vals[tid]);");
    let _ = writeln!(s, "    a.limbs[1] = uint(vals[tid] >> 32);");
    for i in 2..n {
        let _ = writeln!(s, "    a.limbs[{i}] = 0;");
    }
    let _ = writeln!(s, "    result[tid] = fr_to_mont(a);");
    let _ = writeln!(s, "}}");

    s
}

fn generate_fr_struct(s: &mut String, n: usize) {
    let _ = writeln!(s, "struct Fr {{");
    let _ = writeln!(s, "    uint limbs[{n}];");
    let _ = writeln!(s, "}};");
    s.push('\n');
}

fn generate_field_constants(
    s: &mut String,
    n: usize,
    modulus: &[u32],
    inv32: u32,
    r2: &[u32],
    one: &[u32],
) {
    assert_eq!(modulus.len(), n);
    assert_eq!(r2.len(), n);
    assert_eq!(one.len(), n);

    let _ = write!(s, "constant uint FR_MODULUS[{n}] = {{");
    for (i, &v) in modulus.iter().enumerate() {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        let _ = write!(s, "0x{v:08x}u");
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(s, "constant uint FR_INV32 = 0x{inv32:08x}u;");

    let _ = write!(s, "constant uint FR_R2[{n}] = {{");
    for (i, &v) in r2.iter().enumerate() {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        let _ = write!(s, "0x{v:08x}u");
    }
    let _ = writeln!(s, "}};");

    let _ = write!(s, "constant uint FR_ONE[{n}] = {{");
    for (i, &v) in one.iter().enumerate() {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        let _ = write!(s, "0x{v:08x}u");
    }
    let _ = writeln!(s, "}};");
    s.push('\n');
}

fn generate_fr_arithmetic(s: &mut String, n: usize) {
    // fr_gte
    let _ = writeln!(
        s,
        "inline bool fr_gte(Fr a, thread const uint (&m)[{n}]) {{"
    );
    let _ = writeln!(s, "    for (int i = {0}; i >= 0; i--) {{", n - 1);
    let _ = writeln!(s, "        if (a.limbs[i] > m[i]) return true;");
    let _ = writeln!(s, "        if (a.limbs[i] < m[i]) return false;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return true;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_select
    let _ = writeln!(
        s,
        "inline Fr fr_select(bool cond, Fr if_true, Fr if_false) {{"
    );
    let _ = writeln!(s, "    Fr r;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(
        s,
        "        r.limbs[i] = cond ? if_true.limbs[i] : if_false.limbs[i];"
    );
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return r;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_reduce
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_reduce(Fr a) {{");
    let _ = writeln!(s, "    Fr reduced;");
    let _ = writeln!(s, "    uint borrow = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(
        s,
        "        uint2 r = sbb(a.limbs[i], FR_MODULUS[i], borrow);"
    );
    let _ = writeln!(s, "        reduced.limbs[i] = r.x;");
    let _ = writeln!(s, "        borrow = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return fr_select(borrow == 0, reduced, a);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_add
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_add(Fr a, Fr b) {{");
    let _ = writeln!(s, "    Fr result;");
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        uint2 r = adc(a.limbs[i], b.limbs[i], carry);");
    let _ = writeln!(s, "        result.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    Fr reduced;");
    let _ = writeln!(s, "    uint borrow = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(
        s,
        "        uint2 r = sbb(result.limbs[i], FR_MODULUS[i], borrow);"
    );
    let _ = writeln!(s, "        reduced.limbs[i] = r.x;");
    let _ = writeln!(s, "        borrow = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return fr_select(carry >= borrow, reduced, result);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_sub
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_sub(Fr a, Fr b) {{");
    let _ = writeln!(s, "    Fr result;");
    let _ = writeln!(s, "    uint borrow = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        uint2 r = sbb(a.limbs[i], b.limbs[i], borrow);");
    let _ = writeln!(s, "        result.limbs[i] = r.x;");
    let _ = writeln!(s, "        borrow = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    Fr corrected;");
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        uint addend = borrow ? FR_MODULUS[i] : 0u;");
    let _ = writeln!(s, "        uint2 r = adc(result.limbs[i], addend, carry);");
    let _ = writeln!(s, "        corrected.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return corrected;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_neg
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_neg(Fr a) {{");
    let _ = writeln!(s, "    Fr zero;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) zero.limbs[i] = 0;");
    let _ = writeln!(s, "    return fr_sub(zero, a);");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

fn generate_cios_macro(s: &mut String, n: usize) {
    let _ = writeln!(s, "#define FR_CIOS_ROUND(T, a, b_j, T{n}_out) \\");
    let _ = writeln!(s, "{{ \\");
    let _ = writeln!(s, "    uint _carry = 0; \\");
    let _ = writeln!(s, "    for (int _i = 0; _i < {n}; _i++) {{ \\");
    let _ = writeln!(s, "        uint _p_lo = a.limbs[_i] * b_j; \\");
    let _ = writeln!(s, "        uint _p_hi = mulhi(a.limbs[_i], b_j); \\");
    let _ = writeln!(s, "        uint _s1 = T[_i] + _p_lo; \\");
    let _ = writeln!(s, "        uint _c1 = uint(_s1 < T[_i]); \\");
    let _ = writeln!(s, "        uint _s2 = _s1 + _carry; \\");
    let _ = writeln!(s, "        uint _c2 = uint(_s2 < _s1); \\");
    let _ = writeln!(s, "        T[_i] = _s2; \\");
    let _ = writeln!(s, "        _carry = _p_hi + _c1 + _c2; \\");
    let _ = writeln!(s, "    }} \\");
    let _ = writeln!(s, "    {{ \\");
    let _ = writeln!(s, "        uint _s = T[{n}] + _carry; \\");
    let _ = writeln!(s, "        T{n}_out = uint(_s < T[{n}]); \\");
    let _ = writeln!(s, "        T[{n}] = _s; \\");
    let _ = writeln!(s, "    }} \\");
    let _ = writeln!(s, "    uint _m = T[0] * FR_INV32; \\");
    let _ = writeln!(s, "    {{ \\");
    let _ = writeln!(s, "        uint _m_hi = mulhi(_m, (uint)FR_MODULUS[0]); \\");
    let _ = writeln!(s, "        uint _s = T[0] + _m * (uint)FR_MODULUS[0]; \\");
    let _ = writeln!(s, "        _carry = _m_hi + uint(_s < T[0]); \\");
    let _ = writeln!(s, "    }} \\");
    let n_minus_1 = n - 1;
    let _ = writeln!(s, "    for (int _i = 1; _i < {n}; _i++) {{ \\");
    let _ = writeln!(s, "        uint _p_lo = _m * (uint)FR_MODULUS[_i]; \\");
    let _ = writeln!(
        s,
        "        uint _p_hi = mulhi(_m, (uint)FR_MODULUS[_i]); \\"
    );
    let _ = writeln!(s, "        uint _s1 = T[_i] + _p_lo; \\");
    let _ = writeln!(s, "        uint _c1 = uint(_s1 < T[_i]); \\");
    let _ = writeln!(s, "        uint _s2 = _s1 + _carry; \\");
    let _ = writeln!(s, "        uint _c2 = uint(_s2 < _s1); \\");
    let _ = writeln!(s, "        T[_i - 1] = _s2; \\");
    let _ = writeln!(s, "        _carry = _p_hi + _c1 + _c2; \\");
    let _ = writeln!(s, "    }} \\");
    let _ = writeln!(s, "    {{ \\");
    let _ = writeln!(s, "        uint _s = T[{n}] + _carry; \\");
    let _ = writeln!(s, "        uint _c = uint(_s < T[{n}]); \\");
    let _ = writeln!(s, "        T[{n_minus_1}] = _s; \\");
    let _ = writeln!(s, "        T[{n}] = T{n}_out + _c; \\");
    let _ = writeln!(s, "    }} \\");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

fn generate_fr_mul(s: &mut String, n: usize) {
    // fr_mul_unreduced: N unrolled CIOS rounds
    let t_len = n + 1;
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_mul_unreduced(Fr a, Fr b) {{");
    let _ = write!(s, "    uint T[{t_len}] = {{");
    for i in 0..t_len {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        let _ = write!(s, "0");
    }
    let _ = writeln!(s, "}};");
    let _ = writeln!(s, "    uint T{n};");
    for j in 0..n {
        let _ = writeln!(s, "    FR_CIOS_ROUND(T, a, b.limbs[{j}], T{n});");
    }
    let _ = writeln!(s, "    Fr result;");
    let _ = writeln!(
        s,
        "    for (int i = 0; i < {n}; i++) result.limbs[i] = T[i];"
    );
    let _ = writeln!(s, "    return result;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_mul
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_mul(Fr a, Fr b) {{");
    let _ = writeln!(s, "    return fr_reduce(fr_mul_unreduced(a, b));");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_sqr
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_sqr(Fr a) {{");
    let _ = writeln!(s, "    return fr_mul(a, a);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_zero
    let _ = writeln!(s, "inline Fr fr_zero() {{");
    let _ = writeln!(s, "    Fr z;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) z.limbs[i] = 0;");
    let _ = writeln!(s, "    return z;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_one
    let _ = writeln!(s, "inline Fr fr_one() {{");
    let _ = writeln!(s, "    Fr o;");
    let _ = writeln!(
        s,
        "    for (int i = 0; i < {n}; i++) o.limbs[i] = FR_ONE[i];"
    );
    let _ = writeln!(s, "    return o;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_to_mont
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_to_mont(Fr a) {{");
    let _ = writeln!(s, "    Fr r2;");
    let _ = writeln!(
        s,
        "    for (int i = 0; i < {n}; i++) r2.limbs[i] = FR_R2[i];"
    );
    let _ = writeln!(s, "    return fr_mul(a, r2);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_from_mont
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_from_mont(Fr a) {{");
    let _ = writeln!(s, "    Fr one_std;");
    let _ = writeln!(s, "    one_std.limbs[0] = 1;");
    for i in 1..n {
        let _ = writeln!(s, "    one_std.limbs[{i}] = 0;");
    }
    let _ = writeln!(s, "    return fr_reduce(fr_mul(a, one_std));");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_from_u64
    let _ = writeln!(s, "FR_FUNC_ATTR Fr fr_from_u64(ulong val) {{");
    let _ = writeln!(s, "    Fr a;");
    let _ = writeln!(s, "    a.limbs[0] = uint(val);");
    let _ = writeln!(s, "    a.limbs[1] = uint(val >> 32);");
    for i in 2..n {
        let _ = writeln!(s, "    a.limbs[{i}] = 0;");
    }
    let _ = writeln!(s, "    return fr_to_mont(a);");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // fr_eq
    let _ = writeln!(s, "inline bool fr_eq(Fr a, Fr b) {{");
    let _ = writeln!(s, "    Fr ar = fr_reduce(a);");
    let _ = writeln!(s, "    Fr br = fr_reduce(b);");
    let _ = writeln!(s, "    bool eq = true;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        eq = eq && (ar.limbs[i] == br.limbs[i]);");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return eq;");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

fn generate_wide_acc(s: &mut String, n: usize, acc_n: usize) {
    let _ = writeln!(s, "constant int ACC_LIMBS = {acc_n};");
    s.push('\n');

    let _ = writeln!(s, "struct WideAcc {{");
    let _ = writeln!(s, "    uint limbs[ACC_LIMBS];");
    let _ = writeln!(s, "}};");
    s.push('\n');

    // acc_zero
    let _ = writeln!(s, "inline WideAcc acc_zero() {{");
    let _ = writeln!(s, "    WideAcc a;");
    let _ = writeln!(s, "    for (int i = 0; i < ACC_LIMBS; i++) a.limbs[i] = 0;");
    let _ = writeln!(s, "    return a;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_add_fr
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_add_fr(thread WideAcc &acc, Fr a) {{"
    );
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        uint2 r = adc(acc.limbs[i], a.limbs[i], carry);");
    let _ = writeln!(s, "        acc.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    for (int i = {n}; i < ACC_LIMBS && carry; i++) {{");
    let _ = writeln!(s, "        uint2 r = adc(acc.limbs[i], 0u, carry);");
    let _ = writeln!(s, "        acc.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_add_limbs (internal helper)
    let _ = writeln!(s, "inline void acc_add_limbs(thread WideAcc &acc, const thread uint *src, int count, int offset) {{");
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < count; i++) {{");
    let _ = writeln!(
        s,
        "        uint2 r = adc(acc.limbs[offset + i], src[i], carry);"
    );
    let _ = writeln!(s, "        acc.limbs[offset + i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(
        s,
        "    for (int i = offset + count; carry != 0 && i < ACC_LIMBS; i++) {{"
    );
    let _ = writeln!(s, "        uint2 r = adc(acc.limbs[i], 0u, carry);");
    let _ = writeln!(s, "        acc.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_merge (thread overload)
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_merge(thread WideAcc &dst, WideAcc src) {{"
    );
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < ACC_LIMBS; i++) {{");
    let _ = writeln!(
        s,
        "        uint2 r = adc(dst.limbs[i], src.limbs[i], carry);"
    );
    let _ = writeln!(s, "        dst.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_merge (threadgroup overload)
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_merge(threadgroup WideAcc &dst, threadgroup WideAcc &src) {{"
    );
    let _ = writeln!(s, "    uint carry = 0;");
    let _ = writeln!(s, "    for (int i = 0; i < ACC_LIMBS; i++) {{");
    let _ = writeln!(
        s,
        "        uint2 r = adc(dst.limbs[i], src.limbs[i], carry);"
    );
    let _ = writeln!(s, "        dst.limbs[i] = r.x;");
    let _ = writeln!(s, "        carry = r.y;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_reduce: Montgomery reduction of wide accumulator
    let overflow_start = 2 * n; // limbs[2N] and limbs[2N+1]
    let _ = writeln!(s, "FR_FUNC_ATTR Fr acc_reduce(WideAcc acc) {{");
    let _ = writeln!(s, "    for (int round = 0; round < {n}; round++) {{");
    let _ = writeln!(s, "        uint m = acc.limbs[round] * FR_INV32;");
    let _ = writeln!(s, "        uint m_hi = mulhi(m, (uint)FR_MODULUS[0]);");
    let _ = writeln!(
        s,
        "        uint s0 = acc.limbs[round] + m * (uint)FR_MODULUS[0];"
    );
    let _ = writeln!(
        s,
        "        uint carry = m_hi + uint(s0 < acc.limbs[round]);"
    );
    let _ = writeln!(s, "        for (int i = 1; i < {n}; i++) {{");
    let _ = writeln!(s, "            uint p_lo = m * (uint)FR_MODULUS[i];");
    let _ = writeln!(s, "            uint p_hi = mulhi(m, (uint)FR_MODULUS[i]);");
    let _ = writeln!(s, "            uint s1 = acc.limbs[round + i] + p_lo;");
    let _ = writeln!(s, "            uint c1 = uint(s1 < acc.limbs[round + i]);");
    let _ = writeln!(s, "            uint s2 = s1 + carry;");
    let _ = writeln!(s, "            uint c2 = uint(s2 < s1);");
    let _ = writeln!(s, "            acc.limbs[round + i] = s2;");
    let _ = writeln!(s, "            carry = p_hi + c1 + c2;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        int k = round + {n};");
    let _ = writeln!(s, "        while (carry != 0 && k < ACC_LIMBS) {{");
    let _ = writeln!(s, "            uint old = acc.limbs[k];");
    let _ = writeln!(s, "            uint sv = old + carry;");
    let _ = writeln!(s, "            carry = uint(sv < old);");
    let _ = writeln!(s, "            acc.limbs[k] = sv;");
    let _ = writeln!(s, "            k++;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Repeated subtraction to fully reduce
    let _ = writeln!(s, "    while (true) {{");
    let _ = writeln!(
        s,
        "        bool gte = (acc.limbs[{overflow_start}] != 0) || (acc.limbs[{}] != 0);",
        overflow_start + 1
    );
    let _ = writeln!(s, "        if (!gte) {{");
    let _ = writeln!(s, "            gte = true;");
    let _ = writeln!(s, "            for (int i = {}; i >= 0; i--) {{", n - 1);
    let _ = writeln!(
        s,
        "                if (acc.limbs[i + {n}] > FR_MODULUS[i]) break;"
    );
    let _ = writeln!(
        s,
        "                if (acc.limbs[i + {n}] < FR_MODULUS[i]) {{ gte = false; break; }}"
    );
    let _ = writeln!(s, "            }}");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        if (!gte) break;");
    let _ = writeln!(s, "        uint borrow = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(
        s,
        "            uint2 d = sbb(acc.limbs[i + {n}], FR_MODULUS[i], borrow);"
    );
    let _ = writeln!(s, "            acc.limbs[i + {n}] = d.x;");
    let _ = writeln!(s, "            borrow = d.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(
        s,
        "        for (int i = {overflow_start}; i < ACC_LIMBS; i++) {{"
    );
    let _ = writeln!(s, "            uint2 d = sbb(acc.limbs[i], 0, borrow);");
    let _ = writeln!(s, "            acc.limbs[i] = d.x;");
    let _ = writeln!(s, "            borrow = d.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    s.push('\n');

    let _ = writeln!(s, "    Fr result;");
    let _ = writeln!(s, "    for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "        result.limbs[i] = acc.limbs[i + {n}];");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return result;");
    let _ = writeln!(s, "}}");
    s.push('\n');

    // acc_reduce_tg
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR Fr acc_reduce_tg(threadgroup WideAcc &tg_acc) {{"
    );
    let _ = writeln!(s, "    WideAcc acc;");
    let _ = writeln!(
        s,
        "    for (int i = 0; i < ACC_LIMBS; i++) acc.limbs[i] = tg_acc.limbs[i];"
    );
    let _ = writeln!(s, "    return acc_reduce(acc);");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

fn generate_acc_fmadd(s: &mut String, n: usize, _acc_n: usize) {
    let half = n / 2;
    if n == 8 {
        // N=8: Karatsuba on 4-limb halves (3 × 4×4 schoolbook = 48 mul+mulhi)
        generate_schoolbook_4x4(s);
        generate_schoolbook_4x4_wide(s);
        generate_karatsuba_fmadd(s, n, half);
    } else if n == 4 {
        // N=4: Direct 4×4 schoolbook into accumulator
        generate_schoolbook_4x4(s);
        generate_direct_4x4_fmadd(s);
    } else {
        // Generic N: NxN schoolbook
        generate_generic_schoolbook_fmadd(s, n);
    }
}

fn generate_schoolbook_4x4(s: &mut String) {
    let _ = writeln!(s, "inline void schoolbook_4x4(thread uint (&out)[8], const thread uint *a, const thread uint *b) {{");
    let _ = writeln!(s, "    for (int j = 0; j < 4; j++) {{");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < 4; i++) {{");
    let _ = writeln!(s, "            int idx = i + j;");
    let _ = writeln!(s, "            uint p_lo = a[i] * b[j];");
    let _ = writeln!(s, "            uint p_hi = mulhi(a[i], b[j]);");
    let _ = writeln!(s, "            uint s1 = out[idx] + p_lo;");
    let _ = writeln!(s, "            uint c1 = uint(s1 < out[idx]);");
    let _ = writeln!(s, "            uint s2 = s1 + carry;");
    let _ = writeln!(s, "            uint c2 = uint(s2 < s1);");
    let _ = writeln!(s, "            out[idx] = s2;");
    let _ = writeln!(s, "            carry = p_hi + c1 + c2;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(
        s,
        "        for (int k = j + 4; carry != 0 && k < 8; k++) {{"
    );
    let _ = writeln!(s, "            uint old = out[k];");
    let _ = writeln!(s, "            out[k] = old + carry;");
    let _ = writeln!(s, "            carry = uint(out[k] < old);");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

fn generate_schoolbook_4x4_wide(s: &mut String) {
    let _ = writeln!(s, "inline void schoolbook_4x4_wide(thread uint (&out)[9], const thread uint *a, const thread uint *b) {{");
    let _ = writeln!(s, "    for (int j = 0; j < 4; j++) {{");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < 4; i++) {{");
    let _ = writeln!(s, "            int idx = i + j;");
    let _ = writeln!(s, "            uint p_lo = a[i] * b[j];");
    let _ = writeln!(s, "            uint p_hi = mulhi(a[i], b[j]);");
    let _ = writeln!(s, "            uint s1 = out[idx] + p_lo;");
    let _ = writeln!(s, "            uint c1 = uint(s1 < out[idx]);");
    let _ = writeln!(s, "            uint s2 = s1 + carry;");
    let _ = writeln!(s, "            uint c2 = uint(s2 < s1);");
    let _ = writeln!(s, "            out[idx] = s2;");
    let _ = writeln!(s, "            carry = p_hi + c1 + c2;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(
        s,
        "        for (int k = j + 4; carry != 0 && k < 9; k++) {{"
    );
    let _ = writeln!(s, "            uint old = out[k];");
    let _ = writeln!(s, "            out[k] = old + carry;");
    let _ = writeln!(s, "            carry = uint(out[k] < old);");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

/// Karatsuba fmadd for N=8: splits into 4-limb halves.
fn generate_karatsuba_fmadd(s: &mut String, n: usize, half: usize) {
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_fmadd(thread WideAcc &acc, Fr a, Fr b) {{"
    );

    // Step 1: aS = aL + aH, bS = bL + bH
    let _ = writeln!(s, "    uint aS[{half}], bS[{half}];");
    let _ = writeln!(s, "    uint aC, bC;");
    let _ = writeln!(s, "    {{");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {half}; i++) {{");
    let _ = writeln!(
        s,
        "            uint2 r = adc(a.limbs[i], a.limbs[i + {half}], carry);"
    );
    let _ = writeln!(s, "            aS[i] = r.x;");
    let _ = writeln!(s, "            carry = r.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        aC = carry;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    {{");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {half}; i++) {{");
    let _ = writeln!(
        s,
        "            uint2 r = adc(b.limbs[i], b.limbs[i + {half}], carry);"
    );
    let _ = writeln!(s, "            bS[i] = r.x;");
    let _ = writeln!(s, "            carry = r.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        bC = carry;");
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Step 2: Pm = aS * bS (4x4 into 9 limbs) + carry bit cross terms
    let _ = writeln!(s, "    uint Pm[9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};");
    let _ = writeln!(s, "    schoolbook_4x4_wide(Pm, aS, bS);");
    s.push('\n');

    // Pm += aC * bS at offset 4
    let _ = writeln!(s, "    {{");
    let _ = writeln!(s, "        uint mask = uint(-int(aC));");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {half}; i++) {{");
    let _ = writeln!(
        s,
        "            uint2 r = adc(Pm[i + {half}], bS[i] & mask, carry);"
    );
    let _ = writeln!(s, "            Pm[i + {half}] = r.x;");
    let _ = writeln!(s, "            carry = r.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        Pm[{n}] += carry;");
    let _ = writeln!(s, "    }}");

    // Pm += bC * aS at offset 4
    let _ = writeln!(s, "    {{");
    let _ = writeln!(s, "        uint mask = uint(-int(bC));");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {half}; i++) {{");
    let _ = writeln!(
        s,
        "            uint2 r = adc(Pm[i + {half}], aS[i] & mask, carry);"
    );
    let _ = writeln!(s, "            Pm[i + {half}] = r.x;");
    let _ = writeln!(s, "            carry = r.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        Pm[{n}] += carry;");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    Pm[{n}] += aC & bC;");
    s.push('\n');

    // Step 3: P0 = aL * bL, Pm -= P0, acc += P0 at offset 0
    let _ = writeln!(s, "    {{");
    let _ = writeln!(
        s,
        "        uint P0[{n}] = {{{}}};",
        "0, ".repeat(n).trim_end_matches(", ")
    );
    let _ = writeln!(s, "        schoolbook_4x4(P0, a.limbs, b.limbs);");
    let _ = writeln!(s, "        uint borrow = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "            uint2 d = sbb(Pm[i], P0[i], borrow);");
    let _ = writeln!(s, "            Pm[i] = d.x;");
    let _ = writeln!(s, "            borrow = d.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        Pm[{n}] -= borrow;");
    let _ = writeln!(s, "        acc_add_limbs(acc, P0, {n}, 0);");
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Step 4: P2 = aH * bH, Pm -= P2, acc += P2 at offset N
    let _ = writeln!(s, "    {{");
    let _ = writeln!(
        s,
        "        uint P2[{n}] = {{{}}};",
        "0, ".repeat(n).trim_end_matches(", ")
    );
    let _ = writeln!(
        s,
        "        schoolbook_4x4(P2, &a.limbs[{half}], &b.limbs[{half}]);"
    );
    let _ = writeln!(s, "        uint borrow = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "            uint2 d = sbb(Pm[i], P2[i], borrow);");
    let _ = writeln!(s, "            Pm[i] = d.x;");
    let _ = writeln!(s, "            borrow = d.y;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "        Pm[{n}] -= borrow;");
    let _ = writeln!(s, "        acc_add_limbs(acc, P2, {n}, {n});");
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Step 5: acc += Pm at offset half
    let _ = writeln!(s, "    acc_add_limbs(acc, Pm, 9, {half});");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

/// Direct 4×4 fmadd for N=4.
fn generate_direct_4x4_fmadd(s: &mut String) {
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_fmadd(thread WideAcc &acc, Fr a, Fr b) {{"
    );
    let _ = writeln!(s, "    uint prod[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};");
    let _ = writeln!(s, "    schoolbook_4x4(prod, a.limbs, b.limbs);");
    let _ = writeln!(s, "    acc_add_limbs(acc, prod, 8, 0);");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

/// Generic NxN schoolbook fmadd.
fn generate_generic_schoolbook_fmadd(s: &mut String, n: usize) {
    let prod_limbs = 2 * n;
    let _ = writeln!(
        s,
        "FR_FUNC_ATTR void acc_fmadd(thread WideAcc &acc, Fr a, Fr b) {{"
    );
    let _ = write!(s, "    uint prod[{prod_limbs}] = {{");
    for i in 0..prod_limbs {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        let _ = write!(s, "0");
    }
    let _ = writeln!(s, "}};");
    let _ = writeln!(s, "    for (int j = 0; j < {n}; j++) {{");
    let _ = writeln!(s, "        uint carry = 0;");
    let _ = writeln!(s, "        for (int i = 0; i < {n}; i++) {{");
    let _ = writeln!(s, "            int idx = i + j;");
    let _ = writeln!(s, "            uint p_lo = a.limbs[i] * b.limbs[j];");
    let _ = writeln!(s, "            uint p_hi = mulhi(a.limbs[i], b.limbs[j]);");
    let _ = writeln!(s, "            uint s1 = prod[idx] + p_lo;");
    let _ = writeln!(s, "            uint c1 = uint(s1 < prod[idx]);");
    let _ = writeln!(s, "            uint s2 = s1 + carry;");
    let _ = writeln!(s, "            uint c2 = uint(s2 < s1);");
    let _ = writeln!(s, "            prod[idx] = s2;");
    let _ = writeln!(s, "            carry = p_hi + c1 + c2;");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(
        s,
        "        for (int k = j + {n}; carry != 0 && k < {prod_limbs}; k++) {{"
    );
    let _ = writeln!(s, "            uint old = prod[k];");
    let _ = writeln!(s, "            prod[k] = old + carry;");
    let _ = writeln!(s, "            carry = uint(prod[k] < old);");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    acc_add_limbs(acc, prod, {prod_limbs}, 0);");
    let _ = writeln!(s, "}}");
    s.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn preamble_contains_expected_symbols() {
        let preamble = generate_full_preamble::<Fr>();
        assert!(preamble.contains("struct Fr {"));
        assert!(preamble.contains("FR_MODULUS"));
        assert!(preamble.contains("FR_INV32"));
        assert!(preamble.contains("fr_mul"));
        assert!(preamble.contains("fr_add"));
        assert!(preamble.contains("fr_sub"));
        assert!(preamble.contains("fr_neg"));
        assert!(preamble.contains("fr_reduce"));
        assert!(preamble.contains("FR_CIOS_ROUND"));
        assert!(preamble.contains("WideAcc"));
        assert!(preamble.contains("acc_fmadd"));
        assert!(preamble.contains("acc_reduce"));
        assert!(preamble.contains("acc_merge"));
        assert!(preamble.contains("schoolbook_4x4"));
    }

    #[test]
    fn test_kernels_contain_expected_functions() {
        let kernels = generate_test_kernels::<Fr>();
        assert!(kernels.contains("fr_mul_kernel"));
        assert!(kernels.contains("fr_add_kernel"));
        assert!(kernels.contains("fr_sub_kernel"));
        assert!(kernels.contains("fr_sqr_kernel"));
        assert!(kernels.contains("fr_neg_kernel"));
        assert!(kernels.contains("fr_fmadd_kernel"));
        assert!(kernels.contains("fr_from_u64_kernel"));
    }

    #[test]
    fn bn254_constants_in_preamble() {
        let preamble = generate_full_preamble::<Fr>();
        // Verify the BN254 modulus appears in the generated MSL
        assert!(preamble.contains("0xf0000001u"));
        assert!(preamble.contains("0x30644e72u"));
        // Verify INV32
        assert!(preamble.contains("0xefffffffu"));
    }
}
