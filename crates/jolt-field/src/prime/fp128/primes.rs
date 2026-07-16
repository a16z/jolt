use super::*;

/// `p = 2^128 − 275`  (C = 275).
pub type Prime128Offset275 = Fp128<0xfffffffffffffffffffffffffffffeed>;
/// `p = 2^128 − 159`  (C = 159). Split-NTT-only helper prime.
pub type Prime128Offset159 = Fp128<0xffffffffffffffffffffffffffffff61>;
/// `p = 2^128 − 2355`  (C = 2355, p ≡ 5 mod 8).
///
/// Smooth multiplicative subgroup of order 14700 = 2² × 3 × 5² × 7²,
/// supporting mixed-radix FFT up to size 14700 (e.g. 1470 = 2·3·5·7²
/// for RS encoding with 256+1024 ≥ 1280 evaluations).
///
/// Factorization: `p − 1 = 2² · 3 · 5² · 7² · 701 · 2955365183 · 11173595356596918495491`.
pub type Prime128Offset2355 = Fp128<0xfffffffffffffffffffffffffffff6cd>;

impl SmoothFftField for Prime128Offset2355 {
    const SMOOTH_SUBGROUP_ORDER: usize = 14_700;
    /// `2 ^ ((p − 1) / 14_700)` where `g = 2` is a primitive root of `p`.
    /// Verified by `prime_2355_tests::smooth_omega_matches_search` in
    /// `src/fft.rs`.
    const SMOOTH_OMEGA: u128 = 0x2ecd_18d0_8238_2c0c_818c_c05f_446a_8075;
}

/// `p = 2^128 − 2^32 + 22537`  (C = 2^32 − 22537 = 0xFFFFA7F7).
///
/// Solinas-form prime sharing the same CPU reduction cost as
/// `Prime128Offset2355` on x86_64 / AArch64 (both go through the generic
/// 32-bit-C `mul_c_wide` path; neither C is of the form `2^a ± 1`). The
/// multiplicative group contains a smooth subgroup of order
/// `2^3 · 3^7 = 17 496` with a pure radix-3 subgroup of order
/// `3^7 = 2187`, enabling a low-mul mixed-radix FFT.
///
/// Factorization of `p − 1` includes `2^3 · 3^7 · 19 · 41 · 459 647 · …`.
///
/// Subgroup sizes available for FFT-based RS encoding include
/// `1458 = 2 · 3^6`, `2187 = 3^7`, `4374 = 2 · 3^7`, `8748 = 2^2 · 3^7`,
/// and the full `17 496 = 2^3 · 3^7`.
pub type Prime128OffsetA7F7 = Fp128<0xffffffffffffffffffffffff00005809>;

impl SmoothFftField for Prime128OffsetA7F7 {
    const SMOOTH_SUBGROUP_ORDER: usize = 17_496;
    /// `g ^ ((p − 1) / 17_496)` where `g` is the smallest primitive root
    /// found by `find_primitive_nth_root` (note: `g = 2` is a quadratic
    /// residue mod `p` and therefore *not* a primitive root, so the
    /// scanner falls through to the next candidate). Verified by
    /// `prime_a7f7_tests::smooth_omega_matches_search` in
    /// `src/fft.rs`.
    const SMOOTH_OMEGA: u128 = 0x4e9f_650b_7003_d201_9945_e1da_c47c_8b18;
}
