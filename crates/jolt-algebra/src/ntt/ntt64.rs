//! Contract for a guest 64-point `i32` NTT backend (the `ntt64-backend`
//! feature), registered with [`ntt64_backend!`](crate::ntt64_backend).
//!
//! On `riscv64`, for `i32` CRT+NTT parameter sets with `D = 64`, the feature
//! routes exactly two kernels to functions that a downstream crate supplies
//! at link time, typically wrappers around zkVM inlines that this crate
//! cannot depend on: the twisted forward negacyclic NTT (including the
//! centered-digit conversion) and the lazy Montgomery pointwise dot. The
//! inverse and cyclic transforms and every other kernel keep the scalar
//! path. Elsewhere the feature selects nothing and the portable or SIMD
//! kernels run unchanged.

use super::prime::I32_LAZY_DOT_BATCH;

/// Coefficients per transform the backend serves.
pub const DEGREE: usize = 64;

/// Products in one backend pointwise dot: the `i32` lazy-dot batch.
pub const DOT_PRODUCTS: usize = I32_LAZY_DOT_BATCH;

/// Register the 64-point forward NTT and lazy pointwise dot that the
/// `ntt64-backend` feature links against: `ntt64_backend!(forward, pointwise_dot)`.
/// No inverse transform is registered; the crate's inverse NTT stays scalar.
///
/// Expands to the two unmangled `extern "Rust"` symbols the feature's
/// riscv64 kernels call, `__jolt_algebra_ntt64_forward` and
/// `__jolt_algebra_ntt64_pointwise_dot`. Invoke it exactly once in the final
/// program; a riscv64 build with the feature and no registration fails to
/// link, and a second registration is a duplicate-symbol error. The paths
/// must name functions with these signatures and semantics:
///
/// ```text
/// forward(state: &mut [i32; 64], psi: &[i32; 64], twiddles: &[i32; 64], p: i32, pinv: i32)
/// ```
///
/// Twists `state` by `psi` and runs a forward Gentleman-Sande DIF NTT in
/// place, with bit-reversed output. Every multiplication is a Montgomery
/// product modulo `p` with `R = 2^32`, so Montgomery-form tables preserve the
/// input's representation. `psi[i]` is a twist factor for coefficient `i`;
/// stage twiddles occupy `twiddles[len - 1..2 * len - 1]` for half-lengths 32
/// down to 1, and `twiddles[0]` is the Montgomery form of 1, which reduces
/// butterfly sums. Output residues lie in `(-p, p)`.
///
/// ```text
/// pointwise_dot(acc: &mut [i32; 64], lhs: [&[i32; 64]; 6], rhs: [&[i32; 64]; 6], p: i32, pinv: i32)
/// ```
///
/// Computes `acc[i] += Σ_j lhs[j][i] · rhs[j][i] · 2^-32 mod p`, leaving
/// residues in `(-p, p)`. Short dots arrive padded with zero rows.
///
/// Domain for both: odd `p` with `0 < p < 2^30`, `pinv = p^-1 mod 2^32`, and
/// inputs and table entries in `(-p, p)`. Arrays are 4-byte aligned; this
/// crate stores the dot's zero rows and, where the destination allows, the
/// transform's packed input at 8-byte alignment, but the functions must
/// accept any `i32` alignment. `jolt-inlines-ntt`'s `forward_ntt64` and
/// `pointwise_dot64` satisfy this contract.
#[macro_export]
macro_rules! ntt64_backend {
    ($forward:path, $pointwise_dot:path $(,)?) => {
        const _: () = {
            #[unsafe(no_mangle)]
            extern "Rust" fn __jolt_algebra_ntt64_forward(
                state: &mut [i32; $crate::ntt::ntt64::DEGREE],
                psi: &[i32; $crate::ntt::ntt64::DEGREE],
                twiddles: &[i32; $crate::ntt::ntt64::DEGREE],
                p: i32,
                pinv: i32,
            ) {
                $forward(state, psi, twiddles, p, pinv);
            }

            #[unsafe(no_mangle)]
            extern "Rust" fn __jolt_algebra_ntt64_pointwise_dot(
                acc: &mut [i32; $crate::ntt::ntt64::DEGREE],
                lhs: [&[i32; $crate::ntt::ntt64::DEGREE]; $crate::ntt::ntt64::DOT_PRODUCTS],
                rhs: [&[i32; $crate::ntt::ntt64::DEGREE]; $crate::ntt::ntt64::DOT_PRODUCTS],
                p: i32,
                pinv: i32,
            ) {
                $pointwise_dot(acc, lhs, rhs, p, pinv);
            }
        };
    };
}
