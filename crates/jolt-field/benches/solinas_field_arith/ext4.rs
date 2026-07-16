//! Degree-4 extension microbenches.
//!
//! Criterion directory names are capped at 64 characters (`MAX_DIRECTORY_NAME_LEN`).
//! Use the short `label` strings below (≤ 12 chars before `_w{width}`) so groups are not
//! truncated.

use criterion::Criterion;
use jolt_field::packed::HasPacking;
use jolt_field::{FpExt4, Prime31Offset19, Prime32Offset99};

use super::arithmetic::bench_arithmetic_case;
use super::cases::Mersenne31;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext4_matrix(c: &mut Criterion) {
    type F31Mersenne = Mersenne31;
    type F31MersenneFpExt4 = FpExt4<F31Mersenne>;
    type PF31MersenneFpExt4 = <F31MersenneFpExt4 as HasPacking>::Packing;

    type F31 = Prime31Offset19;
    type F31FpExt4 = FpExt4<F31>;
    type PF31FpExt4 = <F31FpExt4 as HasPacking>::Packing;

    type F32 = Prime32Offset99;
    type F32FpExt4 = FpExt4<F32>;
    type PF32FpExt4 = <F32FpExt4 as HasPacking>::Packing;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT4_ARITH", 512, 128);

    bench_arithmetic_case::<F31MersenneFpExt4, PF31MersenneFpExt4>(
        c,
        "ext4",
        "m31_fp_ext4",
        0xe400_3031_00a1,
        params,
    );

    bench_arithmetic_case::<F31FpExt4, PF31FpExt4>(
        c,
        "ext4",
        "p31o19_fp_ext4",
        0xe400_3031,
        params,
    );

    bench_arithmetic_case::<F32FpExt4, PF32FpExt4>(
        c,
        "ext4",
        "p32o99_fp_ext4",
        0xe400_3032,
        params,
    );
}
