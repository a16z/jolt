use criterion::Criterion;
use jolt_field::packed::{HasPacking, PackedFpExt2};
use jolt_field::{FpExt2, Prime31Offset19, Prime32Offset99, Prime64Offset59, TwoNr};

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext2_matrix(c: &mut Criterion) {
    type F31 = Prime31Offset19;
    type PF31 = <F31 as HasPacking>::Packing;
    type F31FpExt2 = FpExt2<F31, TwoNr>;
    type PF31FpExt2 = PackedFpExt2<F31, TwoNr, PF31>;

    type F32 = Prime32Offset99;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32FpExt2 = FpExt2<F32, TwoNr>;
    type PF32FpExt2 = PackedFpExt2<F32, TwoNr, PF32>;

    type F64 = Prime64Offset59;
    type PF64 = <F64 as HasPacking>::Packing;
    type F64FpExt2 = FpExt2<F64, TwoNr>;
    type PF64FpExt2 = PackedFpExt2<F64, TwoNr, PF64>;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT2_ARITH", 512, 128);

    bench_arithmetic_case::<F31FpExt2, PF31FpExt2>(
        c,
        "ext2",
        "prime31_offset19_fp_ext2",
        0xe200_0031,
        params,
    );
    bench_arithmetic_case::<F32FpExt2, PF32FpExt2>(
        c,
        "ext2",
        "prime32_offset99_fp_ext2",
        0xe200_0032,
        params,
    );
    bench_arithmetic_case::<F64FpExt2, PF64FpExt2>(
        c,
        "ext2",
        "prime64_offset59_fp_ext2",
        0xe200_0064,
        params,
    );
}
