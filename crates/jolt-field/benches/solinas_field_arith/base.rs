use criterion::Criterion;
use jolt_field::{
    Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59, Prime56Offset27,
    Prime64Offset59,
};

use super::arithmetic::bench_arithmetic_case;
use super::cases::*;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_base_field_matrix(c: &mut Criterion) {
    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_BASE_ARITH", 2048, 256);

    bench_arithmetic_case::<Prime31Offset19, P31O19>(
        c,
        "base",
        PRIME31_OFFSET19,
        0xba5e_0031,
        params,
    );
    bench_arithmetic_case::<Mersenne31, PackedMersenne31>(
        c,
        "base",
        MERSENNE31,
        0xba5e_3131,
        params,
    );
    bench_arithmetic_case::<Prime32Offset99, P32O99>(
        c,
        "base",
        PRIME32_OFFSET99,
        0xba5e_0032,
        params,
    );
    bench_arithmetic_case::<Prime40Offset195, P40O195>(
        c,
        "base",
        PRIME40_OFFSET195,
        0xba5e_0040,
        params,
    );
    bench_arithmetic_case::<Prime48Offset59, P48O59>(
        c,
        "base",
        PRIME48_OFFSET59,
        0xba5e_0048,
        params,
    );
    bench_arithmetic_case::<Prime56Offset27, P56O27>(
        c,
        "base",
        PRIME56_OFFSET27,
        0xba5e_0056,
        params,
    );
    bench_arithmetic_case::<Prime64Offset59, P64O59>(
        c,
        "base",
        PRIME64_OFFSET59,
        0xba5e_0064,
        params,
    );
    bench_arithmetic_case::<F128, P128O275>(c, "base", PRIME128_OFFSET275, 0xba5e_0128, params);
}
