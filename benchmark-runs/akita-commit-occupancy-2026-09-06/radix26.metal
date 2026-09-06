// D22's representation, cadence and signed bounds are specified in radix26.md.
struct DiagnosticRadix26 {
    int4 d0, d1, d2, d3, d4;
};
static_assert(sizeof(DiagnosticRadix26) == sizeof(AkitaTransposedFp128Accumulator),
    "radix26 retains five source words per coefficient");
static_assert(PACKED_FP128_D128_RANK3_ROWS_PER_TILE == 8u,
    "two tiles must contain at most sixteen selected contributions");

inline DiagnosticRadix26 diagnostic_radix26_zero() {
    DiagnosticRadix26 result;
    result.d0 = result.d1 = result.d2 = result.d3 = result.d4 = int4(0);
    return result;
}

inline void diagnostic_radix26_normalize(thread DiagnosticRadix26 &accumulator) {
    constexpr int mask = (1 << 26) - 1;
    int4 carry = accumulator.d0 >> 26;
    accumulator.d0 &= int4(mask);
    accumulator.d1 += carry;
    carry = accumulator.d1 >> 26;
    accumulator.d1 &= int4(mask);
    accumulator.d2 += carry;
    carry = accumulator.d2 >> 26;
    accumulator.d2 &= int4(mask);
    accumulator.d3 += carry;
    carry = accumulator.d3 >> 26;
    accumulator.d3 &= int4(mask);
    accumulator.d4 += carry;
    int4 high = accumulator.d4 >> 24;
    accumulator.d4 &= int4((1 << 24) - 1);
    constexpr int correction_low = int((1ul << 32ul) - ulong(AKITA_OFFSET));
    accumulator.d0 -= high * int4(correction_low);
    accumulator.d1 += high * int4(64);
}

inline void diagnostic_radix26_add(
    thread DiagnosticRadix26 &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool4 positive)
{
    uint4 v0 = akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources);
    uint4 v1 = akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources);
    uint4 v2 = akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources);
    uint4 v3 = akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources);
    uint4 mask = uint4((1u << 26u) - 1u);
    int4 sign = select(int4(-1), int4(1), positive);
    accumulator.d0 += sign * int4(v0 & mask);
    accumulator.d1 += sign * int4(((v0 >> 26u) | (v1 << 6u)) & mask);
    accumulator.d2 += sign * int4(((v1 >> 20u) | (v2 << 12u)) & mask);
    accumulator.d3 += sign * int4(((v2 >> 14u) | (v3 << 18u)) & mask);
    accumulator.d4 += sign * int4(v3 >> 8u);
}

inline AkitaFp128 diagnostic_radix26_reduce(DiagnosticRadix26 accumulator, uint component) {
    constexpr int mask = (1 << 26) - 1;
    int d0 = accumulator.d0[component];
    int d1 = accumulator.d1[component] + (d0 >> 26);
    d0 &= mask;
    int d2 = accumulator.d2[component] + (d1 >> 26);
    d1 &= mask;
    int d3 = accumulator.d3[component] + (d2 >> 26);
    d2 &= mask;
    int d4 = accumulator.d4[component] + (d3 >> 26);
    d3 &= mask;
    uint w0 = uint(d0) | (uint(d1) << 26u);
    uint w1 = (uint(d1) >> 6u) | (uint(d2) << 20u);
    uint w2 = (uint(d2) >> 12u) | (uint(d3) << 14u);
    uint w3 = (uint(d3) >> 18u) | (uint(d4) << 8u);
    AkitaWideAccumulator digits;
    digits.low_digits = int4(w0 & 65535u, w1 & 65535u, w2 & 65535u, w3 & 65535u);
    // The signed top includes the quotient above bit128; do not truncate it.
    digits.high_digits = int4(w0 >> 16u, w1 >> 16u, w2 >> 16u, d4 >> 8);
    return akita_reduce_wide(digits);
}

kernel void diagnostic_radix26_probe(
    device const DiagnosticRadix26 *input [[buffer(0)]],
    device DiagnosticRadix26 *output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    DiagnosticRadix26 value = input[index];
    diagnostic_radix26_normalize(value);
    output[index] = value;
}
