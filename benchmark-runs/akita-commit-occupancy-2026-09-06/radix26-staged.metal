#define DIAGNOSTIC_RADIX26_TILE_POSITIONS 8u
#define DIAGNOSTIC_RADIX26_ROWS_PER_TILE 4u
#define DIAGNOSTIC_RADIX26_TILE_ELEMENTS 1024u

static_assert(DIAGNOSTIC_RADIX26_TILE_POSITIONS * PACKED_FP128_D128_RANK3_D
    == DIAGNOSTIC_RADIX26_TILE_ELEMENTS, "decoded tile plane geometry");
static_assert(DIAGNOSTIC_RADIX26_ROWS_PER_TILE * 4u == 16u,
    "four staged tiles retain D22's sixteen-contribution bound");
static_assert(17ul * ((1ul << 26ul) + (1ul << 32ul) - ulong(AKITA_OFFSET))
    < (1ul << 31ul), "radix26 signed intermediates fit i32");

inline void diagnostic_radix26_stage(
    threadgroup uint *matrix, uint index, AkitaFp128 value)
{
    constexpr uint mask = (1u << 26u) - 1u;
    matrix[index] = value.limb[0] & mask;
    matrix[DIAGNOSTIC_RADIX26_TILE_ELEMENTS + index] =
        ((value.limb[0] >> 26u) | (value.limb[1] << 6u)) & mask;
    matrix[DIAGNOSTIC_RADIX26_TILE_ELEMENTS * 2u + index] =
        ((value.limb[1] >> 20u) | (value.limb[2] << 12u)) & mask;
    matrix[DIAGNOSTIC_RADIX26_TILE_ELEMENTS * 3u + index] =
        ((value.limb[2] >> 14u) | (value.limb[3] << 18u)) & mask;
    matrix[DIAGNOSTIC_RADIX26_TILE_ELEMENTS * 4u + index] = value.limb[3] >> 8u;
}

inline int4 diagnostic_radix26_gather(
    threadgroup const uint *matrix, uint digit, uint matrix_base, uint4 sources)
{
    uint base = digit * DIAGNOSTIC_RADIX26_TILE_ELEMENTS + matrix_base;
    return int4(matrix[base + sources[0]], matrix[base + sources[1]],
        matrix[base + sources[2]], matrix[base + sources[3]]);
}

inline void diagnostic_radix26_staged_add(
    thread DiagnosticRadix26 &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool4 positive)
{
    int4 sign = select(int4(-1), int4(1), positive);
    accumulator.d0 += sign * diagnostic_radix26_gather(matrix, 0u, matrix_base, sources);
    accumulator.d1 += sign * diagnostic_radix26_gather(matrix, 1u, matrix_base, sources);
    accumulator.d2 += sign * diagnostic_radix26_gather(matrix, 2u, matrix_base, sources);
    accumulator.d3 += sign * diagnostic_radix26_gather(matrix, 3u, matrix_base, sources);
    accumulator.d4 += sign * diagnostic_radix26_gather(matrix, 4u, matrix_base, sources);
}
