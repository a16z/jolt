constant uint diagnostic_tile_positions [[function_constant(20)]];

inline uint4 diagnostic_gather_word(
    threadgroup const uint *matrix, uint word, uint matrix_base, uint4 sources)
{
    uint plane_base = word * diagnostic_tile_positions * 128u;
    return uint4(
        matrix[plane_base + matrix_base + sources[0]],
        matrix[plane_base + matrix_base + sources[1]],
        matrix[plane_base + matrix_base + sources[2]],
        matrix[plane_base + matrix_base + sources[3]]);
}

inline void diagnostic_accumulate_mixed(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix, uint matrix_base, uint4 sources, bool4 positive)
{
    akita_fp128_d512_accumulate_value(
        accumulator,
        diagnostic_gather_word(matrix, 0u, matrix_base, sources),
        diagnostic_gather_word(matrix, 1u, matrix_base, sources),
        diagnostic_gather_word(matrix, 2u, matrix_base, sources),
        diagnostic_gather_word(matrix, 3u, matrix_base, sources),
        positive);
}
