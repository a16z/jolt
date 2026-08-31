#define INSTRUCTION_RA_GROUPS 4u
#define INSTRUCTION_RA_FACTORS_PER_GROUP 4u
#define INSTRUCTION_RA_FACTORS \
    (INSTRUCTION_RA_GROUPS * INSTRUCTION_RA_FACTORS_PER_GROUP)
#define INSTRUCTION_RA_SAMPLES 4u
#define INSTRUCTION_RA_BINS 256u

struct InstructionRaLookup {
    ulong2 limbs;
};

struct InstructionRaFirstMessageParams {
    uint e_in_length;
    uint e_out_length;
    uint2 reserved;
};

struct InstructionRaReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

struct InstructionRaLinear {
    SolinasFp128 at_one;
    SolinasFp128 at_infinity;
};

struct InstructionRaQuadratic {
    SolinasFp128 at_one;
    SolinasFp128 at_two;
    SolinasFp128 at_infinity;
};

inline uint instruction_ra_lookup_byte(InstructionRaLookup lookup, uint factor) {
    uint shift = (INSTRUCTION_RA_FACTORS - 1u - factor) * 8u;
    return shift < 64u
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64u)) & 0xffu;
}

inline InstructionRaLinear instruction_ra_linear(
    uint factor,
    InstructionRaLookup lo_lookup,
    InstructionRaLookup hi_lookup,
    device const SolinasFp128* chunk_tables)
{
    uint table = factor * INSTRUCTION_RA_BINS;
    SolinasFp128 lo = chunk_tables[
        table + instruction_ra_lookup_byte(lo_lookup, factor)];
    SolinasFp128 hi = chunk_tables[
        table + instruction_ra_lookup_byte(hi_lookup, factor)];
    InstructionRaLinear result;
    result.at_one = hi;
    result.at_infinity = solinas_sub(hi, lo);
    return result;
}

inline InstructionRaQuadratic instruction_ra_quadratic(
    InstructionRaLinear lhs,
    InstructionRaLinear rhs)
{
    InstructionRaQuadratic result;
    result.at_one = solinas_mul_wide(lhs.at_one, rhs.at_one);
    result.at_two = solinas_mul_wide(
        solinas_add(lhs.at_one, lhs.at_infinity),
        solinas_add(rhs.at_one, rhs.at_infinity));
    result.at_infinity = solinas_mul_wide(
        lhs.at_infinity,
        rhs.at_infinity);
    return result;
}

inline SolinasFp128 instruction_ra_quadratic_at_three(
    InstructionRaQuadratic value)
{
    SolinasFp128 twice_at_two = solinas_add(value.at_two, value.at_two);
    SolinasFp128 twice_leading = solinas_add(
        value.at_infinity,
        value.at_infinity);
    return solinas_add(solinas_sub(twice_at_two, value.at_one), twice_leading);
}

inline void instruction_ra_accumulate_group(
    uint group,
    InstructionRaLookup lo_lookup,
    InstructionRaLookup hi_lookup,
    device const SolinasFp128* chunk_tables,
    thread SolinasFp128* q)
{
    uint first = group * INSTRUCTION_RA_FACTORS_PER_GROUP;
    InstructionRaLinear f0 = instruction_ra_linear(
        first,
        lo_lookup,
        hi_lookup,
        chunk_tables);
    InstructionRaLinear f1 = instruction_ra_linear(
        first + 1u,
        lo_lookup,
        hi_lookup,
        chunk_tables);
    InstructionRaQuadratic lhs = instruction_ra_quadratic(f0, f1);

    InstructionRaLinear f2 = instruction_ra_linear(
        first + 2u,
        lo_lookup,
        hi_lookup,
        chunk_tables);
    InstructionRaLinear f3 = instruction_ra_linear(
        first + 3u,
        lo_lookup,
        hi_lookup,
        chunk_tables);
    InstructionRaQuadratic rhs = instruction_ra_quadratic(f2, f3);

    q[0] = solinas_add(
        q[0],
        solinas_mul_wide(lhs.at_one, rhs.at_one));
    q[1] = solinas_add(
        q[1],
        solinas_mul_wide(lhs.at_two, rhs.at_two));
    q[2] = solinas_add(
        q[2],
        solinas_mul_wide(
            instruction_ra_quadratic_at_three(lhs),
            instruction_ra_quadratic_at_three(rhs)));
    q[3] = solinas_add(
        q[3],
        solinas_mul_wide(lhs.at_infinity, rhs.at_infinity));
}

inline void instruction_ra_finish_block(
    thread SolinasFp128* lanes,
    SolinasFp128 e_out,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * e_out_length + x_out] =
                    solinas_mul_wide(e_out, sum);
            }
        }
    }
}
