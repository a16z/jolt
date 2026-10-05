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
