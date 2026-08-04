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

inline SolinasFp128 instruction_ra_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
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
        SolinasFp128 sum = instruction_ra_simd_sum(lanes[sample]);
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
            sum = instruction_ra_simd_sum(sum);
            if (lane_in_simd == 0) {
                partials[sample * e_out_length + x_out] =
                    solinas_mul_wide(e_out, sum);
            }
        }
    }
}

kernel void solinas_instruction_ra_first_message(
    device const InstructionRaLookup* lookups [[buffer(0)]],
    device const uint* cycle_to_table_major [[buffer(1)]],
    device const SolinasFp128* chunk_tables [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant InstructionRaFirstMessageParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        uint lo_row = cycle_to_table_major[2u * pair];
        uint hi_row = cycle_to_table_major[2u * pair + 1u];
        InstructionRaLookup lo_lookup = lookups[lo_row];
        InstructionRaLookup hi_lookup = lookups[hi_row];

        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            q[sample] = solinas_zero();
        }
        for (uint group = 0; group < INSTRUCTION_RA_GROUPS; group++) {
            instruction_ra_accumulate_group(
                group,
                lo_lookup,
                hi_lookup,
                chunk_tables,
                q);
        }
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    instruction_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32u);
}

kernel void solinas_instruction_ra_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant InstructionRaReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = instruction_ra_simd_sum(value);
        if (lane_in_simd == 0) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
