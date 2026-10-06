// Appended after instruction_ra_common.metal and booleanity/shader.metal: this
// file reuses the field ABI, quadratic helpers, and booleanity_lazy_reduce.

struct InstructionRaBranchParams {
    uint branch_width;
    uint3 reserved;
};

struct InstructionRaMaterializeParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

constant uint instruction_ra_wide_branch_width [[function_constant(0)]];

inline void instruction_ra_gather_group(
    uint group,
    uint pair,
    uint branch_width,
    device const InstructionRaLookup* lookups,
    device const uint* cycle_to_table_major,
    device const SolinasFp128* branches,
    thread InstructionRaLinear* factors)
{
    // Factor f reads lookup-index bits 8 * (15 - f).. (chunk 0 is the top byte,
    // as in committed_address_chunks), so factor 4 * group + local is byte
    // 3 - local of the little-endian 32-bit word 3 - group.
    uint word = 3u - group;
    uint original = 2u * pair * branch_width;
    if (branch_width == 1u) {
        uint lo_word = ((device const uint*)(lookups + cycle_to_table_major[original]))[word];
        uint hi_word = ((device const uint*)(lookups + cycle_to_table_major[original + 1u]))[word];
        for (uint local = 0; local < INSTRUCTION_RA_FACTORS_PER_GROUP; local++) {
            uint table = (group * INSTRUCTION_RA_FACTORS_PER_GROUP + local) * INSTRUCTION_RA_BINS;
            uint shift = (3u - local) * 8u;
            factors[local].at_one = branches[table + ((hi_word >> shift) & 0xffu)];
            factors[local].at_infinity = solinas_sub(
                factors[local].at_one,
                branches[table + ((lo_word >> shift) & 0xffu)]);
        }
        return;
    }
    SolinasLazySum lo_sums[INSTRUCTION_RA_FACTORS_PER_GROUP] = {};
    SolinasLazySum hi_sums[INSTRUCTION_RA_FACTORS_PER_GROUP] = {};
    for (uint offset = 0; offset < branch_width; offset++) {
        uint lo_row = cycle_to_table_major[original + offset];
        uint hi_row = cycle_to_table_major[original + branch_width + offset];
        uint lo_word = ((device const uint*)(lookups + lo_row))[word];
        uint hi_word = ((device const uint*)(lookups + hi_row))[word];
        for (uint local = 0; local < INSTRUCTION_RA_FACTORS_PER_GROUP; local++) {
            uint factor = group * INSTRUCTION_RA_FACTORS_PER_GROUP + local;
            uint table = (factor * branch_width + offset) * INSTRUCTION_RA_BINS;
            uint shift = (3u - local) * 8u;
            solinas_lazy_add(lo_sums[local], branches[table + ((lo_word >> shift) & 0xffu)]);
            solinas_lazy_add(hi_sums[local], branches[table + ((hi_word >> shift) & 0xffu)]);
        }
    }

    for (uint local = 0; local < INSTRUCTION_RA_FACTORS_PER_GROUP; local++) {
        factors[local].at_one = booleanity_lazy_reduce(hi_sums[local]);
        factors[local].at_infinity = solinas_sub(
            factors[local].at_one,
            booleanity_lazy_reduce(lo_sums[local]));
    }
}

// Lanes 4j..4j+3 of a SIMD group hold groups 0..3 of one pair, so each lane
// carries one group's four factors instead of all sixteen. A lane's products
// are its group's quartic at every sample; `instruction_ra_lane_sample_total`
// hands lane 4j+s the pair's total at sample s.
inline void instruction_ra_group_products(
    thread const InstructionRaLinear* factors,
    thread SolinasFp128* q)
{
    InstructionRaQuadratic lhs = instruction_ra_quadratic(factors[0], factors[1]);
    InstructionRaQuadratic rhs = instruction_ra_quadratic(factors[2], factors[3]);
    q[0] = solinas_mul_wide(lhs.at_one, rhs.at_one);
    q[1] = solinas_mul_wide(lhs.at_two, rhs.at_two);
    q[2] = solinas_mul_wide(
        instruction_ra_quadratic_at_three(lhs),
        instruction_ra_quadratic_at_three(rhs));
    q[3] = solinas_mul_wide(lhs.at_infinity, rhs.at_infinity);
}

inline void instruction_ra_accumulate_linears(
    thread const InstructionRaLinear* factors,
    thread SolinasFp128* q)
{
    SolinasFp128 products[INSTRUCTION_RA_SAMPLES];
    instruction_ra_group_products(factors, products);
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        q[sample] = solinas_add(q[sample], products[sample]);
    }
}

inline SolinasFp128 instruction_ra_sample(thread const SolinasFp128* q, uint sample)
{
    SolinasFp128 low = (sample & 1u) != 0u ? q[1] : q[0];
    SolinasFp128 high = (sample & 1u) != 0u ? q[3] : q[2];
    return (sample & 2u) != 0u ? high : low;
}

// Step k: lane g sends its sample g ^ k, which lane g ^ k needs.
inline SolinasFp128 instruction_ra_lane_sample_total(thread const SolinasFp128* q, uint group)
{
    SolinasFp128 total = instruction_ra_sample(q, group);
    for (ushort step = 1; step < INSTRUCTION_RA_GROUPS; step++) {
        SolinasFp128 other;
        other.limb = simd_shuffle_xor(instruction_ra_sample(q, group ^ step).limb, step);
        total = solinas_add(total, other);
    }
    return total;
}

// Lane l holds sample l & 3 of its pairs; partials get every sample's
// threadgroup total times the outer weight, in instruction_ra_finish_block's layout.
inline void instruction_ra_finish_group_lanes(
    SolinasFp128 lane_sum,
    SolinasFp128 e_out,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (ushort offset = INSTRUCTION_RA_GROUPS; offset < 32; offset <<= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_xor(lane_sum.limb, offset);
        lane_sum = solinas_add(lane_sum, other);
    }
    if (lane_in_simd < INSTRUCTION_RA_SAMPLES) {
        shared[lane_in_simd * simdgroups + simdgroup] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * e_out_length + x_out] = solinas_mul_wide(e_out, sum);
            }
        }
    }
}

inline void instruction_ra_lazy_message_body(
    uint branch_width,
    device const InstructionRaLookup* lookups,
    device const uint* cycle_to_table_major,
    device const SolinasFp128* branches,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    device SolinasFp128* partials,
    constant InstructionRaFirstMessageParams& params,
    threadgroup SolinasFp128* shared,
    uint thread_index,
    uint x_out,
    uint lane_in_simd,
    uint simdgroup,
    uint threads_per_threadgroup)
{
    uint group = thread_index % INSTRUCTION_RA_GROUPS;
    SolinasFp128 lane_sum = solinas_zero();
    for (uint x_in = thread_index / INSTRUCTION_RA_GROUPS; x_in < params.e_in_length;
         x_in += threads_per_threadgroup / INSTRUCTION_RA_GROUPS) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[INSTRUCTION_RA_FACTORS_PER_GROUP];
        instruction_ra_gather_group(
            group,
            pair,
            branch_width,
            lookups,
            cycle_to_table_major,
            branches,
            factors);
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        instruction_ra_group_products(factors, q);
        lane_sum = solinas_add(
            lane_sum,
            solinas_mul_wide(e_in[x_in], instruction_ra_lane_sample_total(q, group)));
    }

    instruction_ra_finish_group_lanes(
        lane_sum,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32u);
}

#define DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(NAME, WIDTH) \
kernel void NAME( \
    device const InstructionRaLookup* lookups [[buffer(0)]], \
    device const uint* cycle_to_table_major [[buffer(1)]], \
    device const SolinasFp128* branches [[buffer(2)]], \
    device const SolinasFp128* e_in [[buffer(3)]], \
    device const SolinasFp128* e_out [[buffer(4)]], \
    device SolinasFp128* partials [[buffer(5)]], \
    constant InstructionRaFirstMessageParams& params [[buffer(6)]], \
    threadgroup SolinasFp128* shared [[threadgroup(0)]], \
    uint thread_index [[thread_index_in_threadgroup]], \
    uint x_out [[threadgroup_position_in_grid]], \
    uint lane_in_simd [[thread_index_in_simdgroup]], \
    uint simdgroup [[simdgroup_index_in_threadgroup]], \
    uint threads [[threads_per_threadgroup]]) \
{ \
    instruction_ra_lazy_message_body( \
        WIDTH, lookups, cycle_to_table_major, branches, e_in, e_out, partials, \
        params, shared, thread_index, x_out, lane_in_simd, simdgroup, threads); \
}

DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(solinas_instruction_ra_first_message, 1u)
DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(solinas_instruction_ra_message_width_2, 2u)
DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(solinas_instruction_ra_message_width_4, 4u)
DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(solinas_instruction_ra_message_width_8, 8u)
DEFINE_INSTRUCTION_RA_MESSAGE_KERNEL(
    solinas_instruction_ra_message_wide,
    instruction_ra_wide_branch_width)

kernel void solinas_instruction_ra_double_branches(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant InstructionRaBranchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_factor = params.branch_width * INSTRUCTION_RA_BINS;
    uint elements = INSTRUCTION_RA_FACTORS * per_factor;
    if (gid >= elements) {
        return;
    }
    uint factor = gid / per_factor;
    uint within = gid - factor * per_factor;
    uint destination_base = factor * 2u * per_factor;
    SolinasFp128 value = source[gid];
    SolinasFp128 one = solinas_zero();
    one.limb[0] = 1u;
    destination[destination_base + within] = solinas_mul_wide(
        solinas_sub(one, challenge),
        value);
    destination[destination_base + per_factor + within] =
        solinas_mul_wide(challenge, value);
}

inline void instruction_ra_materialize_body(
    uint branch_width,
    device const InstructionRaLookup* lookups,
    device const uint* cycle_to_table_major,
    device const SolinasFp128* branches,
    device SolinasFp128* dense,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    device SolinasFp128* partials,
    constant InstructionRaMaterializeParams& params,
    threadgroup SolinasFp128* shared,
    uint thread_index,
    uint x_out,
    uint lane_in_simd,
    uint simdgroup,
    uint threads)
{
    uint group = thread_index % INSTRUCTION_RA_GROUPS;
    SolinasFp128 lane_sum = solinas_zero();
    for (uint x_in = thread_index / INSTRUCTION_RA_GROUPS; x_in < params.e_in_length;
         x_in += threads / INSTRUCTION_RA_GROUPS) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[INSTRUCTION_RA_FACTORS_PER_GROUP];
        instruction_ra_gather_group(
            group,
            pair,
            branch_width,
            lookups,
            cycle_to_table_major,
            branches,
            factors);
        for (uint local = 0; local < INSTRUCTION_RA_FACTORS_PER_GROUP; local++) {
            uint factor = group * INSTRUCTION_RA_FACTORS_PER_GROUP + local;
            uint destination = factor * params.source_elements + 2u * pair;
            dense[destination] = solinas_sub(
                factors[local].at_one,
                factors[local].at_infinity);
            dense[destination + 1u] = factors[local].at_one;
        }
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        instruction_ra_group_products(factors, q);
        lane_sum = solinas_add(
            lane_sum,
            solinas_mul_wide(e_in[x_in], instruction_ra_lane_sample_total(q, group)));
    }

    instruction_ra_finish_group_lanes(
        lane_sum,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_instruction_ra_materialize_width_16(
    device const InstructionRaLookup* lookups [[buffer(0)]],
    device const uint* cycle_to_table_major [[buffer(1)]],
    device const SolinasFp128* branches [[buffer(2)]],
    device SolinasFp128* dense [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant InstructionRaMaterializeParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    instruction_ra_materialize_body(
        16u,
        lookups,
        cycle_to_table_major,
        branches,
        dense,
        e_in,
        e_out,
        partials,
        params,
        shared,
        thread_index,
        x_out,
        lane_in_simd,
        simdgroup,
        threads);
}

kernel void solinas_instruction_ra_materialize_wide(
    device const InstructionRaLookup* lookups [[buffer(0)]],
    device const uint* cycle_to_table_major [[buffer(1)]],
    device const SolinasFp128* branches [[buffer(2)]],
    device SolinasFp128* dense [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant InstructionRaMaterializeParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    instruction_ra_materialize_body(
        instruction_ra_wide_branch_width,
        lookups,
        cycle_to_table_major,
        branches,
        dense,
        e_in,
        e_out,
        partials,
        params,
        shared,
        thread_index,
        x_out,
        lane_in_simd,
        simdgroup,
        threads);
}

kernel void solinas_instruction_ra_dense_transition(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant InstructionRaMaterializeParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            q[sample] = solinas_zero();
        }
        for (uint group = 0; group < INSTRUCTION_RA_GROUPS; group++) {
            InstructionRaLinear factors[INSTRUCTION_RA_FACTORS_PER_GROUP];
            for (uint local = 0; local < INSTRUCTION_RA_FACTORS_PER_GROUP; local++) {
                uint factor = group * INSTRUCTION_RA_FACTORS_PER_GROUP + local;
                uint source = factor * params.source_elements + 4u * pair;
                SolinasFp128 lo_0 = tables[source];
                SolinasFp128 hi_0 = tables[source + 1u];
                SolinasFp128 lo_1 = tables[source + 2u];
                SolinasFp128 hi_1 = tables[source + 3u];
                SolinasFp128 bound_0 = solinas_add(
                    lo_0,
                    solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
                SolinasFp128 bound_1 = solinas_add(
                    lo_1,
                    solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
                uint destination = factor * bound_elements + 2u * pair;
                bound[destination] = bound_0;
                bound[destination + 1u] = bound_1;
                factors[local].at_one = bound_1;
                factors[local].at_infinity = solinas_sub(bound_1, bound_0);
            }
            instruction_ra_accumulate_linears(factors, q);
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
        threads / 32u);
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
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
