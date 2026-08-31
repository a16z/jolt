#define RAM_HAMMING_COLUMNS 2u

struct RamHammingPackParams {
    uint words;
    uint3 reserved;
};

struct RamHammingPrefixParams {
    uint e_in_length;
    uint e_out_length;
    uint q_patterns;
    uint materialize;
};

struct RamHammingDenseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

constant uint ram_hamming_prefix_width [[function_constant(22)]];

inline SolinasFp128 ram_hamming_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline ulong ram_hamming_parent_bits(
    device const uint* access_bits,
    uint pair)
{
    if (ram_hamming_prefix_width == 32u) {
        uint word = 2u * pair;
        return (ulong)access_bits[word]
            | ((ulong)access_bits[word + 1u] << 32);
    }
    uint bits = 2u * ram_hamming_prefix_width;
    uint start = pair * bits;
    uint word = start >> 5;
    uint shift = start & 31u;
    uint mask = bits == 32u ? 0xffffffffu : ((1u << bits) - 1u);
    return (ulong)((access_bits[word] >> shift) & mask);
}

inline SolinasFp128 ram_hamming_value_from_bytes(
    uint mask,
    device const SolinasFp128* value_table)
{
    SolinasFp128 value = solinas_zero();
    uint segments = ram_hamming_prefix_width / 8u;
    for (uint segment = 0u; segment < segments; segment++) {
        uint byte = (mask >> (8u * segment)) & 0xffu;
        value = solinas_add(value, value_table[segment * 256u + byte]);
    }
    return value;
}

inline void ram_hamming_finish_group(
    SolinasFp128 constant_sum,
    SolinasFp128 leading_sum,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint group,
    uint groups,
    uint lane,
    uint simdgroup,
    uint simdgroups)
{
    constant_sum = solinas_simd_sum_32(constant_sum);
    leading_sum = solinas_simd_sum_32(leading_sum);
    if (lane == 0u) {
        shared[simdgroup] = constant_sum;
        shared[simdgroups + simdgroup] = leading_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        SolinasFp128 constant_group = lane < simdgroups
            ? shared[lane]
            : solinas_zero();
        SolinasFp128 leading_group = lane < simdgroups
            ? shared[simdgroups + lane]
            : solinas_zero();
        constant_group = solinas_simd_sum_32(constant_group);
        leading_group = solinas_simd_sum_32(leading_group);
        if (lane == 0u) {
            partials[group] = solinas_mul_wide(outer_weight, constant_group);
            partials[groups + group] = solinas_mul_wide(outer_weight, leading_group);
        }
    }
}

kernel void solinas_ram_hamming_pack(
    device const uint* addresses [[buffer(0)]],
    device uint* access_bits [[buffer(1)]],
    constant RamHammingPackParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.words) {
        return;
    }
    uint bits = 0u;
    uint start = 32u * gid;
    for (uint bit = 0u; bit < 32u; bit++) {
        bits |= (uint)(addresses[start + bit] != 0xffffffffu) << bit;
    }
    access_bits[gid] = bits;
}

kernel void solinas_ram_hamming_prefix(
    device const uint* access_bits [[buffer(0)]],
    device const SolinasFp128* value_table [[buffer(1)]],
    device const SolinasFp128* q_table [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* dense [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RamHammingPrefixParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 constant_sum = solinas_zero();
    SolinasFp128 leading_sum = solinas_zero();
    for (uint x_in = thread_index; x_in < params.e_in_length; x_in += threads) {
        uint pair = group * params.e_in_length + x_in;
        ulong pattern = ram_hamming_parent_bits(access_bits, pair);
        if (ram_hamming_prefix_width == 1u) {
            if ((pattern & 1ul) != ((pattern >> 1) & 1ul)) {
                leading_sum = solinas_add(leading_sum, e_in[x_in]);
            }
            continue;
        }

        SolinasFp128 q_0;
        SolinasFp128 q_2;
        SolinasFp128 h_0 = solinas_zero();
        SolinasFp128 h_1 = solinas_zero();
        if (ram_hamming_prefix_width <= 8u) {
            uint index = (uint)pattern;
            q_0 = q_table[index];
            q_2 = q_table[params.q_patterns + index];
        } else {
            uint child_mask = ram_hamming_prefix_width == 32u
                ? 0xffffffffu
                : ((1u << ram_hamming_prefix_width) - 1u);
            h_0 = ram_hamming_value_from_bytes(
                (uint)pattern & child_mask,
                value_table);
            h_1 = ram_hamming_value_from_bytes(
                (uint)(pattern >> ram_hamming_prefix_width),
                value_table);
            SolinasFp128 one = solinas_zero();
            one.limb[0] = 1u;
            SolinasFp128 delta = solinas_sub(h_1, h_0);
            q_0 = solinas_mul_wide(h_0, solinas_sub(h_0, one));
            q_2 = solinas_mul_wide(delta, delta);
        }
        if (params.materialize != 0u) {
            dense[2u * pair] = h_0;
            dense[2u * pair + 1u] = h_1;
        }
        constant_sum = solinas_add(
            constant_sum,
            solinas_mul_wide(e_in[x_in], q_0));
        leading_sum = solinas_add(
            leading_sum,
            solinas_mul_wide(e_in[x_in], q_2));
    }
    ram_hamming_finish_group(
        constant_sum,
        leading_sum,
        e_out[group],
        partials,
        shared,
        group,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_hamming_dense_transition(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant RamHammingDenseParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 constant_sum = solinas_zero();
    SolinasFp128 leading_sum = solinas_zero();
    for (uint x_in = thread_index; x_in < params.e_in_length; x_in += threads) {
        uint pair = group * params.e_in_length + x_in;
        uint source_base = 4u * pair;
        SolinasFp128 h_0 = ram_hamming_bind(
            source[source_base], source[source_base + 1u], challenge);
        SolinasFp128 h_1 = ram_hamming_bind(
            source[source_base + 2u], source[source_base + 3u], challenge);
        destination[2u * pair] = h_0;
        destination[2u * pair + 1u] = h_1;

        SolinasFp128 one = solinas_zero();
        one.limb[0] = 1u;
        SolinasFp128 delta = solinas_sub(h_1, h_0);
        SolinasFp128 q_0 = solinas_mul_wide(h_0, solinas_sub(h_0, one));
        SolinasFp128 q_2 = solinas_mul_wide(delta, delta);
        constant_sum = solinas_add(
            constant_sum,
            solinas_mul_wide(e_in[x_in], q_0));
        leading_sum = solinas_add(
            leading_sum,
            solinas_mul_wide(e_in[x_in], q_2));
    }
    ram_hamming_finish_group(
        constant_sum,
        leading_sum,
        e_out[group],
        partials,
        shared,
        group,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}
