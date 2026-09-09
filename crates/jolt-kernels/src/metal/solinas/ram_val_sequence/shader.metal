// Topology-free RAM value-check sequence. Sparse increments drive the first
// five rounds; a width-32 materialization starts the resident dense tail.

#define RAM_VAL_SAMPLES 4u

struct RamValPrefixParams {
    uint increment_count;
    uint address_domain;
    uint branch_width;
    uint lt_lo_length;
};

struct RamValMaterializeParams {
    uint increment_count;
    uint source_elements;
    uint address_domain;
    uint lt_lo_length;
};

struct RamValDenseParams {
    uint source_elements;
    uint reserved_0;
    uint2 reserved_1;
};

constant uint ram_val_prefix_width [[function_constant(21)]];

inline SolinasFp128 ram_val_from_i128_twos(ulong low, ulong high) {
    if ((high >> 63) == 0ul) {
        SolinasFp128 value;
        value.limb = uint4(
            (uint)low,
            (uint)(low >> 32),
            (uint)high,
            (uint)(high >> 32));
        SolinasCorrection corrected = solinas_add_offset(value);
        return solinas_select(corrected.carry != 0u, corrected.value, value);
    }

    ulong magnitude_low = ~low + 1ul;
    ulong magnitude_high = ~high + (magnitude_low == 0ul ? 1ul : 0ul);
    SolinasFp128 magnitude;
    magnitude.limb = uint4(
        (uint)magnitude_low,
        (uint)(magnitude_low >> 32),
        (uint)magnitude_high,
        (uint)(magnitude_high >> 32));
    SolinasCorrection corrected = solinas_add_offset(magnitude);
    magnitude = solinas_select(corrected.carry != 0u, corrected.value, magnitude);
    return solinas_sub(solinas_zero(), magnitude);
}

inline SolinasFp128 ram_val_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline SolinasFp128 ram_val_lt_value(
    uint index,
    uint lt_lo_length,
    device const SolinasFp128* lt_lo,
    device const SolinasFp128* lt_hi,
    device const SolinasFp128* eq_hi)
{
    uint high = index / lt_lo_length;
    uint low = index - high * lt_lo_length;
    return solinas_add(
        lt_hi[high],
        solinas_mul_wide(eq_hi[high], lt_lo[low]));
}

inline uint ram_val_lower_bound(
    device const ulong* cycles,
    uint count,
    ulong target)
{
    uint low = 0u;
    uint high = count;
    while (low < high) {
        uint mid = low + (high - low) / 2u;
        if (cycles[mid] < target) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return low;
}

inline SolinasFp128 ram_val_increment_block(
    ulong block_start,
    uint branch_width,
    uint increment_count,
    device const ulong* cycles,
    device const ulong2* increments,
    device const SolinasFp128* cycle_weights)
{
    ulong block_end = block_start + (ulong)branch_width;
    uint cursor = ram_val_lower_bound(cycles, increment_count, block_start);
    SolinasFp128 result = solinas_zero();
    while (cursor < increment_count && cycles[cursor] < block_end) {
        uint offset = (uint)(cycles[cursor] - block_start);
        ulong2 encoded = increments[cursor];
        result = solinas_add(
            result,
            solinas_mul_wide(
                ram_val_from_i128_twos(encoded.x, encoded.y),
                cycle_weights[offset]));
        cursor++;
    }
    return result;
}

inline SolinasFp128 ram_val_address_block(
    uint block_start,
    uint branch_width,
    uint address_domain,
    device const uint* addresses,
    device const SolinasFp128* branches)
{
    SolinasFp128 result = solinas_zero();
    for (uint offset = 0u; offset < branch_width; offset++) {
        uint address = addresses[block_start + offset];
        if (address != 0xffffffffu) {
            result = solinas_add(
                result,
                branches[offset * address_domain + address]);
        }
    }
    return result;
}

inline void ram_val_accumulate_pair(
    SolinasFp128 inc_0,
    SolinasFp128 inc_1,
    SolinasFp128 ra_0,
    SolinasFp128 ra_1,
    SolinasFp128 lt_0,
    SolinasFp128 lt_1,
    thread SolinasFp128* lanes)
{
    SolinasFp128 inc_delta = solinas_sub(inc_1, inc_0);
    SolinasFp128 ra_delta = solinas_sub(ra_1, ra_0);
    SolinasFp128 lt_delta = solinas_sub(lt_1, lt_0);
    SolinasFp128 inc_2 = solinas_add(inc_1, inc_delta);
    SolinasFp128 ra_2 = solinas_add(ra_1, ra_delta);
    SolinasFp128 lt_2 = solinas_add(lt_1, lt_delta);
    SolinasFp128 inc_at[3] = {
        inc_0,
        inc_2,
        solinas_add(inc_2, inc_delta),
    };
    SolinasFp128 ra_at[3] = {
        ra_0,
        ra_2,
        solinas_add(ra_2, ra_delta),
    };
    SolinasFp128 lt_at[3] = {
        lt_0,
        lt_2,
        solinas_add(lt_2, lt_delta),
    };
    for (uint sample = 0u; sample < 3u; sample++) {
        lanes[sample] = solinas_add(
            lanes[sample],
            solinas_mul_wide(
                solinas_mul_wide(inc_at[sample], ra_at[sample]),
                lt_at[sample]));
    }
}

inline void ram_val_finish_group(
    thread SolinasFp128* lanes,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint group,
    uint groups,
    uint lane,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0u; sample < RAM_VAL_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < RAM_VAL_SAMPLES; sample++) {
            SolinasFp128 sum = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane == 0u) {
                partials[sample * groups + group] = sum;
            }
        }
    }
}

kernel void solinas_ram_val_double_branches(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant RamValPrefixParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint elements = params.branch_width * params.address_domain;
    if (gid >= elements) {
        return;
    }
    SolinasFp128 one = solinas_zero();
    one.limb[0] = 1u;
    SolinasFp128 value = source[gid];
    destination[gid] = solinas_mul_wide(solinas_sub(one, challenge), value);
    destination[elements + gid] = solinas_mul_wide(challenge, value);
}

kernel void solinas_ram_val_sparse_prefix(
    device const uint* addresses [[buffer(0)]],
    device const ulong* cycles [[buffer(1)]],
    device const ulong2* increments [[buffer(2)]],
    device const SolinasFp128* branches [[buffer(3)]],
    device const SolinasFp128* cycle_weights [[buffer(4)]],
    device const SolinasFp128* lt_lo [[buffer(5)]],
    device const SolinasFp128* lt_hi [[buffer(6)]],
    device const SolinasFp128* eq_hi [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant RamValPrefixParams& params [[buffer(9)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_VAL_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint gid = group * threads + thread_index;
    if (gid < params.increment_count) {
        ulong cycle = cycles[gid];
        ulong parent = cycle / (2ul * (ulong)ram_val_prefix_width);
        bool leader = gid == 0u
            || cycles[gid - 1u] / (2ul * (ulong)ram_val_prefix_width) != parent;
        if (leader) {
            ulong parent_start = parent * 2ul * (ulong)ram_val_prefix_width;
            ulong child_start = parent_start + (ulong)ram_val_prefix_width;
            SolinasFp128 inc_0 = solinas_zero();
            SolinasFp128 inc_1 = solinas_zero();
            uint cursor = gid;
            ulong parent_end = parent_start + 2ul * (ulong)ram_val_prefix_width;
            while (cursor < params.increment_count && cycles[cursor] < parent_end) {
                ulong record_cycle = cycles[cursor];
                uint offset;
                thread SolinasFp128* destination;
                if (record_cycle < child_start) {
                    offset = (uint)(record_cycle - parent_start);
                    destination = &inc_0;
                } else {
                    offset = (uint)(record_cycle - child_start);
                    destination = &inc_1;
                }
                ulong2 encoded = increments[cursor];
                *destination = solinas_add(
                    *destination,
                    solinas_mul_wide(
                        ram_val_from_i128_twos(encoded.x, encoded.y),
                        cycle_weights[offset]));
                cursor++;
            }
            uint original = (uint)parent_start;
            SolinasFp128 ra_0 = ram_val_address_block(
                original,
                ram_val_prefix_width,
                params.address_domain,
                addresses,
                branches);
            SolinasFp128 ra_1 = ram_val_address_block(
                original + ram_val_prefix_width,
                ram_val_prefix_width,
                params.address_domain,
                addresses,
                branches);
            uint lt_index = (uint)(2ul * parent);
            SolinasFp128 lt_0 = ram_val_lt_value(
                lt_index, params.lt_lo_length, lt_lo, lt_hi, eq_hi);
            SolinasFp128 lt_1 = ram_val_lt_value(
                lt_index + 1u, params.lt_lo_length, lt_lo, lt_hi, eq_hi);
            ram_val_accumulate_pair(
                inc_0, inc_1, ra_0, ra_1, lt_0, lt_1, lanes);
        }
    }

    ram_val_finish_group(
        lanes,
        partials,
        shared,
        group,
        (params.increment_count + threads - 1u) / threads,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_val_materialize_width_32(
    device const uint* addresses [[buffer(0)]],
    device const ulong* cycles [[buffer(1)]],
    device const ulong2* increments [[buffer(2)]],
    device const SolinasFp128* branches [[buffer(3)]],
    device const SolinasFp128* cycle_weights [[buffer(4)]],
    device const SolinasFp128* lt_lo [[buffer(5)]],
    device const SolinasFp128* lt_hi [[buffer(6)]],
    device const SolinasFp128* eq_hi [[buffer(7)]],
    device SolinasFp128* dense [[buffer(8)]],
    device SolinasFp128* partials [[buffer(9)]],
    constant RamValMaterializeParams& params [[buffer(10)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_VAL_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + thread_index;
    uint pairs = params.source_elements / 2u;
    if (pair < pairs) {
        uint low_block = 2u * pair;
        uint high_block = low_block + 1u;
        ulong low_start = (ulong)low_block * 32ul;
        ulong high_start = low_start + 32ul;
        SolinasFp128 inc_0 = ram_val_increment_block(
            low_start,
            32u,
            params.increment_count,
            cycles,
            increments,
            cycle_weights);
        SolinasFp128 inc_1 = ram_val_increment_block(
            high_start,
            32u,
            params.increment_count,
            cycles,
            increments,
            cycle_weights);
        SolinasFp128 ra_0 = ram_val_address_block(
            (uint)low_start, 32u, params.address_domain, addresses, branches);
        SolinasFp128 ra_1 = ram_val_address_block(
            (uint)high_start, 32u, params.address_domain, addresses, branches);
        SolinasFp128 lt_0 = ram_val_lt_value(
            low_block, params.lt_lo_length, lt_lo, lt_hi, eq_hi);
        SolinasFp128 lt_1 = ram_val_lt_value(
            high_block, params.lt_lo_length, lt_lo, lt_hi, eq_hi);

        dense[low_block] = inc_0;
        dense[high_block] = inc_1;
        dense[params.source_elements + low_block] = ra_0;
        dense[params.source_elements + high_block] = ra_1;
        dense[2u * params.source_elements + low_block] = lt_0;
        dense[2u * params.source_elements + high_block] = lt_1;
        ram_val_accumulate_pair(
            inc_0, inc_1, ra_0, ra_1, lt_0, lt_1, lanes);
    }

    ram_val_finish_group(
        lanes,
        partials,
        shared,
        group,
        (pairs + threads - 1u) / threads,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_val_dense_transition(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant SolinasFp128& challenge [[buffer(3)]],
    constant RamValDenseParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_VAL_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + thread_index;
    uint pairs = params.source_elements / 4u;
    uint destination_elements = params.source_elements / 2u;
    if (pair < pairs) {
        SolinasFp128 values[3][2];
        for (uint factor = 0u; factor < 3u; factor++) {
            uint source_base = factor * params.source_elements + 4u * pair;
            SolinasFp128 low = ram_val_bind(
                source[source_base], source[source_base + 1u], challenge);
            SolinasFp128 high = ram_val_bind(
                source[source_base + 2u], source[source_base + 3u], challenge);
            uint destination_base = factor * destination_elements + 2u * pair;
            destination[destination_base] = low;
            destination[destination_base + 1u] = high;
            values[factor][0] = low;
            values[factor][1] = high;
        }
        ram_val_accumulate_pair(
            values[0][0],
            values[0][1],
            values[1][0],
            values[1][1],
            values[2][0],
            values[2][1],
            lanes);
    }

    ram_val_finish_group(
        lanes,
        partials,
        shared,
        group,
        (pairs + threads - 1u) / threads,
        lane,
        simdgroup,
        threads / 32u);
}
