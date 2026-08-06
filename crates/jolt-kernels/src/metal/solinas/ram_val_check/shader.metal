// Concatenate after fp128.metal and simd_reduce.metal.

#define RAM_VAL_CHECK_SAMPLES 3u
#define RAM_VAL_CHECK_NO_ACCESS 0xffffffffu
#define RAM_VAL_CHECK_FLAG_INCREMENT_NONNEGATIVE 1u

struct RamValCheckNativeRow {
    ulong increment_magnitude;
    uint address;
    uint flags;
};

struct RamValCheckDenseRow {
    SolinasFp128 increment;
    SolinasFp128 ram_ra;
};

struct RamValCheckMessageParams {
    uint message_elements;
    uint high_blocks;
    uint lt_lo_length;
    uint reserved;
};

struct RamValCheckReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

inline SolinasFp128 ram_val_check_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 ram_val_check_increment(
    device const RamValCheckNativeRow& row)
{
    SolinasFp128 magnitude = ram_val_check_from_u64(row.increment_magnitude);
    return (row.flags & RAM_VAL_CHECK_FLAG_INCREMENT_NONNEGATIVE) != 0u
        ? magnitude
        : solinas_sub(solinas_zero(), magnitude);
}

inline SolinasFp128 ram_val_check_ra(
    device const RamValCheckNativeRow& row,
    device const SolinasFp128* eq_address)
{
    return row.address == RAM_VAL_CHECK_NO_ACCESS
        ? solinas_zero()
        : eq_address[row.address];
}

inline SolinasFp128 ram_val_check_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline void ram_val_check_accumulate_pair(
    SolinasFp128 increment_0,
    SolinasFp128 increment_1,
    SolinasFp128 ram_ra_0,
    SolinasFp128 ram_ra_1,
    SolinasFp128 lt_0,
    SolinasFp128 lt_1,
    thread SolinasFp128* a,
    thread SolinasFp128* b)
{
    SolinasFp128 increment_delta = solinas_sub(increment_1, increment_0);
    SolinasFp128 ram_ra_delta = solinas_sub(ram_ra_1, ram_ra_0);
    SolinasFp128 lt_delta = solinas_sub(lt_1, lt_0);
    SolinasFp128 increment_2 = solinas_add(increment_1, increment_delta);
    SolinasFp128 ram_ra_2 = solinas_add(ram_ra_1, ram_ra_delta);
    SolinasFp128 lt_2 = solinas_add(lt_1, lt_delta);
    SolinasFp128 increment_at[RAM_VAL_CHECK_SAMPLES] = {
        increment_0,
        increment_2,
        solinas_add(increment_2, increment_delta),
    };
    SolinasFp128 ram_ra_at[RAM_VAL_CHECK_SAMPLES] = {
        ram_ra_0,
        ram_ra_2,
        solinas_add(ram_ra_2, ram_ra_delta),
    };
    SolinasFp128 lt_at[RAM_VAL_CHECK_SAMPLES] = {
        lt_0,
        lt_2,
        solinas_add(lt_2, lt_delta),
    };
    for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
        SolinasFp128 product = solinas_mul_wide(
            increment_at[sample], ram_ra_at[sample]);
        a[sample] = solinas_add(a[sample], product);
        b[sample] = solinas_add(
            b[sample],
            solinas_mul_wide(product, lt_at[sample]));
    }
}

inline void ram_val_check_finish_high(
    thread SolinasFp128* a,
    thread SolinasFp128* b,
    device const SolinasFp128* lt_hi,
    device const SolinasFp128* eq_hi,
    device SolinasFp128* partials,
    constant RamValCheckMessageParams& params,
    threadgroup SolinasFp128* shared,
    uint high,
    uint lane,
    uint simdgroup,
    uint threads)
{
    uint simdgroups = threads / 32u;
    for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
        a[sample] = solinas_simd_sum_32(a[sample]);
        b[sample] = solinas_simd_sum_32(b[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = a[sample];
            shared[(RAM_VAL_CHECK_SAMPLES + sample) * simdgroups + simdgroup] =
                b[sample];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
            SolinasFp128 a_sum = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            SolinasFp128 b_sum = lane < simdgroups
                ? shared[(RAM_VAL_CHECK_SAMPLES + sample) * simdgroups + lane]
                : solinas_zero();
            a_sum = solinas_simd_sum_32(a_sum);
            b_sum = solinas_simd_sum_32(b_sum);
            if (lane == 0u) {
                partials[sample * params.high_blocks + high] = solinas_add(
                    solinas_mul_wide(lt_hi[high], a_sum),
                    solinas_mul_wide(eq_hi[high], b_sum));
            }
        }
    }
}

kernel void solinas_ram_val_check_first_message(
    device const RamValCheckNativeRow* rows [[buffer(0)]],
    device const SolinasFp128* eq_address [[buffer(1)]],
    device const SolinasFp128* lt_lo [[buffer(2)]],
    device const SolinasFp128* lt_hi [[buffer(3)]],
    device const SolinasFp128* eq_hi [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RamValCheckMessageParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (high >= params.high_blocks) {
        return;
    }
    SolinasFp128 a[RAM_VAL_CHECK_SAMPLES];
    SolinasFp128 b[RAM_VAL_CHECK_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint high_base = high * params.lt_lo_length;
    for (uint low_pair = tid; low_pair < low_pairs; low_pair += threads) {
        uint low = 2u * low_pair;
        uint index = high_base + low;
        ram_val_check_accumulate_pair(
            ram_val_check_increment(rows[index]),
            ram_val_check_increment(rows[index + 1u]),
            ram_val_check_ra(rows[index], eq_address),
            ram_val_check_ra(rows[index + 1u], eq_address),
            lt_lo[low],
            lt_lo[low + 1u],
            a,
            b);
    }

    ram_val_check_finish_high(
        a,
        b,
        lt_hi,
        eq_hi,
        partials,
        params,
        shared,
        high,
        lane,
        simdgroup,
        threads);
}

kernel void solinas_ram_val_check_native_transition(
    device const RamValCheckNativeRow* rows [[buffer(0)]],
    device const SolinasFp128* eq_address [[buffer(1)]],
    device const SolinasFp128* lt_lo [[buffer(2)]],
    device const SolinasFp128* lt_hi [[buffer(3)]],
    device const SolinasFp128* eq_hi [[buffer(4)]],
    device RamValCheckDenseRow* dense [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant SolinasFp128& challenge [[buffer(7)]],
    constant RamValCheckMessageParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (high >= params.high_blocks) {
        return;
    }
    SolinasFp128 a[RAM_VAL_CHECK_SAMPLES];
    SolinasFp128 b[RAM_VAL_CHECK_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint source_high_base = high * 2u * params.lt_lo_length;
    uint destination_high_base = high * params.lt_lo_length;
    for (uint low_pair = tid; low_pair < low_pairs; low_pair += threads) {
        uint source = source_high_base + 4u * low_pair;
        uint destination = destination_high_base + 2u * low_pair;
        RamValCheckDenseRow low;
        low.increment = ram_val_check_bind(
            ram_val_check_increment(rows[source]),
            ram_val_check_increment(rows[source + 1u]),
            challenge);
        low.ram_ra = ram_val_check_bind(
            ram_val_check_ra(rows[source], eq_address),
            ram_val_check_ra(rows[source + 1u], eq_address),
            challenge);
        RamValCheckDenseRow high_value;
        high_value.increment = ram_val_check_bind(
            ram_val_check_increment(rows[source + 2u]),
            ram_val_check_increment(rows[source + 3u]),
            challenge);
        high_value.ram_ra = ram_val_check_bind(
            ram_val_check_ra(rows[source + 2u], eq_address),
            ram_val_check_ra(rows[source + 3u], eq_address),
            challenge);
        dense[destination] = low;
        dense[destination + 1u] = high_value;
        ram_val_check_accumulate_pair(
            low.increment,
            high_value.increment,
            low.ram_ra,
            high_value.ram_ra,
            lt_lo[2u * low_pair],
            lt_lo[2u * low_pair + 1u],
            a,
            b);
    }

    ram_val_check_finish_high(
        a,
        b,
        lt_hi,
        eq_hi,
        partials,
        params,
        shared,
        high,
        lane,
        simdgroup,
        threads);
}

kernel void solinas_ram_val_check_dense_transition(
    device const RamValCheckDenseRow* source [[buffer(0)]],
    device RamValCheckDenseRow* destination [[buffer(1)]],
    device const SolinasFp128* lt_lo [[buffer(2)]],
    device const SolinasFp128* lt_hi [[buffer(3)]],
    device const SolinasFp128* eq_hi [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant RamValCheckMessageParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (high >= params.high_blocks) {
        return;
    }
    SolinasFp128 a[RAM_VAL_CHECK_SAMPLES];
    SolinasFp128 b[RAM_VAL_CHECK_SAMPLES];
    for (uint sample = 0u; sample < RAM_VAL_CHECK_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint source_high_base = high * 2u * params.lt_lo_length;
    uint destination_high_base = high * params.lt_lo_length;
    for (uint low_pair = tid; low_pair < low_pairs; low_pair += threads) {
        uint source_index = source_high_base + 4u * low_pair;
        uint destination_index = destination_high_base + 2u * low_pair;
        RamValCheckDenseRow low;
        low.increment = ram_val_check_bind(
            source[source_index].increment,
            source[source_index + 1u].increment,
            challenge);
        low.ram_ra = ram_val_check_bind(
            source[source_index].ram_ra,
            source[source_index + 1u].ram_ra,
            challenge);
        RamValCheckDenseRow high_value;
        high_value.increment = ram_val_check_bind(
            source[source_index + 2u].increment,
            source[source_index + 3u].increment,
            challenge);
        high_value.ram_ra = ram_val_check_bind(
            source[source_index + 2u].ram_ra,
            source[source_index + 3u].ram_ra,
            challenge);
        destination[destination_index] = low;
        destination[destination_index + 1u] = high_value;
        ram_val_check_accumulate_pair(
            low.increment,
            high_value.increment,
            low.ram_ra,
            high_value.ram_ra,
            lt_lo[2u * low_pair],
            lt_lo[2u * low_pair + 1u],
            a,
            b);
    }

    ram_val_check_finish_high(
        a,
        b,
        lt_hi,
        eq_hi,
        partials,
        params,
        shared,
        high,
        lane,
        simdgroup,
        threads);
}

kernel void solinas_ram_val_check_reduce3(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamValCheckReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}
