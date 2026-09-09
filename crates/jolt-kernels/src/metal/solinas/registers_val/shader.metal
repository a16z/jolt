#define REGISTERS_VAL_SAMPLES 3u

struct RegistersValMessageParams {
    uint cycles;
    uint high_blocks;
    uint lt_lo_length;
    uint source_layout;
};

#define REGISTERS_VAL_SOURCE_INSTRUCTION_ROWS 1u

struct RegistersValReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

inline SolinasFp128 registers_val_from_u64(ulong value)
{
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32u);
    return result;
}

inline SolinasFp128 registers_val_inc(
    device const uchar* source_values,
    uint source_layout,
    uint cycles,
    uint index)
{
    if (source_layout == REGISTERS_VAL_SOURCE_INSTRUCTION_ROWS) {
        device const ulong* rows = (device const ulong*)source_values;
        ulong metadata = booleanity_source_word(rows, cycles, 3u, index);
        if (((metadata >> BOOLEANITY_SOURCE_RD_SHIFT) & 0xfful) == 0ul) {
            return solinas_zero();
        }
        SolinasFp128 magnitude = registers_val_from_u64(
            booleanity_source_word(rows, cycles, 2u, index));
        return ((metadata >> BOOLEANITY_SOURCE_RD_SIGN_SHIFT) & 1ul) == 0ul
            ? magnitude
            : solinas_sub(solinas_zero(), magnitude);
    }
    device const SolinasFp128* inc = (device const SolinasFp128*)source_values;
    return inc[index];
}

inline uchar registers_val_rd(
    device const uchar* source_values,
    device const uchar* source_indices,
    uint source_layout,
    uint cycles,
    uint index)
{
    if (source_layout == REGISTERS_VAL_SOURCE_INSTRUCTION_ROWS) {
        device const ulong* rows = (device const ulong*)source_values;
        ulong metadata = booleanity_source_word(rows, cycles, 3u, index);
        uint plus_one = uint((metadata >> BOOLEANITY_SOURCE_RD_SHIFT) & 0xfful);
        return plus_one == 0u ? uchar(255u) : uchar(plus_one - 1u);
    }
    return source_indices[index];
}

inline SolinasFp128 registers_val_wa(
    device const uchar* source_values,
    device const uchar* source_indices,
    device const SolinasFp128* eq_address,
    uint source_layout,
    uint cycles,
    uint index)
{
    uchar index_value = registers_val_rd(
        source_values, source_indices, source_layout, cycles, index);
    return index_value == 255u
        ? solinas_zero()
        : eq_address[(uint)index_value];
}

inline SolinasFp128 registers_val_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(low, solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline void registers_val_accumulate_pair(
    SolinasFp128 inc_0,
    SolinasFp128 inc_1,
    SolinasFp128 wa_0,
    SolinasFp128 wa_1,
    SolinasFp128 lt_0,
    SolinasFp128 lt_1,
    thread SolinasFp128* a,
    thread SolinasFp128* b)
{
    SolinasFp128 inc_delta = solinas_sub(inc_1, inc_0);
    SolinasFp128 wa_delta = solinas_sub(wa_1, wa_0);
    SolinasFp128 lt_delta = solinas_sub(lt_1, lt_0);
    SolinasFp128 inc_2 = solinas_add(inc_1, inc_delta);
    SolinasFp128 wa_2 = solinas_add(wa_1, wa_delta);
    SolinasFp128 lt_2 = solinas_add(lt_1, lt_delta);
    SolinasFp128 inc_at[REGISTERS_VAL_SAMPLES] = {
        inc_0,
        inc_2,
        solinas_add(inc_2, inc_delta),
    };
    SolinasFp128 wa_at[REGISTERS_VAL_SAMPLES] = {
        wa_0,
        wa_2,
        solinas_add(wa_2, wa_delta),
    };
    SolinasFp128 lt_at[REGISTERS_VAL_SAMPLES] = {
        lt_0,
        lt_2,
        solinas_add(lt_2, lt_delta),
    };
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        SolinasFp128 product = solinas_mul_wide(inc_at[sample], wa_at[sample]);
        a[sample] = solinas_add(a[sample], product);
        b[sample] = solinas_add(
            b[sample],
            solinas_mul_wide(product, lt_at[sample]));
    }
}

inline void registers_val_finish_high(
    thread SolinasFp128* a,
    thread SolinasFp128* b,
    device const SolinasFp128* lt_hi,
    device const SolinasFp128* eq_hi,
    device SolinasFp128* partials,
    constant RegistersValMessageParams& params,
    threadgroup SolinasFp128* shared,
    uint high,
    uint lane,
    uint simdgroup,
    uint threads)
{
    uint simdgroups = threads / 32u;
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        a[sample] = solinas_simd_sum_32(a[sample]);
        b[sample] = solinas_simd_sum_32(b[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = a[sample];
            shared[(REGISTERS_VAL_SAMPLES + sample) * simdgroups + simdgroup] = b[sample];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
            SolinasFp128 a_sum = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            SolinasFp128 b_sum = lane < simdgroups
                ? shared[(REGISTERS_VAL_SAMPLES + sample) * simdgroups + lane]
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

kernel void solinas_registers_val_first_message_factorized(
    device const uchar* source_values [[buffer(0)]],
    device const uchar* source_indices [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* lt_lo [[buffer(3)]],
    device const SolinasFp128* lt_hi [[buffer(4)]],
    device const SolinasFp128* eq_hi [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RegistersValMessageParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 a[REGISTERS_VAL_SAMPLES];
    SolinasFp128 b[REGISTERS_VAL_SAMPLES];
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint high_base = high * params.lt_lo_length;
    for (uint low_pair = thread_index; low_pair < low_pairs; low_pair += threads) {
        uint low_0 = 2u * low_pair;
        uint index_0 = high_base + low_0;
        uint index_1 = index_0 + 1u;
        SolinasFp128 inc_0 = registers_val_inc(
            source_values, params.source_layout, params.cycles, index_0);
        SolinasFp128 inc_1 = registers_val_inc(
            source_values, params.source_layout, params.cycles, index_1);
        SolinasFp128 inc_delta = solinas_sub(inc_1, inc_0);
        SolinasFp128 wa_0 = registers_val_wa(
            source_values,
            source_indices,
            eq_address,
            params.source_layout,
            params.cycles,
            index_0);
        SolinasFp128 wa_delta = solinas_sub(
            registers_val_wa(
                source_values,
                source_indices,
                eq_address,
                params.source_layout,
                params.cycles,
                index_1),
            wa_0);
        SolinasFp128 lt_0 = lt_lo[low_0];
        SolinasFp128 lt_delta = solinas_sub(lt_lo[low_0 + 1u], lt_0);
        SolinasFp128 inc_2 = solinas_add(inc_1, inc_delta);
        SolinasFp128 wa_2 = solinas_add(solinas_add(wa_0, wa_delta), wa_delta);
        SolinasFp128 lt_2 = solinas_add(solinas_add(lt_0, lt_delta), lt_delta);
        SolinasFp128 inc_at[REGISTERS_VAL_SAMPLES] = {
            inc_0,
            inc_2,
            solinas_add(inc_2, inc_delta),
        };
        SolinasFp128 wa_at[REGISTERS_VAL_SAMPLES] = {
            wa_0,
            wa_2,
            solinas_add(wa_2, wa_delta),
        };
        SolinasFp128 lt_at[REGISTERS_VAL_SAMPLES] = {
            lt_0,
            lt_2,
            solinas_add(lt_2, lt_delta),
        };
        for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
            SolinasFp128 product = solinas_mul_wide(inc_at[sample], wa_at[sample]);
            a[sample] = solinas_add(a[sample], product);
            b[sample] = solinas_add(
                b[sample],
                solinas_mul_wide(product, lt_at[sample]));
        }
    }

    uint simdgroups = threads / 32u;
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        a[sample] = solinas_simd_sum_32(a[sample]);
        b[sample] = solinas_simd_sum_32(b[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = a[sample];
            shared[(REGISTERS_VAL_SAMPLES + sample) * simdgroups + simdgroup] = b[sample];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
            SolinasFp128 a_sum = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            SolinasFp128 b_sum = lane < simdgroups
                ? shared[(REGISTERS_VAL_SAMPLES + sample) * simdgroups + lane]
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

struct RegistersValDenseRow {
    SolinasFp128 inc;
    SolinasFp128 wa;
};

kernel void solinas_registers_val_native_transition(
    device const uchar* source_values [[buffer(0)]],
    device const uchar* source_indices [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* lt_lo [[buffer(3)]],
    device const SolinasFp128* lt_hi [[buffer(4)]],
    device const SolinasFp128* eq_hi [[buffer(5)]],
    device RegistersValDenseRow* dense [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant SolinasFp128& challenge [[buffer(8)]],
    constant RegistersValMessageParams& params [[buffer(9)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 a[REGISTERS_VAL_SAMPLES];
    SolinasFp128 b[REGISTERS_VAL_SAMPLES];
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint source_high_base = high * 2u * params.lt_lo_length;
    uint destination_high_base = high * params.lt_lo_length;
    for (uint low_pair = thread_index; low_pair < low_pairs; low_pair += threads) {
        uint source = source_high_base + 4u * low_pair;
        uint destination = destination_high_base + 2u * low_pair;
        RegistersValDenseRow low;
        low.inc = registers_val_bind(
            registers_val_inc(
                source_values, params.source_layout, params.cycles, source),
            registers_val_inc(
                source_values, params.source_layout, params.cycles, source + 1u),
            challenge);
        low.wa = registers_val_bind(
            registers_val_wa(
                source_values,
                source_indices,
                eq_address,
                params.source_layout,
                params.cycles,
                source),
            registers_val_wa(
                source_values,
                source_indices,
                eq_address,
                params.source_layout,
                params.cycles,
                source + 1u),
            challenge);
        RegistersValDenseRow high_value;
        high_value.inc = registers_val_bind(
            registers_val_inc(
                source_values, params.source_layout, params.cycles, source + 2u),
            registers_val_inc(
                source_values, params.source_layout, params.cycles, source + 3u),
            challenge);
        high_value.wa = registers_val_bind(
            registers_val_wa(
                source_values,
                source_indices,
                eq_address,
                params.source_layout,
                params.cycles,
                source + 2u),
            registers_val_wa(
                source_values,
                source_indices,
                eq_address,
                params.source_layout,
                params.cycles,
                source + 3u),
            challenge);
        dense[destination] = low;
        dense[destination + 1u] = high_value;
        registers_val_accumulate_pair(
            low.inc,
            high_value.inc,
            low.wa,
            high_value.wa,
            lt_lo[2u * low_pair],
            lt_lo[2u * low_pair + 1u],
            a,
            b);
    }

    registers_val_finish_high(
        a, b, lt_hi, eq_hi, partials, params, shared,
        high, lane, simdgroup, threads);
}

kernel void solinas_registers_val_dense_transition(
    device const RegistersValDenseRow* source [[buffer(0)]],
    device RegistersValDenseRow* destination [[buffer(1)]],
    device const SolinasFp128* lt_lo [[buffer(2)]],
    device const SolinasFp128* lt_hi [[buffer(3)]],
    device const SolinasFp128* eq_hi [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant RegistersValMessageParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 a[REGISTERS_VAL_SAMPLES];
    SolinasFp128 b[REGISTERS_VAL_SAMPLES];
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        a[sample] = solinas_zero();
        b[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    uint source_high_base = high * 2u * params.lt_lo_length;
    uint destination_high_base = high * params.lt_lo_length;
    for (uint low_pair = thread_index; low_pair < low_pairs; low_pair += threads) {
        uint source_index = source_high_base + 4u * low_pair;
        uint destination_index = destination_high_base + 2u * low_pair;
        RegistersValDenseRow low;
        low.inc = registers_val_bind(
            source[source_index].inc,
            source[source_index + 1u].inc,
            challenge);
        low.wa = registers_val_bind(
            source[source_index].wa,
            source[source_index + 1u].wa,
            challenge);
        RegistersValDenseRow high_value;
        high_value.inc = registers_val_bind(
            source[source_index + 2u].inc,
            source[source_index + 3u].inc,
            challenge);
        high_value.wa = registers_val_bind(
            source[source_index + 2u].wa,
            source[source_index + 3u].wa,
            challenge);
        destination[destination_index] = low;
        destination[destination_index + 1u] = high_value;
        registers_val_accumulate_pair(
            low.inc,
            high_value.inc,
            low.wa,
            high_value.wa,
            lt_lo[2u * low_pair],
            lt_lo[2u * low_pair + 1u],
            a,
            b);
    }

    registers_val_finish_high(
        a, b, lt_hi, eq_hi, partials, params, shared,
        high, lane, simdgroup, threads);
}

kernel void solinas_registers_val_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RegistersValReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint sample = 0u; sample < REGISTERS_VAL_SAMPLES; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
