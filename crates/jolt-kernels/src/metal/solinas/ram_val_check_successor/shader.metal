// Concatenate after fp128.metal and simd_reduce.metal.

#define RAM_VAL_SUCCESSOR_COLUMNS 3u
#define RAM_VAL_SUCCESSOR_SIMD_WIDTH 32u
#define RAM_VAL_SUCCESSOR_FLAG_NONNEGATIVE 1u
#define RAM_VAL_SUCCESSOR_FLAG_RAM_INCREMENT 2u
#define RAM_VAL_SUCCESSOR_VALID_FLAGS 3u
#define RAM_VAL_SUCCESSOR_STATUS_UNSUPPORTED 1u
#define RAM_VAL_SUCCESSOR_STATUS_INVALID_ROW 2u

struct RamValSuccessorRow {
    ulong increment_magnitude;
    uint ram_address;
    uint flags;
};

struct RamValSuccessorFirstMessageParams {
    uint rows;
    uint high_blocks;
    uint low_length;
    uint address_domain;
    uint threads;
    uint no_address;
    uint reserved_0;
    uint reserved_1;
};

struct RamValSuccessorReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct RamValActivePair {
    uint pair_index;
    uint signs;
    ulong lo_magnitude;
    ulong hi_magnitude;
};

struct RamValSparseFirstMessageParams {
    uint active_pairs;
    uint rows;
    uint low_length;
    uint address_domain;
};

inline SolinasFp128 ram_val_successor_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 ram_val_sparse_signed_magnitude(
    ulong magnitude,
    bool negative)
{
    SolinasFp128 value = ram_val_successor_from_u64(magnitude);
    return negative ? solinas_sub(solinas_zero(), value) : value;
}

inline bool ram_val_successor_row_valid(
    RamValSuccessorRow row,
    constant RamValSuccessorFirstMessageParams& params)
{
    bool flags_valid = (row.flags & ~RAM_VAL_SUCCESSOR_VALID_FLAGS) == 0u;
    bool zero_valid = row.increment_magnitude != 0ul
        || (row.flags & RAM_VAL_SUCCESSOR_FLAG_NONNEGATIVE) != 0u;
    bool address_valid = row.ram_address == params.no_address
        || row.ram_address < params.address_domain;
    return flags_valid && zero_valid && address_valid;
}

inline bool ram_val_successor_nonzero_ram_increment(
    RamValSuccessorRow row)
{
    return row.increment_magnitude != 0ul
        && (row.flags & RAM_VAL_SUCCESSOR_FLAG_RAM_INCREMENT) != 0u;
}

inline SolinasFp128 ram_val_successor_ram_increment(
    RamValSuccessorRow row)
{
    if ((row.flags & RAM_VAL_SUCCESSOR_FLAG_RAM_INCREMENT) == 0u) {
        return solinas_zero();
    }
    SolinasFp128 magnitude = ram_val_successor_from_u64(
        row.increment_magnitude);
    return (row.flags & RAM_VAL_SUCCESSOR_FLAG_NONNEGATIVE) != 0u
        ? magnitude
        : solinas_sub(solinas_zero(), magnitude);
}

inline SolinasFp128 ram_val_successor_ra(
    RamValSuccessorRow row,
    device const SolinasFp128* eq_address,
    uint no_address)
{
    return row.ram_address == no_address
        ? solinas_zero()
        : eq_address[row.ram_address];
}

kernel void solinas_ram_val_check_successor_first_message(
    device const RamValSuccessorRow* rows [[buffer(0)]],
    device const SolinasFp128* eq_address [[buffer(1)]],
    device const SolinasFp128* lt_low [[buffer(2)]],
    device const SolinasFp128* lt_high [[buffer(3)]],
    device const SolinasFp128* eq_high [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RamValSuccessorFirstMessageParams& params [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint high [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool supported = threads == RAM_VAL_SUCCESSOR_SIMD_WIDTH
        && params.threads == RAM_VAL_SUCCESSOR_SIMD_WIDTH
        && params.no_address == 0xffffffffu
        && params.rows >= 2u
        && (params.rows & (params.rows - 1u)) == 0u
        && params.address_domain != 0u
        && (params.address_domain & (params.address_domain - 1u)) == 0u
        && params.low_length >= 2u
        && (params.low_length & (params.low_length - 1u)) == 0u
        && params.high_blocks != 0u
        && (params.high_blocks & (params.high_blocks - 1u)) == 0u
        && params.reserved_0 == 0u
        && params.reserved_1 == 0u
        && (ulong)params.high_blocks * (ulong)params.low_length
            == (ulong)params.rows;
    if (!supported) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_VAL_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (high >= params.high_blocks) {
        return;
    }

    SolinasFp128 a0 = solinas_zero();
    SolinasFp128 a2 = solinas_zero();
    SolinasFp128 a3 = solinas_zero();
    SolinasFp128 b0 = solinas_zero();
    SolinasFp128 b2 = solinas_zero();
    SolinasFp128 b3 = solinas_zero();

    uint low_pairs = params.low_length / 2u;
    uint high_base = high * params.low_length;
    for (uint low_pair = tid; low_pair < low_pairs; low_pair += threads) {
        uint low = 2u * low_pair;
        uint index = high_base + low;
        RamValSuccessorRow row0 = rows[index];
        RamValSuccessorRow row1 = rows[index + 1u];
        bool valid0 = ram_val_successor_row_valid(row0, params);
        bool valid1 = ram_val_successor_row_valid(row1, params);
        if (!valid0 || !valid1) {
            atomic_fetch_or_explicit(
                status,
                RAM_VAL_SUCCESSOR_STATUS_INVALID_ROW,
                memory_order_relaxed);
        }
        bool pair_active = valid0 && valid1
            && (ram_val_successor_nonzero_ram_increment(row0)
                || ram_val_successor_nonzero_ram_increment(row1));
        if (!simd_any(pair_active)) {
            continue;
        }
        if (pair_active) {
            SolinasFp128 inc0 = ram_val_successor_ram_increment(row0);
            SolinasFp128 inc1 = ram_val_successor_ram_increment(row1);
            SolinasFp128 ra0 = ram_val_successor_ra(
                row0, eq_address, params.no_address);
            SolinasFp128 ra1 = ram_val_successor_ra(
                row1, eq_address, params.no_address);
            SolinasFp128 lt0 = lt_low[low];
            SolinasFp128 lt1 = lt_low[low + 1u];

            SolinasFp128 inc_delta = solinas_sub(inc1, inc0);
            SolinasFp128 ra_delta = solinas_sub(ra1, ra0);
            SolinasFp128 lt_delta = solinas_sub(lt1, lt0);

            SolinasFp128 product = solinas_mul_wide(inc0, ra0);
            a0 = solinas_add(a0, product);
            b0 = solinas_add(b0, solinas_mul_wide(product, lt0));

            inc1 = solinas_add(inc1, inc_delta);
            ra1 = solinas_add(ra1, ra_delta);
            lt1 = solinas_add(lt1, lt_delta);
            product = solinas_mul_wide(inc1, ra1);
            a2 = solinas_add(a2, product);
            b2 = solinas_add(b2, solinas_mul_wide(product, lt1));

            inc1 = solinas_add(inc1, inc_delta);
            ra1 = solinas_add(ra1, ra_delta);
            lt1 = solinas_add(lt1, lt_delta);
            product = solinas_mul_wide(inc1, ra1);
            a3 = solinas_add(a3, product);
            b3 = solinas_add(b3, solinas_mul_wide(product, lt1));
        }
    }

    a0 = solinas_simd_sum_32(a0);
    a2 = solinas_simd_sum_32(a2);
    a3 = solinas_simd_sum_32(a3);
    b0 = solinas_simd_sum_32(b0);
    b2 = solinas_simd_sum_32(b2);
    b3 = solinas_simd_sum_32(b3);
    if (lane == 0u) {
        SolinasFp128 lt = lt_high[high];
        SolinasFp128 eq = eq_high[high];
        partials[high] = solinas_add(
            solinas_mul_wide(lt, a0), solinas_mul_wide(eq, b0));
        partials[params.high_blocks + high] = solinas_add(
            solinas_mul_wide(lt, a2), solinas_mul_wide(eq, b2));
        partials[2u * params.high_blocks + high] = solinas_add(
            solinas_mul_wide(lt, a3), solinas_mul_wide(eq, b3));
    }
}

kernel void solinas_ram_val_check_sparse_first_message(
    device const RamValActivePair* active_pairs [[buffer(0)]],
    device const uint* addresses [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* lt_low [[buffer(3)]],
    device const SolinasFp128* lt_high [[buffer(4)]],
    device const SolinasFp128* eq_high [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RamValSparseFirstMessageParams& params [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool supported = threads == RAM_VAL_SUCCESSOR_SIMD_WIDTH
        && params.active_pairs != 0u
        && params.rows >= 2u
        && (params.rows & (params.rows - 1u)) == 0u
        && params.low_length >= 2u
        && (params.low_length & (params.low_length - 1u)) == 0u
        && params.rows % params.low_length == 0u
        && params.address_domain != 0u
        && (params.address_domain & (params.address_domain - 1u)) == 0u;
    if (!supported) {
        if (lane == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_VAL_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }

    uint active_index = group * RAM_VAL_SUCCESSOR_SIMD_WIDTH + lane;
    SolinasFp128 eval0 = solinas_zero();
    SolinasFp128 eval2 = solinas_zero();
    SolinasFp128 eval3 = solinas_zero();
    if (active_index < params.active_pairs) {
        RamValActivePair active = active_pairs[active_index];
        ulong cycle = 2ul * (ulong)active.pair_index;
        bool pair_valid = cycle + 1ul < (ulong)params.rows;
        if (!pair_valid) {
            atomic_fetch_or_explicit(
                status,
                RAM_VAL_SUCCESSOR_STATUS_INVALID_ROW,
                memory_order_relaxed);
        } else {
            uint address0 = addresses[cycle];
            uint address1 = addresses[cycle + 1ul];
            bool address0_valid = address0 == 0xffffffffu
                || address0 < params.address_domain;
            bool address1_valid = address1 == 0xffffffffu
                || address1 < params.address_domain;
            if (!address0_valid || !address1_valid) {
                atomic_fetch_or_explicit(
                    status,
                    RAM_VAL_SUCCESSOR_STATUS_INVALID_ROW,
                    memory_order_relaxed);
            } else {
                SolinasFp128 inc0 = ram_val_sparse_signed_magnitude(
                    active.lo_magnitude,
                    (active.signs & 1u) != 0u);
                SolinasFp128 inc1 = ram_val_sparse_signed_magnitude(
                    active.hi_magnitude,
                    (active.signs & 2u) != 0u);
                SolinasFp128 ra0 = address0 == 0xffffffffu
                    ? solinas_zero()
                    : eq_address[address0];
                SolinasFp128 ra1 = address1 == 0xffffffffu
                    ? solinas_zero()
                    : eq_address[address1];
                uint low0 = (uint)cycle & (params.low_length - 1u);
                uint high = (uint)(cycle / (ulong)params.low_length);
                SolinasFp128 lt0 = solinas_add(
                    lt_high[high],
                    solinas_mul_wide(eq_high[high], lt_low[low0]));
                SolinasFp128 lt1 = solinas_add(
                    lt_high[high],
                    solinas_mul_wide(eq_high[high], lt_low[low0 + 1u]));

                SolinasFp128 inc_delta = solinas_sub(inc1, inc0);
                SolinasFp128 ra_delta = solinas_sub(ra1, ra0);
                SolinasFp128 lt_delta = solinas_sub(lt1, lt0);

                eval0 = solinas_mul_wide(solinas_mul_wide(inc0, ra0), lt0);
                inc1 = solinas_add(inc1, inc_delta);
                ra1 = solinas_add(ra1, ra_delta);
                lt1 = solinas_add(lt1, lt_delta);
                eval2 = solinas_mul_wide(solinas_mul_wide(inc1, ra1), lt1);
                inc1 = solinas_add(inc1, inc_delta);
                ra1 = solinas_add(ra1, ra_delta);
                lt1 = solinas_add(lt1, lt_delta);
                eval3 = solinas_mul_wide(solinas_mul_wide(inc1, ra1), lt1);
            }
        }
    }

    eval0 = solinas_simd_sum_32(eval0);
    eval2 = solinas_simd_sum_32(eval2);
    eval3 = solinas_simd_sum_32(eval3);
    if (lane == 0u) {
        uint groups = (params.active_pairs + RAM_VAL_SUCCESSOR_SIMD_WIDTH - 1u)
            / RAM_VAL_SUCCESSOR_SIMD_WIDTH;
        partials[group] = eval0;
        partials[groups + group] = eval2;
        partials[2u * groups + group] = eval3;
    }
}

kernel void solinas_ram_val_check_successor_reduce3(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamValSuccessorReductionParams& params [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    ulong expected_output =
        ((ulong)params.input_count + (ulong)RAM_VAL_SUCCESSOR_SIMD_WIDTH - 1ul)
        / (ulong)RAM_VAL_SUCCESSOR_SIMD_WIDTH;
    bool supported = threads == RAM_VAL_SUCCESSOR_SIMD_WIDTH
        && params.input_count != 0u
        && (ulong)params.output_count == expected_output
        && params.columns == RAM_VAL_SUCCESSOR_COLUMNS
        && params.reserved == 0u;
    if (!supported) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_VAL_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    uint output_index = gid / RAM_VAL_SUCCESSOR_SIMD_WIDTH;
    if (output_index >= params.output_count) {
        return;
    }
    for (uint column = 0u; column < RAM_VAL_SUCCESSOR_COLUMNS; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[column * params.output_count + output_index] = value;
        }
    }
}
