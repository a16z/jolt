#define REGISTERS_VAL_DIRECT_SAMPLES 3u
#define REGISTERS_VAL_DIRECT_SIMD_WIDTH 32u
#define REGISTERS_VAL_DIRECT_INVALID_INDEX 0u
#define REGISTERS_VAL_DIRECT_INVALID_PARAMS 1u

struct RegistersValDirectFirstParams {
    uint cycles;
    uint pairs;
    uint lt_lo_length;
    uint high_blocks;
    uint threadgroups;
    uint threads_per_threadgroup;
    uint address_domain;
    uint absent_register;
};

inline SolinasFp128 registers_val_direct_wa(
    device const uchar* rd_index,
    device const SolinasFp128* eq_address,
    device atomic_uint* audit,
    uint cycle,
    uint address_domain,
    uint absent_register)
{
    uint index = (uint)rd_index[cycle];
    if (index == absent_register) {
        return solinas_zero();
    }
    if (index >= address_domain) {
        atomic_fetch_add_explicit(
            &audit[REGISTERS_VAL_DIRECT_INVALID_INDEX],
            1u,
            memory_order_relaxed);
        return solinas_zero();
    }
    return eq_address[index];
}

inline void registers_val_direct_accumulate(
    SolinasFp128 inc,
    SolinasFp128 wa,
    SolinasFp128 lt_lo,
    SolinasFp128 lt_hi,
    SolinasFp128 eq_hi,
    thread SolinasFp128& sum)
{
    SolinasFp128 lt = solinas_add(
        lt_hi,
        solinas_mul_wide(eq_hi, lt_lo));
    SolinasFp128 inc_wa = solinas_mul_wide(inc, wa);
    sum = solinas_add(sum, solinas_mul_wide(inc_wa, lt));
}

inline void registers_val_direct_finish_group(
    thread SolinasFp128* sums,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint group,
    uint groups,
    uint lane,
    uint simdgroup,
    uint threads)
{
    uint simdgroups = threads / REGISTERS_VAL_DIRECT_SIMD_WIDTH;
    for (uint sample = 0u; sample < REGISTERS_VAL_DIRECT_SAMPLES; sample++) {
        sums[sample] = solinas_simd_sum_32(sums[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = sums[sample];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < REGISTERS_VAL_DIRECT_SAMPLES; sample++) {
            SolinasFp128 value = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            value = solinas_simd_sum_32(value);
            if (lane == 0u) {
                partials[sample * groups + group] = value;
            }
        }
    }
}

kernel void solinas_registers_val_direct_first_message(
    device const SolinasFp128* rd_inc [[buffer(0)]],
    device const uchar* rd_index [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* lt_lo [[buffer(3)]],
    device const SolinasFp128* lt_hi [[buffer(4)]],
    device const SolinasFp128* eq_hi [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RegistersValDirectFirstParams& params [[buffer(7)]],
    device atomic_uint* audit [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 actual_threadgroups [[threadgroups_per_grid]],
    uint threads [[threads_per_threadgroup]])
{
    ulong grid_threads =
        (ulong)params.threadgroups * (ulong)params.threads_per_threadgroup;
    bool supported = params.cycles >= 4u
        && (params.cycles & (params.cycles - 1u)) == 0u
        && params.pairs == params.cycles / 2u
        && params.lt_lo_length >= 2u
        && (params.lt_lo_length & (params.lt_lo_length - 1u)) == 0u
        && params.high_blocks > 0u
        && (params.high_blocks & (params.high_blocks - 1u)) == 0u
        && params.cycles % params.lt_lo_length == 0u
        && params.high_blocks == params.cycles / params.lt_lo_length
        && params.threadgroups > 0u
        && params.threads_per_threadgroup == threads
        && actual_threadgroups.x == params.threadgroups
        && actual_threadgroups.y == 1u
        && actual_threadgroups.z == 1u
        && params.address_domain == 128u
        && params.absent_register == 255u
        && threads >= REGISTERS_VAL_DIRECT_SIMD_WIDTH
        && threads % REGISTERS_VAL_DIRECT_SIMD_WIDTH == 0u
        && threads / REGISTERS_VAL_DIRECT_SIMD_WIDTH
            <= REGISTERS_VAL_DIRECT_SIMD_WIDTH
        && grid_threads <= 0xfffffffful;
    if (!supported) {
        if (group == 0u && tid == 0u) {
            atomic_fetch_add_explicit(
                &audit[REGISTERS_VAL_DIRECT_INVALID_PARAMS],
                1u,
                memory_order_relaxed);
        }
        return;
    }

    SolinasFp128 sums[REGISTERS_VAL_DIRECT_SAMPLES];
    for (uint sample = 0u; sample < REGISTERS_VAL_DIRECT_SAMPLES; sample++) {
        sums[sample] = solinas_zero();
    }

    uint low_pairs = params.lt_lo_length / 2u;
    for (ulong pair_wide = (ulong)gid;
         pair_wide < (ulong)params.pairs;
         pair_wide += grid_threads) {
        uint pair = (uint)pair_wide;
        uint high = pair / low_pairs;
        uint low_0 = 2u * (pair - high * low_pairs);
        uint cycle_0 = high * params.lt_lo_length + low_0;
        uint cycle_1 = cycle_0 + 1u;

        SolinasFp128 inc_0 = rd_inc[cycle_0];
        SolinasFp128 inc_1 = rd_inc[cycle_1];
        SolinasFp128 inc_delta = solinas_sub(inc_1, inc_0);
        SolinasFp128 wa_0 = registers_val_direct_wa(
            rd_index,
            eq_address,
            audit,
            cycle_0,
            params.address_domain,
            params.absent_register);
        SolinasFp128 wa_1 = registers_val_direct_wa(
            rd_index,
            eq_address,
            audit,
            cycle_1,
            params.address_domain,
            params.absent_register);
        SolinasFp128 wa_delta = solinas_sub(wa_1, wa_0);
        SolinasFp128 lt_0 = lt_lo[low_0];
        SolinasFp128 lt_1 = lt_lo[low_0 + 1u];
        SolinasFp128 lt_delta = solinas_sub(lt_1, lt_0);
        SolinasFp128 block_lt = lt_hi[high];
        SolinasFp128 block_eq = eq_hi[high];

        registers_val_direct_accumulate(
            inc_0, wa_0, lt_0, block_lt, block_eq, sums[0]);

        SolinasFp128 inc_2 = solinas_add(inc_1, inc_delta);
        SolinasFp128 wa_2 = solinas_add(wa_1, wa_delta);
        SolinasFp128 lt_2 = solinas_add(lt_1, lt_delta);
        registers_val_direct_accumulate(
            inc_2, wa_2, lt_2, block_lt, block_eq, sums[1]);

        registers_val_direct_accumulate(
            solinas_add(inc_2, inc_delta),
            solinas_add(wa_2, wa_delta),
            solinas_add(lt_2, lt_delta),
            block_lt,
            block_eq,
            sums[2]);
    }

    registers_val_direct_finish_group(
        sums,
        partials,
        shared,
        group,
        params.threadgroups,
        lane,
        simdgroup,
        threads);
}
