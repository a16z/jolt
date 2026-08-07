#if SOLINAS_OFFSET != 0xffffa7f7u
#error "bytecode address successor requires the Akita Solinas offset"
#endif

#define BYTECODE_ADDRESS_STAGES 9u
#define BYTECODE_ADDRESS_BASE_STAGES 5u
#define BYTECODE_ADDRESS_SIMD_WIDTH 32u
#define BYTECODE_ADDRESS_CSR_THREADS 1024u
#define BYTECODE_ADDRESS_BINS_PER_THREAD 8u
#define BYTECODE_ADDRESS_INNER_LENGTH 32768u
#define BYTECODE_ADDRESS_ACCUMULATOR_WORDS 5u
#define BYTECODE_ADDRESS_STATUS_SHORT_RUNS 0u
#define BYTECODE_ADDRESS_STATUS_LONG_RUNS 1u
#define BYTECODE_ADDRESS_STATUS_INVALID_ROWS 2u
#define BYTECODE_ADDRESS_STATUS_COMPLETED_GROUPS 3u
#define BYTECODE_ADDRESS_STATUS_OCCURRENCE_ROWS 4u
#define BYTECODE_ADDRESS_DIAGNOSTIC_SHORT_OCCURRENCES 0u
#define BYTECODE_ADDRESS_DIAGNOSTIC_LONG_OCCURRENCES 1u
#define BYTECODE_ADDRESS_DIAGNOSTIC_MAXIMUM_RUN 2u
#define BYTECODE_ADDRESS_DIAGNOSTIC_HISTOGRAM_BASE 4u

struct BytecodeReadRafRowWords {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

struct BytecodeReadRafRun {
    uint start;
    uint count;
    uint outer;
    uint address;
};

struct BytecodeReadRafCsrParams {
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint run_capacity;
    uint short_threshold;
    uint bins_per_thread;
    uint reserved;
};

struct BytecodeReadRafPushforwardParams {
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint run_capacity;
    uint short_threshold;
    uint short_threads;
    uint long_threads;
    uint stages;
    uint base_stages;
    uint accumulator_words;
    uint reserved;
};

struct BytecodeReadRafIndirectGrid {
    uint4 words;
};

struct BytecodeReadRafDispatchArgs {
    BytecodeReadRafIndirectGrid short_runs;
    BytecodeReadRafIndirectGrid long_runs;
};

struct BytecodeReadRafWide192 {
    uint limb[6];
};

inline uint bytecode_address_push_pc(
    BytecodeReadRafRowWords row,
    uint addresses,
    thread bool& valid)
{
    ulong plus_one = row.packed_pc_and_flags & 0x00fffffffffffffful;
    ulong pc = plus_one == 0ul ? 0ul : plus_one - 1ul;
    valid = pc < (ulong)addresses;
    return valid ? (uint)pc : 0u;
}

inline uint bytecode_address_simd_inclusive_sum(uint value, uint lane) {
    for (ushort offset = 1; offset < BYTECODE_ADDRESS_SIMD_WIDTH; offset <<= 1) {
        uint previous = simd_shuffle_up(value, offset);
        if (lane >= (uint)offset) {
            value += previous;
        }
    }
    return value;
}

inline uint bytecode_address_threadgroup_exclusive_sum(
    uint local_total,
    uint lane,
    uint simdgroup,
    threadgroup atomic_uint* scratch,
    thread uint& total)
{
    uint simd_inclusive = bytecode_address_simd_inclusive_sum(local_total, lane);
    if (lane == BYTECODE_ADDRESS_SIMD_WIDTH - 1u) {
        atomic_store_explicit(
            &scratch[simdgroup],
            simd_inclusive,
            memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        uint group_total = atomic_load_explicit(&scratch[lane], memory_order_relaxed);
        uint group_inclusive = bytecode_address_simd_inclusive_sum(group_total, lane);
        atomic_store_explicit(
            &scratch[lane],
            group_inclusive - group_total,
            memory_order_relaxed);
        if (lane == BYTECODE_ADDRESS_SIMD_WIDTH - 1u) {
            atomic_store_explicit(
                &scratch[BYTECODE_ADDRESS_SIMD_WIDTH],
                group_inclusive,
                memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint group_prefix = atomic_load_explicit(&scratch[simdgroup], memory_order_relaxed);
    total = atomic_load_explicit(
        &scratch[BYTECODE_ADDRESS_SIMD_WIDTH],
        memory_order_relaxed);
    uint exclusive = group_prefix + simd_inclusive - local_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return exclusive;
}

inline SolinasFp128 bytecode_address_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32u);
    return result;
}

inline SolinasFp128 bytecode_address_neg(SolinasFp128 value) {
    return solinas_sub(solinas_zero(), value);
}

inline BytecodeReadRafWide192 bytecode_address_product_u64(
    SolinasFp128 lhs,
    ulong rhs)
{
    uint rhs_limb[2] = {(uint)rhs, (uint)(rhs >> 32u)};
    BytecodeReadRafWide192 product;
    for (uint i = 0u; i < 6u; i++) {
        product.limb[i] = 0u;
    }
    for (uint i = 0u; i < 4u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 2u; j++) {
            uint k = i + j;
            ulong word = (ulong)lhs.limb[i] * (ulong)rhs_limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32u;
        }
        product.limb[i + 2u] = (uint)carry;
    }
    return product;
}

inline SolinasFp128 bytecode_address_reduce_u192(BytecodeReadRafWide192 product) {
    SolinasFp128 folded;
    ulong carry = 0ul;
    for (uint i = 0u; i < 2u; i++) {
        ulong word = (ulong)product.limb[i + 4u] * (ulong)SOLINAS_OFFSET
            + (ulong)product.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }
    for (uint i = 2u; i < 4u; i++) {
        ulong word = (ulong)product.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }

    ulong word = (ulong)folded.limb[0] + carry * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32u;
    for (uint i = 1u; i < 4u; i++) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }

    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(
        carry != 0ul || corrected.carry != 0u,
        corrected.value,
        folded);
}

inline SolinasFp128 bytecode_address_mul_signed_u64(
    SolinasFp128 coefficient,
    ulong magnitude,
    bool negative)
{
    SolinasFp128 product = bytecode_address_reduce_u192(
        bytecode_address_product_u64(coefficient, magnitude));
    return negative ? bytecode_address_neg(product) : product;
}

inline SolinasFp128 bytecode_address_mul_signed_full(
    SolinasFp128 coefficient,
    ulong magnitude,
    bool negative)
{
    SolinasFp128 product = solinas_mul_wide(
        coefficient,
        bytecode_address_from_u64(magnitude));
    return negative ? bytecode_address_neg(product) : product;
}

inline void bytecode_address_device_atomic_add_5(
    device atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * BYTECODE_ADDRESS_ACCUMULATOR_WORDS;
    uint carry = 0u;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32u) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0u) {
        atomic_fetch_add_explicit(
            &sums[base + 4u],
            carry,
            memory_order_relaxed);
    }
}

inline SolinasFp128 bytecode_address_device_atomic_reduce_5(
    device atomic_uint* sums,
    uint field)
{
    uint base = field * BYTECODE_ADDRESS_ACCUMULATOR_WORDS;
    SolinasFp128 low;
    for (uint limb = 0u; limb < 4u; limb++) {
        low.limb[limb] = atomic_load_explicit(
            &sums[base + limb],
            memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(
        &sums[base + 4u],
        memory_order_relaxed);

    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0u, canonical.value, low);
    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return solinas_add(low, correction);
}

kernel void solinas_bytecode_address_build_csr(
    device const BytecodeReadRafRowWords* rows [[buffer(0)]],
    device uint* occurrences [[buffer(1)]],
    device BytecodeReadRafRun* runs [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant BytecodeReadRafCsrParams& params [[buffer(4)]],
    device atomic_uint* diagnostics [[buffer(5)]],
    threadgroup atomic_uint* bins [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint outer [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool supported = threads == BYTECODE_ADDRESS_CSR_THREADS
        && params.addresses == BYTECODE_ADDRESS_CSR_THREADS
            * BYTECODE_ADDRESS_BINS_PER_THREAD
        && params.bins_per_thread == BYTECODE_ADDRESS_BINS_PER_THREAD
        && params.inner_length == BYTECODE_ADDRESS_INNER_LENGTH
        && params.rows == params.inner_length * params.outer_length
        && params.run_capacity != 0u
        && outer < params.outer_length;
    if (!supported) {
        if (tid == 0u && outer < params.outer_length) {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
                1u,
                memory_order_relaxed);
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_STATUS_COMPLETED_GROUPS],
                1u,
                memory_order_relaxed);
        }
        return;
    }

    for (uint address = tid; address < params.addresses; address += threads) {
        atomic_store_explicit(&bins[address], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outer_base = outer * params.inner_length;
    for (uint inner = tid; inner < params.inner_length; inner += threads) {
        uint row_index = outer_base + inner;
        bool valid;
        uint address = bytecode_address_push_pc(rows[row_index], params.addresses, valid);
        if (!valid) {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
                1u,
                memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&bins[address], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_counts[BYTECODE_ADDRESS_BINS_PER_THREAD];
    uint local_total = 0u;
    uint first_address = tid * BYTECODE_ADDRESS_BINS_PER_THREAD;
    for (uint i = 0u; i < BYTECODE_ADDRESS_BINS_PER_THREAD; i++) {
        uint count = atomic_load_explicit(
            &bins[first_address + i],
            memory_order_relaxed);
        local_counts[i] = count;
        local_total += count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint occurrence_total;
    uint occurrence_prefix = bytecode_address_threadgroup_exclusive_sum(
        local_total, lane, simdgroup, bins, occurrence_total);
    if (tid == 0u && occurrence_total != params.inner_length) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
            1u,
            memory_order_relaxed);
    }

    uint local_short_runs = 0u;
    uint local_long_runs = 0u;
    for (uint i = 0u; i < BYTECODE_ADDRESS_BINS_PER_THREAD; i++) {
        uint count = local_counts[i];
        local_short_runs += (uint)(count != 0u && count <= params.short_threshold);
        local_long_runs += (uint)(count > params.short_threshold);
    }
    uint short_total;
    uint short_prefix = bytecode_address_threadgroup_exclusive_sum(
        local_short_runs, lane, simdgroup, bins, short_total);
    uint long_total;
    uint long_prefix = bytecode_address_threadgroup_exclusive_sum(
        local_long_runs, lane, simdgroup, bins, long_total);

    if (tid == 0u) {
        uint short_base = atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_STATUS_SHORT_RUNS],
            short_total,
            memory_order_relaxed);
        uint long_base = atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_STATUS_LONG_RUNS],
            long_total,
            memory_order_relaxed);
        atomic_store_explicit(&bins[0], short_base, memory_order_relaxed);
        atomic_store_explicit(&bins[1], long_base, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint short_base = atomic_load_explicit(&bins[0], memory_order_relaxed);
    uint long_base = atomic_load_explicit(&bins[1], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint short_index = short_base + short_prefix;
    uint long_index = long_base + long_prefix;
    uint running = occurrence_prefix;
    for (uint i = 0u; i < BYTECODE_ADDRESS_BINS_PER_THREAD; i++) {
        uint address = first_address + i;
        uint count = local_counts[i];
        uint run_start = running;
        atomic_store_explicit(&bins[address], run_start, memory_order_relaxed);
        if (count != 0u) {
            uint occurrence_index = count <= params.short_threshold
                ? BYTECODE_ADDRESS_DIAGNOSTIC_SHORT_OCCURRENCES
                : BYTECODE_ADDRESS_DIAGNOSTIC_LONG_OCCURRENCES;
            atomic_fetch_add_explicit(
                &diagnostics[occurrence_index],
                count,
                memory_order_relaxed);
            atomic_fetch_max_explicit(
                &diagnostics[BYTECODE_ADDRESS_DIAGNOSTIC_MAXIMUM_RUN],
                count,
                memory_order_relaxed);
            uint histogram_bucket = 31u - clz(count);
            atomic_fetch_add_explicit(
                &diagnostics[
                    BYTECODE_ADDRESS_DIAGNOSTIC_HISTOGRAM_BASE + histogram_bucket],
                1u,
                memory_order_relaxed);
            BytecodeReadRafRun run;
            run.start = outer_base + run_start;
            run.count = count;
            run.outer = outer;
            run.address = address;
            if (count <= params.short_threshold) {
                if (short_index < params.run_capacity) {
                    runs[short_index] = run;
                } else {
                    atomic_fetch_add_explicit(
                        &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
                        1u,
                        memory_order_relaxed);
                }
                short_index++;
            } else {
                if (long_index < params.run_capacity) {
                    runs[params.run_capacity - 1u - long_index] = run;
                } else {
                    atomic_fetch_add_explicit(
                        &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
                        1u,
                        memory_order_relaxed);
                }
                long_index++;
            }
        }
        running += count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint inner = tid; inner < params.inner_length; inner += threads) {
        uint row_index = outer_base + inner;
        bool valid;
        uint address = bytecode_address_push_pc(rows[row_index], params.addresses, valid);
        (void)valid;
        uint destination = atomic_fetch_add_explicit(
            &bins[address],
            1u,
            memory_order_relaxed);
        if (destination < params.inner_length) {
            occurrences[outer_base + destination] = row_index;
        } else {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS],
                1u,
                memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_STATUS_OCCURRENCE_ROWS],
            occurrence_total,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_STATUS_COMPLETED_GROUPS],
            1u,
            memory_order_relaxed);
    }
}

kernel void solinas_bytecode_address_write_dispatch(
    device const uint* status [[buffer(0)]],
    device BytecodeReadRafDispatchArgs* output [[buffer(1)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) {
        return;
    }
    uint short_runs = status[BYTECODE_ADDRESS_STATUS_SHORT_RUNS];
    uint long_runs = status[BYTECODE_ADDRESS_STATUS_LONG_RUNS];
    uint total_runs = short_runs + long_runs;
    bool valid = params.short_threads != 0u
        && params.long_threads >= BYTECODE_ADDRESS_SIMD_WIDTH
        && params.long_threads % BYTECODE_ADDRESS_SIMD_WIDTH == 0u
        && status[BYTECODE_ADDRESS_STATUS_INVALID_ROWS] == 0u
        && status[BYTECODE_ADDRESS_STATUS_COMPLETED_GROUPS] == params.outer_length
        && status[BYTECODE_ADDRESS_STATUS_OCCURRENCE_ROWS] == params.rows
        && total_runs >= params.outer_length
        && total_runs <= params.run_capacity
        && total_runs >= short_runs;
    if (!valid) {
        output[0].short_runs.words = uint4(0u);
        output[0].long_runs.words = uint4(0u);
        return;
    }
    uint short_groups = (short_runs
        + params.short_threads - 1u) / params.short_threads;
    uint long_workers = params.long_threads / BYTECODE_ADDRESS_SIMD_WIDTH;
    uint long_groups = (long_runs
        + long_workers - 1u) / long_workers;
    output[0].short_runs.words = uint4(short_groups, 1u, 1u, 0u);
    output[0].long_runs.words = uint4(long_groups, 1u, 1u, 0u);
}

template <bool use_u64>
inline SolinasFp128 bytecode_address_fused_product(
    SolinasFp128 coefficient,
    BytecodeReadRafRowWords row)
{
    bool negative = (row.packed_pc_and_flags >> 63u) != 0ul;
    if (use_u64) {
        return bytecode_address_mul_signed_u64(
            coefficient,
            row.fused_inc_magnitude,
            negative);
    }
    return bytecode_address_mul_signed_full(
        coefficient,
        row.fused_inc_magnitude,
        negative);
}

template <bool use_u64>
inline void bytecode_address_short_runs_impl(
    device const BytecodeReadRafRowWords* rows,
    device const uint* occurrences,
    device const BytecodeReadRafRun* runs,
    device const uint* counters,
    device const SolinasFp128* e_lo,
    device const SolinasFp128* e_hi,
    device atomic_uint* output,
    constant BytecodeReadRafPushforwardParams& params,
    uint run_index)
{
    if (run_index >= counters[BYTECODE_ADDRESS_STATUS_SHORT_RUNS]) {
        return;
    }
    BytecodeReadRafRun run = runs[run_index];
    SolinasFp128 sums[BYTECODE_ADDRESS_STAGES];
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_STAGES; stage++) {
        sums[stage] = solinas_zero();
    }
    uint inner_mask = params.inner_length - 1u;
    for (uint offset = 0u; offset < run.count; offset++) {
        uint row_index = occurrences[run.start + offset];
        BytecodeReadRafRowWords row = rows[row_index];
        uint inner = row_index & inner_mask;
        for (uint stage = 0u; stage < BYTECODE_ADDRESS_BASE_STAGES; stage++) {
            sums[stage] = solinas_add(
                sums[stage],
                e_lo[stage * params.inner_length + inner]);
        }
        for (uint stage = BYTECODE_ADDRESS_BASE_STAGES;
             stage < BYTECODE_ADDRESS_STAGES;
             stage++) {
            SolinasFp128 term = bytecode_address_fused_product<use_u64>(
                e_lo[stage * params.inner_length + inner],
                row);
            sums[stage] = solinas_add(sums[stage], term);
        }
    }
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_STAGES; stage++) {
        SolinasFp128 value = solinas_mul_wide(
            sums[stage],
            e_hi[stage * params.outer_length + run.outer]);
        bytecode_address_device_atomic_add_5(
            output,
            stage * params.addresses + run.address,
            value);
    }
}

template <bool use_u64>
inline void bytecode_address_long_runs_impl(
    device const BytecodeReadRafRowWords* rows,
    device const uint* occurrences,
    device const BytecodeReadRafRun* runs,
    device const uint* counters,
    device const SolinasFp128* e_lo,
    device const SolinasFp128* e_hi,
    device atomic_uint* output,
    constant BytecodeReadRafPushforwardParams& params,
    uint run_index,
    uint lane)
{
    if (run_index >= counters[BYTECODE_ADDRESS_STATUS_LONG_RUNS]) {
        return;
    }
    BytecodeReadRafRun run = runs[params.run_capacity - 1u - run_index];
    SolinasFp128 sums[BYTECODE_ADDRESS_STAGES];
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_STAGES; stage++) {
        sums[stage] = solinas_zero();
    }
    uint inner_mask = params.inner_length - 1u;
    for (uint offset = lane; offset < run.count; offset += BYTECODE_ADDRESS_SIMD_WIDTH) {
        uint row_index = occurrences[run.start + offset];
        BytecodeReadRafRowWords row = rows[row_index];
        uint inner = row_index & inner_mask;
        for (uint stage = 0u; stage < BYTECODE_ADDRESS_BASE_STAGES; stage++) {
            sums[stage] = solinas_add(
                sums[stage],
                e_lo[stage * params.inner_length + inner]);
        }
        for (uint stage = BYTECODE_ADDRESS_BASE_STAGES;
             stage < BYTECODE_ADDRESS_STAGES;
             stage++) {
            SolinasFp128 term = bytecode_address_fused_product<use_u64>(
                e_lo[stage * params.inner_length + inner],
                row);
            sums[stage] = solinas_add(sums[stage], term);
        }
    }
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_STAGES; stage++) {
        SolinasFp128 sum = solinas_simd_sum_32(sums[stage]);
        if (lane == 0u) {
            SolinasFp128 value = solinas_mul_wide(
                sum,
                e_hi[stage * params.outer_length + run.outer]);
            bytecode_address_device_atomic_add_5(
                output,
                stage * params.addresses + run.address,
                value);
        }
    }
}

kernel void solinas_bytecode_address_short_runs_u64(
    device const BytecodeReadRafRowWords* rows [[buffer(0)]],
    device const uint* occurrences [[buffer(1)]],
    device const BytecodeReadRafRun* runs [[buffer(2)]],
    device const uint* counters [[buffer(3)]],
    device const SolinasFp128* e_lo [[buffer(4)]],
    device const SolinasFp128* e_hi [[buffer(5)]],
    device atomic_uint* output [[buffer(6)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    bytecode_address_short_runs_impl<true>(
        rows, occurrences, runs, counters, e_lo, e_hi, output, params, gid);
}

kernel void solinas_bytecode_address_short_runs_full(
    device const BytecodeReadRafRowWords* rows [[buffer(0)]],
    device const uint* occurrences [[buffer(1)]],
    device const BytecodeReadRafRun* runs [[buffer(2)]],
    device const uint* counters [[buffer(3)]],
    device const SolinasFp128* e_lo [[buffer(4)]],
    device const SolinasFp128* e_hi [[buffer(5)]],
    device atomic_uint* output [[buffer(6)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    bytecode_address_short_runs_impl<false>(
        rows, occurrences, runs, counters, e_lo, e_hi, output, params, gid);
}

kernel void solinas_bytecode_address_long_runs_u64(
    device const BytecodeReadRafRowWords* rows [[buffer(0)]],
    device const uint* occurrences [[buffer(1)]],
    device const BytecodeReadRafRun* runs [[buffer(2)]],
    device const uint* counters [[buffer(3)]],
    device const SolinasFp128* e_lo [[buffer(4)]],
    device const SolinasFp128* e_hi [[buffer(5)]],
    device atomic_uint* output [[buffer(6)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]])
{
    uint run_index = group * (threads / BYTECODE_ADDRESS_SIMD_WIDTH) + simdgroup;
    bytecode_address_long_runs_impl<true>(
        rows, occurrences, runs, counters, e_lo, e_hi, output, params, run_index, lane);
}

kernel void solinas_bytecode_address_long_runs_full(
    device const BytecodeReadRafRowWords* rows [[buffer(0)]],
    device const uint* occurrences [[buffer(1)]],
    device const BytecodeReadRafRun* runs [[buffer(2)]],
    device const uint* counters [[buffer(3)]],
    device const SolinasFp128* e_lo [[buffer(4)]],
    device const SolinasFp128* e_hi [[buffer(5)]],
    device atomic_uint* output [[buffer(6)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]])
{
    uint run_index = group * (threads / BYTECODE_ADDRESS_SIMD_WIDTH) + simdgroup;
    bytecode_address_long_runs_impl<false>(
        rows, occurrences, runs, counters, e_lo, e_hi, output, params, run_index, lane);
}

kernel void solinas_bytecode_address_finalize(
    device atomic_uint* sums [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BytecodeReadRafPushforwardParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint fields = params.stages * params.addresses;
    if (gid < fields) {
        output[gid] = bytecode_address_device_atomic_reduce_5(sums, gid);
    }
}
