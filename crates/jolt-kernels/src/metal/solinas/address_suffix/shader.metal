#define ADDRESS_SUFFIX_BINS 256u
#define ADDRESS_SUFFIX_WORDS 5u

struct AddressSuffixJob {
    uint start;
    uint end;
    uint table;
    uint reserved;
};

struct AddressSuffixTableJobs {
    uint start;
    uint end;
};

inline void address_suffix_atomic_add(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * ADDRESS_SUFFIX_WORDS;
    uint carry = 0;
    for (uint limb = 0; limb < 4; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0) {
        atomic_fetch_add_explicit(&sums[base + 4], carry, memory_order_relaxed);
    }
}

inline SolinasFp128 address_suffix_reduce_atomic_sum(
    threadgroup atomic_uint* sums,
    uint field)
{
    uint base = field * ADDRESS_SUFFIX_WORDS;
    SolinasFp128 low;
    for (uint limb = 0; limb < 4; limb++) {
        low.limb[limb] = atomic_load_explicit(&sums[base + limb], memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(&sums[base + 4], memory_order_relaxed);
    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0, canonical.value, low);

    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32);
    return solinas_add(low, correction);
}

kernel void solinas_address_suffix_one_tile(
    device const ushort* keys [[buffer(0)]],
    device const SolinasFp128* weights [[buffer(1)]],
    device const AddressSuffixJob* jobs [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint job_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint counter = tid; counter < ADDRESS_SUFFIX_BINS * ADDRESS_SUFFIX_WORDS; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AddressSuffixJob job = jobs[job_index];
    for (uint row = job.start + tid; row < job.end; row += threads) {
        uint chunk = keys[row] & (ADDRESS_SUFFIX_BINS - 1);
        address_suffix_atomic_add(sums, chunk, weights[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_base = job_index * ADDRESS_SUFFIX_BINS;
    for (uint chunk = tid; chunk < ADDRESS_SUFFIX_BINS; chunk += threads) {
        partials[output_base + chunk] = address_suffix_reduce_atomic_sum(sums, chunk);
    }
}

kernel void solinas_address_suffix_one_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device const AddressSuffixTableJobs* table_jobs [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    uint table [[threadgroup_position_in_grid]],
    uint chunk [[thread_index_in_threadgroup]])
{
    AddressSuffixTableJobs jobs = table_jobs[table];
    SolinasFp128 sum = solinas_zero();
    for (uint job = jobs.start; job < jobs.end; job++) {
        sum = solinas_add(sum, partials[job * ADDRESS_SUFFIX_BINS + chunk]);
    }
    output[table * ADDRESS_SUFFIX_BINS + chunk] = sum;
}
