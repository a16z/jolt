// Concatenate after fp128.metal and simd_reduce.metal.

struct RamRaClaimReductionParams {
    uint prefix_elements;
    uint suffix_elements;
    uint active_high_elements;
    uint no_access;
    uint q_slices;
    uint active_q_slices;
};

struct RamRaCompactRecord {
    uint cycle;
    uint address;
};

struct RamRaQRecord {
    uint x_hi;
    uint address;
};

struct RamRaWide288 {
    uint limb[9];
};

constant uint RAM_RA_ACTIVITY_BLOCK = 32u;

inline RamRaWide288 ram_ra_wide_zero() {
    RamRaWide288 result;
    for (uint i = 0u; i < 9u; i++) {
        result.limb[i] = 0u;
    }
    return result;
}

inline void ram_ra_wide_add_product(
    thread RamRaWide288& accumulator,
    SolinasFp128 lhs,
    SolinasFp128 rhs)
{
    SolinasWide256 product = solinas_product_wide(lhs, rhs);
    ulong carry = 0ul;
    for (uint i = 0u; i < 8u; i++) {
        ulong word = (ulong)accumulator.limb[i]
            + (ulong)product.limb[i]
            + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
    accumulator.limb[8] += (uint)carry;
}

inline SolinasFp128 ram_ra_wide_reduce(RamRaWide288 accumulator) {
    SolinasFp128 folded;
    ulong carry = 0ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong word = (ulong)accumulator.limb[i]
            + (ulong)accumulator.limb[i + 4u] * (ulong)SOLINAS_OFFSET
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }
    ulong high = (ulong)accumulator.limb[8] * (ulong)SOLINAS_OFFSET + carry;
    uint high_low = (uint)high;
    uint high_high = (uint)(high >> 32);
    ulong word = (ulong)folded.limb[0]
        + (ulong)high_low * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32;
    word = (ulong)folded.limb[1]
        + (ulong)high_high * (ulong)SOLINAS_OFFSET
        + carry;
    folded.limb[1] = (uint)word;
    carry = word >> 32;
    for (uint i = 2u; i < 4u; i++) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }
    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(carry != 0ul || corrected.carry != 0u, corrected.value, folded);
}

kernel void solinas_ram_ra_claim_build_q(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* eq_address [[buffer(1)]],
    device const SolinasFp128* eq_hi [[buffer(2)]],
    device SolinasFp128* q [[buffer(3)]],
    constant RamRaClaimReductionParams& params [[buffer(4)]],
    uint work [[thread_position_in_grid]])
{
    uint work_items = params.prefix_elements * params.active_q_slices;
    if (work >= work_items) {
        return;
    }
    uint slice = work / params.prefix_elements;
    uint x_lo = work - slice * params.prefix_elements;
    uint high_per_slice = params.suffix_elements / params.q_slices;
    uint high_start = slice * high_per_slice;
    uint high_end = min(
        high_start + high_per_slice, params.active_high_elements);
    RamRaWide288 sums[3];
    sums[0] = ram_ra_wide_zero();
    sums[1] = ram_ra_wide_zero();
    sums[2] = ram_ra_wide_zero();
    for (uint block_start = high_start;
         block_start < high_end;
         block_start += RAM_RA_ACTIVITY_BLOCK) {
        uint active = 0u;
        for (uint offset = 0u; offset < RAM_RA_ACTIVITY_BLOCK; offset++) {
            uint x_hi = block_start + offset;
            if (x_hi < high_end) {
                uint row = x_hi * params.prefix_elements + x_lo;
                active |= (uint)(addresses[row] != params.no_access) << offset;
            }
        }
        while (active != 0u) {
            uint offset = ctz(active);
            active &= active - 1u;
            uint x_hi = block_start + offset;
            uint row = x_hi * params.prefix_elements + x_lo;
            uint address = addresses[row];
            SolinasFp128 h = eq_address[address];
            ram_ra_wide_add_product(sums[0], h, eq_hi[x_hi]);
            ram_ra_wide_add_product(
                sums[1], h, eq_hi[params.suffix_elements + x_hi]);
            ram_ra_wide_add_product(
                sums[2], h, eq_hi[2u * params.suffix_elements + x_hi]);
        }
    }
    uint output_base = slice * 3u * params.prefix_elements + x_lo;
    q[output_base] = ram_ra_wide_reduce(sums[0]);
    q[output_base + params.prefix_elements] = ram_ra_wide_reduce(sums[1]);
    q[output_base + 2u * params.prefix_elements] = ram_ra_wide_reduce(sums[2]);
}

kernel void solinas_ram_ra_claim_build_q_sparse(
    device const uint* offsets [[buffer(0)]],
    device const RamRaQRecord* records [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* eq_hi [[buffer(3)]],
    device SolinasFp128* q [[buffer(4)]],
    constant RamRaClaimReductionParams& params [[buffer(5)]],
    uint x_lo [[thread_position_in_grid]])
{
    if (x_lo >= params.prefix_elements) {
        return;
    }
    RamRaWide288 sums[3];
    sums[0] = ram_ra_wide_zero();
    sums[1] = ram_ra_wide_zero();
    sums[2] = ram_ra_wide_zero();
    uint end = offsets[x_lo + 1u];
    for (uint index = offsets[x_lo]; index < end; index++) {
        RamRaQRecord record = records[index];
        SolinasFp128 h = eq_address[record.address];
        ram_ra_wide_add_product(sums[0], h, eq_hi[record.x_hi]);
        ram_ra_wide_add_product(
            sums[1], h, eq_hi[params.suffix_elements + record.x_hi]);
        ram_ra_wide_add_product(
            sums[2], h, eq_hi[2u * params.suffix_elements + record.x_hi]);
    }
    q[x_lo] = ram_ra_wide_reduce(sums[0]);
    q[params.prefix_elements + x_lo] = ram_ra_wide_reduce(sums[1]);
    q[2u * params.prefix_elements + x_lo] = ram_ra_wide_reduce(sums[2]);
}

kernel void solinas_ram_ra_claim_reduce_q(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* q [[buffer(1)]],
    constant RamRaClaimReductionParams& params [[buffer(2)]],
    uint x_lo [[thread_position_in_grid]])
{
    if (x_lo >= params.prefix_elements) {
        return;
    }
    SolinasFp128 sums[3];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();
    sums[2] = solinas_zero();
    for (uint slice = 0u; slice < params.active_q_slices; slice++) {
        uint input_base = slice * 3u * params.prefix_elements + x_lo;
        sums[0] = solinas_add(sums[0], partials[input_base]);
        sums[1] = solinas_add(
            sums[1], partials[input_base + params.prefix_elements]);
        sums[2] = solinas_add(
            sums[2], partials[input_base + 2u * params.prefix_elements]);
    }
    q[x_lo] = sums[0];
    q[params.prefix_elements + x_lo] = sums[1];
    q[2u * params.prefix_elements + x_lo] = sums[2];
}

kernel void solinas_ram_ra_claim_gather_h(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* eq_address [[buffer(1)]],
    device const SolinasFp128* eq_prefix [[buffer(2)]],
    device SolinasFp128* h_prime [[buffer(3)]],
    constant RamRaClaimReductionParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_hi >= params.suffix_elements) {
        return;
    }
    if (x_hi >= params.active_high_elements) {
        if (tid == 0u) {
            h_prime[x_hi] = solinas_zero();
        }
        return;
    }
    RamRaWide288 wide_sum = ram_ra_wide_zero();
    uint row_start = x_hi * params.prefix_elements;
    uint block_stride = threads * RAM_RA_ACTIVITY_BLOCK;
    for (uint block_start = tid;
         block_start < params.prefix_elements;
         block_start += block_stride) {
        uint active = 0u;
        for (uint offset = 0u; offset < RAM_RA_ACTIVITY_BLOCK; offset++) {
            uint x_lo = block_start + offset * threads;
            if (x_lo < params.prefix_elements) {
                active |=
                    (uint)(addresses[row_start + x_lo] != params.no_access) << offset;
            }
        }
        for (uint rank = 0u; rank < RAM_RA_ACTIVITY_BLOCK; rank++) {
            if (active != 0u) {
                uint offset = ctz(active);
                active &= active - 1u;
                uint x_lo = block_start + offset * threads;
                uint address = addresses[row_start + x_lo];
                ram_ra_wide_add_product(
                    wide_sum, eq_address[address], eq_prefix[x_lo]);
            }
        }
    }
    SolinasFp128 sum = ram_ra_wide_reduce(wide_sum);
    sum = solinas_simd_sum_32(sum);
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane == 0u) {
        SolinasFp128 total = solinas_zero();
        uint simdgroups = threads / 32u;
        for (uint group = 0u; group < simdgroups; group++) {
            total = solinas_add(total, shared[group]);
        }
        h_prime[x_hi] = total;
    }
}

kernel void solinas_ram_ra_claim_gather_h_sparse(
    device const uint* offsets [[buffer(0)]],
    device const RamRaCompactRecord* records [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* eq_prefix [[buffer(3)]],
    device SolinasFp128* h_prime [[buffer(4)]],
    constant RamRaClaimReductionParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_hi >= params.active_high_elements) {
        return;
    }
    RamRaWide288 wide_sum = ram_ra_wide_zero();
    uint end = offsets[x_hi + 1u];
    for (uint index = offsets[x_hi] + tid; index < end; index += threads) {
        RamRaCompactRecord record = records[index];
        uint x_lo = record.cycle & (params.prefix_elements - 1u);
        ram_ra_wide_add_product(
            wide_sum, eq_address[record.address], eq_prefix[x_lo]);
    }
    SolinasFp128 sum = ram_ra_wide_reduce(wide_sum);
    sum = solinas_simd_sum_32(sum);
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane == 0u) {
        SolinasFp128 total = solinas_zero();
        uint simdgroups = threads / 32u;
        for (uint group = 0u; group < simdgroups; group++) {
            total = solinas_add(total, shared[group]);
        }
        h_prime[x_hi] = total;
    }
}
