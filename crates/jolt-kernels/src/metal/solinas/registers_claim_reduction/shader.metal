// Concatenate after the offset-specialized fp128.metal and simd_reduce.metal.

#define REGISTERS_CLAIM_OUTPUT_COLUMNS 3u
#define REGISTERS_CLAIM_WIDE_LIMBS 7u

struct RegistersClaimParams {
    uint rows;
    uint prefix_elements;
    uint suffix_elements;
    uint reserved;
};

struct RegistersClaimWide224 {
    uint limb[REGISTERS_CLAIM_WIDE_LIMBS];
};

inline RegistersClaimWide224 registers_claim_wide_zero() {
    RegistersClaimWide224 result;
    for (uint i = 0u; i < REGISTERS_CLAIM_WIDE_LIMBS; i++) {
        result.limb[i] = 0u;
    }
    return result;
}

inline void registers_claim_accumulate_u64(
    thread RegistersClaimWide224& accumulator,
    SolinasFp128 coefficient,
    ulong scalar)
{
    uint scalar_limb[2] = {(uint)scalar, (uint)(scalar >> 32)};
    for (uint i = 0u; i < 4u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 2u; j++) {
            uint k = i + j;
            ulong word = (ulong)coefficient.limb[i] * (ulong)scalar_limb[j]
                + (ulong)accumulator.limb[k]
                + carry;
            accumulator.limb[k] = (uint)word;
            carry = word >> 32;
        }

        uint k = i + 2u;
        while (carry != 0ul && k < REGISTERS_CLAIM_WIDE_LIMBS) {
            ulong word = (ulong)accumulator.limb[k] + carry;
            accumulator.limb[k] = (uint)word;
            carry = word >> 32;
            k++;
        }
    }
}

inline SolinasFp128 registers_claim_reduce_wide(
    RegistersClaimWide224 accumulator)
{
    // Fold the 96 bits above 2^128 with 2^128 = SOLINAS_OFFSET (mod p).
    SolinasFp128 folded;
    ulong carry = 0ul;
    for (uint i = 0u; i < 3u; i++) {
        ulong word = (ulong)accumulator.limb[i + 4u] * (ulong)SOLINAS_OFFSET
            + (ulong)accumulator.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    ulong word = (ulong)accumulator.limb[3] + carry;
    folded.limb[3] = (uint)word;
    carry = word >> 32;

    word = (ulong)folded.limb[0] + carry * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32;
    for (uint i = 1u; i < 4u; i++) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(
        carry != 0ul || corrected.carry != 0u,
        corrected.value,
        folded);
}

kernel void solinas_registers_claim_build_linear_q(
    device const ulong* rd_write_value [[buffer(0)]],
    device const ulong* rs1_value [[buffer(1)]],
    device const ulong* rs2_value [[buffer(2)]],
    device const SolinasFp128* gamma_powers [[buffer(3)]],
    device const SolinasFp128* eq_suffix [[buffer(4)]],
    device SolinasFp128* q [[buffer(5)]],
    constant RegistersClaimParams& params [[buffer(6)]],
    uint x_lo [[thread_position_in_grid]])
{
    if (x_lo >= params.prefix_elements) {
        return;
    }

    RegistersClaimWide224 rd = registers_claim_wide_zero();
    RegistersClaimWide224 rs1 = registers_claim_wide_zero();
    RegistersClaimWide224 rs2 = registers_claim_wide_zero();

    for (uint x_hi = 0u; x_hi < params.suffix_elements; x_hi++) {
        uint row = x_hi * params.prefix_elements + x_lo;
        SolinasFp128 weight = eq_suffix[x_hi];
        registers_claim_accumulate_u64(rd, weight, rd_write_value[row]);
        registers_claim_accumulate_u64(rs1, weight, rs1_value[row]);
        registers_claim_accumulate_u64(rs2, weight, rs2_value[row]);
    }

    SolinasFp128 value = registers_claim_reduce_wide(rd);
    value = solinas_add(
        value,
        solinas_mul_wide(gamma_powers[0], registers_claim_reduce_wide(rs1)));
    value = solinas_add(
        value,
        solinas_mul_wide(gamma_powers[1], registers_claim_reduce_wide(rs2)));
    q[x_lo] = value;
}

kernel void solinas_registers_claim_fold_alias_rd(
    device const ulong* rd_write_value [[buffer(0)]],
    device const SolinasFp128* eq_prefix [[buffer(1)]],
    device SolinasFp128* rd_dense [[buffer(2)]],
    constant RegistersClaimParams& params [[buffer(3)]],
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

    RegistersClaimWide224 accumulator = registers_claim_wide_zero();
    uint row_start = x_hi * params.prefix_elements;
    for (uint x_lo = tid; x_lo < params.prefix_elements; x_lo += threads) {
        registers_claim_accumulate_u64(
            accumulator,
            eq_prefix[x_lo],
            rd_write_value[row_start + x_lo]);
    }

    SolinasFp128 sum = solinas_simd_sum_32(
        registers_claim_reduce_wide(accumulator));
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        SolinasFp128 total = solinas_zero();
        uint simdgroups = threads / 32u;
        for (uint group = 0u; group < simdgroups; group++) {
            total = solinas_add(total, shared[group]);
        }
        rd_dense[x_hi] = total;
    }
}

kernel void solinas_registers_claim_fold_direct(
    device const ulong* rd_write_value [[buffer(0)]],
    device const ulong* rs1_value [[buffer(1)]],
    device const ulong* rs2_value [[buffer(2)]],
    device const SolinasFp128* eq_prefix [[buffer(3)]],
    device SolinasFp128* dense_outputs [[buffer(4)]],
    constant RegistersClaimParams& params [[buffer(5)]],
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

    RegistersClaimWide224 sums[REGISTERS_CLAIM_OUTPUT_COLUMNS];
    for (uint column = 0u; column < REGISTERS_CLAIM_OUTPUT_COLUMNS; column++) {
        sums[column] = registers_claim_wide_zero();
    }

    uint row_start = x_hi * params.prefix_elements;
    for (uint x_lo = tid; x_lo < params.prefix_elements; x_lo += threads) {
        uint row = row_start + x_lo;
        SolinasFp128 weight = eq_prefix[x_lo];
        registers_claim_accumulate_u64(sums[0], weight, rd_write_value[row]);
        registers_claim_accumulate_u64(sums[1], weight, rs1_value[row]);
        registers_claim_accumulate_u64(sums[2], weight, rs2_value[row]);
    }

    uint simdgroups = threads / 32u;
    for (uint column = 0u; column < REGISTERS_CLAIM_OUTPUT_COLUMNS; column++) {
        SolinasFp128 sum = solinas_simd_sum_32(
            registers_claim_reduce_wide(sums[column]));
        if (lane == 0u) {
            shared[column * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane < REGISTERS_CLAIM_OUTPUT_COLUMNS) {
        SolinasFp128 total = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            total = solinas_add(total, shared[lane * simdgroups + group]);
        }
        dense_outputs[lane * params.suffix_elements + x_hi] = total;
    }
}
