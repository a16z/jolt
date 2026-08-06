#define REGISTERS_RW_SAMPLES 2u
#define REGISTERS_RW_ABSENT 255u

struct RegistersRwRow {
    ulong rs1_value;
    ulong rs2_value;
    ulong rd_pre_value;
    ulong rd_post_value;
    ulong metadata;
};

struct RegistersRwFirstMessageParams {
    uint e_in_length;
    uint e_out_length;
    uint2 reserved;
};

struct RegistersRwReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

struct RegistersRwCell {
    SolinasFp128 value;
    ulong prev_value;
    ulong next_value;
    SolinasFp128 ra;
    bool present;
    bool read;
    bool write;
};

inline SolinasFp128 registers_rw_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32u);
    return result;
}

inline uint registers_rw_index(RegistersRwRow row, uint slot) {
    return (uint)(row.metadata >> (8u * slot)) & 0xffu;
}

inline RegistersRwCell registers_rw_cell(
    RegistersRwRow row,
    uint column,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq)
{
    bool rs1 = registers_rw_index(row, 0u) == column;
    bool rs2 = registers_rw_index(row, 1u) == column;
    bool rd = registers_rw_index(row, 2u) == column;
    ulong value = rs1 ? row.rs1_value : (rs2 ? row.rs2_value : row.rd_pre_value);

    RegistersRwCell cell;
    cell.value = registers_rw_from_u64(value);
    cell.prev_value = value;
    cell.next_value = rd ? row.rd_post_value : value;
    cell.ra = solinas_zero();
    if (rs1) {
        cell.ra = gamma;
    }
    if (rs2) {
        cell.ra = solinas_add(cell.ra, gamma_sq);
    }
    cell.present = rs1 || rs2 || rd;
    cell.read = rs1 || rs2;
    cell.write = rd;
    return cell;
}

inline uint registers_rw_candidate(RegistersRwRow lo, RegistersRwRow hi, uint slot) {
    return slot < 3u
        ? registers_rw_index(lo, slot)
        : registers_rw_index(hi, slot - 3u);
}

inline bool registers_rw_first_occurrence(
    RegistersRwRow lo,
    RegistersRwRow hi,
    uint slot,
    uint column)
{
    for (uint previous = 0u; previous < slot; previous++) {
        if (registers_rw_candidate(lo, hi, previous) == column) {
            return false;
        }
    }
    return true;
}

inline SolinasFp128 registers_rw_bind(
    SolinasFp128 lo,
    SolinasFp128 hi,
    SolinasFp128 challenge)
{
    return solinas_add(lo, solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

inline RegistersRwRow registers_rw_select_row(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint cycle)
{
    switch (cycle) {
        case 0u: return row0;
        case 1u: return row1;
        case 2u: return row2;
        default: return row3;
    }
}

inline ulong registers_rw_value_from_access(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint source_cycle,
    uint target_cycle,
    uint column,
    ulong source_value)
{
    ulong value = source_value;
    if (target_cycle < source_cycle) {
        for (uint cycle = source_cycle; cycle > target_cycle; cycle--) {
            RegistersRwRow previous = registers_rw_select_row(
                row0, row1, row2, row3, cycle - 1u);
            if (registers_rw_index(previous, 2u) == column) {
                value = previous.rd_pre_value;
            }
        }
    } else {
        for (uint cycle = source_cycle; cycle < target_cycle; cycle++) {
            RegistersRwRow current = registers_rw_select_row(
                row0, row1, row2, row3, cycle);
            if (registers_rw_index(current, 2u) == column) {
                value = current.rd_post_value;
            }
        }
    }
    return value;
}

inline SolinasFp128 registers_rw_ra_dot_value(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint source_cycle,
    uint target_cycle,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq)
{
    RegistersRwRow source = registers_rw_select_row(
        row0, row1, row2, row3, source_cycle);
    SolinasFp128 result = solinas_zero();
    uint rs1 = registers_rw_index(source, 0u);
    if (rs1 != REGISTERS_RW_ABSENT) {
        ulong value = registers_rw_value_from_access(
            row0, row1, row2, row3,
            source_cycle, target_cycle, rs1, source.rs1_value);
        result = solinas_mul_wide(gamma, registers_rw_from_u64(value));
    }
    uint rs2 = registers_rw_index(source, 1u);
    if (rs2 != REGISTERS_RW_ABSENT) {
        ulong value = registers_rw_value_from_access(
            row0, row1, row2, row3,
            source_cycle, target_cycle, rs2, source.rs2_value);
        result = solinas_add(
            result,
            solinas_mul_wide(gamma_sq, registers_rw_from_u64(value)));
    }
    return result;
}

inline SolinasFp128 registers_rw_wa_dot_value(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint source_cycle,
    uint target_cycle)
{
    RegistersRwRow source = registers_rw_select_row(
        row0, row1, row2, row3, source_cycle);
    uint rd = registers_rw_index(source, 2u);
    if (rd == REGISTERS_RW_ABSENT) {
        return solinas_zero();
    }
    ulong value = registers_rw_value_from_access(
        row0, row1, row2, row3,
        source_cycle, target_cycle, rd, source.rd_pre_value);
    return registers_rw_from_u64(value);
}

inline SolinasFp128 registers_rw_bound_ra_dot_value(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint source_pair,
    uint target_pair,
    SolinasFp128 challenge,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq)
{
    SolinasFp128 source_lo = registers_rw_bind(
        registers_rw_ra_dot_value(
            row0, row1, row2, row3,
            source_pair, target_pair, gamma, gamma_sq),
        registers_rw_ra_dot_value(
            row0, row1, row2, row3,
            source_pair, target_pair + 1u, gamma, gamma_sq),
        challenge);
    SolinasFp128 source_hi = registers_rw_bind(
        registers_rw_ra_dot_value(
            row0, row1, row2, row3,
            source_pair + 1u, target_pair, gamma, gamma_sq),
        registers_rw_ra_dot_value(
            row0, row1, row2, row3,
            source_pair + 1u, target_pair + 1u, gamma, gamma_sq),
        challenge);
    return registers_rw_bind(source_lo, source_hi, challenge);
}

inline SolinasFp128 registers_rw_bound_wa_dot_value(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    uint source_pair,
    uint target_pair,
    SolinasFp128 challenge)
{
    SolinasFp128 source_lo = registers_rw_bind(
        registers_rw_wa_dot_value(
            row0, row1, row2, row3, source_pair, target_pair),
        registers_rw_wa_dot_value(
            row0, row1, row2, row3, source_pair, target_pair + 1u),
        challenge);
    SolinasFp128 source_hi = registers_rw_bind(
        registers_rw_wa_dot_value(
            row0, row1, row2, row3, source_pair + 1u, target_pair),
        registers_rw_wa_dot_value(
            row0, row1, row2, row3, source_pair + 1u, target_pair + 1u),
        challenge);
    return registers_rw_bind(source_lo, source_hi, challenge);
}

inline SolinasFp128 registers_rw_bound_wa_sum(
    RegistersRwRow lo,
    RegistersRwRow hi,
    SolinasFp128 challenge)
{
    return registers_rw_bind(
        registers_rw_from_u64(
            registers_rw_index(lo, 2u) == REGISTERS_RW_ABSENT ? 0u : 1u),
        registers_rw_from_u64(
            registers_rw_index(hi, 2u) == REGISTERS_RW_ABSENT ? 0u : 1u),
        challenge);
}

inline void registers_rw_second_pair_endpoints(
    RegistersRwRow row0,
    RegistersRwRow row1,
    RegistersRwRow row2,
    RegistersRwRow row3,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_infinity,
    SolinasFp128 challenge,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq,
    thread SolinasFp128& q_zero,
    thread SolinasFp128& q_infinity)
{
    SolinasFp128 ra_aa = registers_rw_bound_ra_dot_value(
        row0, row1, row2, row3, 0u, 0u, challenge, gamma, gamma_sq);
    SolinasFp128 ra_ab = registers_rw_bound_ra_dot_value(
        row0, row1, row2, row3, 0u, 2u, challenge, gamma, gamma_sq);
    SolinasFp128 ra_ba = registers_rw_bound_ra_dot_value(
        row0, row1, row2, row3, 2u, 0u, challenge, gamma, gamma_sq);
    SolinasFp128 ra_bb = registers_rw_bound_ra_dot_value(
        row0, row1, row2, row3, 2u, 2u, challenge, gamma, gamma_sq);
    SolinasFp128 wa_aa = registers_rw_bound_wa_dot_value(
        row0, row1, row2, row3, 0u, 0u, challenge);
    SolinasFp128 wa_ab = registers_rw_bound_wa_dot_value(
        row0, row1, row2, row3, 0u, 2u, challenge);
    SolinasFp128 wa_ba = registers_rw_bound_wa_dot_value(
        row0, row1, row2, row3, 2u, 0u, challenge);
    SolinasFp128 wa_bb = registers_rw_bound_wa_dot_value(
        row0, row1, row2, row3, 2u, 2u, challenge);
    SolinasFp128 wa_sum_a = registers_rw_bound_wa_sum(row0, row1, challenge);
    SolinasFp128 wa_sum_b = registers_rw_bound_wa_sum(row2, row3, challenge);

    q_zero = solinas_add(
        solinas_add(ra_aa, wa_aa),
        solinas_mul_wide(inc_zero, wa_sum_a));
    SolinasFp128 ra_infinity = solinas_add(
        solinas_sub(solinas_sub(ra_bb, ra_ba), ra_ab),
        ra_aa);
    SolinasFp128 wa_infinity = solinas_add(
        solinas_sub(solinas_sub(wa_bb, wa_ba), wa_ab),
        wa_aa);
    q_infinity = solinas_add(
        solinas_add(ra_infinity, wa_infinity),
        solinas_mul_wide(
            inc_infinity,
            solinas_sub(wa_sum_b, wa_sum_a)));
}

inline void registers_rw_pair_endpoints(
    RegistersRwRow lo,
    RegistersRwRow hi,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_infinity,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq,
    thread SolinasFp128& q_zero,
    thread SolinasFp128& q_infinity)
{
    q_zero = solinas_zero();
    q_infinity = solinas_zero();
    for (uint slot = 0u; slot < 6u; slot++) {
        uint column = registers_rw_candidate(lo, hi, slot);
        if (column == REGISTERS_RW_ABSENT ||
            !registers_rw_first_occurrence(lo, hi, slot, column)) {
            continue;
        }

        RegistersRwCell even = registers_rw_cell(lo, column, gamma, gamma_sq);
        RegistersRwCell odd = registers_rw_cell(hi, column, gamma, gamma_sq);
        if (even.read) {
            q_zero = solinas_add(
                q_zero,
                solinas_mul_wide(even.ra, even.value));
        }
        if (even.write) {
            q_zero = solinas_add(q_zero, solinas_add(even.value, inc_zero));
        }

        SolinasFp128 value_infinity;
        if (even.present && odd.present) {
            value_infinity = solinas_sub(odd.value, even.value);
        } else if (even.present) {
            value_infinity = solinas_sub(
                registers_rw_from_u64(even.next_value),
                even.value);
        } else {
            value_infinity = solinas_zero();
        }
        if (even.read || odd.read) {
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(
                    solinas_sub(odd.ra, even.ra),
                    value_infinity));
        }
        SolinasFp128 write_term = solinas_add(value_infinity, inc_infinity);
        if (odd.write && !even.write) {
            q_infinity = solinas_add(q_infinity, write_term);
        } else if (even.write && !odd.write) {
            q_infinity = solinas_sub(q_infinity, write_term);
        }
    }
}

inline void registers_rw_finish_block(
    thread SolinasFp128* lanes,
    SolinasFp128 e_out,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0u; sample < REGISTERS_RW_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0u) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < REGISTERS_RW_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0u) {
                partials[sample * e_out_length + x_out] =
                    solinas_mul_wide(e_out, sum);
            }
        }
    }
}

kernel void solinas_registers_rw_first_message(
    device const RegistersRwRow* rows [[buffer(0)]],
    device const SolinasFp128* inc [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device const SolinasFp128* gamma [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RegistersRwFirstMessageParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[REGISTERS_RW_SAMPLES];
    lanes[0] = solinas_zero();
    lanes[1] = solinas_zero();

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        uint lo_index = 2u * pair;
        SolinasFp128 q_zero;
        SolinasFp128 q_infinity;
        registers_rw_pair_endpoints(
            rows[lo_index],
            rows[lo_index + 1u],
            inc[lo_index],
            solinas_sub(inc[lo_index + 1u], inc[lo_index]),
            gamma[0],
            gamma[1],
            q_zero,
            q_infinity);
        SolinasFp128 weight = e_in[x_in];
        lanes[0] = solinas_add(lanes[0], solinas_mul_wide(weight, q_zero));
        lanes[1] = solinas_add(lanes[1], solinas_mul_wide(weight, q_infinity));
    }

    registers_rw_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32u);
}

kernel void solinas_registers_rw_second_message(
    device const RegistersRwRow* rows [[buffer(0)]],
    device const SolinasFp128* inc [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device const SolinasFp128* gamma [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant RegistersRwFirstMessageParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[REGISTERS_RW_SAMPLES];
    lanes[0] = solinas_zero();
    lanes[1] = solinas_zero();

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        uint base = 4u * pair;
        SolinasFp128 inc0 = registers_rw_bind(inc[base], inc[base + 1u], challenge);
        SolinasFp128 inc1 = registers_rw_bind(inc[base + 2u], inc[base + 3u], challenge);
        SolinasFp128 q_zero;
        SolinasFp128 q_infinity;
        registers_rw_second_pair_endpoints(
            rows[base],
            rows[base + 1u],
            rows[base + 2u],
            rows[base + 3u],
            inc0,
            solinas_sub(inc1, inc0),
            challenge,
            gamma[0],
            gamma[1],
            q_zero,
            q_infinity);
        SolinasFp128 weight = e_in[x_in];
        lanes[0] = solinas_add(lanes[0], solinas_mul_wide(weight, q_zero));
        lanes[1] = solinas_add(lanes[1], solinas_mul_wide(weight, q_infinity));
    }

    registers_rw_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32u);
}

kernel void solinas_registers_rw_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RegistersRwReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint sample = 0u; sample < REGISTERS_RW_SAMPLES; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0u) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
