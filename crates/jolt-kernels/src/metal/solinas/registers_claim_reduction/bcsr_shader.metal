struct RegistersClaimBcsrComponentParams {
    uint cycles;
    uint blocks;
    uint prefix_elements;
    uint suffix_elements;
    uint partial_blocks;
    uint low_blocks;
    uint suffixes_per_partial;
    uint columns;
};

struct RegistersClaimBcsrReduceParams {
    uint partial_blocks;
    uint prefix_elements;
    uint columns;
    uint reserved;
};

struct RegistersClaimBcsrMidpointParams {
    uint blocks;
    uint prefix_elements;
    uint suffix_elements;
    uint low_blocks;
    uint columns;
    uint offset_stride;
    uint position_stride;
    uint reserved;
};

struct RegistersClaimBcsrWorkspace {
    ulong rs1_values[256];
    ulong rs2_values[256];
    ulong rd_values[256];
    SolinasFp128 weight;
};

struct RegistersClaimBcsrIndexedWorkspace {
    ushort rd_event[256];
    SolinasFp128 weight;
};

struct RegistersClaimBcsrMidpointWorkspace {
    ushort rd_event[256];
    SolinasFp128 sums[8];
};

inline ulong registers_claim_bcsr_value_before(
    device const ulong* start_values,
    device const ushort* rd_offsets,
    device const uchar* rd_positions,
    device const ulong* rd_post_values,
    uint block,
    uint position,
    uint reg,
    uint columns)
{
    if (reg >= columns) {
        return 0ul;
    }

    uint offset_base = block * (columns + 1u);
    uint position_base = block * 256u;
    uint begin = (uint)rd_offsets[offset_base + reg];
    uint low = begin;
    uint high = (uint)rd_offsets[offset_base + reg + 1u];
    while (low < high) {
        uint midpoint = low + (high - low) / 2u;
        if ((uint)rd_positions[position_base + midpoint] < position) {
            low = midpoint + 1u;
        } else {
            high = midpoint;
        }
    }
    if (low == begin) {
        return start_values[block * columns + reg];
    }
    return rd_post_values[position_base + low - 1u];
}

kernel void solinas_registers_claim_bcsr_components(
    device const ulong* start_values [[buffer(0)]],
    device const ushort* rs1_offsets [[buffer(1)]],
    device const uchar* rs1_positions [[buffer(2)]],
    device const ushort* rs2_offsets [[buffer(3)]],
    device const uchar* rs2_positions [[buffer(4)]],
    device const ushort* rd_offsets [[buffer(5)]],
    device const uchar* rd_positions [[buffer(6)]],
    device const ulong* rd_post_values [[buffer(7)]],
    device const SolinasFp128* eq_suffix [[buffer(8)]],
    device SolinasFp128* partials [[buffer(9)]],
    constant RegistersClaimBcsrComponentParams& params [[buffer(10)]],
    threadgroup RegistersClaimBcsrWorkspace* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint low_block = group % params.low_blocks;
    uint partial = group / params.low_blocks;
    if (partial >= params.partial_blocks || tid >= 256u) {
        return;
    }

    SolinasFp128 rs1_sum = solinas_zero();
    SolinasFp128 rs2_sum = solinas_zero();
    SolinasFp128 rd_sum = solinas_zero();
    uint suffix_start = partial * params.suffixes_per_partial;
    uint suffix_end = suffix_start + params.suffixes_per_partial;

    for (uint x_hi = suffix_start; x_hi < suffix_end; x_hi++) {
        workspace->rs1_values[tid] = 0ul;
        workspace->rs2_values[tid] = 0ul;
        workspace->rd_values[tid] = 0ul;
        if (tid == 0u) {
            workspace->weight = eq_suffix[x_hi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block = x_hi * params.low_blocks + low_block;
        if (tid < params.columns && block < params.blocks) {
            uint offset_base = block * (params.columns + 1u);
            uint position_base = block * 256u;
            uint rs1_cursor = (uint)rs1_offsets[offset_base + tid];
            uint rs1_end = (uint)rs1_offsets[offset_base + tid + 1u];
            uint rs2_cursor = (uint)rs2_offsets[offset_base + tid];
            uint rs2_end = (uint)rs2_offsets[offset_base + tid + 1u];
            uint rd_cursor = (uint)rd_offsets[offset_base + tid];
            uint rd_end = (uint)rd_offsets[offset_base + tid + 1u];
            ulong state = start_values[block * params.columns + tid];

            while (rs1_cursor < rs1_end || rs2_cursor < rs2_end || rd_cursor < rd_end) {
                uint position = 256u;
                if (rs1_cursor < rs1_end) {
                    position = min(position, (uint)rs1_positions[position_base + rs1_cursor]);
                }
                if (rs2_cursor < rs2_end) {
                    position = min(position, (uint)rs2_positions[position_base + rs2_cursor]);
                }
                if (rd_cursor < rd_end) {
                    position = min(position, (uint)rd_positions[position_base + rd_cursor]);
                }

                if (rs1_cursor < rs1_end
                    && (uint)rs1_positions[position_base + rs1_cursor] == position) {
                    workspace->rs1_values[position] = state;
                    rs1_cursor++;
                }
                if (rs2_cursor < rs2_end
                    && (uint)rs2_positions[position_base + rs2_cursor] == position) {
                    workspace->rs2_values[position] = state;
                    rs2_cursor++;
                }
                if (rd_cursor < rd_end
                    && (uint)rd_positions[position_base + rd_cursor] == position) {
                    state = rd_post_values[position_base + rd_cursor];
                    workspace->rd_values[position] = state;
                    rd_cursor++;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        SolinasFp128 weight = workspace->weight;
        ulong rd_value = workspace->rd_values[tid];
        ulong rs1_value = workspace->rs1_values[tid];
        ulong rs2_value = workspace->rs2_values[tid];
        if (rd_value != 0ul) {
            rd_sum = solinas_add(rd_sum, solinas_half_width_mul_u64(weight, rd_value));
        }
        if (rs1_value != 0ul) {
            rs1_sum = solinas_add(rs1_sum, solinas_half_width_mul_u64(weight, rs1_value));
        }
        if (rs2_value != 0ul) {
            rs2_sum = solinas_add(rs2_sum, solinas_half_width_mul_u64(weight, rs2_value));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint x_lo = low_block * 256u + tid;
    uint partial_stride = params.partial_blocks * params.prefix_elements;
    uint partial_index = partial * params.prefix_elements + x_lo;
    partials[partial_index] = rd_sum;
    partials[partial_stride + partial_index] = rs1_sum;
    partials[2u * partial_stride + partial_index] = rs2_sum;
}

kernel void solinas_registers_claim_bcsr_indexed_components(
    device const ulong* start_values [[buffer(0)]],
    device const ushort* rd_offsets [[buffer(1)]],
    device const uchar* rd_positions [[buffer(2)]],
    device const ulong* rd_post_values [[buffer(3)]],
    device const uchar* rs1_index [[buffer(4)]],
    device const uchar* rs2_index [[buffer(5)]],
    device const SolinasFp128* eq_suffix [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant RegistersClaimBcsrComponentParams& params [[buffer(8)]],
    threadgroup RegistersClaimBcsrIndexedWorkspace* workspace [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint low_block = group % params.low_blocks;
    uint partial = group / params.low_blocks;
    if (partial >= params.partial_blocks || tid >= 256u) {
        return;
    }

    SolinasFp128 rs1_sum = solinas_zero();
    SolinasFp128 rs2_sum = solinas_zero();
    SolinasFp128 rd_sum = solinas_zero();
    uint suffix_start = partial * params.suffixes_per_partial;
    uint suffix_end = suffix_start + params.suffixes_per_partial;

    for (uint x_hi = suffix_start; x_hi < suffix_end; x_hi++) {
        workspace->rd_event[tid] = 0xffffu;
        if (tid == 0u) {
            workspace->weight = eq_suffix[x_hi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block = x_hi * params.low_blocks + low_block;
        uint position_base = block * 256u;
        if (tid < params.columns && block < params.blocks) {
            uint offset_base = block * (params.columns + 1u);
            uint cursor = (uint)rd_offsets[offset_base + tid];
            uint end = (uint)rd_offsets[offset_base + tid + 1u];
            while (cursor < end) {
                uint position = (uint)rd_positions[position_base + cursor];
                workspace->rd_event[position] = (ushort)cursor;
                cursor++;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint cycle = position_base + tid;
        uchar rs1_reg = rs1_index[cycle];
        uchar rs2_reg = rs2_index[cycle];
        ulong rs1_value = registers_claim_bcsr_value_before(
            start_values,
            rd_offsets,
            rd_positions,
            rd_post_values,
            block,
            tid,
            (uint)rs1_reg,
            params.columns);
        ulong rs2_value = registers_claim_bcsr_value_before(
            start_values,
            rd_offsets,
            rd_positions,
            rd_post_values,
            block,
            tid,
            (uint)rs2_reg,
            params.columns);
        ushort rd_event = workspace->rd_event[tid];
        ulong rd_value = rd_event == 0xffffu
            ? 0ul
            : rd_post_values[position_base + (uint)rd_event];
        SolinasFp128 weight = workspace->weight;
        if (rd_value != 0ul) {
            rd_sum = solinas_add(rd_sum, solinas_half_width_mul_u64(weight, rd_value));
        }
        if (rs1_value != 0ul) {
            rs1_sum = solinas_add(rs1_sum, solinas_half_width_mul_u64(weight, rs1_value));
        }
        if (rs2_value != 0ul) {
            rs2_sum = solinas_add(rs2_sum, solinas_half_width_mul_u64(weight, rs2_value));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint x_lo = low_block * 256u + tid;
    uint partial_stride = params.partial_blocks * params.prefix_elements;
    uint partial_index = partial * params.prefix_elements + x_lo;
    partials[partial_index] = rd_sum;
    partials[partial_stride + partial_index] = rs1_sum;
    partials[2u * partial_stride + partial_index] = rs2_sum;
}

kernel void solinas_registers_claim_bcsr_reduce_components(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* components [[buffer(1)]],
    constant RegistersClaimBcsrReduceParams& params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint low_blocks = params.prefix_elements / 256u;
    uint component = group / low_blocks;
    uint low_block = group % low_blocks;
    if (component >= params.columns || tid >= 256u) {
        return;
    }

    uint x_lo = low_block * 256u + tid;
    uint partial_stride = params.partial_blocks * params.prefix_elements;
    uint base = component * partial_stride + x_lo;
    SolinasFp128 total = solinas_zero();
    for (uint partial = 0u; partial < params.partial_blocks; partial++) {
        total = solinas_add(total, partials[base + partial * params.prefix_elements]);
    }
    components[component * params.prefix_elements + x_lo] = total;
}

kernel void solinas_registers_claim_bcsr_fold_rd_midpoint(
    device const ushort* rd_offsets [[buffer(0)]],
    device const uchar* rd_positions [[buffer(1)]],
    device const ulong* rd_post_values [[buffer(2)]],
    device const SolinasFp128* eq_prefix [[buffer(3)]],
    device SolinasFp128* rd_dense [[buffer(4)]],
    constant RegistersClaimBcsrMidpointParams& params [[buffer(5)]],
    threadgroup RegistersClaimBcsrMidpointWorkspace* workspace [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    if (x_hi >= params.suffix_elements || tid >= 256u) {
        return;
    }

    SolinasFp128 accumulator = solinas_zero();
    for (uint low_block = 0u; low_block < params.low_blocks; low_block++) {
        uint block = x_hi * params.low_blocks + low_block;
        uint offset_base = block * params.offset_stride;
        uint position_base = block * params.position_stride;
        workspace->rd_event[tid] = 0xffffu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < params.columns) {
            uint cursor = (uint)rd_offsets[offset_base + tid];
            uint end = (uint)rd_offsets[offset_base + tid + 1u];
            while (cursor < end) {
                uint position = (uint)rd_positions[position_base + cursor];
                workspace->rd_event[position] = (ushort)cursor;
                cursor++;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ushort event = workspace->rd_event[tid];
        if (event != 0xffffu) {
            uint x_lo = low_block * params.position_stride + tid;
            accumulator = solinas_add(
                accumulator,
                solinas_half_width_mul_u64(
                    eq_prefix[x_lo],
                    rd_post_values[position_base + (uint)event]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    SolinasFp128 sum = solinas_simd_sum_32(accumulator);
    if (lane == 0u) {
        workspace->sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        SolinasFp128 total = solinas_zero();
        for (uint group = 0u; group < 8u; group++) {
            total = solinas_add(total, workspace->sums[group]);
        }
        rd_dense[x_hi] = total;
    }
}
