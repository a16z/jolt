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

struct RegistersClaimBcsrWorkspace {
    ulong rs1_values[256];
    ulong rs2_values[256];
    ulong rd_values[256];
    SolinasFp128 weight;
};

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
