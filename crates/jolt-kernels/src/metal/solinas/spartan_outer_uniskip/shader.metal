#define SPARTAN_OUTER_NODES 9u
#define SPARTAN_OUTER_SIMD_WIDTH 32u
#define SPARTAN_OUTER_FIRST_NODES 6u
#define SPARTAN_OUTER_FIRST_ROWS_PER_SIMD 5u
#define SPARTAN_OUTER_SECOND_NODES 3u
#define SPARTAN_OUTER_SECOND_ROWS_PER_SIMD 10u
#define SPARTAN_STAGE1_PRIMER_SOURCES 6u

struct SpartanOuterUniskipParams {
    uint rows;
    uint pairs_per_block;
    uint blocks;
    uint reserved;
};

struct SpartanStage1SourcePrimerParams {
    ulong word_counts[SPARTAN_STAGE1_PRIMER_SOURCES];
    uint page_words;
    uint total_threads;
};

kernel void solinas_spartan_stage1_source_primer(
    device const uint* instruction_input [[buffer(0)]],
    device const uint* successor [[buffer(1)]],
    device const uint* cold [[buffer(2)]],
    device const uint* unexpanded_pc [[buffer(3)]],
    device const uint* pc [[buffer(4)]],
    device const uint* shift_flags [[buffer(5)]],
    device uint* checksums [[buffer(6)]],
    constant SpartanStage1SourcePrimerParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.total_threads) {
        return;
    }

    ulong pages[SPARTAN_STAGE1_PRIMER_SOURCES];
    ulong total_pages = 0;
    for (uint source = 0; source < SPARTAN_STAGE1_PRIMER_SOURCES; source++) {
        pages[source] = (params.word_counts[source] + params.page_words - 1u)
            / params.page_words;
        total_pages += pages[source];
    }

    uint checksum = 0x9e3779b9u ^ gid;
    for (ulong page = gid; page < total_pages; page += params.total_threads) {
        ulong local_page = page;
        uint value;
        if (local_page < pages[0]) {
            value = instruction_input[local_page * params.page_words];
        } else if ((local_page -= pages[0]) < pages[1]) {
            value = successor[local_page * params.page_words];
        } else if ((local_page -= pages[1]) < pages[2]) {
            value = cold[local_page * params.page_words];
        } else if ((local_page -= pages[2]) < pages[3]) {
            value = unexpanded_pc[local_page * params.page_words];
        } else if ((local_page -= pages[3]) < pages[4]) {
            value = pc[local_page * params.page_words];
        } else {
            local_page -= pages[4];
            value = shift_flags[local_page * params.page_words];
        }
        checksum ^= value ^ (uint)page;
        checksum = ((checksum << 5u) | (checksum >> 27u)) * 0x85ebca6bu;
    }
    checksums[gid] = checksum;
}

struct SpartanFieldSum192 {
    uint limb[6];
};

constant int SPARTAN_OUTER_EXTENSION[SPARTAN_OUTER_NODES][10] = {
    { 2002, -15015, 51480, -105105, 140140, -126126, 76440, -30030, 6930, -715 },
    { 715, -5148, 17160, -34320, 45045, -40040, 24024, -9360, 2145, -220 },
    { 220, -1485, 4752, -9240, 11880, -10395, 6160, -2376, 540, -55 },
    { 55, -330, 990, -1848, 2310, -1980, 1155, -440, 99, -10 },
    { 10, -45, 120, -210, 252, -210, 120, -45, 10, -1 },
    { -1, 10, -45, 120, -210, 252, -210, 120, -45, 10 },
    { -10, 99, -440, 1155, -1980, 2310, -1848, 990, -330, 55 },
    { -55, 540, -2376, 6160, -10395, 11880, -9240, 4752, -1485, 220 },
    { -220, 2145, -9360, 24024, -40040, 45045, -34320, 17160, -5148, 715 },
};

inline bool spartan_outer_flag(ulong flags, uint bit) {
    return ((flags >> bit) & 1u) != 0;
}

inline SpartanFieldSum192 spartan_field_sum_zero() {
    SpartanFieldSum192 value;
    for (uint i = 0; i < 6; i++) {
        value.limb[i] = 0;
    }
    return value;
}

inline void spartan_field_sum_add(
    thread SpartanFieldSum192& accumulator,
    SolinasFp128 value)
{
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)accumulator.limb[i] + (ulong)value.limb[i] + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
    for (uint i = 4; i < 6; i++) {
        ulong word = (ulong)accumulator.limb[i] + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
}

inline SolinasFp128 spartan_field_sum_reduce(SpartanFieldSum192 accumulator) {
    SolinasWide256 wide;
    for (uint i = 0; i < 6; i++) {
        wide.limb[i] = accumulator.limb[i];
    }
    wide.limb[6] = 0;
    wide.limb[7] = 0;
    return solinas_reduce(wide);
}

inline void spartan_outer_accumulate_contribution(
    device const InstructionInputRow& instruction_input,
    device const SpartanOuterSuccessorRow& successor,
    device const SpartanOuterColdRow& cold,
    constant const int* c,
    SolinasFp128 even_weight,
    SolinasFp128 odd_weight,
    thread SpartanFieldSum192& accumulator)
{
    ulong flags = instruction_input_row_word(instruction_input, 5u);
    int load = (int)spartan_outer_flag(flags, 0);
    int store = (int)spartan_outer_flag(flags, 1);
    int add = (int)spartan_outer_flag(flags, 2);
    int sub = (int)spartan_outer_flag(flags, 3);
    int mul = (int)spartan_outer_flag(flags, 4);
    int jump = (int)spartan_outer_flag(flags, 5);
    int should_branch = (int)spartan_outer_flag(flags, 6);
    int assert_flag = (int)spartan_outer_flag(flags, 7);
    int should_jump = (int)spartan_outer_flag(flags, 8);
    int virtual_instruction = (int)spartan_outer_flag(flags, 9);
    int is_last = (int)spartan_outer_flag(flags, 10);
    int next_is_virtual = (int)spartan_outer_flag(flags, 11);
    int next_is_first = (int)spartan_outer_flag(flags, 12);
    int advice = (int)spartan_outer_flag(flags, 13);
    int write_lookup = (int)spartan_outer_flag(flags, 14);
    int do_not_update = (int)spartan_outer_flag(flags, 15);
    int is_compressed = (int)spartan_outer_flag(flags, 16);

    ulong rs1 = instruction_input_row_word(instruction_input, 0u);
    ulong rs2 = instruction_input_row_word(instruction_input, 2u);
    ulong memory_0 = spartan_outer_residual_word(successor, cold, 6u);
    ulong memory_1 = spartan_outer_residual_word(successor, cold, 7u);
    ulong ram_address = load != 0 || store != 0 ? memory_0 : 0;
    ulong rd_write = store != 0 ? 0 : (load != 0 ? memory_1 : memory_0);
    ulong ram_read = load != 0 || store != 0 ? memory_1 : 0;
    ulong ram_write = load != 0 ? memory_1 : (store != 0 ? rs2 : 0);

    int operations = add + sub + mul;
    int az_first = c[0]
        + load * (-c[0] + c[1] + c[2])
        + store * (-c[0] + c[3])
        + operations * (c[4] - c[5])
        + c[5]
        + assert_flag * c[6]
        + should_jump * c[7]
        + (virtual_instruction - is_last) * c[8]
        + (next_is_virtual - next_is_first) * c[9];

    SpartanSigned192 bz_first = spartan_s192_zero();
    spartan_accumulate_scaled_u64(bz_first, ram_address, c[0]);
    spartan_accumulate_scaled_u64(bz_first, ram_read, c[1] + c[2]);
    spartan_accumulate_scaled_u64(bz_first, ram_write, -c[1] - c[3]);
    spartan_accumulate_scaled_u64(bz_first, rd_write, -c[2]);
    spartan_accumulate_scaled_u64(bz_first, rs2, c[3]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 8u), c[4] + c[5]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 0u), -c[5]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 13u), c[6] - c[7]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 11u), c[7]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 12u), c[8]);
    spartan_accumulate_scaled_u64(
        bz_first, spartan_outer_residual_word(successor, cold, 5u), -c[8]);
    spartan_accumulate_i32(bz_first, -c[6] - c[8] + c[9] - do_not_update * c[9]);
    SolinasFp128 first = spartan_small_times_s192(az_first, bz_first);

    int az_second = c[4] + c[8]
        + (load + store) * c[0]
        + add * (c[1] - c[4])
        + sub * (c[2] - c[4])
        + mul * (c[3] - c[4])
        - advice * c[4]
        + write_lookup * c[5]
        + jump * (c[6] - c[8])
        + should_branch * (c[7] - c[8]);

    SpartanSigned192 bz_second = spartan_s192_zero();
    spartan_accumulate_scaled_u64(bz_second, ram_address, c[0]);
    spartan_accumulate_scaled_u64(bz_second, rs1, -c[0]);
    spartan_accumulate_scaled_u128(
        bz_second,
        instruction_input_row_word(instruction_input, 3u),
        instruction_input_row_word(instruction_input, 4u),
        spartan_outer_flag(flags, 18),
        -c[0] - c[7]);
    spartan_accumulate_scaled_u128(
        bz_second,
        spartan_outer_residual_word(successor, cold, 9u),
        spartan_outer_residual_word(successor, cold, 10u),
        true,
        c[1] + c[2] + c[3] + c[4]);
    spartan_accumulate_scaled_u64(
        bz_second, spartan_outer_residual_word(successor, cold, 0u), -c[1] - c[2]);
    spartan_accumulate_scaled_u128(
        bz_second,
        spartan_outer_residual_word(successor, cold, 1u),
        spartan_outer_residual_word(successor, cold, 2u),
        spartan_outer_flag(flags, 17),
        -c[1] + c[2] - c[4]);
    spartan_accumulate_pow64(bz_second, -c[2]);
    spartan_accumulate_scaled_u128(
        bz_second,
        spartan_outer_residual_word(successor, cold, 3u),
        spartan_outer_residual_word(successor, cold, 4u),
        spartan_outer_flag(flags, 19),
        -c[3]);
    spartan_accumulate_scaled_u64(bz_second, rd_write, c[5] + c[6]);
    spartan_accumulate_scaled_u64(
        bz_second, spartan_outer_residual_word(successor, cold, 13u), -c[5]);
    spartan_accumulate_scaled_u64(
        bz_second,
        instruction_input_row_word(instruction_input, 1u),
        -c[6] - c[7] - c[8]);
    spartan_accumulate_scaled_u64(
        bz_second, spartan_outer_residual_word(successor, cold, 11u), c[7] + c[8]);
    spartan_accumulate_i32(
        bz_second,
        -4 * c[6] - 4 * c[8]
            + is_compressed * (2 * c[6] + 2 * c[8])
            + do_not_update * 4 * c[8]);
    SolinasFp128 second = spartan_small_times_s192(az_second, bz_second);

    spartan_field_sum_add(accumulator, solinas_mul_wide(even_weight, first));
    spartan_field_sum_add(accumulator, solinas_mul_wide(odd_weight, second));
}

kernel void solinas_spartan_outer_uniskip_blocks(
    device const InstructionInputRow* instruction_input_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* successor_rows [[buffer(1)]],
    device const SpartanOuterColdRow* cold_rows [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* block_sums [[buffer(5)]],
    constant SpartanOuterUniskipParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint lane = tid & (SPARTAN_OUTER_SIMD_WIDTH - 1);
    uint simdgroup = tid / SPARTAN_OUTER_SIMD_WIDTH;
    uint simdgroups = threads / SPARTAN_OUTER_SIMD_WIDTH;
    {
        uint cluster = lane / SPARTAN_OUTER_FIRST_NODES;
        uint node = lane - cluster * SPARTAN_OUTER_FIRST_NODES;
        uint row_slot = simdgroup * SPARTAN_OUTER_FIRST_ROWS_PER_SIMD + cluster;
        uint row_stride = simdgroups * SPARTAN_OUTER_FIRST_ROWS_PER_SIMD;
        SpartanFieldSum192 sum = spartan_field_sum_zero();
        if (cluster < SPARTAN_OUTER_FIRST_ROWS_PER_SIMD) {
            for (uint pair = row_slot; pair < params.pairs_per_block; pair += row_stride) {
                uint row_index = block * params.pairs_per_block + pair;
                if (row_index < params.rows) {
                    spartan_outer_accumulate_contribution(
                        instruction_input_rows[row_index],
                        successor_rows[row_index],
                        cold_rows[row_index],
                        SPARTAN_OUTER_EXTENSION[node],
                        e_in[2 * pair],
                        e_in[2 * pair + 1],
                        sum);
                }
            }
        }
        shared[tid] = spartan_field_sum_reduce(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < SPARTAN_OUTER_FIRST_NODES) {
        SolinasFp128 block_sum = solinas_zero();
        for (uint group = 0; group < simdgroups; group++) {
            for (uint row_cluster = 0;
                 row_cluster < SPARTAN_OUTER_FIRST_ROWS_PER_SIMD;
                 row_cluster++) {
                block_sum = solinas_add(
                    block_sum,
                    shared[group * SPARTAN_OUTER_SIMD_WIDTH
                        + row_cluster * SPARTAN_OUTER_FIRST_NODES + tid]);
            }
        }
        block_sums[block * SPARTAN_OUTER_NODES + tid] =
            solinas_mul_wide(e_out[block], block_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint cluster = lane / SPARTAN_OUTER_SECOND_NODES;
        uint node = lane - cluster * SPARTAN_OUTER_SECOND_NODES;
        uint row_slot = simdgroup * SPARTAN_OUTER_SECOND_ROWS_PER_SIMD + cluster;
        uint row_stride = simdgroups * SPARTAN_OUTER_SECOND_ROWS_PER_SIMD;
        SpartanFieldSum192 sum = spartan_field_sum_zero();
        if (cluster < SPARTAN_OUTER_SECOND_ROWS_PER_SIMD) {
            for (uint pair = row_slot; pair < params.pairs_per_block; pair += row_stride) {
                uint row_index = block * params.pairs_per_block + pair;
                if (row_index < params.rows) {
                    spartan_outer_accumulate_contribution(
                        instruction_input_rows[row_index],
                        successor_rows[row_index],
                        cold_rows[row_index],
                        SPARTAN_OUTER_EXTENSION[node + SPARTAN_OUTER_FIRST_NODES],
                        e_in[2 * pair],
                        e_in[2 * pair + 1],
                        sum);
                }
            }
        }
        shared[tid] = spartan_field_sum_reduce(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < SPARTAN_OUTER_SECOND_NODES) {
        SolinasFp128 block_sum = solinas_zero();
        for (uint group = 0; group < simdgroups; group++) {
            for (uint row_cluster = 0;
                 row_cluster < SPARTAN_OUTER_SECOND_ROWS_PER_SIMD;
                 row_cluster++) {
                block_sum = solinas_add(
                    block_sum,
                    shared[group * SPARTAN_OUTER_SIMD_WIDTH
                        + row_cluster * SPARTAN_OUTER_SECOND_NODES + tid]);
            }
        }
        uint output_node = tid + SPARTAN_OUTER_FIRST_NODES;
        block_sums[block * SPARTAN_OUTER_NODES + output_node] =
            solinas_mul_wide(e_out[block], block_sum);
    }
}

kernel void solinas_spartan_outer_uniskip_reduce(
    device const SolinasFp128* block_sums [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant SpartanOuterUniskipParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint node [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sum = solinas_zero();
    for (uint block = tid; block < params.blocks; block += threads) {
        sum = solinas_add(sum, block_sums[block * SPARTAN_OUTER_NODES + node]);
    }
    sum = solinas_simd_sum_32(sum);
    if (lane == 0) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        uint simdgroups = threads / SPARTAN_OUTER_SIMD_WIDTH;
        sum = lane < simdgroups ? shared[lane] : solinas_zero();
        sum = solinas_simd_sum_32(sum);
        if (lane == 0) {
            output[node] = sum;
        }
    }
}
