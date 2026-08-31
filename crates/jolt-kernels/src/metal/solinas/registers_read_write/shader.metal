// Concatenate after fp128.metal, simd_reduce.metal, and ram_read_write.

#define REGISTERS_READ_WRITE_SIMD_WIDTH 32u
#define REGISTERS_READ_WRITE_THREADS 256u
#define REGISTERS_READ_WRITE_SIMDGROUPS \
    (REGISTERS_READ_WRITE_THREADS / REGISTERS_READ_WRITE_SIMD_WIDTH)
#define REGISTERS_READ_WRITE_NO_REGISTER 255u

constant bool registers_read_write_remap_enabled [[function_constant(2)]];
constant bool registers_read_write_stage1_source [[function_constant(3)]];

#define REGISTERS_READ_WRITE_RS1_INDEX_SHIFT 32u
#define REGISTERS_READ_WRITE_RS2_INDEX_SHIFT 40u
#define REGISTERS_READ_WRITE_RD_INDEX_SHIFT 48u
#define REGISTERS_READ_WRITE_RD_PRE_REGISTERS 64u
#define REGISTERS_READ_WRITE_RD_PRE_CHUNK_ROWS 4096u
#define REGISTERS_READ_WRITE_RD_PRE_LOCAL_BITS 12u
#define REGISTERS_READ_WRITE_RD_PRE_LOCAL_MASK 0xfffu
#define REGISTERS_READ_WRITE_RD_PRE_NO_OFFSET 0xffffu
#define REGISTERS_READ_WRITE_RD_PRE_NO_KEY 0xffffffffu
#define REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS 32u
#define REGISTERS_READ_WRITE_RD_PRE_FIXUP_REGISTERS_PER_GROUP 8u

struct PackedRegisterCycleRow {
    ulong rs1_value;
    ulong rs2_value;
    ulong rd_pre_value;
    ulong rd_post_value;
    uchar rs1_index;
    uchar rs2_index;
    uchar rd_index;
    uchar padding[5];
};

struct RegistersReadWriteFirstMessageParams {
    uint row_count;
    uint pair_count;
    uint output_stride;
    uint e_in_length;
    uint source_stride;
};

struct RegistersReadWriteSourcePrimerParams {
    ulong word_counts[3];
    uint page_words;
    uint total_threads;
};

struct RegistersReadWriteRdPreParams {
    uint row_count;
    uint chunk_rows;
    uint chunks;
    uint source_stride;
};

struct RegistersReadWriteCell {
    bool present;
    bool write;
    uint read_mask;
    ulong previous;
    ulong next;
    ulong value;
};

struct RegistersReadWriteSignedU64 {
    ulong magnitude;
    bool negative;
};

inline SolinasFp128 registers_read_write_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline PackedRegisterCycleRow registers_read_write_empty_row() {
    PackedRegisterCycleRow row;
    row.rs1_value = 0ul;
    row.rs2_value = 0ul;
    row.rd_pre_value = 0ul;
    row.rd_post_value = 0ul;
    row.rs1_index = REGISTERS_READ_WRITE_NO_REGISTER;
    row.rs2_index = REGISTERS_READ_WRITE_NO_REGISTER;
    row.rd_index = REGISTERS_READ_WRITE_NO_REGISTER;
    for (uint i = 0u; i < 5u; i++) {
        row.padding[i] = REGISTERS_READ_WRITE_NO_REGISTER;
    }
    return row;
}

inline PackedRegisterCycleRow registers_read_write_remap_row(
    PackedRegisterCycleRow row)
{
    if (registers_read_write_remap_enabled) {
        row.rs1_index = row.padding[0];
        row.rs2_index = row.padding[1];
        row.rd_index = row.padding[2];
    }
    return row;
}

inline uchar registers_read_write_decode_index(ulong flags, uint shift)
{
    uchar plus_one = uchar((flags >> shift) & 0xfful);
    return plus_one == 0u
        ? uchar(REGISTERS_READ_WRITE_NO_REGISTER)
        : uchar(plus_one - 1u);
}

inline uchar registers_read_write_remap_index(
    uchar index,
    device const uchar* register_map)
{
    if (index == REGISTERS_READ_WRITE_NO_REGISTER) {
        return index;
    }
    return registers_read_write_remap_enabled
        ? register_map[uint(index)]
        : index;
}

inline PackedRegisterCycleRow registers_read_write_load_source_row(
    uint row_index,
    device const PackedRegisterCycleRow* packed_rows,
    device const ulong* instruction_input,
    device const ulong* rd_pre,
    device const ulong* rd_post,
    device const uchar* register_map,
    uint source_stride)
{
    if (!registers_read_write_stage1_source) {
        return registers_read_write_remap_row(packed_rows[row_index]);
    }
    uint instruction_offset = 6u * row_index;
    ulong flags = instruction_input[instruction_offset + 5u];
    uchar rd_plus_one = uchar(
        (flags >> REGISTERS_READ_WRITE_RD_INDEX_SHIFT) & 0xfful);
    uchar rd_index = rd_plus_one == 0u
        ? uchar(REGISTERS_READ_WRITE_NO_REGISTER)
        : uchar(rd_plus_one - 1u);
    ulong rd_post_value = rd_post[row_index];
    uchar rs1_index = registers_read_write_decode_index(
        flags, REGISTERS_READ_WRITE_RS1_INDEX_SHIFT);
    uchar rs2_index = registers_read_write_decode_index(
        flags, REGISTERS_READ_WRITE_RS2_INDEX_SHIFT);
    ulong rd_pre_value = rd_plus_one == 0u ? 0ul : rd_pre[row_index];
    PackedRegisterCycleRow row;
    row.rs1_value = instruction_input[instruction_offset];
    row.rs2_value = instruction_input[instruction_offset + 2u];
    row.rd_pre_value = rd_pre_value;
    row.rd_post_value = rd_post_value;
    row.rs1_index = registers_read_write_remap_index(rs1_index, register_map);
    row.rs2_index = registers_read_write_remap_index(rs2_index, register_map);
    row.rd_index = registers_read_write_remap_index(rd_index, register_map);
    for (uint index = 0u; index < 5u; index++) {
        row.padding[index] = 0u;
    }
    return row;
}

kernel void solinas_registers_read_write_derive_rd_pre_chunks(
    device const uchar* rd_indices [[buffer(0)]],
    device const ulong* rd_post [[buffer(1)]],
    device const uchar* register_map [[buffer(2)]],
    device ulong* rd_pre [[buffer(3)]],
    device ulong* last_values [[buffer(4)]],
    device ushort* first_offsets [[buffer(5)]],
    constant RegistersReadWriteRdPreParams& params [[buffer(6)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (chunk >= params.chunks) {
        return;
    }
    threadgroup ulong register_values[REGISTERS_READ_WRITE_RD_PRE_REGISTERS];
    threadgroup ushort register_first[REGISTERS_READ_WRITE_RD_PRE_REGISTERS];
    for (uint reg = lane;
         reg < REGISTERS_READ_WRITE_RD_PRE_REGISTERS;
         reg += REGISTERS_READ_WRITE_SIMD_WIDTH) {
        register_values[reg] = 0ul;
        register_first[reg] = ushort(REGISTERS_READ_WRITE_RD_PRE_NO_OFFSET);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint row_start = chunk * params.chunk_rows;
    uint row_end = min(row_start + params.chunk_rows, params.row_count);
    for (uint batch = 0u; batch < params.chunk_rows;
         batch += REGISTERS_READ_WRITE_SIMD_WIDTH) {
        uint local = batch + lane;
        uint row = row_start + local;
        uchar rd_index = row < row_end
            ? rd_indices[row]
            : uchar(REGISTERS_READ_WRITE_NO_REGISTER);
        uint key = rd_index == uchar(REGISTERS_READ_WRITE_NO_REGISTER)
            ? REGISTERS_READ_WRITE_RD_PRE_NO_KEY
            : (uint(register_map[uint(rd_index)])
                << REGISTERS_READ_WRITE_RD_PRE_LOCAL_BITS) | local;
        ulong post = row < row_end ? rd_post[row] : 0ul;

        for (uint size = 2u; size <= REGISTERS_READ_WRITE_SIMD_WIDTH; size <<= 1u) {
            for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
                uint other_key = simd_shuffle_xor(key, stride);
                uint other_post_low = simd_shuffle_xor(uint(post), stride);
                uint other_post_high = simd_shuffle_xor(uint(post >> 32u), stride);
                ulong other_post = (ulong(other_post_high) << 32u)
                    | ulong(other_post_low);
                bool ascending = (lane & size) == 0u;
                bool lower_lane = (lane & stride) == 0u;
                bool take_minimum = ascending == lower_lane;
                bool take_other = take_minimum
                    ? other_key < key
                    : other_key > key;
                if (take_other) {
                    key = other_key;
                    post = other_post;
                }
            }
        }

        uint reg = key >> REGISTERS_READ_WRITE_RD_PRE_LOCAL_BITS;
        bool write = key != REGISTERS_READ_WRITE_RD_PRE_NO_KEY;
        uint sorted_local = key & REGISTERS_READ_WRITE_RD_PRE_LOCAL_MASK;
        uint prior_key = simd_shuffle_up(key, 1u);
        uint prior_post_low = simd_shuffle_up(uint(post), 1u);
        uint prior_post_high = simd_shuffle_up(uint(post >> 32u), 1u);
        ulong prior_post = (ulong(prior_post_high) << 32u)
            | ulong(prior_post_low);
        bool segment_head = write
            && (lane == 0u
                || (prior_key >> REGISTERS_READ_WRITE_RD_PRE_LOCAL_BITS) != reg);
        if (write) {
            ulong previous = segment_head ? register_values[reg] : prior_post;
            rd_pre[row_start + sorted_local] = previous;
            if (segment_head
                && register_first[reg]
                    == ushort(REGISTERS_READ_WRITE_RD_PRE_NO_OFFSET)) {
                register_first[reg] = ushort(sorted_local);
            }
        }
        uint next_key = simd_shuffle_down(key, 1u);
        bool segment_tail = write
            && (lane + 1u == REGISTERS_READ_WRITE_SIMD_WIDTH
                || (next_key >> REGISTERS_READ_WRITE_RD_PRE_LOCAL_BITS) != reg);
        if (segment_tail) {
            register_values[reg] = post;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    ulong summary = ulong(chunk) * REGISTERS_READ_WRITE_RD_PRE_REGISTERS;
    for (uint reg = lane;
         reg < REGISTERS_READ_WRITE_RD_PRE_REGISTERS;
         reg += REGISTERS_READ_WRITE_SIMD_WIDTH) {
        ulong index = summary + ulong(reg);
        last_values[index] = register_values[reg];
        first_offsets[index] = register_first[reg];
    }
}

kernel void solinas_registers_read_write_fixup_rd_pre(
    device ulong* rd_pre [[buffer(3)]],
    device const ulong* last_values [[buffer(4)]],
    device const ushort* first_offsets [[buffer(5)]],
    constant RegistersReadWriteRdPreParams& params [[buffer(6)]],
    uint register_group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup ulong partition_last[REGISTERS_READ_WRITE_THREADS];
    threadgroup uint partition_first_row[REGISTERS_READ_WRITE_THREADS];

    uint local_register = tid / REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS;
    uint partition = tid % REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS;
    uint reg = register_group
        * REGISTERS_READ_WRITE_RD_PRE_FIXUP_REGISTERS_PER_GROUP
        + local_register;
    uint chunk_start = uint(
        (ulong(params.chunks) * ulong(partition))
        / ulong(REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS));
    uint chunk_end = uint(
        (ulong(params.chunks) * ulong(partition + 1u))
        / ulong(REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS));
    ulong previous = 0ul;
    uint first_row = REGISTERS_READ_WRITE_RD_PRE_NO_KEY;
    for (uint chunk = chunk_start; chunk < chunk_end; chunk++) {
        ulong index = ulong(chunk) * REGISTERS_READ_WRITE_RD_PRE_REGISTERS
            + ulong(reg);
        ushort first = first_offsets[index];
        if (first == ushort(REGISTERS_READ_WRITE_RD_PRE_NO_OFFSET)) {
            continue;
        }
        uint row = chunk * params.chunk_rows + uint(first);
        if (first_row == REGISTERS_READ_WRITE_RD_PRE_NO_KEY) {
            first_row = row;
        }
        rd_pre[row] = previous;
        previous = last_values[index];
    }
    partition_last[tid] = previous;
    partition_first_row[tid] = first_row;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong partition_carry = 0ul;
    uint partition_base = local_register
        * REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS;
    for (uint prior = 0u; prior < partition; prior++) {
        uint index = partition_base + prior;
        if (partition_first_row[index] != REGISTERS_READ_WRITE_RD_PRE_NO_KEY) {
            partition_carry = partition_last[index];
        }
    }
    if (first_row != REGISTERS_READ_WRITE_RD_PRE_NO_KEY) {
        rd_pre[first_row] = partition_carry;
    }
}

kernel void solinas_registers_read_write_source_primer(
    device const ulong* instruction_input [[buffer(0)]],
    device const ulong* instruction_read_raf [[buffer(1)]],
    device const ulong* rd_post [[buffer(2)]],
    device uint* checksums [[buffer(3)]],
    constant RegistersReadWriteSourcePrimerParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.total_threads) {
        return;
    }
    const ulong instruction_pages =
        (params.word_counts[0] + params.page_words - 1u) / params.page_words;
    const ulong read_raf_pages =
        (params.word_counts[1] + params.page_words - 1u) / params.page_words;
    const ulong rd_post_pages =
        (params.word_counts[2] + params.page_words - 1u) / params.page_words;
    const ulong total_pages = instruction_pages + read_raf_pages + rd_post_pages;
    uint checksum = 0x9e3779b9u ^ gid;
    for (ulong page = gid; page < total_pages; page += params.total_threads) {
        ulong value;
        if (page < instruction_pages) {
            value = instruction_input[page * params.page_words];
        } else if (page < instruction_pages + read_raf_pages) {
            value = instruction_read_raf[
                (page - instruction_pages) * params.page_words];
        } else {
            value = rd_post[
                (page - instruction_pages - read_raf_pages) * params.page_words];
        }
        checksum ^= uint(value) ^ uint(value >> 32u) ^ uint(page);
        checksum = ((checksum << 5u) | (checksum >> 27u)) * 0x85ebca6bu;
    }
    checksums[gid] = checksum;
}

inline uint registers_read_write_mask_popcount(ulong mask)
{
    return popcount((uint)mask) + popcount((uint)(mask >> 32u));
}

inline uint registers_read_write_mask_first_column(ulong mask)
{
    uint low = (uint)mask;
    return low != 0u ? ctz(low) : 32u + ctz((uint)(mask >> 32u));
}

inline ulong registers_read_write_row_mask(PackedRegisterCycleRow row)
{
    ulong mask = 0ul;
    if (row.rs1_index != REGISTERS_READ_WRITE_NO_REGISTER) {
        mask |= 1ul << uint(row.rs1_index);
    }
    if (row.rs2_index != REGISTERS_READ_WRITE_NO_REGISTER) {
        mask |= 1ul << uint(row.rs2_index);
    }
    if (row.rd_index != REGISTERS_READ_WRITE_NO_REGISTER) {
        mask |= 1ul << uint(row.rd_index);
    }
    return mask;
}

inline SolinasFp128 registers_read_write_increment(
    PackedRegisterCycleRow row)
{
    if (row.rd_index == REGISTERS_READ_WRITE_NO_REGISTER) {
        return solinas_zero();
    }
    if (row.rd_post_value >= row.rd_pre_value) {
        return registers_read_write_from_u64(
            row.rd_post_value - row.rd_pre_value);
    }
    return solinas_sub(
        solinas_zero(),
        registers_read_write_from_u64(
            row.rd_pre_value - row.rd_post_value));
}

inline RegistersReadWriteCell registers_read_write_cell(
    PackedRegisterCycleRow row,
    uchar column)
{
    bool rs1 = row.rs1_index == column;
    bool rs2 = row.rs2_index == column;
    bool rd = row.rd_index == column;
    RegistersReadWriteCell cell;
    cell.present = rs1 || rs2 || rd;
    cell.write = rd;
    cell.read_mask = uint(rs1) | (uint(rs2) << 1u);
    ulong value = rs1
        ? row.rs1_value
        : (rs2 ? row.rs2_value : row.rd_pre_value);
    cell.previous = value;
    cell.next = rd ? row.rd_post_value : value;
    cell.value = value;
    return cell;
}

inline SolinasFp128 registers_read_write_read_coefficient(
    uint mask,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq)
{
    if (mask == 1u) {
        return gamma;
    }
    if (mask == 2u) {
        return gamma_sq;
    }
    if (mask == 3u) {
        return solinas_add(gamma, gamma_sq);
    }
    return solinas_zero();
}

inline RegistersReadWriteSignedU64 registers_read_write_signed_delta(
    ulong high,
    ulong low)
{
    RegistersReadWriteSignedU64 result;
    result.negative = high < low;
    result.magnitude = result.negative ? low - high : high - low;
    return result;
}

inline SolinasFp128 registers_read_write_delta_field(
    RegistersReadWriteSignedU64 delta)
{
    SolinasFp128 magnitude = registers_read_write_from_u64(delta.magnitude);
    return delta.negative ? solinas_sub(solinas_zero(), magnitude) : magnitude;
}

inline SolinasFp128 registers_read_write_head(
    uint parent,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    uint e_in_length)
{
    uint in_bits = ctz(e_in_length);
    return solinas_mul_wide(
        e_out[parent >> in_bits],
        e_in[parent & (e_in_length - 1u)]);
}

inline void registers_read_write_store_geometry(
    uint count,
    ulong mask,
    device uint* geometry_counts,
    device ushort* geometry_offsets,
    device ulong* geometry_masks,
    threadgroup uint* count_sums,
    uint group,
    uint item,
    bool valid,
    uint tid,
    uint lane,
    uint simdgroup)
{
    uint local_offset = simd_prefix_exclusive_sum(count);
    uint simd_count = simd_sum(count);
    if (lane == 0u) {
        count_sums[simdgroup] = simd_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint cursor = 0u;
        for (uint index = 0u; index < REGISTERS_READ_WRITE_SIMDGROUPS; index++) {
            uint group_count = count_sums[index];
            count_sums[index] = cursor;
            cursor += group_count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        geometry_offsets[item] = (ushort)(count_sums[simdgroup] + local_offset);
        geometry_masks[item] = mask;
    }
    if (simdgroup + 1u == REGISTERS_READ_WRITE_SIMDGROUPS && lane == 0u) {
        geometry_counts[group] = count_sums[simdgroup] + simd_count;
    }
}

inline void registers_read_write_accumulate_first_infinity(
    RegistersReadWriteCell low_cell,
    RegistersReadWriteCell high_cell,
    SolinasFp128 inc_slope,
    SolinasFp128 gamma,
    SolinasFp128 gamma_sq,
    thread SolinasFp128& q_infinity)
{
    ulong low_value = low_cell.present
        ? low_cell.value
        : high_cell.previous;
    ulong high_value = high_cell.present
        ? high_cell.value
        : low_cell.next;
    RegistersReadWriteSignedU64 value_delta =
        registers_read_write_signed_delta(high_value, low_value);
    if (low_cell.read_mask != high_cell.read_mask) {
        SolinasFp128 ra_slope = solinas_sub(
            registers_read_write_read_coefficient(
                high_cell.read_mask, gamma, gamma_sq),
            registers_read_write_read_coefficient(
                low_cell.read_mask, gamma, gamma_sq));
        q_infinity = solinas_add(
            q_infinity,
            solinas_half_width_mul_signed_u64(
                ra_slope,
                value_delta.magnitude,
                value_delta.negative));
    }
    if (low_cell.write != high_cell.write) {
        SolinasFp128 write_term = solinas_add(
            registers_read_write_delta_field(value_delta),
            inc_slope);
        q_infinity = low_cell.write
            ? solinas_sub(q_infinity, write_term)
            : solinas_add(q_infinity, write_term);
    }
}

kernel void solinas_registers_read_write_first_message(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant RegistersReadWriteFirstMessageParams& params [[buffer(4)]],
    constant SolinasFp128& gamma [[buffer(5)]],
    constant SolinasFp128& gamma_sq [[buffer(6)]],
    device uint* geometry_counts [[buffer(7)]],
    device ushort* geometry_offsets [[buffer(8)]],
    device ulong* geometry_masks [[buffer(9)]],
    device const ulong* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_rd_pre [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    device uchar* stage1_compact_rs1 [[buffer(27)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    uint pair = group * REGISTERS_READ_WRITE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    uint union_count = 0u;
    ulong union_mask = 0ul;
    if (pair < params.pair_count) {
        uint low_index = 2u * pair;
        uint high_index = low_index + 1u;
        PackedRegisterCycleRow low = low_index < params.row_count
            ? registers_read_write_load_source_row(
                low_index,
                rows,
                stage1_instruction_input,
                stage1_rd_pre,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
        PackedRegisterCycleRow high = high_index < params.row_count
            ? registers_read_write_load_source_row(
                high_index,
                rows,
                stage1_instruction_input,
                stage1_rd_pre,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
        if (registers_read_write_stage1_source) {
            stage1_compact_rs1[low_index] = registers_read_write_decode_index(
                stage1_instruction_input[6u * low_index + 5u],
                REGISTERS_READ_WRITE_RS1_INDEX_SHIFT);
            if (high_index < params.row_count) {
                stage1_compact_rs1[high_index] = registers_read_write_decode_index(
                    stage1_instruction_input[6u * high_index + 5u],
                    REGISTERS_READ_WRITE_RS1_INDEX_SHIFT);
            }
        }
        SolinasFp128 inc_low = registers_read_write_increment(low);
        SolinasFp128 inc_high = registers_read_write_increment(high);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_low);
        uchar columns[6] = {
            low.rs1_index,
            low.rs2_index,
            low.rd_index,
            high.rs1_index,
            high.rs2_index,
            high.rd_index,
        };
        for (uint candidate = 0u; candidate < 6u; candidate++) {
            uchar column = columns[candidate];
            if (column == REGISTERS_READ_WRITE_NO_REGISTER) {
                continue;
            }
            bool duplicate = false;
            for (uint prior = 0u; prior < candidate; prior++) {
                duplicate = duplicate || columns[prior] == column;
            }
            if (!duplicate) {
                union_count += 1u;
                union_mask |= 1ul << uint(column);
                RegistersReadWriteCell low_cell = registers_read_write_cell(
                    low, column);
                RegistersReadWriteCell high_cell = registers_read_write_cell(
                    high, column);
                ulong low_value = low_cell.present
                    ? low_cell.value
                    : high_cell.previous;
                ulong high_value = high_cell.present
                    ? high_cell.value
                    : low_cell.next;
                RegistersReadWriteSignedU64 value_delta =
                    registers_read_write_signed_delta(high_value, low_value);
                SolinasFp128 value_slope =
                    registers_read_write_delta_field(value_delta);
                if (low_cell.read_mask != 0u) {
                    q_zero = solinas_add(
                        q_zero,
                        solinas_half_width_mul_u64(
                            registers_read_write_read_coefficient(
                                low_cell.read_mask, gamma, gamma_sq),
                            low_value));
                }
                if (low_cell.write) {
                    q_zero = solinas_add(
                        q_zero,
                        solinas_add(
                            registers_read_write_from_u64(low_value),
                            inc_low));
                }
                if (low_cell.read_mask != high_cell.read_mask) {
                    SolinasFp128 ra_slope = solinas_sub(
                        registers_read_write_read_coefficient(
                            high_cell.read_mask, gamma, gamma_sq),
                        registers_read_write_read_coefficient(
                            low_cell.read_mask, gamma, gamma_sq));
                    q_infinity = solinas_add(
                        q_infinity,
                        solinas_half_width_mul_signed_u64(
                            ra_slope,
                            value_delta.magnitude,
                            value_delta.negative));
                }
                if (low_cell.write != high_cell.write) {
                    SolinasFp128 write_term = solinas_add(value_slope, inc_slope);
                    q_infinity = low_cell.write
                        ? solinas_sub(q_infinity, write_term)
                        : solinas_add(q_infinity, write_term);
                }
            }
        }
        SolinasFp128 head = registers_read_write_head(
            pair, e_in, e_out, params.e_in_length);
        q_zero = solinas_mul_wide(head, q_zero);
        q_infinity = solinas_mul_wide(head, q_infinity);
    }

    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SIMDGROUPS];
    registers_read_write_store_geometry(
        union_count,
        union_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        pair,
        pair < params.pair_count,
        tid,
        lane,
        simdgroup);
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < REGISTERS_READ_WRITE_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < REGISTERS_READ_WRITE_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[group] = q_zero;
            partials[params.output_stride + group] = q_infinity;
        }
    }
}

kernel void solinas_registers_read_write_first_message_intersection(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant RegistersReadWriteFirstMessageParams& params [[buffer(4)]],
    constant SolinasFp128& gamma [[buffer(5)]],
    constant SolinasFp128& gamma_sq [[buffer(6)]],
    device const ulong* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_rd_pre [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    device uchar* stage1_compact_rs1 [[buffer(27)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    uint pair = group * REGISTERS_READ_WRITE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (pair < params.pair_count) {
        uint low_index = 2u * pair;
        uint high_index = low_index + 1u;
        PackedRegisterCycleRow low = low_index < params.row_count
            ? registers_read_write_load_source_row(
                low_index,
                rows,
                stage1_instruction_input,
                stage1_rd_pre,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
        PackedRegisterCycleRow high = high_index < params.row_count
            ? registers_read_write_load_source_row(
                high_index,
                rows,
                stage1_instruction_input,
                stage1_rd_pre,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
        if (registers_read_write_stage1_source) {
            stage1_compact_rs1[low_index] = registers_read_write_decode_index(
                stage1_instruction_input[6u * low_index + 5u],
                REGISTERS_READ_WRITE_RS1_INDEX_SHIFT);
            if (high_index < params.row_count) {
                stage1_compact_rs1[high_index] = registers_read_write_decode_index(
                    stage1_instruction_input[6u * high_index + 5u],
                    REGISTERS_READ_WRITE_RS1_INDEX_SHIFT);
            }
        }
        SolinasFp128 inc_low = registers_read_write_increment(low);
        SolinasFp128 inc_high = registers_read_write_increment(high);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_low);
        if (low.rs1_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            q_zero = solinas_add(
                q_zero,
                solinas_half_width_mul_u64(gamma, low.rs1_value));
        }
        if (low.rs2_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            q_zero = solinas_add(
                q_zero,
                solinas_half_width_mul_u64(gamma_sq, low.rs2_value));
        }
        if (low.rd_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            q_zero = solinas_add(
                q_zero,
                registers_read_write_from_u64(low.rd_post_value));
        }
        ulong low_mask = registers_read_write_row_mask(low);
        ulong high_mask = registers_read_write_row_mask(high);
        ulong intersection_mask = low_mask & high_mask;
        while (intersection_mask != 0ul) {
            uchar column = (uchar)registers_read_write_mask_first_column(
                intersection_mask);
            intersection_mask &= intersection_mask - 1ul;
            RegistersReadWriteCell low_cell = registers_read_write_cell(low, column);
            RegistersReadWriteCell high_cell = registers_read_write_cell(high, column);
            registers_read_write_accumulate_first_infinity(
                low_cell,
                high_cell,
                inc_slope,
                gamma,
                gamma_sq,
                q_infinity);
        }
        if (low.rd_index != REGISTERS_READ_WRITE_NO_REGISTER
            && (high_mask & (1ul << uint(low.rd_index))) == 0ul) {
            RegistersReadWriteCell low_cell = registers_read_write_cell(
                low, low.rd_index);
            if (low_cell.read_mask != 0u) {
                RegistersReadWriteSignedU64 write_delta =
                    registers_read_write_signed_delta(
                        low.rd_post_value, low.rd_pre_value);
                q_infinity = solinas_sub(
                    q_infinity,
                    solinas_half_width_mul_signed_u64(
                        registers_read_write_read_coefficient(
                            low_cell.read_mask, gamma, gamma_sq),
                        write_delta.magnitude,
                        write_delta.negative));
            }
            q_infinity = solinas_sub(q_infinity, inc_high);
        }
        if (high.rd_index != REGISTERS_READ_WRITE_NO_REGISTER
            && (low_mask & (1ul << uint(high.rd_index))) == 0ul) {
            q_infinity = solinas_add(q_infinity, inc_slope);
        }
        SolinasFp128 head = registers_read_write_head(
            pair, e_in, e_out, params.e_in_length);
        q_zero = solinas_mul_wide(head, q_zero);
        q_infinity = solinas_mul_wide(head, q_infinity);
    }

    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SIMDGROUPS];
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < REGISTERS_READ_WRITE_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < REGISTERS_READ_WRITE_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[group] = q_zero;
            partials[params.output_stride + group] = q_infinity;
        }
    }
}
