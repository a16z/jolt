#define SPARTAN_OUTER_NODES 9u
#define SPARTAN_OUTER_ROW_WORDS 20u
#define SPARTAN_OUTER_SIMD_WIDTH 32u
#define SPARTAN_OUTER_FIRST_NODES 6u
#define SPARTAN_OUTER_FIRST_ROWS_PER_SIMD 5u
#define SPARTAN_OUTER_SECOND_NODES 3u
#define SPARTAN_OUTER_SECOND_ROWS_PER_SIMD 10u

struct SpartanOuterUniskipRow {
    ulong words[SPARTAN_OUTER_ROW_WORDS];
};

struct SpartanOuterUniskipParams {
    uint rows;
    uint pairs_per_block;
    uint blocks;
    uint reserved;
};

struct SpartanSigned192 {
    uint limb[6];
};

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

inline SpartanSigned192 spartan_s192_zero() {
    SpartanSigned192 value;
    for (uint i = 0; i < 6; i++) {
        value.limb[i] = 0;
    }
    return value;
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

inline SpartanSigned192 spartan_s192_negate(SpartanSigned192 value) {
    ulong carry = 1;
    for (uint i = 0; i < 6; i++) {
        ulong word = (ulong)(~value.limb[i]) + carry;
        value.limb[i] = (uint)word;
        carry = word >> 32;
    }
    return value;
}

inline void spartan_s192_add(
    thread SpartanSigned192& accumulator,
    SpartanSigned192 value)
{
    ulong carry = 0;
    for (uint i = 0; i < 6; i++) {
        ulong word = (ulong)accumulator.limb[i] + (ulong)value.limb[i] + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
}

inline SpartanSigned192 spartan_scaled_u64(ulong value, uint scale) {
    SpartanSigned192 product = spartan_s192_zero();
    ulong word = (ulong)(uint)value * (ulong)scale;
    product.limb[0] = (uint)word;
    ulong carry = word >> 32;
    word = (ulong)(uint)(value >> 32) * (ulong)scale + carry;
    product.limb[1] = (uint)word;
    product.limb[2] = (uint)(word >> 32);
    return product;
}

inline SpartanSigned192 spartan_scaled_u128(ulong low, ulong high, uint scale) {
    SpartanSigned192 product = spartan_s192_zero();
    uint source[4] = {
        (uint)low,
        (uint)(low >> 32),
        (uint)high,
        (uint)(high >> 32),
    };
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)source[i] * (ulong)scale + carry;
        product.limb[i] = (uint)word;
        carry = word >> 32;
    }
    product.limb[4] = (uint)carry;
    return product;
}

inline void spartan_accumulate_scaled_u64(
    thread SpartanSigned192& accumulator,
    ulong value,
    int coefficient)
{
    if (coefficient == 0 || value == 0) {
        return;
    }
    bool negative = coefficient < 0;
    uint scale = negative ? (uint)(-coefficient) : (uint)coefficient;
    SpartanSigned192 product = spartan_scaled_u64(value, scale);
    spartan_s192_add(accumulator, negative ? spartan_s192_negate(product) : product);
}

inline void spartan_accumulate_scaled_u128(
    thread SpartanSigned192& accumulator,
    ulong low,
    ulong high,
    bool positive,
    int coefficient)
{
    if (coefficient == 0 || (low == 0 && high == 0)) {
        return;
    }
    bool negative = (coefficient < 0) == positive;
    uint scale = coefficient < 0 ? (uint)(-coefficient) : (uint)coefficient;
    SpartanSigned192 product = spartan_scaled_u128(low, high, scale);
    spartan_s192_add(accumulator, negative ? spartan_s192_negate(product) : product);
}

inline void spartan_accumulate_i32(
    thread SpartanSigned192& accumulator,
    int value)
{
    if (value == 0) {
        return;
    }
    bool negative = value < 0;
    uint magnitude = negative ? (uint)(-value) : (uint)value;
    SpartanSigned192 encoded = spartan_s192_zero();
    encoded.limb[0] = magnitude;
    spartan_s192_add(accumulator, negative ? spartan_s192_negate(encoded) : encoded);
}

inline void spartan_accumulate_pow64(
    thread SpartanSigned192& accumulator,
    int coefficient)
{
    if (coefficient == 0) {
        return;
    }
    bool negative = coefficient < 0;
    SpartanSigned192 encoded = spartan_s192_zero();
    encoded.limb[2] = negative ? (uint)(-coefficient) : (uint)coefficient;
    spartan_s192_add(accumulator, negative ? spartan_s192_negate(encoded) : encoded);
}

inline SolinasFp128 spartan_small_times_s192(int small, SpartanSigned192 wide) {
    bool wide_negative = (wide.limb[5] & 0x80000000u) != 0;
    if (wide_negative) {
        wide = spartan_s192_negate(wide);
    }
    bool small_negative = small < 0;
    uint scale = small_negative ? (uint)(-small) : (uint)small;
    SolinasWide256 product;
    for (uint i = 0; i < 8; i++) {
        product.limb[i] = 0;
    }
    ulong carry = 0;
    for (uint i = 0; i < 6; i++) {
        ulong word = (ulong)wide.limb[i] * (ulong)scale + carry;
        product.limb[i] = (uint)word;
        carry = word >> 32;
    }
    product.limb[6] = (uint)carry;
    SolinasFp128 reduced = solinas_reduce(product);
    return wide_negative != small_negative
        ? solinas_sub(solinas_zero(), reduced)
        : reduced;
}

inline void spartan_outer_accumulate_contribution(
    device const SpartanOuterUniskipRow& row,
    constant const int* c,
    SolinasFp128 even_weight,
    SolinasFp128 odd_weight,
    thread SpartanFieldSum192& accumulator)
{
    ulong flags = row.words[19];
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

    ulong rs1 = row.words[9];
    ulong slot1 = row.words[10];
    ulong slot2 = row.words[11];
    ulong slot3 = row.words[12];
    ulong ram_address = load != 0 ? slot1 : (store != 0 ? slot3 : 0);
    ulong rs2 = load != 0 ? 0 : slot1;
    ulong rd_write = store != 0 ? 0 : slot3;
    ulong ram_read = load != 0 ? slot3 : (store != 0 ? slot2 : 0);
    ulong ram_write = load != 0 ? slot3 : (store != 0 ? slot1 : 0);

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
    spartan_accumulate_scaled_u64(bz_first, row.words[13], c[4] + c[5]);
    spartan_accumulate_scaled_u64(bz_first, row.words[0], -c[5]);
    spartan_accumulate_scaled_u64(bz_first, row.words[18], c[6] - c[7]);
    spartan_accumulate_scaled_u64(bz_first, row.words[16], c[7]);
    spartan_accumulate_scaled_u64(bz_first, row.words[17], c[8]);
    spartan_accumulate_scaled_u64(bz_first, row.words[5], -c[8]);
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
        row.words[7],
        row.words[8],
        spartan_outer_flag(flags, 18),
        -c[0] - c[7]);
    spartan_accumulate_scaled_u128(
        bz_second,
        row.words[14],
        row.words[15],
        true,
        c[1] + c[2] + c[3] + c[4]);
    spartan_accumulate_scaled_u64(bz_second, row.words[0], -c[1] - c[2]);
    spartan_accumulate_scaled_u128(
        bz_second,
        row.words[1],
        row.words[2],
        spartan_outer_flag(flags, 17),
        -c[1] + c[2] - c[4]);
    spartan_accumulate_pow64(bz_second, -c[2]);
    spartan_accumulate_scaled_u128(
        bz_second,
        row.words[3],
        row.words[4],
        spartan_outer_flag(flags, 19),
        -c[3]);
    spartan_accumulate_scaled_u64(bz_second, rd_write, c[5] + c[6]);
    spartan_accumulate_scaled_u64(bz_second, row.words[18], -c[5]);
    spartan_accumulate_scaled_u64(bz_second, row.words[6], -c[6] - c[7] - c[8]);
    spartan_accumulate_scaled_u64(bz_second, row.words[16], c[7] + c[8]);
    spartan_accumulate_i32(
        bz_second,
        -4 * c[6] - 4 * c[8]
            + is_compressed * (2 * c[6] + 2 * c[8])
            + do_not_update * 4 * c[8]);
    SolinasFp128 second = spartan_small_times_s192(az_second, bz_second);

    spartan_field_sum_add(accumulator, solinas_mul_wide(even_weight, first));
    spartan_field_sum_add(accumulator, solinas_mul_wide(odd_weight, second));
}

inline SolinasFp128 spartan_outer_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

kernel void solinas_spartan_outer_uniskip_blocks(
    device const SpartanOuterUniskipRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* block_sums [[buffer(3)]],
    constant SpartanOuterUniskipParams& params [[buffer(4)]],
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
                        rows[row_index],
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
                        rows[row_index],
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
    sum = spartan_outer_simd_sum(sum);
    if (lane == 0) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        uint simdgroups = threads / SPARTAN_OUTER_SIMD_WIDTH;
        sum = lane < simdgroups ? shared[lane] : solinas_zero();
        sum = spartan_outer_simd_sum(sum);
        if (lane == 0) {
            output[node] = sum;
        }
    }
}
