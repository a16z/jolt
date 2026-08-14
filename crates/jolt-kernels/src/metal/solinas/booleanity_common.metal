struct BooleanityRow {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

#define BOOLEANITY_SOURCE_RAM_MASK 0xfffffffful
#define BOOLEANITY_SOURCE_PC_MASK 0x3ffful
#define BOOLEANITY_SOURCE_PC_SHIFT 32u
#define BOOLEANITY_SOURCE_RD_SHIFT 46u
#define BOOLEANITY_SOURCE_RANK_SHIFT 54u
#define BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT 61u
#define BOOLEANITY_SOURCE_RD_SIGN_SHIFT 62u

inline ulong booleanity_source_word(
    device const ulong* rows,
    uint row_count,
    uint word,
    uint row)
{
    return rows[word * row_count + row];
}

inline ulong booleanity_row_word(
    device const ulong* rows,
    uint row_count,
    uint word,
    uint row)
{
    if (word < 2u) {
        return booleanity_source_word(rows, row_count, word, row);
    }
    if (word == 3u) {
        return booleanity_source_word(rows, row_count, 2u, row);
    }
    ulong metadata = booleanity_source_word(rows, row_count, 3u, row);
    if (word == 2u) {
        return metadata & BOOLEANITY_SOURCE_RAM_MASK;
    }
    ulong pc_plus_one =
        (metadata >> BOOLEANITY_SOURCE_PC_SHIFT) & BOOLEANITY_SOURCE_PC_MASK;
    ulong rank = (metadata >> BOOLEANITY_SOURCE_RANK_SHIFT) & 0x7ful;
    ulong fused_negative =
        (metadata >> BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT) & 1ul;
    return pc_plus_one | (rank << 56u) | (fused_negative << 63u);
}

inline BooleanityRow booleanity_row_load(
    device const ulong* rows,
    uint row_count,
    uint row)
{
    BooleanityRow value;
    value.lookup_lo = booleanity_row_word(rows, row_count, 0u, row);
    value.lookup_hi = booleanity_row_word(rows, row_count, 1u, row);
    value.ram_address_plus_one = booleanity_row_word(rows, row_count, 2u, row);
    value.fused_inc_magnitude = booleanity_row_word(rows, row_count, 3u, row);
    value.packed_pc_and_flags = booleanity_row_word(rows, row_count, 4u, row);
    return value;
}

struct BooleanitySelector {
    uint kind;
    uint shift;
};

inline bool booleanity_hot_index(
    BooleanityRow row,
    BooleanitySelector selector,
    uint chunk_bits,
    ulong inc_bias,
    thread uint& hot)
{
    uint mask = (1u << chunk_bits) - 1u;
    if (selector.kind == 0u) {
        ulong word = selector.shift < 64u ? row.lookup_lo : row.lookup_hi;
        uint shift = selector.shift < 64u ? selector.shift : selector.shift - 64u;
        hot = (uint)(word >> shift) & mask;
        return true;
    }
    if (selector.kind == 1u) {
        ulong plus_one = row.packed_pc_and_flags & 0x00ffFFFFFFFFFFFFul;
        if (plus_one == 0ul) {
            return false;
        }
        hot = (uint)((plus_one - 1ul) >> selector.shift) & mask;
        return true;
    }
    if (selector.kind == 2u) {
        ulong plus_one = row.ram_address_plus_one & 0x00ffFFFFFFFFFFFFul;
        if (plus_one == 0ul) {
            return false;
        }
        hot = (uint)((plus_one - 1ul) >> selector.shift) & mask;
        return true;
    }

    bool negative = (row.packed_pc_and_flags >> 63) != 0ul;
    ulong biased;
    int carry;
    if (negative) {
        biased = inc_bias - row.fused_inc_magnitude;
        carry = row.fused_inc_magnitude > inc_bias ? -1 : 0;
    } else {
        biased = inc_bias + row.fused_inc_magnitude;
        carry = biased < inc_bias ? 1 : 0;
    }
    if (selector.kind == 3u) {
        uint standard = (uint)(biased >> selector.shift) & mask;
        hot = (standard + (1u << (chunk_bits - 1u))) & mask;
    } else {
        hot = (uint)carry & mask;
    }
    return true;
}
