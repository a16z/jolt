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

// Chunk of an offset-by-one source (bytecode PC + 1, RAM address + 1); a zero
// source is a cold cycle.
inline bool booleanity_offset_chunk(ulong plus_one, uint shift, uint mask, thread uint& hot)
{
    if (plus_one == 0ul) {
        return false;
    }
    hot = (uint)((plus_one - 1ul) >> shift) & mask;
    return true;
}

// Balanced fused-increment chunk: digit (kind 3) or carry (kind 4).
inline uint booleanity_inc_chunk(
    ulong magnitude,
    bool negative,
    BooleanitySelector selector,
    uint chunk_bits,
    ulong inc_bias)
{
    uint mask = (1u << chunk_bits) - 1u;
    ulong biased = negative ? inc_bias - magnitude : inc_bias + magnitude;
    if (selector.kind == 3u) {
        // Adding 2^(chunk_bits - 1) modulo 2^chunk_bits flips the chunk's top bit.
        return ((uint)(biased >> selector.shift) & mask) ^ (1u << (chunk_bits - 1u));
    }
    if (negative) {
        return magnitude > inc_bias ? mask : 0u;
    }
    return biased < inc_bias ? 1u : 0u;
}

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
        return booleanity_offset_chunk(
            row.packed_pc_and_flags & 0x00ffFFFFFFFFFFFFul, selector.shift, mask, hot);
    }
    if (selector.kind == 2u) {
        return booleanity_offset_chunk(
            row.ram_address_plus_one & 0x00ffFFFFFFFFFFFFul, selector.shift, mask, hot);
    }
    hot = booleanity_inc_chunk(
        row.fused_inc_magnitude,
        (row.packed_pc_and_flags >> 63) != 0ul,
        selector,
        chunk_bits,
        inc_bias);
    return true;
}
