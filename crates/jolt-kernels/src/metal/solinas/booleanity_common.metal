struct BooleanityRow {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

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
        if (row.ram_address_plus_one == 0ul) {
            return false;
        }
        hot = (uint)((row.ram_address_plus_one - 1ul) >> selector.shift) & mask;
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
