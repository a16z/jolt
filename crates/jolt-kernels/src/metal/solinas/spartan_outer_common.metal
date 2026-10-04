struct InstructionInputRow {
    ulong2 chunks[3];
};

// [pc, memory_0, memory_1, lookup_output]; SpartanRawRow in spartan_outer_uniskip/mod.rs.
struct SpartanRawRow {
    ulong2 chunks[2];
};

inline ulong instruction_input_row_word(
    device const InstructionInputRow& row,
    uint word)
{
    return row.chunks[word >> 1][word & 1u];
}

inline ulong spartan_raw_row_word(device const SpartanRawRow& row, uint word) {
    return row.chunks[word >> 1][word & 1u];
}

// RdWriteValue from the memory slots of SpartanOuterUniskipRow::split.
inline ulong spartan_rd_write_value(ulong flags, ulong memory_0, ulong memory_1) {
    bool load = (flags & 1ul) != 0;
    bool store = ((flags >> 1) & 1ul) != 0;
    return store ? 0ul : (load ? memory_1 : memory_0);
}

inline ulong spartan_row_rd_write_value(
    device const InstructionInputRow& compact,
    device const SpartanRawRow& raw)
{
    return spartan_rd_write_value(
        instruction_input_row_word(compact, 5u),
        spartan_raw_row_word(raw, 1u),
        spartan_raw_row_word(raw, 2u));
}

// Two's-complement 128-bit value: lo, hi.
struct SpartanU128 {
    ulong lo;
    ulong hi;
};

inline SpartanU128 spartan_u128(ulong lo, ulong hi) {
    SpartanU128 value;
    value.lo = lo;
    value.hi = hi;
    return value;
}

inline SpartanU128 spartan_u128_add(SpartanU128 left, SpartanU128 right) {
    ulong lo = left.lo + right.lo;
    return spartan_u128(lo, left.hi + right.hi + (lo < left.lo ? 1ul : 0ul));
}

inline SpartanU128 spartan_u128_negate(SpartanU128 value) {
    return spartan_u128_add(spartan_u128(~value.lo, ~value.hi), spartan_u128(1ul, 0ul));
}

inline SpartanU128 spartan_u128_mul_u64(ulong left, ulong right) {
    return spartan_u128(left * right, mulhi(left, right));
}

// The residual words r0..r13 of the 20-word row (index map in
// SpartanOuterUniskipRow::split), rebuilt from the compact and raw rows. The
// derived words assume the expansion's normalized operands: the left/right
// inputs are the flag-selected rs1/upc and rs2/imm, and the lookup operands
// follow the per-family formulas below (R1CS rows 5-10 pin all but the Advice
// right operand, which equals the lookup output). The witness test
// stage1_rows_cover_every_expanded_instruction pins every instruction kind.
struct SpartanOuterResidual {
    ulong word[14];
};

// Every residual word except the successor words r11/r12 (left zero), which
// spartan_outer_decode_residual reads from row t + 1.
inline SpartanOuterResidual spartan_outer_decode_current(
    device const InstructionInputRow& current,
    device const SpartanRawRow& current_raw)
{
    ulong rs1 = instruction_input_row_word(current, 0u);
    ulong unexpanded_pc = instruction_input_row_word(current, 1u);
    ulong rs2 = instruction_input_row_word(current, 2u);
    ulong flags = instruction_input_row_word(current, 5u);
    ulong left = (((flags >> 20) & 1ul) != 0 ? rs1 : 0ul)
        + (((flags >> 21) & 1ul) != 0 ? unexpanded_pc : 0ul);
    SpartanU128 imm = spartan_u128(
        instruction_input_row_word(current, 3u),
        instruction_input_row_word(current, 4u));
    if (((flags >> 18) & 1ul) == 0) {
        imm = spartan_u128_negate(imm);
    }
    SpartanU128 right = spartan_u128_add(
        spartan_u128(((flags >> 22) & 1ul) != 0 ? rs2 : 0ul, 0ul),
        ((flags >> 23) & 1ul) != 0 ? imm : spartan_u128(0ul, 0ul));
    SpartanU128 right_magnitude =
        (right.hi >> 63) != 0 ? spartan_u128_negate(right) : right;
    SpartanU128 product = spartan_u128_mul_u64(left, right_magnitude.lo);
    product.hi += left * right_magnitude.hi;

    bool add = ((flags >> 2) & 1ul) != 0;
    bool sub = ((flags >> 3) & 1ul) != 0;
    bool mul = ((flags >> 4) & 1ul) != 0;
    ulong lookup_output = spartan_raw_row_word(current_raw, 3u);
    SpartanU128 right_lookup;
    if (add) {
        right_lookup = spartan_u128_add(spartan_u128(left, 0ul), right);
    } else if (sub) {
        right_lookup = spartan_u128_add(
            spartan_u128(left, 1ul), spartan_u128_negate(right));
    } else if (mul) {
        right_lookup = spartan_u128_mul_u64(left, right.lo);
    } else if (((flags >> 13) & 1ul) != 0) {
        right_lookup = spartan_u128(lookup_output, 0ul);
    } else {
        right_lookup = spartan_u128(right.lo, 0ul);
    }

    SpartanOuterResidual residual;
    residual.word[0] = left;
    residual.word[1] = right_magnitude.lo;
    residual.word[2] = right_magnitude.hi;
    residual.word[3] = product.lo;
    residual.word[4] = product.hi;
    residual.word[5] = spartan_raw_row_word(current_raw, 0u);
    residual.word[6] = spartan_raw_row_word(current_raw, 1u);
    residual.word[7] = spartan_raw_row_word(current_raw, 2u);
    residual.word[8] = add || sub || mul ? 0ul : left;
    residual.word[9] = right_lookup.lo;
    residual.word[10] = right_lookup.hi;
    residual.word[11] = 0ul;
    residual.word[12] = 0ul;
    residual.word[13] = lookup_output;
    return residual;
}

inline SpartanOuterResidual spartan_outer_decode_residual(
    device const InstructionInputRow* compact,
    device const SpartanRawRow* raw,
    uint row,
    uint rows)
{
    SpartanOuterResidual residual = spartan_outer_decode_current(compact[row], raw[row]);
    if (row + 1u < rows) {
        residual.word[11] = instruction_input_row_word(compact[row + 1u], 1u);
        residual.word[12] = spartan_raw_row_word(raw[row + 1u], 0u);
    }
    return residual;
}

struct SpartanSigned192 {
    uint limb[6];
};

inline SpartanSigned192 spartan_s192_zero() {
    SpartanSigned192 value;
    for (uint i = 0; i < 6; i++) {
        value.limb[i] = 0;
    }
    return value;
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
