struct InstructionInputRow {
    ulong2 chunks[3];
};

struct SpartanOuterSuccessorRow {
    ulong2 chunks[4];
};

struct SpartanOuterColdRow {
    ulong2 chunks[3];
};

inline ulong instruction_input_row_word(
    device const InstructionInputRow& row,
    uint word)
{
    return row.chunks[word >> 1][word & 1u];
}

inline ulong spartan_outer_successor_word(
    device const SpartanOuterSuccessorRow& row,
    uint word)
{
    uint stored_word;
    switch (word) {
        case 0u: stored_word = 0u; break;
        case 1u: stored_word = 1u; break;
        case 2u: stored_word = 2u; break;
        case 7u: stored_word = 3u; break;
        case 8u: stored_word = 4u; break;
        case 9u: stored_word = 5u; break;
        case 10u: stored_word = 6u; break;
        default: stored_word = 7u; break;
    }
    return row.chunks[stored_word >> 1][stored_word & 1u];
}

inline ulong spartan_outer_cold_word(
    device const SpartanOuterColdRow& row,
    uint word)
{
    uint stored_word;
    switch (word) {
        case 3u: stored_word = 0u; break;
        case 4u: stored_word = 1u; break;
        case 5u: stored_word = 2u; break;
        case 6u: stored_word = 3u; break;
        case 11u: stored_word = 4u; break;
        default: stored_word = 5u; break;
    }
    return row.chunks[stored_word >> 1][stored_word & 1u];
}

inline ulong spartan_outer_residual_word(
    device const SpartanOuterSuccessorRow& successor,
    device const SpartanOuterColdRow& cold,
    uint word)
{
    switch (word) {
        case 0u:
        case 1u:
        case 2u:
        case 7u:
        case 8u:
        case 9u:
        case 10u:
        case 13u:
            return spartan_outer_successor_word(successor, word);
        default:
            return spartan_outer_cold_word(cold, word);
    }
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
