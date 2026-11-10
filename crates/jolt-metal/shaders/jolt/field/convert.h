namespace jolt {
// Full signed i128, including MIN, without signed negation or overflow.
template <typename F> F from_i128(ulong2 bits) {
    F lo = F::from_u64(bits.x);
    if (bits.y == 0) return lo;
    F radix = F::from_u64(1ul << 63) + F::from_u64(1ul << 63);
    if (bits.y == ~0ul) return lo - radix;
    return lo + F::from_i64(as_type<long>(bits.y)) * radix;
}
}
