inline SolinasFp128 solinas_simd_sum_32(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}
