// Donor registers_val first-message/native/dense transitions, adapted to
// guarded explicit prefixes and the canonical jolt field/accumulator owner.
struct RegisterValParams {
    uint previous_prefix, prefix, inc_prefix, rd_prefix, lo_length, mode;
};
template <typename F>
F register_val_source(device const ulong2* inc, device const uchar* rd,
    device const F* eq_address, device const F* previous,
    constant RegisterValParams& p, ulong row, uint factor) {
    if (p.mode == 2) {
        if (row < p.previous_prefix) return previous[2 * row + factor];
        return F::zero();
    }
    if (factor == 0) return row < p.inc_prefix ? jolt::from_i128<F>(inc[row]) : F::zero();
    if (row >= p.rd_prefix || rd[row] == 255u) return F::zero();
    return eq_address[rd[row]];
}
template <typename F>
kernel void registers_val_round(device const ulong2* inc [[buffer(0)]],
    device const uchar* rd [[buffer(1)]], device const F* eq_address [[buffer(2)]],
    device const F* previous [[buffer(3)]], device F* output [[buffer(4)]],
    device const F* lt_lo [[buffer(5)]], device const F* lt_hi [[buffer(6)]],
    device const F* eq_hi [[buffer(7)]], device F* partials [[buffer(8)]],
    constant RegisterValParams& p [[buffer(9)]], constant F& challenge [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 16384, "logT<=31 low-block product bound");
    threadgroup Acc scratch[6][8];
    Acc sums[6];
    for (uint i = 0; i < 6; i++) sums[i] = Acc::zero();
    for (uint pair = local; pair < p.lo_length / 2; pair += 256) {
        ulong first = ulong(group) * p.lo_length + 2 * pair;
        if (first >= p.prefix) break;
        F values[2][2];
        for (uint side = 0; side < 2; side++) {
            ulong row = first + side;
            for (uint factor = 0; factor < 2; factor++) {
                F value = F::zero();
                if (row < p.prefix) {
                    ulong source = p.mode == 0 ? row : 2 * row;
                    F low = register_val_source(inc, rd, eq_address, previous, p, source, factor);
                    value = low;
                    if (p.mode != 0) {
                        F high = register_val_source(inc, rd, eq_address, previous, p, source + 1, factor);
                        value = low + challenge * (high - low);
                        output[2 * row + factor] = value;
                    }
                }
                values[side][factor] = value;
            }
        }
        F inc_delta = values[1][0] - values[0][0];
        F wa_delta = values[1][1] - values[0][1];
        F lt_delta = lt_lo[2 * pair + 1] - lt_lo[2 * pair];
        F a[3] = {values[0][0], values[1][0] + inc_delta, values[1][0] + inc_delta + inc_delta};
        F b[3] = {values[0][1], values[1][1] + wa_delta, values[1][1] + wa_delta + wa_delta};
        F c[3] = {lt_lo[2 * pair], lt_lo[2 * pair + 1] + lt_delta, lt_lo[2 * pair + 1] + lt_delta + lt_delta};
        for (uint sample = 0; sample < 3; sample++) {
            F product = a[sample] * b[sample];
            sums[sample].add(product);
            sums[sample + 3].fmadd(product, c[sample]);
        }
    }
    for (uint sample = 0; sample < 3; sample++) {
        auto a = jolt::threadgroup_merge(sums[sample], scratch[sample], lane, sg, 8, 32);
        auto b = jolt::threadgroup_merge(sums[sample + 3], scratch[sample + 3], lane, sg, 8, 32);
        if (lane == 0 && sg == 0)
            partials[ulong(group) * 3 + sample] = lt_hi[group] * a.reduce() + eq_hi[group] * b.reduce();
    }
}
