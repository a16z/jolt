uint ra_fold_index(ulong4 row, uint2 selector, uint radix) {
    ulong value = row[selector.x];
    if (selector.x == 2) value &= 0x00FFFFFFFFFFFFFFul;
    if (selector.x == 3) {
        if (value == 0) return radix;
        value--;
    }
    return uint(value >> selector.y) & (radix - 1);
}
template <typename F>
kernel void ra_materialize(
    device const ulong4* rows [[buffer(0)]], device const F* tables [[buffer(1)]],
    device const uint2* selectors [[buffer(2)]], device F* output [[buffer(3)]],
    constant uint* p [[buffer(4)]], constant ulong4& tail [[buffer(5)]],
    constant F& challenge [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    ulong count = ulong(p[1]) * p[3];
    if (index >= count) return;
    uint row = index / p[3], column = index % p[3];
    F low = F::zero(), high = F::zero();
    for (uint branch = 0; branch < p[2]; branch++) {
        ulong lo = 2ul * row * p[2] + branch, hi = lo + p[2];
        uint l = ra_fold_index(lo < p[0] ? rows[lo] : tail, selectors[column], p[4]);
        uint h = ra_fold_index(hi < p[0] ? rows[hi] : tail, selectors[column], p[4]);
        uint base = (column * p[2] + branch) * p[4];
        if (l < p[4]) low = low + tables[base + l];
        if (h < p[4]) high = high + tables[base + h];
    }
    output[index] = low + challenge * (high - low);
}
template <typename F>
kernel void ra_fold(
    device const F* previous [[buffer(0)]], device const F* tails [[buffer(1)]],
    device F* output [[buffer(2)]], constant uint* p [[buffer(3)]],
    constant F& challenge [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    ulong count = ulong(p[1]) * p[2];
    if (index >= count) return;
    uint row = index / p[2], column = index % p[2];
    ulong low_row = 2ul * row, high_row = low_row + 1;
    F low = low_row < p[0] ? previous[low_row * p[2] + column] : tails[column];
    F high = high_row < p[0] ? previous[high_row * p[2] + column] : tails[column];
    output[index] = low + challenge * (high - low);
}
