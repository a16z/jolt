struct RaMaterializeParams {
    uint source_rows;
    uint output_rows;
    uint width;
    uint columns;
    uint radix;
};
struct RaFoldParams {
    uint source_rows;
    uint output_rows;
    uint columns;
};
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
    constant RaMaterializeParams& p [[buffer(4)]], constant ulong4& tail [[buffer(5)]],
    constant F& challenge [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    ulong count = ulong(p.output_rows) * p.columns;
    if (index >= count) return;
    uint row = index / p.columns, column = index % p.columns;
    F low = F::zero(), high = F::zero();
    for (uint branch = 0; branch < p.width; branch++) {
        ulong lo = 2ul * row * p.width + branch, hi = lo + p.width;
        uint l = ra_fold_index(lo < p.source_rows ? rows[lo] : tail, selectors[column], p.radix);
        uint h = ra_fold_index(hi < p.source_rows ? rows[hi] : tail, selectors[column], p.radix);
        uint base = (column * p.width + branch) * p.radix;
        if (l < p.radix) low = low + tables[base + l];
        if (h < p.radix) high = high + tables[base + h];
    }
    output[index] = low + challenge * (high - low);
}
template <typename F>
kernel void ra_fold(
    device const F* previous [[buffer(0)]], device const F* tails [[buffer(1)]],
    device F* output [[buffer(2)]], constant RaFoldParams& p [[buffer(3)]],
    constant F& challenge [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    ulong count = ulong(p.output_rows) * p.columns;
    if (index >= count) return;
    uint row = index / p.columns, column = index % p.columns;
    ulong low_row = 2ul * row, high_row = low_row + 1;
    F low = low_row < p.source_rows ? previous[low_row * p.columns + column] : tails[column];
    F high = high_row < p.source_rows ? previous[high_row * p.columns + column] : tails[column];
    output[index] = low + challenge * (high - low);
}
