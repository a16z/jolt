// Parent-leader traversal from donor solinas_ram_val_sparse_prefix. The host
// supplies current LazyFoldedRa branches and SplitLt factors, not donor schedules.
struct RamValParams {
    uint records, address_prefix, width, radix, lo_length;
};
template <typename F>
kernel void ram_val_sparse_round(
    device const uint* cycles [[buffer(0)]], device const F* increments [[buffer(1)]],
    device const uint* addresses [[buffer(2)]], device const F* table [[buffer(3)]],
    device const F* weights [[buffer(4)]], device const F* lt_lo [[buffer(5)]],
    device const F* lt_hi [[buffer(6)]], device const F* eq_hi [[buffer(7)]],
    device F* partials [[buffer(8)]], constant RamValParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 256, "at most one product per participating tid");
    threadgroup Acc scratch[3][8];
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    if (tid < p.records) {
        uint block_width = 2 * p.width;
        uint parent = cycles[tid] / block_width;
        if (tid == 0 || cycles[tid - 1] / block_width != parent) {
            F inc[2] = {F::zero(), F::zero()};
            F ra[2] = {F::zero(), F::zero()};
            for (uint i = tid; i < p.records && cycles[i] / block_width == parent; i++) {
                uint offset = cycles[i] % block_width;
                inc[offset / p.width] = inc[offset / p.width] + increments[i] * weights[offset % p.width];
            }
            // RA includes zero-increment siblings. Omitting them would change
            // the mixed terms after a cycle bind and the terminal RA opening.
            for (uint side = 0; side < 2; side++) {
                for (uint branch = 0; branch < p.width; branch++) {
                    ulong row = ulong(parent) * block_width + side * p.width + branch;
                    if (row < p.address_prefix) {
                        uint address = addresses[row];
                        if (address != 0xffffffffu)
                            ra[side] = ra[side] + table[ulong(branch) * p.radix + address];
                    }
                }
            }
            uint index = 2 * parent;
            uint high = index / p.lo_length;
            uint low = index % p.lo_length;
            F lt[2] = {lt_hi[high] + eq_hi[high] * lt_lo[low],
                       lt_hi[high] + eq_hi[high] * lt_lo[low + 1]};
            F di = inc[1] - inc[0], dr = ra[1] - ra[0], dl = lt[1] - lt[0];
            sums[0].fmadd(inc[0] * ra[0], lt[0]);
            sums[1].fmadd((inc[1] + di) * (ra[1] + dr), lt[1] + dl);
            sums[2].fmadd((inc[1] + di + di) * (ra[1] + dr + dr), lt[1] + dl + dl);
        }
    }
    for (uint sample = 0; sample < 3; sample++) {
        auto sum = jolt::threadgroup_merge(sums[sample], scratch[sample], lane, sg, 8, 32);
        if (lane == 0 && sg == 0) partials[ulong(group) * 3 + sample] = sum.reduce();
    }
}
