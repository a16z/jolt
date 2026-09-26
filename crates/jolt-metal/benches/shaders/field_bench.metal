// Benchmark kernels, generic over the field type. Elementwise add, mul, and
// square reuse tests/shaders/field_ops.metal.
//
// Every kernel writes a value that depends on all of its work, and the
// benchmark checks it against the CPU, so the compiler cannot drop work.

// `rounds` dependent multiplies per thread: x <- x * y. Throughput is bounded
// by how many threads the core keeps resident to hide the dependency.
template <typename F>
kernel void jolt_bench_field_mul_chain(device const F* a [[buffer(0)]],
                                       device const F* b [[buffer(1)]],
                                       constant uint& rounds [[buffer(2)]],
                                       device F* out [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    F x = a[i];
    F y = b[i];
    for (uint r = 0; r < rounds; r++) {
        x = x * y;
    }
    out[i] = x;
}

// Four independent chains of `rounds` multiplies per thread, then their sum:
// the same work as mul_chain with four-way instruction-level parallelism and
// four times the live registers.
template <typename F>
kernel void jolt_bench_field_mul_chain4(device const F* a [[buffer(0)]],
                                        device const F* b [[buffer(1)]],
                                        constant uint& rounds [[buffer(2)]],
                                        device F* out [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    F y = b[i];
    F x0 = a[i];
    F x1 = x0 + y;
    F x2 = x1 + y;
    F x3 = x2 + y;
    for (uint r = 0; r < rounds; r++) {
        x0 = x0 * y;
        x1 = x1 * y;
        x2 = x2 * y;
        x3 = x3 * y;
    }
    out[i] = (x0 + x1) + (x2 + x3);
}

// `rounds` dependent squarings per thread.
template <typename F>
kernel void jolt_bench_field_square_chain(device const F* a [[buffer(0)]],
                                          constant uint& rounds [[buffer(1)]],
                                          device F* out [[buffer(2)]],
                                          uint i [[thread_position_in_grid]]) {
    F x = a[i];
    for (uint r = 0; r < rounds; r++) {
        x = square(x);
    }
    out[i] = x;
}

// `rounds` dependent additions per thread.
template <typename F>
kernel void jolt_bench_field_add_chain(device const F* a [[buffer(0)]],
                                       device const F* b [[buffer(1)]],
                                       constant uint& rounds [[buffer(2)]],
                                       device F* out [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    F x = a[i];
    F y = b[i];
    for (uint r = 0; r < rounds; r++) {
        x = x + y;
    }
    out[i] = x;
}

// Threads per threadgroup of jolt_bench_field_inner_product.
constant constexpr uint INNER_PRODUCT_GROUP = 256;

// Partial inner products of a and b, one per threadgroup: the shape of a
// sumcheck round. Thread t of a grid of T threads sums a[j] * b[j] over
// j = t, t + T, ... < n, and each threadgroup of INNER_PRODUCT_GROUP threads
// sums its threads' values with a tree in threadgroup memory.
template <typename F>
kernel void jolt_bench_field_inner_product(device const F* a [[buffer(0)]],
                                           device const F* b [[buffer(1)]],
                                           constant uint& n [[buffer(2)]],
                                           device F* out [[buffer(3)]],
                                           uint i [[thread_position_in_grid]],
                                           uint threads [[threads_per_grid]],
                                           uint lane [[thread_position_in_threadgroup]],
                                           uint group [[threadgroup_position_in_grid]]) {
    threadgroup F partial[INNER_PRODUCT_GROUP];
    F sum = F::zero();
    for (uint j = i; j < n; j += threads) {
        sum = sum + a[j] * b[j];
    }
    partial[lane] = sum;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint stride = INNER_PRODUCT_GROUP / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        out[group] = partial[0];
    }
}
