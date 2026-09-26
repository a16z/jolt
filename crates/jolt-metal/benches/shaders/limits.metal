// Machine-limit benchmarks that no field kernel measures: memory bandwidth,
// for copies and for reads, and threadgroup memory bandwidth. All move 16 B
// words, the size of an Fp128 element.

// out[i] = in[i]: 16 B read and 16 B written per thread.
kernel void jolt_bench_copy(device const uint4* in [[buffer(0)]],
                            device uint4* out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = in[i];
}

// Words each thread of jolt_bench_read sums.
constant constexpr uint READ_WORDS = 4;

// Streaming read: in a grid of T threads, thread i sums in[i + k T] for
// k < READ_WORDS, with wrapping lane-wise addition, and writes the sum. A
// simdgroup's loads are consecutive, and there is one write per READ_WORDS
// reads. Four words per thread keeps enough threads in flight at in-cache
// sizes: with 16, reading 4 MiB ran at 1.3 TB/s instead of 1.5 on an M4 Max.
kernel void jolt_bench_read(device const uint4* in [[buffer(0)]],
                            device uint4* out [[buffer(1)]],
                            uint i [[thread_position_in_grid]],
                            uint threads [[threads_per_grid]]) {
    uint4 acc = uint4(0u);
    for (uint k = 0; k < READ_WORDS; k++) {
        acc += in[i + k * threads];
    }
    out[i] = acc;
}

// Words in the threadgroup tile, and loads per thread.
constant constexpr uint TILE_WORDS = 256;
constant constexpr uint TILE_ROUNDS = 1024;

// 16 B loads from threadgroup memory, in threadgroups of exactly TILE_WORDS
// threads. Each threadgroup copies in[0..TILE_WORDS) to its tile, then
// thread t sums tile[(t + 32 r) mod TILE_WORDS] over TILE_ROUNDS rounds r,
// with wrapping lane-wise addition: each round a simdgroup reads 32
// consecutive words, and consecutive rounds of a thread read different
// words, so no load is loop-invariant. (XOR would cancel: the index repeats
// every 8 rounds, an even number of times.)
kernel void jolt_bench_threadgroup_load(device const uint4* in [[buffer(0)]],
                                        device uint4* out [[buffer(1)]],
                                        uint i [[thread_position_in_grid]],
                                        ushort t [[thread_index_in_threadgroup]]) {
    threadgroup uint4 tile[TILE_WORDS];
    tile[t] = in[t];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint4 acc = uint4(0u);
    for (uint r = 0; r < TILE_ROUNDS; r++) {
        acc += tile[(t + 32 * r) % TILE_WORDS];
    }
    out[i] = acc;
}
