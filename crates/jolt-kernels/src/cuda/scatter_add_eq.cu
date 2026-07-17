extern "C" __global__ void scatter_add_eq(
    u64 *__restrict__ worker_banks,
    const u64 *__restrict__ eq,
    const int *__restrict__ addr,
    unsigned long trace_len,
    unsigned long k,
    unsigned long num_workers
) {
}
