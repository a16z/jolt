extern "C" __global__ void scatter_add_eq(
    u64 *__restrict__ worker_banks,
    const u64 *__restrict__ eq,
    const int *__restrict__ addr,
    unsigned long trace_len,
    unsigned long k,
    unsigned long num_workers
) {
    unsigned long worker = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= num_workers) return;

    for (unsigned long c = worker; c < trace_len; c += num_workers) {
        int a = addr[c];
        if (a < 0 || (unsigned long)a >= k) continue;
        u64 w[4];
        load4(eq + c * 4, w);
        u64 *slot = worker_banks + ((unsigned long)a * num_workers + worker) * 4UL;
        raf_add_inplace(slot, w);
    }
}
