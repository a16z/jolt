extern "C" __global__ void add_scalar(
    u64 *__restrict__ io,
    const u64 *__restrict__ scalar,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // STUB: body implemented after review. Intended: io[i] += *scalar (broadcast add).
    }
}
