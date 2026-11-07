extern "C" __global__ void scan_u32_block_kernel(const unsigned int *__restrict__ in,
                                                 unsigned int *__restrict__ out,
                                                 unsigned int *__restrict__ block_sums,
                                                 unsigned int n) {
    __shared__ unsigned int scratch[BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    scratch[tid] = (i < n) ? in[i] : 0u;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = (tid >= stride) ? scratch[tid - stride] : 0u;
        __syncthreads();
        scratch[tid] += addend;
        __syncthreads();
    }

    unsigned int inclusive = scratch[tid];
    unsigned int self = (i < n) ? in[i] : 0u;
    if (i < n) out[i] = inclusive - self;
    if (tid == blockDim.x - 1) block_sums[blockIdx.x] = inclusive;
}

extern "C" __global__ void scan_u32_add_offsets_kernel(unsigned int *__restrict__ out,
                                                      const unsigned int *__restrict__ block_offsets,
                                                      unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += block_offsets[blockIdx.x];
}
