extern "C" __global__ void scan_u32_block(
    unsigned int *__restrict__ out,
    unsigned int *__restrict__ block_sums,
    const unsigned int *__restrict__ in,
    unsigned long n
) {
    extern __shared__ unsigned int scan_sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + tid;

    unsigned int self = (i < n) ? in[i] : 0u;
    scan_sdata[tid] = self;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int v = (tid >= offset) ? scan_sdata[tid - offset] : 0u;
        __syncthreads();
        scan_sdata[tid] += v;
        __syncthreads();
    }

    if (i < n) {
        out[i] = scan_sdata[tid] - self;
    }
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = scan_sdata[tid];
    }
}

extern "C" __global__ void scan_u32_add_offsets(
    unsigned int *__restrict__ out,
    const unsigned int *__restrict__ block_offsets,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += block_offsets[blockIdx.x];
    }
}
