extern "C" __global__ void sum_reduce(u64 *__restrict__ out, const u64 *__restrict__ in, unsigned long n) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        load4(in + i * 4, acc);
    } else {
        for (int k = 0; k < 4; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_add(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store4(out + blockIdx.x * 4, acc);
    }
}

extern "C" __global__ void product_reduce(u64 *__restrict__ out, const u64 *__restrict__ in, unsigned long n, const u64 *__restrict__ one) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        load4(in + i * 4, acc);
    } else {
        for (int k = 0; k < 4; k++) acc[k] = one[k];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_mul(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store4(out + blockIdx.x * 4, acc);
    }
}
