extern "C" __global__ void core_booleanity_sparse_round1_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const u64 *__restrict__ present_mask,
    const unsigned char *__restrict__ values,
    const u64 *__restrict__ rho,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long num_polys,
    unsigned long chunk_domain,
    unsigned long poly_stride,
    unsigned long half,
    unsigned int in_bits
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        // STUB: real body gathers (h0,h1) from tables via the sparse row indices, applies
        // c_sum += h0*(h0-rho), q_sum += (h1-h0)^2 over polys, weighted by e_in*e_out.
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * 8;
            u64 s[4];
            fr_add(acc + 0, other + 0, s);
            for (int k = 0; k < 4; k++) acc[k] = s[k];
            fr_add(acc + 4, other + 4, s);
            for (int k = 0; k < 4; k++) acc[4 + k] = s[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 8; k++) out[blockIdx.x * 8 + k] = acc[k];
    }
}
