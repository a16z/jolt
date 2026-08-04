__device__ __forceinline__ void lt_split_get(const u64 *__restrict__ lt_lo,
                                             const u64 *__restrict__ lt_hi,
                                             const u64 *__restrict__ eq_hi,
                                             unsigned int lo_bits,
                                             unsigned int lo_mask,
                                             unsigned int idx,
                                             u64 *out) {
    unsigned int i_hi = idx >> lo_bits;
    unsigned int i_lo = idx & lo_mask;
    u64 h[LIMBS], e[LIMBS], l[LIMBS], product[LIMBS];
    load4(lt_hi + (unsigned long long)i_hi * LIMBS, h);
    load4(eq_hi + (unsigned long long)i_hi * LIMBS, e);
    load4(lt_lo + (unsigned long long)i_lo * LIMBS, l);
    fr_mul(e, l, product);
    fr_add(h, product, out);
}

extern "C" __global__ void lt_reconstruct_kernel(const u64 *__restrict__ lt_lo,
                                                 const u64 *__restrict__ lt_hi,
                                                 const u64 *__restrict__ eq_hi,
                                                 unsigned int lo_bits,
                                                 unsigned int lo_mask,
                                                 u64 *__restrict__ out,
                                                 unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    u64 value[LIMBS];
    lt_split_get(lt_lo, lt_hi, eq_hi, lo_bits, lo_mask, j, value);
    store4(out + (unsigned long long)j * LIMBS, value);
}
