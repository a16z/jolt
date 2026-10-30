extern "C" __global__ void pfx_mle_round_kernel(
    const u64 *__restrict__ tables, unsigned int half, unsigned int prefixes,
    u64 *__restrict__ out) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prefix = blockIdx.y;
    if (b >= half || prefix >= prefixes) return;
    unsigned long long row = (unsigned long long)prefix * 2u * half;
    u64 lo[LIMBS], hi[LIMBS], doubled[LIMBS], at_two[LIMBS];
    load4(tables + (row + b) * LIMBS, lo);
    load4(tables + (row + half + b) * LIMBS, hi);
    fr_add(hi, hi, doubled);
    fr_sub(doubled, lo, at_two);
    store4(out + ((unsigned long long)prefix * half + b) * LIMBS, lo);
    store4(out + ((unsigned long long)(prefixes + prefix) * half + b) * LIMBS, at_two);
}
