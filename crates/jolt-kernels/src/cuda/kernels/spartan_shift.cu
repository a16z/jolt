extern "C" __global__ void ss_packed_columns_kernel(
    const unsigned int *__restrict__ pc_words,
    const unsigned int *__restrict__ flags, u64 *__restrict__ packed,
    unsigned int virtual_bit, unsigned int first_bit, unsigned int noop_bit,
    unsigned int flag_base, unsigned long long *__restrict__ unmapped,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int word = pc_words[t];
    if (word == 0xFFFFFFFFu) {
        atomicMin(unmapped, (unsigned long long)t);
        return;
    }

    unsigned int mask = flags[t];
    u64 out = (u64)word;
    out |= (u64)((mask >> virtual_bit) & 1u) << flag_base;
    out |= (u64)((mask >> first_bit) & 1u) << (flag_base + 1u);
    out |= (u64)((mask >> noop_bit) & 1u) << (flag_base + 2u);
    packed[t] = out;
}
