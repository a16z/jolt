#define SS_NARROW 2
#define SS_UNEXPANDED_PC_SLOT 0
#define SS_PC_SLOT 1

extern "C" __global__ void ss_columns_kernel(
    const u64 *__restrict__ narrow, const unsigned int *__restrict__ flags,
    u64 *__restrict__ unexpanded_pc, u64 *__restrict__ pc,
    u64 *__restrict__ is_virtual, u64 *__restrict__ is_first_in_sequence,
    u64 *__restrict__ is_noop, unsigned int virtual_bit, unsigned int first_bit,
    unsigned int noop_bit, unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    const u64 *row = narrow + (unsigned long long)t * SS_NARROW;
    unsigned long long offset = (unsigned long long)t * LIMBS;

    u64 raw[LIMBS] = {row[SS_UNEXPANDED_PC_SLOT], 0, 0, 0};
    u64 out[LIMBS];
    fr_to_mont(raw, out);
    store4(unexpanded_pc + offset, out);

    raw[0] = row[SS_PC_SLOT];
    fr_to_mont(raw, out);
    store4(pc + offset, out);

    unsigned int mask = flags[t];
    u64 one[LIMBS];
    load4(FR_ONE, one);
    u64 zero[LIMBS] = {0, 0, 0, 0};

    store4(is_virtual + offset, ((mask >> virtual_bit) & 1u) ? one : zero);
    store4(is_first_in_sequence + offset, ((mask >> first_bit) & 1u) ? one : zero);
    store4(is_noop + offset, ((mask >> noop_bit) & 1u) ? one : zero);
}

extern "C" __global__ void ss_columns_device_kernel(
    const u64 *__restrict__ address, const unsigned int *__restrict__ pc_words,
    const unsigned int *__restrict__ flags, u64 *__restrict__ unexpanded_pc,
    u64 *__restrict__ pc, u64 *__restrict__ is_virtual,
    u64 *__restrict__ is_first_in_sequence, u64 *__restrict__ is_noop,
    unsigned int virtual_bit, unsigned int first_bit, unsigned int noop_bit,
    unsigned long long *__restrict__ unmapped, unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int word = pc_words[t];
    if (word == 0xFFFFFFFFu) {
        atomicMin(unmapped, (unsigned long long)t);
        return;
    }

    unsigned long long offset = (unsigned long long)t * LIMBS;
    u64 raw[LIMBS] = {address[t], 0, 0, 0};
    u64 out[LIMBS];
    fr_to_mont(raw, out);
    store4(unexpanded_pc + offset, out);

    raw[0] = (u64)word;
    fr_to_mont(raw, out);
    store4(pc + offset, out);

    unsigned int mask = flags[t];
    u64 one[LIMBS];
    load4(FR_ONE, one);
    u64 zero[LIMBS] = {0, 0, 0, 0};

    store4(is_virtual + offset, ((mask >> virtual_bit) & 1u) ? one : zero);
    store4(is_first_in_sequence + offset, ((mask >> first_bit) & 1u) ? one : zero);
    store4(is_noop + offset, ((mask >> noop_bit) & 1u) ? one : zero);
}
