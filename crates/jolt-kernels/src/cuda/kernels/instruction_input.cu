extern "C" __global__ void ii_flag_words_kernel(
    const unsigned int *__restrict__ canonical, u64 *__restrict__ words,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;
    words[t] = (u64)canonical[t];
}
