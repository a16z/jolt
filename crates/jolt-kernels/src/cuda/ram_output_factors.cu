extern "C" __global__ void ram_output_factors(
    u64 *__restrict__ io_mask_out,
    u64 *__restrict__ diff_out,
    const u64 *__restrict__ final_ram,
    unsigned long io_start,
    unsigned long io_end,
    unsigned long k
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    u64 io_mask[4] = {0, 0, 0, 0};
    u64 diff[4] = {0, 0, 0, 0};

    if (i >= io_start && i < io_end) {
        u64 one[4] = {1, 0, 0, 0};
        raf_to_mont(one, io_mask);
    } else if (final_ram[i] != 0) {
        u64 raw[4] = {final_ram[i], 0, 0, 0};
        raf_to_mont(raw, diff);
    }

    store4(io_mask_out + i * 4, io_mask);
    store4(diff_out + i * 4, diff);
}
