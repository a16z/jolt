extern "C" __global__ void register_merge_count(
    unsigned int *__restrict__ counts,
    const int *__restrict__ rs1_addr,
    const int *__restrict__ rs2_addr,
    const int *__restrict__ rd_addr,
    unsigned long n
) {
}

extern "C" __global__ void register_merge_scatter(
    unsigned int *__restrict__ cols_out,
    u64 *__restrict__ prev_out,
    u64 *__restrict__ next_out,
    unsigned char *__restrict__ rs1_flag_out,
    unsigned char *__restrict__ rs2_flag_out,
    unsigned char *__restrict__ rd_flag_out,
    const unsigned int *__restrict__ offsets,
    const int *__restrict__ rs1_addr,
    const u64 *__restrict__ rs1_val,
    const int *__restrict__ rs2_addr,
    const u64 *__restrict__ rs2_val,
    const int *__restrict__ rd_addr,
    const u64 *__restrict__ rd_pre,
    const u64 *__restrict__ rd_post,
    unsigned long n
) {
}
