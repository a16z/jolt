extern "C" __global__ void schedule_round_count(
    unsigned int *__restrict__ counts,
    const unsigned int *__restrict__ cur_rows,
    const unsigned int *__restrict__ cur_cols,
    unsigned long len
) {
}

extern "C" __global__ void schedule_round_emit(
    int *__restrict__ even_idx,
    int *__restrict__ odd_idx,
    unsigned int *__restrict__ pair_out,
    unsigned int *__restrict__ next_rows,
    unsigned int *__restrict__ next_cols,
    const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ cur_rows,
    const unsigned int *__restrict__ cur_cols,
    unsigned long len
) {
}
