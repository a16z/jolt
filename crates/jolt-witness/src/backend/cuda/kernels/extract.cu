#define COLD 0xFFFFFFFFu
#define NO_SEQUENCE 0xFFFFFFFFu
#define RAM_NO_ACCESS 0xFFFFFFFFFFFFFFFFull
#define U32_MAX 0xFFFFFFFFull

extern "C" __global__ void mapped_pc_words_kernel(
    const unsigned long long *extras,
    unsigned int *out,
    unsigned long long *rejected,
    unsigned int cycles) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) return;
  unsigned long long pc = extras[(size_t)index * 11u + 10u];
  if (pc >= U32_MAX) atomicMin(rejected, pc);
  out[index] = (unsigned int)pc;
}

extern "C" __global__ void remapped_ram_words_kernel(
    const unsigned long long *ram_address,
    unsigned long long lowest_address,
    unsigned long long addresses,
    unsigned int *out,
    unsigned long long *rejected,
    unsigned long long *span,
    unsigned int cycles) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }

  unsigned int word = COLD;
  unsigned long long raw = ram_address[index];
  if (raw != RAM_NO_ACCESS && raw != 0ull && raw >= lowest_address) {
    unsigned long long remapped = (raw - lowest_address) / 8ull;
    if (remapped >= U32_MAX || remapped >= addresses) {
      atomicMin(rejected, remapped);
    } else {
      word = (unsigned int)remapped;
      atomicMax(span, remapped + 1ull);
    }
  }
  out[index] = word;
}
