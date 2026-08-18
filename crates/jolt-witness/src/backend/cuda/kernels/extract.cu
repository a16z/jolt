#define COLD 0xFFFFFFFFu
#define NO_SEQUENCE 0xFFFFFFFFu
#define RAM_NO_ACCESS 0xFFFFFFFFFFFFFFFFull
#define U32_MAX 0xFFFFFFFFull

extern "C" __global__ void mapped_pc_words_kernel(
    const unsigned char *is_noop,
    const unsigned long long *address,
    const unsigned int *virtual_sequence,
    const unsigned int *bucket_offsets,
    const unsigned int *sequences,
    const unsigned long long *values,
    unsigned int buckets,
    unsigned long long ram_start,
    unsigned long long alignment,
    unsigned int *out,
    unsigned long long *rejected,
    unsigned int cycles) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }

  unsigned int word = COLD;
  if (is_noop[index] != 0) {
    word = 0u;
  } else {
    unsigned long long slot_address = address[index];
    if (slot_address >= ram_start && (slot_address % alignment) == 0ull) {
      unsigned long long bucket = (slot_address - ram_start) / alignment + 1ull;
      if (bucket < (unsigned long long)buckets) {
        unsigned int sequence = virtual_sequence[index];
        if (sequence == NO_SEQUENCE) {
          sequence = 0u;
        }
        unsigned int begin = bucket_offsets[bucket];
        unsigned int end = bucket_offsets[bucket + 1ull];
        for (unsigned int entry = begin; entry < end; ++entry) {
          if (sequences[entry] == sequence) {
            unsigned long long slot = values[entry];
            if (slot >= U32_MAX) {
              atomicMin(rejected, slot);
            } else {
              word = (unsigned int)slot;
            }
            break;
          }
        }
      }
    }
  }
  out[index] = word;
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
