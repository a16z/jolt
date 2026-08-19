typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed long long i64;
typedef __uint128_t u128;
typedef __int128_t i128;

#define KIND_UNMAPPED 0xFFFFu
#define NO_SEQUENCE 0xFFFFFFFFu
#define EXTRA_WORDS 10
#define X_RS1 0
#define X_RS2 1
#define X_RD_POST 2
#define X_RAM_READ 3
#define X_RAM_WRITE 4
#define X_IMM_LO 5
#define X_IMM_HI 6
#define X_KIND_BITS 7
#define X_REGISTERS 8
#define X_RD_PRE 9
#define VARIANTS 12
#define FLAG_BIT_JUMP 5
#define FLAG_BIT_BRANCH 18
#define FLAG_BIT_NOOP_ROW 21
#define FLAG_BIT_NEXT_IS_NOOP 22
#define FLAG_BIT_RAM_HAMMING 23
#define FLAG_BIT_SHOULD_BRANCH 24
#define FLAG_BIT_SHOULD_JUMP 25
#define FLAG_BIT_PRODUCT_NEGATIVE 26

__device__ __forceinline__ u64 rev8w(u64 v) {
  u32 lo = __byte_perm((u32)v, 0, 0x0123);
  u32 hi = __byte_perm((u32)(v >> 32), 0, 0x0123);
  return (u64)lo + ((u64)hi << 32);
}

__device__ __forceinline__ u64 tz64(u64 v) { return v == 0ull ? 64ull : (u64)__ffsll((long long)v) - 1ull; }

__device__ __forceinline__ void inputs_of(u32 mode, u64 rs1, u64 rs2, i128 imm, u64 address,
                                          u64 *x, i128 *y) {
  switch (mode) {
    case 0:  *x = 0ull;     *y = imm; break;
    case 1:  *x = rs1;      *y = imm; break;
    case 2:  *x = rs1;      *y = (i128)rs2; break;
    case 3:  *x = 0ull;     *y = 0; break;
    case 4:  *x = rs1;      *y = 0; break;
    case 5:  *x = 0ull;     *y = (i128)(u64)imm; break;
    case 6:  *x = address;  *y = (i128)(u64)imm; break;
    case 7:  *x = address;  *y = imm; break;
    case 8:  *x = rs1;      *y = (i128)rs2; break;
    case 9:  *x = rs1;      *y = (i128)(u64)imm; break;
    case 10: *x = rs1;      *y = imm; break;
    default: *x = rs1;      *y = 0; break;
  }
}

__device__ __forceinline__ void operands_of(u32 mode, u64 x, i128 y, u64 rs1, u64 rd_post,
                                            u64 *left, u128 *right) {
  switch (mode) {
    case 0: *left = 0ull; *right = (u128)rs1; break;
    case 1: *left = x;    *right = (u128)(u64)y; break;
    case 2:
    case 3:
    case 4: *left = 0ull; *right = (u128)((i128)x + y); break;
    case 5: *left = 0ull; *right = (u128)x * (u128)(u64)y; break;
    case 6: *left = 0ull; *right = (u128)x + (u128)(u64)y; break;
    case 7: *left = 0ull; *right = (u128)x + (((u128)1 << 64) - (u128)(u64)y); break;
    default: *left = 0ull; *right = (u128)rd_post; break;
  }
}

__device__ __forceinline__ u64 half_rot(u64 x, u64 yl, u32 k) {
  u64 v = (x ^ yl) & 0xFFFFFFFFull;
  return ((v >> k) | (v << (32u - k))) & 0xFFFFFFFFull;
}

__device__ __forceinline__ u64 full_rot(u64 x, u64 yl, u32 k) {
  u64 v = x ^ yl;
  return (v >> k) | (v << (64u - k));
}

__device__ __forceinline__ u64 output_of(u32 op, u64 x, i128 y, u128 index, u128 right,
                                         u64 rs1, u64 rd_post, i128 imm) {
  u64 yl = (u64)y;
  switch (op) {
    case 0: return 0ull;
    case 1: return (index % 2u == 0) ? 1ull : 0ull;
    case 2: return (index % 4u == 0) ? 1ull : 0ull;
    case 3: {
      u64 lower = (u64)right & 0xFFFFFFFFull;
      return ((lower >> 31) & 1ull) ? (lower | 0xFFFFFFFF00000000ull) : lower;
    }
    case 4: return (u64)right & 0xFFFFFFFFull;
    case 5: {
      int dividend = (int)(u32)x;
      int divisor = (int)(u32)yl;
      if (dividend == (-2147483647 - 1) && divisor == -1) return 1ull;
      return (u64)(i64)divisor;
    }
    case 6: {
      i64 sd = (i64)x, sv = (i64)yl;
      if (sd == (i64)0x8000000000000000ull && sv == -1) return 1ull;
      return (u64)sv;
    }
    case 7: return (x == 0ull) ? ((yl == 0xFFFFFFFFFFFFFFFFull) ? 1ull : 0ull) : 1ull;
    case 8: return x + yl;
    case 9: return (yl == 0ull || x < yl) ? 1ull : 0ull;
    case 10: {
      u64 half_mask = 0xFFFFFFFFull;
      u64 r = tz64(yl); if (r > 32ull) r = 32ull;
      u64 v = x & half_mask;
      if (r == 0ull || r == 32ull) return v;
      return ((v >> r) | (v << (32ull - r))) & half_mask;
    }
    case 11: {
      u64 r = tz64(yl) % 64ull;
      if (r == 0ull) return x;
      return (x >> r) | (x << (64ull - r));
    }
    case 12: return x >> (tz64(yl) & 63ull);
    case 13: return (u64)((i64)x >> (tz64(yl) & 63ull));
    case 14: return half_rot(x, yl, 12u);
    case 15: return half_rot(x, yl, 16u);
    case 16: return half_rot(x, yl, 7u);
    case 17: return half_rot(x, yl, 8u);
    case 18: return x >> (tz64(yl) & 63ull);
    case 19: return (u64)((i64)x >> (tz64(yl) & 63ull));
    case 20: return full_rot(x, yl, 16u);
    case 21: return full_rot(x, yl, 24u);
    case 22: return full_rot(x, yl, 32u);
    case 23: return full_rot(x, yl, 63u);
    case 24: return ((u128)x * (u128)yl) <= (u128)0xFFFFFFFFFFFFFFFFull ? 1ull : 0ull;
    case 25: return (x & 0x8000000000000000ull) ? 0xFFFFFFFFFFFFFFFFull : 0ull;
    case 26: return (u64)(((u128)x * (u128)yl) >> 64);
    case 27: return (x != yl) ? 1ull : 0ull;
    case 28: return (x < yl) ? 1ull : 0ull;
    case 29: return (x <= yl) ? 1ull : 0ull;
    case 30: return (x == yl) ? 1ull : 0ull;
    case 31: return (x >= yl) ? 1ull : 0ull;
    case 32: return x & ~yl;
    case 33: return x & yl;
    case 34: return x + yl;
    case 35: return (x + yl) & ~1ull;
    case 36: return x * yl;
    case 37: return x - yl;
    case 38: return ((i64)x < (i64)yl) ? 1ull : 0ull;
    case 39: return ((i64)x >= (i64)yl) ? 1ull : 0ull;
    case 40: return (u64)((i64)x * (i64)yl);
    case 41: return ((i64)x < (i64)yl) ? 1ull : 0ull;
    case 42: return x ^ yl;
    case 43: return x | yl;
    case 44: return (u64)(i64)(int)(u32)(x + yl);
    case 45: return (u64)(i64)(int)(u32)(x * yl);
    case 46: return (u64)(i64)(int)(u32)(x - yl);
    case 47: return (u64)imm;
    case 48: return rd_post;
    case 49: return 1ull << (u64)(index & 31u);
    case 50: return 1ull << (u64)(index & 63u);
    case 51: {
      u64 shift = (u64)(index & 63u);
      return (u64)((((u128)1 << (64ull - shift)) - 1) << shift);
    }
    default: return rev8w(rs1);
  }
}

__device__ __forceinline__ u128 spread_bits(u64 v) {
  u128 b = (u128)v;
  b = (b | (b << 32)) & (((u128)0x00000000FFFFFFFFull << 64) | (u128)0x00000000FFFFFFFFull);
  b = (b | (b << 16)) & (((u128)0x0000FFFF0000FFFFull << 64) | (u128)0x0000FFFF0000FFFFull);
  b = (b | (b << 8)) & (((u128)0x00FF00FF00FF00FFull << 64) | (u128)0x00FF00FF00FF00FFull);
  b = (b | (b << 4)) & (((u128)0x0F0F0F0F0F0F0F0Full << 64) | (u128)0x0F0F0F0F0F0F0F0Full);
  b = (b | (b << 2)) & (((u128)0x3333333333333333ull << 64) | (u128)0x3333333333333333ull);
  b = (b | (b << 1)) & (((u128)0x5555555555555555ull << 64) | (u128)0x5555555555555555ull);
  return b;
}

extern "C" __global__ void lookup_index_limbs_kernel(
    const u64 *extras,
    const u64 *address,
    const u8 *kind_input,
    const u8 *kind_operand,
    const u8 *kind_index,
    u32 kind_count,
    u64 *out,
    u32 *unmapped,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }
  const u64 *words = extras + (size_t)index * EXTRA_WORDS;
  u64 rs1 = words[X_RS1];
  u64 rs2 = words[X_RS2];
  u64 rd_post = words[X_RD_POST];
  i128 imm = (i128)(((u128)words[X_IMM_HI] << 64) | (u128)words[X_IMM_LO]);
  u32 kind = (u32)(words[X_KIND_BITS] & 0xFFFFull);
  if (kind == KIND_UNMAPPED || kind >= kind_count) {
    atomicExch(unmapped, 1u);
    return;
  }

  u64 x;
  i128 y;
  inputs_of(kind_input[kind], rs1, rs2, imm, address[index], &x, &y);
  u64 left;
  u128 right;
  operands_of(kind_operand[kind], x, y, rs1, rd_post, &left, &right);

  u128 value;
  switch (kind_index[kind]) {
    case 1: value = right; break;
    case 2: value = (u128)rs1; break;
    default: value = (spread_bits(left) << 1) | spread_bits((u64)right); break;
  }
  out[(size_t)index * 2] = (u64)value;
  out[(size_t)index * 2 + 1] = (u64)(value >> 64);
}

extern "C" __global__ void atom_columns_kernel(
    const u8 *is_noop,
    const u64 *address,
    const u64 *extras,
    const u32 *virtual_sequence,
    const u64 *ram_address,
    const u32 *pc_bucket_offsets,
    const u32 *pc_sequences,
    const u64 *pc_values,
    u32 pc_buckets,
    u64 ram_start,
    u64 bytecode_alignment,
    const u32 *kind_flags,
    const u32 *kind_table_index,
    const u8 *kind_input,
    const u8 *kind_operand,
    const u8 *kind_output,
    const u8 *kind_index,
    u32 kind_count,
    u32 *out_flags,
    u32 *out_table_index,
    u64 *out_bytecode_pc,
    u64 *out_rd_pre,
    u32 *out_rs1_address,
    u32 *out_rs2_address,
    u32 *out_rd_address,
    u64 *out_rd_inc,
    u64 *out_ram_inc,
    u64 *out_left_input,
    u64 *out_right_input,
    u64 *out_left_operand,
    u64 *out_right_operand,
    u64 *out_lookup_output,
    u64 *out_product,
    u32 *unmapped,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }

  const u64 *words = extras + (size_t)index * EXTRA_WORDS;
  u64 kind_bits = words[X_KIND_BITS];
  u32 kind = (u32)(kind_bits & 0xFFFFull);
  if (kind == KIND_UNMAPPED || kind >= kind_count) {
    atomicExch(unmapped, 1u);
    return;
  }

  u32 sequence = virtual_sequence[index];
  u32 sequence_class = (sequence == NO_SEQUENCE) ? 0u : (sequence == 0u ? 1u : 2u);
  u32 variant = sequence_class * 4u + (u32)((kind_bits >> 16) & 1ull) * 2u +
                (u32)((kind_bits >> 17) & 1ull);

  u32 mask = kind_flags[kind * VARIANTS + variant];
  bool row_is_noop = is_noop[index] != 0;
  if (row_is_noop) {
    mask |= 1u << FLAG_BIT_NOOP_ROW;
  }
  bool next_is_noop = (index + 1u >= cycles) || (is_noop[index + 1u] != 0);
  if (next_is_noop) {
    mask |= 1u << FLAG_BIT_NEXT_IS_NOOP;
  }
  u64 raw_ram = ram_address[index];
  if (raw_ram != 0xFFFFFFFFFFFFFFFFull && raw_ram != 0ull) {
    mask |= 1u << FLAG_BIT_RAM_HAMMING;
  }
  out_table_index[index] = kind_table_index[kind];

  u64 row_address = address[index];
  u64 mapped_pc = 0ull;
  if (row_address >= ram_start && (row_address % bytecode_alignment) == 0ull) {
    u64 bucket = (row_address - ram_start) / bytecode_alignment + 1ull;
    if (bucket < (u64)pc_buckets) {
      u32 want = (sequence == NO_SEQUENCE) ? 0u : sequence;
      u32 begin = pc_bucket_offsets[bucket];
      u32 end = pc_bucket_offsets[bucket + 1ull];
      for (u32 entry = begin; entry < end; ++entry) {
        if (pc_sequences[entry] == want) {
          mapped_pc = pc_values[entry];
          break;
        }
      }
    }
  }
  out_bytecode_pc[index] = row_is_noop ? 0ull : mapped_pc;

  u64 registers = words[X_REGISTERS];
  u64 rs1_slot = registers & 0xFFull;
  u64 rs2_slot = (registers >> 8) & 0xFFull;
  u64 rd_slot = (registers >> 16) & 0xFFull;
  out_rs1_address[index] = (rs1_slot == 0xFFull) ? 0xFFFFFFFFu : (u32)rs1_slot;
  out_rs2_address[index] = (rs2_slot == 0xFFull) ? 0xFFFFFFFFu : (u32)rs2_slot;
  out_rd_address[index] = (rd_slot == 0xFFull) ? 0xFFFFFFFFu : (u32)rd_slot;

  u64 rd_pre = words[X_RD_PRE];
  out_rd_pre[index] = rd_pre;

  i128 rd_inc = (rd_slot == 0xFFull)
                    ? (i128)0
                    : ((i128)(u128)words[X_RD_POST] - (i128)(u128)rd_pre);
  out_rd_inc[(size_t)index * 2] = (u64)(u128)rd_inc;
  out_rd_inc[(size_t)index * 2 + 1] = (u64)((u128)rd_inc >> 64);

  i128 ram_inc = (i128)(u128)words[X_RAM_WRITE] - (i128)(u128)words[X_RAM_READ];
  out_ram_inc[(size_t)index * 2] = (u64)(u128)ram_inc;
  out_ram_inc[(size_t)index * 2 + 1] = (u64)((u128)ram_inc >> 64);

  u64 rs1 = words[X_RS1];
  u64 rd_post = words[X_RD_POST];
  i128 imm = (i128)(((u128)words[X_IMM_HI] << 64) | (u128)words[X_IMM_LO]);

  u64 x;
  i128 y;
  inputs_of(kind_input[kind], rs1, words[X_RS2], imm, row_address, &x, &y);
  u64 left;
  u128 right;
  operands_of(kind_operand[kind], x, y, rs1, rd_post, &left, &right);

  u128 lookup_index;
  switch (kind_index[kind]) {
    case 1: lookup_index = right; break;
    case 2: lookup_index = (u128)rs1; break;
    default: lookup_index = (spread_bits(left) << 1) | spread_bits((u64)right); break;
  }
  u64 lookup_output =
      output_of(kind_output[kind], x, y, lookup_index, right, rs1, rd_post, imm);

  out_left_input[index] = x;
  out_right_input[(size_t)index * 2] = (u64)(u128)y;
  out_right_input[(size_t)index * 2 + 1] = (u64)((u128)y >> 64);
  out_left_operand[index] = left;
  out_right_operand[(size_t)index * 2] = (u64)right;
  out_right_operand[(size_t)index * 2 + 1] = (u64)(right >> 64);
  out_lookup_output[index] = lookup_output;

  bool right_is_negative = y < 0;
  u128 magnitude = right_is_negative ? (~(u128)y + (u128)1) : (u128)y;
  u128 product = (u128)x * magnitude;
  out_product[(size_t)index * 2] = (u64)product;
  out_product[(size_t)index * 2 + 1] = (u64)(product >> 64);
  if (right_is_negative && product != (u128)0) {
    mask |= 1u << FLAG_BIT_PRODUCT_NEGATIVE;
  }
  if (((mask >> FLAG_BIT_BRANCH) & 1u) != 0u && lookup_output == 1ull) {
    mask |= 1u << FLAG_BIT_SHOULD_BRANCH;
  }
  bool successor_is_noop = (index + 1u < cycles) && (is_noop[index + 1u] != 0);
  if (((mask >> FLAG_BIT_JUMP) & 1u) != 0u && !successor_is_noop) {
    mask |= 1u << FLAG_BIT_SHOULD_JUMP;
  }
  out_flags[index] = mask;
}

extern "C" __global__ void flag_bit_column_kernel(
    const u32 *flags,
    u32 bit,
    u64 *out,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }
  out[index] = (u64)((flags[index] >> bit) & 1u);
}

extern "C" __global__ void hot_chunk_limbs_kernel(
    const u64 *limbs,
    u32 shift,
    u64 mask,
    u32 *out,
    u64 *spans,
    u32 slot,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }
  u128 value = (u128)limbs[(size_t)index * 2] | ((u128)limbs[(size_t)index * 2 + 1] << 64);
  u64 chunk = (u64)((value >> shift) & (u128)mask);
  out[index] = (u32)chunk;
  atomicMax(spans + slot, chunk + 1ull);
}

extern "C" __global__ void hot_chunk_words_kernel(
    const u32 *words,
    u32 shift,
    u64 mask,
    u32 *out,
    u64 *spans,
    u32 slot,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }
  u32 word = words[index];
  if (word == COLD) {
    out[index] = COLD;
    return;
  }
  u64 chunk = ((u64)word >> shift) & mask;
  out[index] = (u32)chunk;
  atomicMax(spans + slot, chunk + 1ull);
}

#define NARROW_REJECTED 0
#define NARROW_SPAN 1
#define NARROW_FIRST 2

extern "C" __global__ void narrow_u64_kernel(
    const u64 *source,
    u64 bound,
    u32 *out,
    u64 *facts,
    u32 cycles) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cycles) {
    return;
  }
  u64 value = source[index];
  if (index == 0u) {
    facts[NARROW_FIRST] = value;
  }
  if (value >= bound) {
    atomicMin(facts + NARROW_REJECTED, value);
    return;
  }
  out[index] = (u32)value;
  atomicMax(facts + NARROW_SPAN, value + 1ull);
}
