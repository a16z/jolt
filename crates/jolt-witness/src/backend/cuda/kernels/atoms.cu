typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed long long i64;
typedef __uint128_t u128;
typedef __int128_t i128;

#define KIND_UNMAPPED 0xFFFFu
#define NO_SEQUENCE 0xFFFFFFFFu
#define EXTRA_WORDS 11
#define X_BYTECODE_PC 10
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
