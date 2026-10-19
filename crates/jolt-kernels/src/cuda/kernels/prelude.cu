typedef unsigned long long u64;

#define LIMBS 4
#define BLOCK 256

__device__ __forceinline__ void load4(const u64 *__restrict__ p, u64 *r) {
    ulonglong4 v = *reinterpret_cast<const ulonglong4 *>(p);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
}

__device__ __forceinline__ void store4(u64 *p, const u64 *r) {
    ulonglong4 v;
    v.x = r[0]; v.y = r[1]; v.z = r[2]; v.w = r[3];
    *reinterpret_cast<ulonglong4 *>(p) = v;
}
