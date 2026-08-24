#define CI_EXTRA_WORDS 10
#define CI_EXTRA_RD_POST 2
#define CI_EXTRA_RAM_READ 3
#define CI_EXTRA_RAM_WRITE 4
#define CI_EXTRA_REGISTERS 8
#define CI_EXTRA_RD_PRE 9
#define CI_REGISTER_ABSENT 255
#define CI_RD_SLOT_SHIFT 16
#define CI_KIND_RD 0
#define CI_KIND_RAM 1

extern "C" __global__ void commit_increment_column_kernel(
    const u64 *__restrict__ extras, unsigned int kind,
    u64 *__restrict__ magnitudes, unsigned char *__restrict__ signs,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    const u64 *words = extras + (unsigned long long)t * CI_EXTRA_WORDS;
    u64 post;
    u64 pre;
    if (kind == CI_KIND_RD) {
        u64 slot = (words[CI_EXTRA_REGISTERS] >> CI_RD_SLOT_SHIFT) & 0xFFull;
        bool absent = slot == (u64)CI_REGISTER_ABSENT;
        post = absent ? 0ull : words[CI_EXTRA_RD_POST];
        pre = absent ? 0ull : words[CI_EXTRA_RD_PRE];
    } else {
        post = words[CI_EXTRA_RAM_WRITE];
        pre = words[CI_EXTRA_RAM_READ];
    }

    bool negative = post < pre;
    magnitudes[t] = negative ? pre - post : post - pre;
    signs[t] = negative ? 1 : 0;
}
