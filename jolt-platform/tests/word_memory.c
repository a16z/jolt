/* Unit tests for the actual RV64 assembly, executed under qemu-riscv64.
 * Compile with builtins and loop-to-libcall conversion disabled. The oracle
 * reads volatile bytes, independent of either memory routine under test.
 */
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void *(*volatile copy_fn)(void *, const void *, size_t) = memcpy;
static int (*volatile compare_fn)(const void *, const void *, size_t) = memcmp;
extern int word_memory_section_probe(void);

static int byte_compare(const volatile unsigned char *a,
                        const volatile unsigned char *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        unsigned char x = a[i], y = b[i];
        if (x != y) return (int)x - (int)y;
    }
    return 0;
}

static void fill(volatile unsigned char *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = (unsigned char)((i * 131u + seed) ^ (i >> 3));
}

static void all_copy_alignments(void) {
    _Alignas(64) unsigned char source[640], dest[640];
    fill(source, sizeof source, 19);
    for (size_t s = 0; s < 8; ++s) {
        for (size_t d = 0; d < 8; ++d) {
            for (size_t n = 0; n <= 257; ++n) {
                for (size_t i = 0; i < sizeof dest; ++i) dest[i] = 0xa5;
                assert(copy_fn(dest + 32 + d, source + 32 + s, n) == dest + 32 + d);
                for (size_t i = 0; i < sizeof dest; ++i) {
                    unsigned char expected = 0xa5;
                    if (i >= 32 + d && i - (32 + d) < n)
                        expected = source[32 + s + i - (32 + d)];
                    assert(((volatile unsigned char *)dest)[i] == expected);
                }
            }
        }
    }
    puts("PASS copy_all_alignments_lengths_and_canaries");
}

static void all_compare_alignments(void) {
    _Alignas(64) unsigned char a[640], b[640];
    for (size_t x = 0; x < 8; ++x) {
        for (size_t y = 0; y < 8; ++y) {
            for (size_t n = 0; n <= 257; ++n) {
                fill(a + x, n, 31);
                fill(b + y, n, 31);
                assert(compare_fn(a + x, b + y, n) == 0);
                for (size_t i = 0; i < n; ++i) {
                    if (n > 65 && i != 0 && i != n / 2 && i + 1 != n) continue;
                    b[y + i] ^= 0x80;
                    assert(compare_fn(a + x, b + y, n) == byte_compare(a + x, b + y, n));
                    assert(compare_fn(b + y, a + x, n) == byte_compare(b + y, a + x, n));
                    b[y + i] ^= 0x80;
                }
            }
        }
    }
    puts("PASS compare_all_alignments_first_difference_and_unsigned_order");
}

static unsigned char *guarded(size_t page) {
    void *p = mmap(NULL, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    unsigned char *middle = (unsigned char *)p + page;
    assert(mprotect(middle, page, PROT_READ | PROT_WRITE) == 0);
    return middle;
}

static void guard_pages_and_long_copies(void) {
    long page_result = sysconf(_SC_PAGESIZE);
    assert(page_result >= 4096);
    size_t page = (size_t)page_result;
    unsigned char *a = guarded(page), *b = guarded(page);
    for (size_t n = 0; n <= page - 16; n += (n < 160 ? 1 : 97)) {
        for (size_t offset = 0; offset < 8; ++offset) {
            fill(a, page, 83);
            fill(b, page, 17);
            unsigned char *source = a + page - n;
            unsigned char *dest = b + offset;
            assert(copy_fn(dest, source, n) == dest);
            assert(byte_compare(dest, source, n) == 0);
            assert(compare_fn(dest, source, n) == 0);
            source = a + offset;
            dest = b + page - n;
            assert(copy_fn(dest, source, n) == dest);
            assert(byte_compare(dest, source, n) == 0);
            assert(compare_fn(dest, source, n) == 0);
        }
    }
    assert(copy_fn(b, a, page) == b);
    assert(byte_compare(a, b, page) == 0);
    assert(compare_fn(a, b, page) == 0);
    assert(compare_fn(a, a, page) == 0);
    /* Zero-length calls must not dereference either inaccessible page. */
    assert(copy_fn(b - page, a - page, 0) == b - page);
    assert(compare_fn(a - page, b - page, 0) == 0);
    assert(munmap(a - page, page * 3) == 0);
    assert(munmap(b - page, page * 3) == 0);
    puts("PASS guard_pages_both_ends_long_copies_equal_alias_and_zero_length");
}

static void uninitialized_object_copy(void) {
    unsigned char *a = malloc(4096), *b = malloc(4096);
    assert(a != NULL && b != NULL);
    /* Inspecting these representations as integers in Rust would be invalid.
     * The assembly copies them without any language-level integer read. */
    for (size_t x = 0; x < 8; ++x)
        for (size_t y = 0; y < 8; ++y)
            assert(copy_fn(b + y, a + x, 2048) == b + y);
    free(a);
    free(b);
    puts("PASS uninitialized_object_representations");
}

int main(void) {
    assert(word_memory_section_probe() == 291);
    puts("PASS assembler_section_restored_for_subsequent_code");
    all_copy_alignments();
    all_compare_alignments();
    guard_pages_and_long_copies();
    uninitialized_object_copy();
    puts("WORD_MEMORY_UNIT_TESTS_PASS");
    return 0;
}
