#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "CENSUS_FAILURE %s\n", message);
        std::exit(1);
    }
}

struct Mapping {
    void *data;
    size_t size;
    explicit Mapping(const char *path) {
        const int fd = open(path, O_RDONLY);
        require(fd >= 0, "open captured source");
        struct stat info{};
        require(fstat(fd, &info) == 0 && info.st_size > 0, "capture file size");
        size = size_t(info.st_size);
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        require(data != MAP_FAILED, "map captured source");
    }
    ~Mapping() { munmap(data, size); }
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
};

struct Census {
    const uint8_t *lanes;
    const uint64_t *zeros;
    uint64_t columns, rows_per_block, blocks, zero_mask;
    std::vector<uint64_t> hot, fingerprints;
    std::vector<uint8_t> tile_counts;
    uint64_t total_hot = 0, unique_hot = 0, duplicate_tasks = 0, zero_tasks = 0;
    uint64_t barrier_sum = 0, barrier_max_sum = 0;
    std::array<uint64_t, 17> maximum_histogram{};
    std::vector<uint64_t> column_hot, column_unique_hot;

    uint16_t symbol(uint64_t row, uint64_t column) const {
        const uint8_t value = lanes[row * columns + column];
        if (value) return value;
        const bool selected_zero = ((zero_mask >> column) & 1)
            && ((zeros[row / 64] >> (row % 64)) & 1);
        return selected_zero ? 256 : 0;
    }

    bool equal(uint64_t lhs, uint64_t rhs) const {
        for (uint64_t row = 0; row < rows_per_block; ++row) {
            if (symbol((lhs / columns) * rows_per_block + row, lhs % columns)
                != symbol((rhs / columns) * rows_per_block + row, rhs % columns)) return false;
        }
        return true;
    }

    Census(const uint8_t *source, const uint64_t *active_zero, uint64_t cols,
           uint64_t block_rows, uint64_t full_blocks, uint64_t mask)
        : lanes(source), zeros(active_zero), columns(cols), rows_per_block(block_rows),
          blocks(full_blocks), zero_mask(mask), hot(cols * full_blocks),
          fingerprints(cols * full_blocks, 14695981039346656037ull),
          tile_counts(((block_rows + 7) / 8) * cols * full_blocks),
          column_hot(cols), column_unique_hot(cols) {
        require(cols > 0 && cols <= 32 && block_rows > 0 && full_blocks > 0,
            "bounded census shape");
        const uint64_t tiles = (rows_per_block + 7) / 8;
        for (uint64_t block = 0; block < blocks; ++block) {
            for (uint64_t row = 0; row < rows_per_block; ++row) {
                const uint64_t global_row = block * rows_per_block + row;
                for (uint64_t column = 0; column < columns; ++column) {
                    const uint64_t task = block * columns + column;
                    const uint16_t value = symbol(global_row, column);
                    fingerprints[task] = (fingerprints[task] ^ value) * 1099511628211ull;
                    if (value) {
                        ++hot[task];
                        ++tile_counts[(block * tiles + row / 8) * columns + column];
                    }
                }
            }
        }
        std::map<uint64_t, std::vector<uint64_t>> representatives;
        for (uint64_t task = 0; task < hot.size(); ++task) {
            total_hot += hot[task];
            column_hot[task % columns] += hot[task];
            if (!hot[task]) { ++zero_tasks; continue; }
            bool duplicate = false;
            auto &bucket = representatives[fingerprints[task]];
            for (uint64_t previous : bucket) {
                if (hot[previous] == hot[task] && equal(previous, task)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                ++duplicate_tasks;
            } else {
                bucket.push_back(task);
                unique_hot += hot[task];
                column_unique_hot[task % columns] += hot[task];
            }
        }
        for (uint64_t first = 0; first < hot.size(); first += 64) {
            for (uint64_t tile = 0; tile < tiles; ++tile) {
                uint64_t sum = 0, maximum = 0;
                for (uint64_t simd = 0; simd < 32; ++simd) {
                    uint64_t iterations = 0;
                    for (uint64_t offset = 0; offset < 2; ++offset) {
                        const uint64_t task = first + simd * 2 + offset;
                        if (task < hot.size())
                            iterations += tile_counts[((task / columns) * tiles + tile)
                                * columns + task % columns];
                    }
                    sum += iterations;
                    maximum = std::max(maximum, iterations);
                }
                barrier_sum += sum;
                barrier_max_sum += 32 * maximum;
                ++maximum_histogram[maximum];
            }
        }
        require(barrier_sum == total_hot, "barrier census conserves selected work");
    }

    void print() const {
        std::printf("CENSUS_SUMMARY tasks=%llu hot=%llu unique_hot=%llu duplicate_tasks=%llu zero_tasks=%llu removable_update_fraction=%.9f barrier_work=%llu barrier_max_times32=%llu selector_balance_ratio=%.9f\n",
            (unsigned long long)hot.size(), (unsigned long long)total_hot,
            (unsigned long long)unique_hot, (unsigned long long)duplicate_tasks,
            (unsigned long long)zero_tasks, 1.0 - double(unique_hot) / total_hot,
            (unsigned long long)barrier_sum, (unsigned long long)barrier_max_sum,
            double(barrier_sum) / barrier_max_sum);
        for (uint64_t column = 0; column < columns; ++column)
            std::printf("CENSUS_COLUMN column=%llu hot=%llu representative_hot=%llu\n",
                (unsigned long long)column, (unsigned long long)column_hot[column],
                (unsigned long long)column_unique_hot[column]);
        for (uint64_t count = 0; count < maximum_histogram.size(); ++count)
            std::printf("CENSUS_BARRIER maximum_iterations=%llu count=%llu\n",
                (unsigned long long)count, (unsigned long long)maximum_histogram[count]);
    }
};

int main(int argc, const char **argv) {
    const uint8_t fixture[] = {1,0,0,0,2,1,0,0,1,0,0,0,2,1,0,0};
    const uint64_t active_zero = (1ull << 1) | (1ull << 5);
    Census test(fixture, &active_zero, 2, 4, 2, 2);
    require(test.total_hot == 8 && test.unique_hot == 4 && test.duplicate_tasks == 2
        && test.zero_tasks == 0 && test.barrier_max_sum == 128, "independent hand-counted fixture");
    std::puts("CENSUS_SELFTEST pass=true");
    if (argc == 1) return 0;
    require(argc == 9, "usage: census lanes zeros rows columns positions full_blocks zero_mask expected_hot");
    const auto start = std::chrono::steady_clock::now();
    Mapping lanes(argv[1]), zeros(argv[2]);
    const uint64_t rows = std::stoull(argv[3]), columns = std::stoull(argv[4]);
    const uint64_t positions = std::stoull(argv[5]), blocks = std::stoull(argv[6]);
    const uint64_t mask = std::stoull(argv[7]), expected_hot = std::stoull(argv[8]);
    require(rows == (1ull << 28) && columns > 0 && columns <= 32 && positions == (1ull << 19)
        && blocks > 0 && blocks <= 1024, "production census envelope");
    require(lanes.size == rows * columns && zeros.size == rows / 8, "capture lengths");
    Census result(static_cast<const uint8_t *>(lanes.data),
        static_cast<const uint64_t *>(zeros.data), columns, positions / 2, blocks, mask);
    require(result.total_hot == expected_hot, "producer hot-entry count equality");
    result.print();
    std::printf("CENSUS_COMPLETE elapsed_s=%.6f producer_hot_match=true\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}
