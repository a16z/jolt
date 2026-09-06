#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <numeric>
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

static uint16_t selected_symbol(const uint8_t *lanes, const uint64_t *zeros,
                               uint64_t columns, uint64_t zero_mask,
                               uint64_t row, uint64_t column) {
    const uint8_t value = lanes[row * columns + column];
    if (value) return value;
    const bool selected_zero = ((zero_mask >> column) & 1)
        && ((zeros[row / 64] >> (row % 64)) & 1);
    return selected_zero ? 256 : 0;
}

struct TemplateCensus {
    uint64_t columns, rows, total_hot = 0, correction_hot = 0;
    std::vector<std::array<uint64_t, 257>> histogram, sample;
    std::vector<uint32_t> choices;

    TemplateCensus(const uint8_t *lanes, const uint64_t *zeros, uint64_t cols,
                   uint64_t count, uint64_t mask, uint64_t live_rows)
        : columns(cols), rows(count), histogram(cols), sample(cols), choices(cols) {
        require(live_rows > 0 && live_rows <= rows, "template sample prefix");
        for (uint64_t index = 0; index < 1024; ++index) {
            uint64_t random = index + 0x9e3779b97f4a7c15ull;
            random = (random ^ (random >> 30)) * 0xbf58476d1ce4e5b9ull;
            random = (random ^ (random >> 27)) * 0x94d049bb133111ebull;
            random ^= random >> 31;
            for (uint64_t column = 0; column < columns; ++column)
                ++sample[column][selected_symbol(lanes, zeros, columns, mask,
                    random % live_rows, column)];
        }
        for (uint64_t column = 0; column < columns; ++column) {
            for (uint32_t value = 1; value <= 256; ++value)
                if (sample[column][value] > 512) choices[column] = value;
        }
        for (uint64_t row = 0; row < rows; ++row) {
            for (uint64_t column = 0; column < columns; ++column)
                ++histogram[column][selected_symbol(lanes, zeros, columns, mask, row, column)];
        }
        for (uint64_t column = 0; column < columns; ++column) {
            const uint64_t hot = rows - histogram[column][0];
            total_hot += hot;
            correction_hot += cost(column, choices[column]);
        }
    }

    uint64_t cost(uint64_t column, uint32_t choice) const {
        const uint64_t hot = rows - histogram[column][0];
        return choice ? rows + hot - 2 * histogram[column][choice] : hot;
    }

    void print(const char *path) const {
        uint64_t oracle_hot = 0;
        std::array<bool, 257> used{};
        for (uint64_t column = 0; column < columns; ++column) {
            uint32_t best = 0;
            for (uint32_t value = 1; value <= 256; ++value)
                if (cost(column, value) < cost(column, best)) best = value;
            oracle_hot += cost(column, best);
            used[choices[column]] = true;
            std::printf("TEMPLATE_COLUMN column=%llu hot=%llu choice=%u sampled_matches=%llu exact_matches=%llu correction_hot=%llu oracle_choice=%u oracle_hot=%llu\n",
                (unsigned long long)column, (unsigned long long)(rows - histogram[column][0]),
                choices[column], (unsigned long long)sample[column][choices[column]],
                (unsigned long long)histogram[column][choices[column]],
                (unsigned long long)cost(column, choices[column]), best,
                (unsigned long long)cost(column, best));
        }
        const uint64_t templates = std::count(used.begin() + 1, used.end(), true);
        std::printf("TEMPLATE_SUMMARY rows=%llu hot=%llu correction_hot=%llu reduction=%.9f oracle_hot=%llu oracle_reduction=%.9f templates=%llu\n",
            (unsigned long long)rows, (unsigned long long)total_hot,
            (unsigned long long)correction_hot, 1.0 - double(correction_hot) / total_hot,
            (unsigned long long)oracle_hot, 1.0 - double(oracle_hot) / total_hot,
            (unsigned long long)templates);
        const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        require(fd >= 0, "fresh template choices artifact");
        const size_t bytes = choices.size() * sizeof(uint32_t);
        require(write(fd, choices.data(), bytes) == ssize_t(bytes), "template choices write");
        require(fsync(fd) == 0 && close(fd) == 0, "template choices sync");
    }
};

struct Census {
    const uint8_t *lanes;
    const uint64_t *zeros;
    uint64_t columns, rows_per_block, blocks, zero_mask, domain_count;
    std::vector<uint64_t> hot, fingerprints;
    std::vector<uint8_t> tile_counts;
    uint64_t total_hot = 0, unique_hot = 0, duplicate_tasks = 0, zero_tasks = 0;
    uint64_t barrier_sum = 0, barrier_max_sum = 0;
    std::array<uint64_t, 17> maximum_histogram{};
    std::vector<uint64_t> column_hot, column_unique_hot;
    std::vector<uint64_t> domain_unique_tasks;

    uint16_t symbol(uint64_t row, uint64_t column) const {
        return selected_symbol(lanes, zeros, columns, zero_mask, row, column);
    }

    bool equal(uint64_t lhs, uint64_t rhs) const {
        for (uint64_t row = 0; row < rows_per_block; ++row) {
            if (symbol((lhs / columns) * rows_per_block + row, lhs % columns)
                != symbol((rhs / columns) * rows_per_block + row, rhs % columns)) return false;
        }
        return true;
    }

    Census(const uint8_t *source, const uint64_t *active_zero, uint64_t cols,
           uint64_t block_rows, uint64_t full_blocks, uint64_t mask, uint64_t domains = 1)
        : lanes(source), zeros(active_zero), columns(cols), rows_per_block(block_rows),
          blocks(full_blocks), zero_mask(mask), domain_count(domains), hot(cols * full_blocks),
          fingerprints(cols * full_blocks, 14695981039346656037ull),
          tile_counts(domains == 1 ? ((block_rows + 7) / 8) * cols * full_blocks : 0),
          column_hot(cols), column_unique_hot(cols), domain_unique_tasks(domains) {
        require(cols > 0 && cols <= 32 && block_rows > 0 && full_blocks > 0
            && (domains == 1 || domains == 16) && full_blocks % domains == 0,
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
                        if (domains == 1)
                            ++tile_counts[(block * tiles + row / 8) * columns + column];
                    }
                }
            }
        }
        std::map<std::pair<uint64_t, uint64_t>, std::vector<uint64_t>> representatives;
        for (uint64_t task = 0; task < hot.size(); ++task) {
            total_hot += hot[task];
            column_hot[task % columns] += hot[task];
            if (!hot[task]) { ++zero_tasks; continue; }
            bool duplicate = false;
            const uint64_t domain = (task / columns) % domains;
            auto &bucket = representatives[{domain, fingerprints[task]}];
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
                ++domain_unique_tasks[domain];
            }
        }
        if (domains != 1) return;
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
        if (domain_count != 1) {
            const uint64_t original_tasks = blocks / domain_count * columns;
            const uint64_t original_groups = ((original_tasks + 63) / 64) * domain_count;
            uint64_t unique_groups = 0;
            for (uint64_t count : domain_unique_tasks) unique_groups += (count + 63) / 64;
            std::printf("FRAGMENT_SUMMARY domains=%llu rows_per_fragment=%llu fragments=%llu hot=%llu unique_hot=%llu duplicate_fragments=%llu zero_fragments=%llu removable_update_fraction=%.9f original_groups_per_rank=%llu unique_groups_per_rank=%llu matrix_group_reduction=%.9f\n",
                (unsigned long long)domain_count, (unsigned long long)rows_per_block,
                (unsigned long long)hot.size(), (unsigned long long)total_hot,
                (unsigned long long)unique_hot, (unsigned long long)duplicate_tasks,
                (unsigned long long)zero_tasks, 1.0 - double(unique_hot) / total_hot,
                (unsigned long long)original_groups, (unsigned long long)unique_groups,
                1.0 - double(unique_groups) / original_groups);
            for (uint64_t domain = 0; domain < domain_count; ++domain)
                std::printf("FRAGMENT_DOMAIN partial=%llu unique_tasks=%llu\n",
                    (unsigned long long)domain, (unsigned long long)domain_unique_tasks[domain]);
            for (uint64_t column = 0; column < columns; ++column)
                std::printf("FRAGMENT_COLUMN column=%llu hot=%llu representative_hot=%llu\n",
                    (unsigned long long)column, (unsigned long long)column_hot[column],
                    (unsigned long long)column_unique_hot[column]);
            return;
        }
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

    void price_pairing(uint64_t live_rows, const char *map_path) const {
        require(domain_count == 1, "pairing uses full original task groups");
        require(live_rows > 0 && live_rows <= blocks * rows_per_block, "live prefix bounds");
        std::vector<uint64_t> sampled_hot(columns), score(hot.size());
        std::vector<uint32_t> mapping(hot.size());
        for (uint64_t sample = 0; sample < 1024; ++sample) {
            uint64_t random = sample + 0x9e3779b97f4a7c15ull;
            random = (random ^ (random >> 30)) * 0xbf58476d1ce4e5b9ull;
            random = (random ^ (random >> 27)) * 0x94d049bb133111ebull;
            random ^= random >> 31;
            const uint64_t row = random % live_rows;
            for (uint64_t column = 0; column < columns; ++column)
                sampled_hot[column] += symbol(row, column) != 0;
        }
        for (uint64_t task = 0; task < hot.size(); ++task) {
            const uint64_t first_row = (task / columns) * rows_per_block;
            const uint64_t valid_rows = first_row < live_rows
                ? std::min(rows_per_block, live_rows - first_row) : 0;
            score[task] = sampled_hot[task % columns] * valid_rows;
        }
        for (uint64_t first = 0; first < hot.size(); first += 64) {
            const size_t count = std::min(uint64_t(64), hot.size() - first);
            std::vector<uint32_t> tasks(count);
            std::iota(tasks.begin(), tasks.end(), uint32_t(first));
            std::sort(tasks.begin(), tasks.end(), [&](uint32_t lhs, uint32_t rhs) {
                return score[lhs] == score[rhs] ? lhs < rhs : score[lhs] < score[rhs];
            });
            size_t lower = 0, upper = count, output = first;
            while (lower < upper) {
                mapping[output++] = tasks[lower++];
                if (lower < upper) mapping[output++] = tasks[--upper];
            }
            auto check = std::vector<uint32_t>(mapping.begin() + first, mapping.begin() + first + count);
            std::sort(check.begin(), check.end());
            for (size_t index = 0; index < count; ++index)
                require(check[index] == first + index, "permutation preserves each original group");
        }
        const uint64_t tiles = (rows_per_block + 7) / 8;
        uint64_t new_work = 0, new_max_sum = 0;
        for (uint64_t first = 0; first < hot.size(); first += 64) {
            for (uint64_t tile = 0; tile < tiles; ++tile) {
                uint64_t maximum = 0;
                for (uint64_t simd = 0; simd < 32; ++simd) {
                    uint64_t iterations = 0;
                    for (uint64_t offset = 0; offset < 2; ++offset) {
                        const uint64_t local_task = first + simd * 2 + offset;
                        if (local_task < mapping.size()) {
                            const uint64_t task = mapping[local_task];
                            iterations += tile_counts[((task / columns) * tiles + tile)
                                * columns + task % columns];
                        }
                    }
                    new_work += iterations;
                    maximum = std::max(maximum, iterations);
                }
                new_max_sum += maximum * 32;
            }
        }
        require(new_work == total_hot, "permuted work conservation");
        for (unsigned artifact = 0; artifact < 2; ++artifact) {
            const std::string path = std::string(map_path) + (artifact == 0 ? "" : ".hot.u64le");
            const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
            require(fd >= 0, "fresh permutation artifact");
            const void *data = artifact == 0 ? static_cast<const void *>(mapping.data())
                : static_cast<const void *>(hot.data());
            const size_t bytes = artifact == 0 ? mapping.size() * sizeof(uint32_t)
                : hot.size() * sizeof(uint64_t);
            size_t written = 0;
            while (written < bytes) {
                const ssize_t count = write(fd, static_cast<const uint8_t *>(data) + written, bytes - written);
                require(count > 0, "permutation artifact write");
                written += size_t(count);
            }
            require(fsync(fd) == 0 && close(fd) == 0, "permutation artifact sync");
        }
        std::printf("PAIRING_PRICE original_max_times32=%llu paired_max_times32=%llu conserved_hot=%llu envelope_reduction=%.9f original_balance=%.9f paired_balance=%.9f map_bytes=%llu\n",
            (unsigned long long)barrier_max_sum, (unsigned long long)new_max_sum,
            (unsigned long long)new_work, 1.0 - double(new_max_sum) / barrier_max_sum,
            double(total_hot) / barrier_max_sum, double(total_hot) / new_max_sum,
            (unsigned long long)(mapping.size() * sizeof(uint32_t)));
        for (uint64_t column = 0; column < columns; ++column)
            std::printf("PAIRING_SAMPLE column=%llu hot_of_1024=%llu\n",
                (unsigned long long)column, (unsigned long long)sampled_hot[column]);
    }
};

int main(int argc, const char **argv) {
    const uint8_t fixture[] = {1,0,0,0,2,1,0,0,1,0,0,0,2,1,0,0};
    const uint64_t active_zero = (1ull << 1) | (1ull << 5);
    Census test(fixture, &active_zero, 2, 4, 2, 2);
    require(test.total_hot == 8 && test.unique_hot == 4 && test.duplicate_tasks == 2
        && test.zero_tasks == 0 && test.barrier_max_sum == 128, "independent hand-counted fixture");
    std::array<uint8_t, 64> domain_fixture;
    domain_fixture.fill(1);
    domain_fixture[1] = domain_fixture[33] = 0;
    const uint64_t domain_zeros = 1ull | (1ull << 16);
    Census domains(domain_fixture.data(), &domain_zeros, 2, 1, 32, 2, 16);
    require(domains.total_hot == 64 && domains.unique_hot == 17
        && domains.duplicate_tasks == 47 && domains.zero_tasks == 0
        && domains.domain_unique_tasks[0] == 2 && domains.domain_unique_tasks[1] == 1,
        "fragment equality restricted to the same public-A domain");
    const uint8_t template_fixture[] = {7,0,7,0,7,3,0,3};
    const uint64_t template_zeros = 1;
    TemplateCensus templates(template_fixture, &template_zeros, 2, 4, 2, 4);
    require(templates.total_hot == 6 && templates.cost(0, 7) == 1
        && templates.cost(1, 3) == 3 && templates.cost(1, 256) == 5
        && templates.histogram[1][0] == 1 && templates.histogram[1][256] == 1,
        "template correction additions and subtractions independently counted");
    std::puts("CENSUS_SELFTEST pass=true");
    if (argc == 1) return 0;
    require(argc == 9 || argc == 10 || argc == 11 || argc == 12,
        "usage: census lanes zeros rows columns positions full_blocks zero_mask expected_hot [--fragments | live_rows map_path | --templates live_rows choices_path]");
    const bool fragments = argc == 10;
    require(!fragments || std::string(argv[9]) == "--fragments", "fragment mode flag");
    const auto start = std::chrono::steady_clock::now();
    Mapping lanes(argv[1]), zeros(argv[2]);
    const uint64_t rows = std::stoull(argv[3]), columns = std::stoull(argv[4]);
    const uint64_t positions = std::stoull(argv[5]), blocks = std::stoull(argv[6]);
    const uint64_t mask = std::stoull(argv[7]), expected_hot = std::stoull(argv[8]);
    require(rows == (1ull << 28) && columns > 0 && columns <= 32 && positions == (1ull << 19)
        && blocks > 0 && blocks <= 1024, "production census envelope");
    require(lanes.size == rows * columns && zeros.size == rows / 8, "capture lengths");
    if (argc == 12) {
        require(std::string(argv[9]) == "--templates", "template mode flag");
        TemplateCensus result(static_cast<const uint8_t *>(lanes.data),
            static_cast<const uint64_t *>(zeros.data), columns, blocks * positions / 2,
            mask, std::stoull(argv[10]));
        require(result.total_hot == expected_hot, "producer hot-entry count equality");
        result.print(argv[11]);
        std::printf("CENSUS_COMPLETE elapsed_s=%.6f producer_hot_match=true\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return 0;
    }
    Census result(static_cast<const uint8_t *>(lanes.data),
        static_cast<const uint64_t *>(zeros.data), columns, positions / (fragments ? 32 : 2),
        blocks * (fragments ? 16 : 1), mask, fragments ? 16 : 1);
    require(result.total_hot == expected_hot, "producer hot-entry count equality");
    result.print();
    if (argc == 11) result.price_pairing(std::stoull(argv[9]), argv[10]);
    std::printf("CENSUS_COMPLETE elapsed_s=%.6f producer_hot_match=true\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}
