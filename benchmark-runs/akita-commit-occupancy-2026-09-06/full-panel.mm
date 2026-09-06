#define COMMIT_DIAGNOSTIC_LIBRARY
#include "saturation.mm"
#include "radix26-probe.mm"

enum class PanelSchedule { RowMajor, ColumnMajor, WidenedCarry, SingleTask, Radix26, Radix26Staged };

struct PanelReplay {
    Case &test;
    NSData *source;
    NSData *zero_bits;

    id<MTLBuffer> run(id<MTLDevice> device, id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> panel, id<MTLComputePipelineState> reducer,
                      PanelSchedule schedule) {
        const bool column_major = schedule == PanelSchedule::ColumnMajor;
        const uint64_t tasks_per_stream = schedule == PanelSchedule::SingleTask ? 32 : 64;
        const auto started = Clock::now();
        const PackedParams original = test.params;
        require(original.tasks == original.full_blocks * original.columns
            && original.task_offset == 0 && original.dispatch_tasks == original.tasks,
            "full-panel original task geometry");
        test.device_output = [device newBufferWithLength:test.output.length options:MTLResourceStorageModePrivate];
        id<MTLBuffer> final = [device newBufferWithLength:original.output_coefficients * sizeof(U128)
            options:MTLResourceStorageModeShared];
        test.zero_rows = [device newBufferWithBytes:zero_bits.bytes length:zero_bits.length
            options:MTLResourceStorageModeShared];
        require(test.device_output && final && test.zero_rows, "full-panel output and zero storage");
        id<MTLBuffer> whole_source = nil;
        if (column_major) {
            whole_source = [device newBufferWithBytesNoCopy:const_cast<void *>(source.bytes)
                length:source.length options:MTLResourceStorageModeShared deallocator:nil];
            require(whole_source != nil, "column-major zero-copy view without fallback");
        }
        NSMutableArray<id<MTLCommandBuffer>> *commands = [NSMutableArray new];
        NSMutableArray<id<MTLBuffer>> *views = [NSMutableArray new];
        const uint64_t rows_per_block = original.positions / 2;
        const double epoch = [[NSDate date] timeIntervalSince1970];
        for (uint64_t first = 0; first < original.tasks; first += 512) {
            PackedParams params = original;
            params.task_offset = first;
            params.dispatch_tasks = std::min(uint64_t(512), original.tasks - first);
            id<MTLBuffer> lanes = whole_source;
            if (!column_major) {
                const uint64_t first_row = (first / original.columns) * rows_per_block;
                const uint64_t final_row = ((first + params.dispatch_tasks - 1) / original.columns + 1)
                    * rows_per_block;
                const uint64_t offset = first_row * original.columns;
                const uint64_t bytes = (final_row - first_row) * original.columns;
                require(offset + bytes <= source.length, "parent selector slice bounds");
                lanes = [device newBufferWithBytesNoCopy:const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(source.bytes)) + offset length:bytes
                    options:MTLResourceStorageModeShared deallocator:nil];
                require(lanes != nil, "parent zero-copy slice without fallback");
                params.lane_row_offset = first_row;
            }
            [views addObject:lanes];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:panel];
            [encoder setBuffer:test.device_matrix offset:0 atIndex:0];
            [encoder setBuffer:lanes offset:0 atIndex:1];
            [encoder setBuffer:test.device_output offset:0 atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder setBuffer:test.zero_rows offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(((params.dispatch_tasks + tasks_per_stream - 1)
                / tasks_per_stream) * 48, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [commands addObject:command];
        }
        id<MTLCommandBuffer> reduction = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [reduction computeCommandEncoder];
        [encoder setComputePipelineState:reducer];
        [encoder setBuffer:test.device_output offset:0 atIndex:0];
        [encoder setBuffer:final offset:0 atIndex:1];
        [encoder setBytes:&original length:sizeof(original) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(original.output_coefficients, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [reduction commit];
        for (id<MTLCommandBuffer> command in commands) finish_command(command, Clock::now());
        finish_command(reduction, Clock::now());
        const double wall = std::chrono::duration<double>(Clock::now() - started).count();
        double active = 0;
        unsigned index = 0;
        for (id<MTLCommandBuffer> command in commands) {
            const double gpu = command.GPUEndTime - command.GPUStartTime;
            require(std::isfinite(gpu) && gpu > 0, "full-panel per-command timestamp");
            active += gpu;
            std::printf("PANEL_COMMAND index=%u gpu_ms=%.6f gpu_start=%.9f gpu_end=%.9f\n",
                index++, gpu * 1e3, command.GPUStartTime, command.GPUEndTime);
        }
        const double panel_span = commands.lastObject.GPUEndTime - commands.firstObject.GPUStartTime;
        const double reduction_gpu = reduction.GPUEndTime - reduction.GPUStartTime;
        uint64_t hot = 0;
        for (uint64_t count : test.task_hot) hot += count;
        require(commands.count == 44 && hot == 3263846381ull, "full-panel command and useful-work identity");
        std::printf("FULL_PANEL variant=%u tasks=%llu commands=%lu hot=%llu panel_gpu_ms=%.6f panel_span_ms=%.6f reduce_gpu_ms=%.6f wall_ms=%.6f giga_updates_s=%.6f epoch=%.6f zero_copy=true\n",
            unsigned(schedule != PanelSchedule::RowMajor), (unsigned long long)original.tasks, (unsigned long)commands.count,
            (unsigned long long)hot, active * 1e3, panel_span * 1e3, reduction_gpu * 1e3,
            wall * 1e3, double(hot) * 384 / active / 1e9, epoch);
        std::fflush(stdout);
        // The parent gets a whole CPU-oracle view only after its timed slice schedule completes.
        test.lanes = whole_source ? whole_source : [device newBufferWithBytesNoCopy:const_cast<void *>(source.bytes)
            length:source.length options:MTLResourceStorageModeShared deallocator:nil];
        require(test.lanes != nil, "post-timing oracle selector view");
        return final;
    }
};

int main(int argc, const char **argv) {
    require(argc >= 6 && argc <= 9, "usage: full-panel production.metal capture-directory reference.fp128le archive-prefix variant [--widened-carry | --single-task | --radix26 helper.metal | --staged-radix26 helper.metal staged.metal]");
    const bool widened = argc == 7 && std::string(argv[6]) == "--widened-carry";
    const bool single_task = argc == 7 && std::string(argv[6]) == "--single-task";
    const bool staged_radix26 = argc == 9 && std::string(argv[6]) == "--staged-radix26";
    const bool radix26 = staged_radix26 || (argc == 8 && std::string(argv[6]) == "--radix26");
    require(argc == 6 || widened || single_task || radix26, "full-panel mechanism flag");
    const unsigned variant = unsigned(std::stoul(argv[5]));
    require(variant < 2, "full-panel variant");
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "accepted source");
        NSRange start = [source rangeOfString:@"kernel void akita_packed_onehot_commit_fp128_d128_rank3("];
        NSRange end = [source rangeOfString:@"// Packed decompose-fold for the D128 rank-3 row."];
        require(start.location != NSNotFound && end.location > start.location, "root kernel boundary");
        NSString *body = [source substringWithRange:NSMakeRange(start.location, end.location - start.location)];
        NSString *variant_name = staged_radix26 ? @"diagnostic_radix26_staged_commit"
            : radix26 ? @"diagnostic_radix26_commit" : single_task ? @"diagnostic_single_task_commit"
            : widened ? @"diagnostic_widened_carry_commit" : @"diagnostic_column_major_commit";
        body = [body stringByReplacingOccurrencesOfString:@"akita_packed_onehot_commit_fp128_d128_rank3"
            withString:variant_name];
        NSString *combined;
        if (radix26) {
            NSString *radix_source = [NSString stringWithContentsOfFile:@(argv[7]) encoding:NSUTF8StringEncoding error:&error];
            require(radix_source && !error, "radix26 shader source");
            NSRange tile_start = [source rangeOfString:@"inline void akita_fp128_d128_rank3_accumulate_task_tile("];
            NSRange store_start = [source rangeOfString:@"inline void akita_store_fp128_d128_rank3("];
            require(tile_start.location != NSNotFound && store_start.location > tile_start.location
                && start.location > store_start.location, "radix26 original helper boundaries");
            NSString *helpers = [source substringWithRange:NSMakeRange(tile_start.location, start.location - tile_start.location)];
            for (NSArray<NSString *> *rename in @[
                @[@"AkitaTransposedFp128Accumulator", @"DiagnosticRadix26"],
                @[@"akita_transposed_fp128_zero", @"diagnostic_radix26_zero"],
                @[@"akita_fp128_d128_rank3_accumulate_task_tile", @"diagnostic_radix26_task_tile"],
                @[@"akita_fp128_d512_accumulate_mixed", @"diagnostic_radix26_add"],
                @[@"akita_store_fp128_d128_rank3", @"diagnostic_radix26_store"],
                @[@"akita_reduce_transposed_fp128", @"diagnostic_radix26_reduce"]]) {
                helpers = [helpers stringByReplacingOccurrencesOfString:rename[0] withString:rename[1]];
                body = [body stringByReplacingOccurrencesOfString:rename[0] withString:rename[1]];
            }
            NSString *anchor = @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n        matrix_cursor += (ulong)PACKED_FP128_D512_PANEL_TILE_ELEMENTS;";
            require([body containsString:anchor], "radix26 normalization cadence anchor");
            body = [body stringByReplacingOccurrencesOfString:anchor withString:
                @"        if ((tile & 1u) != 0u) {\n            if (active_0) diagnostic_radix26_normalize(accumulator_0);\n            if (active_1) diagnostic_radix26_normalize(accumulator_1);\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        matrix_cursor += (ulong)PACKED_FP128_D512_PANEL_TILE_ELEMENTS;"];
            if (staged_radix26) {
                NSString *staged_source = [NSString stringWithContentsOfFile:@(argv[8])
                    encoding:NSUTF8StringEncoding error:&error];
                require(staged_source && !error, "staged radix26 shader source");
                NSString *stores = @"            shared_matrix[shared_index] = value.limb[0];\n            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS + shared_index] = value.limb[1];\n            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 2u + shared_index] =\n                value.limb[2];\n            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 3u + shared_index] =\n                value.limb[3];";
                require([body containsString:stores], "staged radix26 cooperative-copy anchor");
                body = [body stringByReplacingOccurrencesOfString:stores
                    withString:@"            diagnostic_radix26_stage(shared_matrix, shared_index, value);"];
                for (NSArray<NSString *> *replacement in @[
                    @[@"PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4", @"DIAGNOSTIC_RADIX26_TILE_ELEMENTS * 5"],
                    @[@"PACKED_FP128_D512_PANEL_TILE_ELEMENTS", @"DIAGNOSTIC_RADIX26_TILE_ELEMENTS"],
                    @[@"PACKED_FP128_D128_RANK3_TILE_POSITIONS", @"DIAGNOSTIC_RADIX26_TILE_POSITIONS"],
                    @[@"PACKED_FP128_D128_RANK3_ROWS_PER_TILE", @"DIAGNOSTIC_RADIX26_ROWS_PER_TILE"],
                    @[@"diagnostic_radix26_add", @"diagnostic_radix26_staged_add"],
                    @[@"(tile & 1u) != 0u", @"(tile & 3u) == 3u"]]) {
                    require([body containsString:replacement[0]] || [helpers containsString:replacement[0]],
                        "staged radix26 geometry/cadence anchor");
                    helpers = [helpers stringByReplacingOccurrencesOfString:replacement[0] withString:replacement[1]];
                    body = [body stringByReplacingOccurrencesOfString:replacement[0] withString:replacement[1]];
                }
                radix_source = [radix_source stringByAppendingString:staged_source];
            }
            combined = [[[source stringByAppendingString:radix_source] stringByAppendingString:helpers] stringByAppendingString:body];
        } else if (widened) {
            NSRange word_start = [source rangeOfString:@"inline uint4 akita_add_transposed_word("];
            NSRange word_end = [source rangeOfString:@"kernel void akita_packed_onehot_reduce_partials("];
            NSRange value_start = [source rangeOfString:@"inline void akita_fp128_d512_accumulate_value("];
            NSRange value_end = [source rangeOfString:@"inline void akita_fp128_d512_accumulate_positive("];
            NSRange mixed_start = [source rangeOfString:@"inline void akita_fp128_d512_accumulate_mixed("];
            NSRange mixed_end = [source rangeOfString:@"inline void akita_fp128_d512_accumulate_pair("];
            NSRange tile_start = [source rangeOfString:@"inline void akita_fp128_d128_rank3_accumulate_task_tile("];
            NSRange tile_end = [source rangeOfString:@"inline void akita_store_fp128_d128_rank3("];
            require(word_start.location != NSNotFound && word_end.location > word_start.location
                && value_start.location != NSNotFound && value_end.location > value_start.location
                && mixed_start.location != NSNotFound && mixed_end.location > mixed_start.location
                && tile_start.location != NSNotFound && tile_end.location > tile_start.location,
                "widened carry helper boundaries");
            NSString *word = @"\ninline uint4 diagnostic_widened_word(uint4 lhs, uint4 rhs, thread uint4 &carry) {\n    ulong4 wide = ulong4(lhs) + ulong4(rhs) + ulong4(carry);\n    carry = uint4(wide >> 32ul);\n    return uint4(wide);\n}\n";
            NSString *value = [source substringWithRange:NSMakeRange(value_start.location, value_end.location - value_start.location)];
            NSString *mixed = [source substringWithRange:NSMakeRange(mixed_start.location, mixed_end.location - mixed_start.location)];
            NSString *tile = [source substringWithRange:NSMakeRange(tile_start.location, tile_end.location - tile_start.location)];
            NSString *helpers = [[[word stringByAppendingString:value] stringByAppendingString:mixed] stringByAppendingString:tile];
            for (NSArray<NSString *> *rename in @[
                @[@"akita_add_transposed_word", @"diagnostic_widened_word"],
                @[@"akita_fp128_d512_accumulate_value", @"diagnostic_widened_value"],
                @[@"akita_fp128_d512_accumulate_mixed", @"diagnostic_widened_mixed"],
                @[@"akita_fp128_d128_rank3_accumulate_task_tile", @"diagnostic_widened_tile"]]) {
                helpers = [helpers stringByReplacingOccurrencesOfString:rename[0] withString:rename[1]];
                body = [body stringByReplacingOccurrencesOfString:rename[0] withString:rename[1]];
            }
            combined = [[source stringByAppendingString:helpers] stringByAppendingString:body];
        } else if (single_task) {
            for (NSArray<NSString *> *replacement in @[
                @[@"constexpr uint tasks_per_stream = PACKED_FP128_D128_RANK3_TASKS_PER_STREAM;",
                    @"constexpr uint tasks_per_stream = 32u;"],
                @[@"simdgroup * PACKED_FP128_D128_RANK3_TASKS_PER_SIMDGROUP", @"simdgroup"],
                @[@"bool active_1 = dispatch_task_0 + 1u < num_tasks;", @"bool active_1 = false;"]]) {
                require([body containsString:replacement[0]], "single-task production mapping anchor");
                body = [body stringByReplacingOccurrencesOfString:replacement[0] withString:replacement[1]];
            }
            combined = [source stringByAppendingString:body];
        } else {
        NSString *anchor = @"    uint global_1 = global_0 + 1u;";
        require([body containsString:anchor], "original task index anchor");
        body = [body stringByReplacingOccurrencesOfString:anchor withString:
            @"    uint global_1 = global_0 + 1u;\n    uint column_blocks = (uint)params.full_blocks_per_column;\n    if (active_0) global_0 = (global_0 % column_blocks) * live_columns + global_0 / column_blocks;\n    if (active_1) global_1 = (global_1 % column_blocks) * live_columns + global_1 / column_blocks;"];
        combined = [source stringByAppendingString:body];
        }
        require([combined writeToFile:[NSString stringWithFormat:@"%s.generated.metal", argv[4]]
            atomically:NO encoding:NSUTF8StringEncoding error:&error], "full-panel shader snapshot");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:combined options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "full-panel library");
        NSArray<NSString *> *names = radix26
            ? @[@"akita_packed_onehot_commit_fp128_d128_rank3", variant_name,
                @"akita_packed_onehot_reduce_partials", @"diagnostic_radix26_probe"]
            : @[@"akita_packed_onehot_commit_fp128_d128_rank3", variant_name,
                @"akita_packed_onehot_reduce_partials"];
        id<MTLComputePipelineState> pipelines[4];
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        for (unsigned index = 0; index < names.count; ++index) {
            id<MTLFunction> function = [library newFunctionWithName:names[index]];
            pipelines[index] = [device newComputePipelineStateWithFunction:function error:&error];
            require(pipelines[index] && !error, "full-panel pipeline");
            if (index < 2) require(pipelines[index].maxTotalThreadsPerThreadgroup == 1024
                && pipelines[index].threadExecutionWidth == 32
                && pipelines[index].staticThreadgroupMemoryLength == (staged_radix26 && index == 1 ? 20480 : 32768),
                "preregistered panel resource fingerprint");
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "full-panel archive");
        }
        require([archive serializeToURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s.bin", argv[4]]]
            error:&error], "full-panel archive serialization");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (radix26) check_radix26_normalizer(device, queue, pipelines[3]);
        for (unsigned candidate = 0; candidate < 2; ++candidate) {
            for (unsigned fixture = 0; fixture < (radix26 ? 5u : widened || single_task ? 3u : 2u); ++fixture) {
                Case small(device, radix26 ? 1024 : 256, 4, 5, 15, 10);
                small.params.full_blocks = 3;
                small.params.tasks = 15;
                if (fixture == 1) {
                    std::memset(small.lanes.contents, 0, small.lanes.length);
                    std::memset(small.zero_rows.contents, 0, small.zero_rows.length);
                    std::fill(small.task_hot.begin(), small.task_hot.end(), 0);
                }
                if (fixture == 2) {
                    const U128 values[] = {0, 1, MODULUS - 1, MODULUS - 2,
                        (U128(1) << 32) - 1, (U128(1) << 64) - 1,
                        (U128(1) << 96) - 1, U128(1) << 127};
                    auto *matrix_values = static_cast<U128 *>(small.matrix.contents);
                    for (size_t index = 0; index < small.matrix.length / sizeof(U128); ++index)
                        matrix_values[index] = values[index % 8];
                }
                if (fixture >= 3) {
                    auto *matrix_values = static_cast<U128 *>(small.matrix.contents);
                    std::fill_n(matrix_values, small.matrix.length / sizeof(U128), MODULUS - 1);
                    std::memset(small.lanes.contents, 0, small.lanes.length);
                    std::memset(small.zero_rows.contents, 0, small.zero_rows.length);
                    const uint64_t rows_per_block = small.params.positions / 2;
                    std::memset(small.lanes.contents, fixture == 3 ? 128 : 255,
                        small.params.full_blocks * rows_per_block * small.params.columns);
                    small.task_hot.assign(small.params.tasks, rows_per_block);
                }
                small.make_private(device, queue);
                small.run(queue, pipelines[candidate], candidate, false, true);
            }
        }
        NSString *directory = @(argv[2]);
        NSData *shape_bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"metadata.json"]];
        NSDictionary *shape = [NSJSONSerialization JSONObjectWithData:shape_bytes options:0 error:&error];
        require(shape && !error && [shape[@"rows"] unsignedLongLongValue] == (1ull << 28)
            && [shape[@"columns"] unsignedLongLongValue] == 29
            && [shape[@"positions"] unsignedLongLongValue] == (1ull << 19)
            && [shape[@"full_blocks"] unsignedLongLongValue] == 769
            && [shape[@"zero_mask"] unsignedLongLongValue] == 402653184, "frozen full-panel shape");
        NSData *lanes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"lanes.u8"]
            options:NSDataReadingMappedAlways error:&error];
        require(lanes && !error && lanes.length == (1ull << 28) * 29, "captured full selectors");
        NSData *zeros = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"active_zero_rows.u64le"]
            options:NSDataReadingMappedAlways error:&error];
        require(zeros && !error && zeros.length == (1ull << 25), "captured zero bits");
        NSData *hot = [NSData dataWithContentsOfFile:[[directory stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"d3-task-map.u32le.hot.u64le"]];
        require(hot.length == 22301 * 8, "frozen exact hot counts");
        Case target(device, 1 << 19, 1024, 29, 512, 0);
        target.params.tasks = target.params.dispatch_tasks = 22301;
        target.params.full_blocks = 769;
        target.params.zero_mask = 402653184;
        target.task_hot.resize(22301);
        std::memcpy(target.task_hot.data(), hot.bytes, hot.length);
        target.device_matrix = [device newBufferWithLength:target.matrix.length options:MTLResourceStorageModePrivate];
        require(target.device_matrix != nil, "resident private public matrix");
        id<MTLCommandBuffer> setup = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [setup blitCommandEncoder];
        [blit copyFromBuffer:target.matrix sourceOffset:0 toBuffer:target.device_matrix destinationOffset:0
            size:target.matrix.length];
        [blit endEncoding];
        const auto setup_started = Clock::now();
        [setup commit];
        finish_command(setup, setup_started);
        PanelReplay replay{target, lanes, zeros};
        const PanelSchedule schedule = variant == 0 ? PanelSchedule::RowMajor
            : staged_radix26 ? PanelSchedule::Radix26Staged
            : radix26 ? PanelSchedule::Radix26 : single_task ? PanelSchedule::SingleTask
            : widened ? PanelSchedule::WidenedCarry : PanelSchedule::ColumnMajor;
        id<MTLBuffer> result = replay.run(device, queue, pipelines[variant], pipelines[2], schedule);
        id<MTLCommandBuffer> readback = [queue commandBuffer];
        id<MTLBlitCommandEncoder> copy = [readback blitCommandEncoder];
        [copy copyFromBuffer:target.device_output sourceOffset:0 toBuffer:target.output destinationOffset:0
            size:target.output.length];
        [copy endEncoding];
        const auto readback_started = Clock::now();
        [readback commit];
        finish_command(readback, readback_started);
        const uint64_t oracle_hash = target.verify(false);
        const auto *final = static_cast<const U128 *>(result.contents);
        for (uint64_t column = 0; column < 32; ++column)
            for (uint64_t block = 0; block < 1024; ++block)
                if (column >= 29 || block >= 769)
                    for (uint64_t index = 0; index < 384; ++index)
                        require(final[(column * 1024 + block) * 384 + index] == 0, "all final padding zero");
        NSString *reference_path = @(argv[3]);
        if ([[NSFileManager defaultManager] fileExistsAtPath:reference_path]) {
            NSData *reference = [NSData dataWithContentsOfFile:reference_path options:NSDataReadingMappedAlways error:&error];
            require(reference && !error && reference.length == result.length
                && std::memcmp(reference.bytes, result.contents, result.length) == 0,
                "every final coefficient equals frozen first-parent output");
        } else {
            require(variant == 0, "only first parent establishes output artifact");
            NSData *reference = [NSData dataWithBytesNoCopy:result.contents length:result.length freeWhenDone:NO];
            require([reference writeToFile:reference_path options:NSDataWritingWithoutOverwriting error:&error]
                && !error, "fresh parent final output artifact");
        }
        std::printf("FULL_PANEL_COMPLETE target_observations=1 parity=pass oracle_hash=%016llx final_bytes=%lu\n",
            (unsigned long long)oracle_hash, (unsigned long)result.length);
    }
}
