macro_rules! provable_with_config {
    ($item: item) => {
        #[jolt::provable(
            max_input_size = 4096,
            max_output_size = 4096,
            memory_size = 33554432,
            stack_size = 4096,
            max_trace_length = 16777216
        )]
        $item
    };
}