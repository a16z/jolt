macro_rules! provable_with_config {
    ($item: item) => {
        #[jolt::provable(
            max_input_size = 4096,
            max_output_size = 4096,
            max_untrusted_advice_size = 0,
            max_trusted_advice_size = 0,
            memory_size = 33554432,
            stack_size = 1048576,
            max_trace_length = 67108864
        )]
        $item
    };
}