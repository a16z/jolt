use jolt_eval::objective::performance::hamming_weight_pushforward::HammingWeightPushforwardObjective;

jolt_eval::bench_objective!(
    HammingWeightPushforwardObjective,
    config:
        sample_size(10),
        sampling_mode(::criterion::SamplingMode::Flat),
        measurement_time(std::time::Duration::from_secs(12)),
);
