use jolt_eval::guests::Sha2Chain;
use jolt_eval::objective::performance::prover_time::ProverTimeObjective;

jolt_eval::bench_objective!(
    ProverTimeObjective::new(Sha2Chain::default()),
    config:
        sample_size(10),
        sampling_mode(::criterion::SamplingMode::Flat),
        measurement_time(std::time::Duration::from_secs(60)),
);
