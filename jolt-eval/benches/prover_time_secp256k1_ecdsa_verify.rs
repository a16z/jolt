use jolt_eval::guests::Secp256k1EcdsaVerify;
use jolt_eval::objective::performance::prover_time::ProverTimeObjective;

jolt_eval::bench_objective!(
    ProverTimeObjective::new(Secp256k1EcdsaVerify::default()),
    config:
        sample_size(10),
        sampling_mode(::criterion::SamplingMode::Flat),
        measurement_time(std::time::Duration::from_secs(60)),
);
