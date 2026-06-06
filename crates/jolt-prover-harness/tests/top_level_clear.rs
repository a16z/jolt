#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
use jolt_prover_harness::{FeatureMode, FixtureKind};

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn top_level_clear_prover_outputs_verify() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let fixture = jolt_prover_harness::load_top_level_clear_prover_fixture(&request)
            .map_err(|error| error.to_string())?;
        fixture
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}
