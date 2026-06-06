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
fn stage8_opening_structure_matches_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage8_opening_structure_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.modular, checkpoint.expected);
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage8_ra_constituents_evaluate_to_reduced_claims() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint =
            jolt_prover_harness::load_stage8_ra_constituent_checkpoint_fixture(&request)
                .map_err(|error| error.to_string())?;

        assert_eq!(checkpoint.evaluated, checkpoint.expected);
        assert_eq!(checkpoint.dense_evaluated, checkpoint.dense_expected);
        assert_eq!(checkpoint.joint_evaluation, checkpoint.joint_claim);
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage8_joint_opening_proof_verifies_against_core_fixtures() -> Result<(), String> {
    for kind in [FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer] {
        let request = jolt_prover_harness::FixtureRequest::new(kind, FeatureMode::Transparent);
        let checkpoint = jolt_prover_harness::load_stage8_joint_opening_replay_fixture(&request)
            .map_err(|error| error.to_string())?;
        checkpoint
            .fixture
            .verify()
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}
