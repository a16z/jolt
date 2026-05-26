use jolt_prover_harness::{
    validate_frontier_optimization_ids, AcceptanceMode, FeatureMode, FixtureKind, FrontierSpec,
    KnownOptimizationIds, ParityTarget,
};

const INVENTORY: &str = include_str!("../../../specs/jolt-core-prover-optimization-inventory.md");
const FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const PARITY: &[ParityTarget] = &[ParityTarget::VerifierAcceptance];

fn frontier(optimization_ids: &'static [&'static str]) -> FrontierSpec {
    FrontierSpec {
        name: "optimization_inventory_test",
        mode: AcceptanceMode::PrefixCheckpoint,
        fixtures: FIXTURES,
        features: FEATURES,
        parity: PARITY,
        perf: None,
        optimization_ids,
    }
}

#[test]
fn optimization_inventory_parser_finds_known_ids() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;

    assert!(known.contains("OPT-COM-001"));
    assert!(known.contains("OPT-OPEN-003"));
    assert!(known.contains("OPT-ZK-001"));
    Ok(())
}

#[test]
fn frontier_optimization_ids_must_exist_in_inventory() -> Result<(), String> {
    let known = KnownOptimizationIds::parse_inventory(INVENTORY).map_err(|e| e.to_string())?;

    validate_frontier_optimization_ids(frontier(&["OPT-COM-001"]), &known)
        .map_err(|e| e.to_string())?;
    assert!(validate_frontier_optimization_ids(frontier(&["OPT-NOPE-000"]), &known).is_err());
    Ok(())
}
