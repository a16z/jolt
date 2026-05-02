#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedStageRegistry {
    pub stages: Vec<&'static str>,
    pub prover_stage_count: usize,
    pub verifier_stage_count: usize,
}

pub fn generated_stage_registry() -> Option<GeneratedStageRegistry> {
    let prover_stages = jolt_prover::generated_stage_names().collect::<Vec<_>>();
    let verifier_stages = jolt_verifier::generated_stage_names().collect::<Vec<_>>();
    if prover_stages != verifier_stages {
        return None;
    }
    Some(GeneratedStageRegistry {
        prover_stage_count: prover_stages.len(),
        verifier_stage_count: verifier_stages.len(),
        stages: prover_stages,
    })
}

pub fn generated_stage_names() -> Vec<&'static str> {
    jolt_prover::generated_stage_names().collect()
}
