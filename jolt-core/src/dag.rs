use crate::{
    field::JoltField, subprotocols::sumcheck::BatchableSumcheckInstance,
    utils::transcript::Transcript,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
}

pub trait StageContributor<F: JoltField, ProofTranscript: Transcript, SM> {
    fn stage(&self) -> Stage;
    
    /// Create prover instances for this stage
    fn prover_instances(
        &self,
        state_manager: &SM,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>>;
    
    /// Create verifier instances for this stage
    /// The state manager holds all proofs and verification data
    fn verifier_instances(
        &self,
        state_manager: &SM,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>>;
}

/// Registry that collects all StageContributors
///
/// # Example Usage
/// ```rust,ignore
/// // Registration Phase
/// let mut registry = StageRegistry::new();
///
/// // Register all stage contributors
/// registry.register(Box::new(SpartanStage2::new()));
/// // Later: registry.register(Box::new(RAMStage1::new()));
/// // Later: registry.register(Box::new(RegistersStage2::new()));
///
/// // Execution Phase (in the state manager)
/// // Get all stages in order
/// let stages = registry.get_stages(); // Returns [Stage1, Stage2, ...]
///
/// // Prover: Execute stages in order
/// for stage in stages {
///     // Get all prover sumcheck instances for this stage
///     let instances = registry.get_stage_prover_instances(stage, &state_manager);
///     
///     // Execute all instances in parallel
///     let results = instances.par_iter_mut()
///         .map(|instance| instance.prove_single(transcript))
///         .collect();
/// }
/// 
/// // Verifier: Verify stages in order
/// for stage in stages {
///     // Get all verifier sumcheck instances for this stage
///     // State manager holds all proofs and verification data
///     let verifier_instances = registry.get_stage_verifier_instances(stage, &state_manager);
///     
///     // Verify all instances - can be done in parallel
///     let results: Result<Vec<_>, _> = verifier_instances.par_iter()
///         .map(|instance| {
///             // Extract the appropriate proof from state_manager
///             let proof = state_manager.get_proof_for(instance);
///             instance.verify_single(&proof, transcript)
///         })
///         .collect();
/// }
/// ```
pub struct StageRegistry<F: JoltField, ProofTranscript: Transcript, SM> {
    contributors: Vec<Box<dyn StageContributor<F, ProofTranscript, SM>>>,
}

impl<F: JoltField, ProofTranscript: Transcript, SM> StageRegistry<F, ProofTranscript, SM> {
    pub fn new() -> Self {
        Self {
            contributors: Vec::new(),
        }
    }

    /// Register a new StageContributor
    pub fn register(&mut self, contributor: Box<dyn StageContributor<F, ProofTranscript, SM>>) {
        self.contributors.push(contributor);
    }

    /// Get all prover sumcheck instances for a specific stage
    pub fn get_stage_prover_instances(
        &self,
        stage: Stage,
        state_manager: &SM,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        self.contributors
            .iter()
            .filter(|contributor| contributor.stage() == stage)
            .flat_map(|contributor| contributor.prover_instances(state_manager))
            .collect()
    }

    /// Get all verifier sumcheck instances for a specific stage
    pub fn get_stage_verifier_instances(
        &self,
        stage: Stage,
        state_manager: &SM,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        self.contributors
            .iter()
            .filter(|contributor| contributor.stage() == stage)
            .flat_map(|contributor| contributor.verifier_instances(state_manager))
            .collect()
    }

    /// Get all stages that have contributors
    pub fn get_stages(&self) -> Vec<Stage> {
        let mut stages: Vec<Stage> = self
            .contributors
            .iter()
            .map(|contributor| contributor.stage())
            .collect();
        stages.sort_by_key(|s| match s {
            Stage::Stage1 => 1,
            Stage::Stage2 => 2,
            Stage::Stage3 => 3,
            Stage::Stage4 => 4,
            Stage::Stage5 => 5,
        });
        stages.dedup();
        stages
    }
}
