use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::utils::transcript::Transcript;

pub struct JoltDAG<'a, F: JoltField, ProofTranscript: Transcript> {
    state_manager: StateManager<'a, F, ProofTranscript>,
    transcript: ProofTranscript,
    registry: Vec<Box<dyn SumcheckStages<F, ProofTranscript> + 'a>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript> JoltDAG<'a, F, ProofTranscript> {
    pub fn new(state_manager: StateManager<'a, F, ProofTranscript>, transcript: ProofTranscript) -> Self {
        Self {
            state_manager,
            transcript,
            registry: Vec::new(),
        }
    }

    pub fn register(&mut self, stage_impl: Box<dyn SumcheckStages<F, ProofTranscript> + 'a>) {
        self.registry.push(stage_impl);
    }

    pub fn prove(&mut self) {
        // Stage 1 - Special case that returns proofs directly
        let mut stage1_results = Vec::new();
        for impl_ in &self.registry {
            stage1_results.extend(impl_.stage1_prove(&mut self.state_manager, &mut self.transcript));
        }

        // Stage 2 - Collect and prove
        self.execute_stage(|impl_, sm| impl_.stage2_prover_instances(sm));

        // Stage 3 - Collect and prove
        self.execute_stage(|impl_, sm| impl_.stage3_prover_instances(sm));

        // Stage 4 - Collect and prove
        self.execute_stage(|impl_, sm| impl_.stage4_prover_instances(sm));

        // Stage 5 - Collect and prove
        self.execute_stage(|impl_, sm| impl_.stage5_prover_instances(sm));
    }

    /// Helper function to execute a stage
    fn execute_stage<G>(&mut self, get_instances: G)
    where
        G: Fn(&dyn SumcheckStages<F, ProofTranscript>, &mut StateManager<F, ProofTranscript>) -> Vec<Box<dyn crate::subprotocols::sumcheck::BatchableSumcheckInstance<F, ProofTranscript>>>,
    {
        // Collect all instances for this stage
        let mut all_instances = Vec::new();
        for impl_ in &self.registry {
            let instances = get_instances(impl_.as_ref(), &mut self.state_manager);
            all_instances.extend(instances);
        }

        // Process them if there are any
        if !all_instances.is_empty() {
            // Create references manually to control lifetime
            let mut refs: Vec<&mut dyn crate::subprotocols::sumcheck::BatchableSumcheckInstance<F, ProofTranscript>> = Vec::new();
            for instance in &mut all_instances {
                refs.push(instance.as_mut());
            }
            
            let (_proof, _r) = BatchedSumcheck::prove(
                refs,
                &mut self.transcript,
            );
        }
    }

    pub fn verify(&mut self) {
        // Stage 1 verification
        let stage1_proofs = vec![]; // TODO: Get from state manager
        for impl_ in &self.registry {
            let _results = impl_.stage1_verify(&stage1_proofs, &mut self.state_manager, &mut self.transcript);
        }
        // TODO: Implement verification stages
    }
}