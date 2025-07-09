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

        // Stage 2
        self.execute_stage(|impl_, sm| impl_.stage2_prover_instances(sm));

        // Stage 3
        self.execute_stage(|impl_, sm| impl_.stage3_prover_instances(sm));

        // Stage 4
        self.execute_stage(|impl_, sm| impl_.stage4_prover_instances(sm));

        // Stage 5
        self.execute_stage(|impl_, sm| impl_.stage5_prover_instances(sm));
    }

    /// Execute a single stage
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

    pub fn verify(&mut self) -> Result<(), crate::utils::errors::ProofVerifyError> {
        // Stage 1 verification - Verify the outer Spartan proof

        let stage1_proofs = {
            let proofs_guard = self.state_manager.proofs.lock().unwrap();
            if let Some(crate::dag::state_manager::ProofData::Spartan(spartan_proof)) = 
                proofs_guard.get(&crate::dag::state_manager::ProofKeys::SpartanOuterSumcheck) {
                // Extract the necessary data from the Spartan proof for verification
                vec![(
                    spartan_proof.outer_sumcheck_proof.clone(),
                    vec![], // The opening point will be filled by stage1_verify
                    [spartan_proof.outer_sumcheck_claims.0, 
                     spartan_proof.outer_sumcheck_claims.1, 
                     spartan_proof.outer_sumcheck_claims.2],
                )]
            } else {
                return Err(crate::utils::errors::ProofVerifyError::InternalError);
            }
        }; // MutexGuard is dropped here
        
        // Now we can mutably borrow state_manager
        for impl_ in &self.registry {
            let results = impl_.stage1_verify(&stage1_proofs, &mut self.state_manager, &mut self.transcript)?;
            // Process results if needed
            for (opening_point, claims) in results {
                // Store the opening point and claims for subsequent stages
                // This will be used in stages 2-5
                println!("Stage 1 verified with opening point: {:?}, claims: {:?}", opening_point, claims);
            }
        }
        
        // TODO: Implement verification for stages 2-5
        
        Ok(())
    }
}