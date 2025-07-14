use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::{ProofData, ProofKeys, Proofs, StateManager};
use crate::field::JoltField;
use crate::jolt::vm::instruction_lookups::LookupsDag;
use crate::jolt::vm::registers::RegistersDag;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::ALL_COMMITTED_POLYNOMIALS;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::spartan::SpartanDag;
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck};
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
pub struct JoltDAG<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
{
    prover_state_manager: Option<StateManager<'a, F, ProofTranscript, PCS>>,
    verifier_state_manager: Option<StateManager<'a, F, ProofTranscript, PCS>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltDagProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript
{
    pub verifier_preprocessing: crate::jolt::vm::JoltVerifierPreprocessing<F, PCS>,
    pub program_io: tracer::JoltDevice,
    pub trace_length: usize,
    pub sumcheck_switch_index_registers: usize,
    pub sumcheck_switch_index_ram: usize,
    
    pub commitments: JoltCommitments<F, PCS>,
    pub dag_proofs: Proofs<F, ProofTranscript>,
    pub claims: crate::dag::state_manager::Claims<F>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    JoltDAG<'a, F, ProofTranscript, PCS>
{
    pub fn new() -> Self {
        Self {
            prover_state_manager: None,
            verifier_state_manager: None,
        }
    }

    pub fn new_prover(prover_state_manager: StateManager<'a, F, ProofTranscript, PCS>) -> Self {
        Self {
            prover_state_manager: Some(prover_state_manager),
            verifier_state_manager: None,
        }
    }

    pub fn new_verifier(verifier_state_manager: StateManager<'a, F, ProofTranscript, PCS>) -> Self {
        Self {
            prover_state_manager: None,
            verifier_state_manager: Some(verifier_state_manager),
        }
    }

    pub fn prove<const WORD_SIZE: usize>(&mut self) -> Result<JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>, anyhow::Error> {
        // Initialize DoryGlobals first
        let _guard = {
            let prover_state_manager = self.prover_state_manager.as_mut()
                .ok_or_else(|| anyhow::anyhow!("Prover state not initialized"))?;
            
            let (preprocessing, trace, _, _) = prover_state_manager.get_prover_data();
            let trace_length = trace.len();
            let padded_trace_length = trace_length.next_power_of_two();

            // Calculate K for DoryGlobals initialization
            let ram_addresses: Vec<_> = trace
                .par_iter()
                .map(|cycle| {
                    crate::jolt::vm::ram::remap_address(
                        cycle.ram_access().address() as u64,
                        &preprocessing.shared.memory_layout,
                    ) as usize
                })
                .collect();
            let ram_K = ram_addresses.par_iter().max().unwrap().next_power_of_two();

            let K = [
                preprocessing.shared.bytecode.code_size,
                ram_K,
                1 << 16, // K for instruction lookups
            ]
            .into_iter()
            .max()
            .unwrap();

            crate::poly::commitment::dory::DoryGlobals::initialize(K, padded_trace_length)
        };

        // Generate and commit to all witness polynomials
        {
            let prover_state_manager = self.prover_state_manager.as_mut()
                .ok_or_else(|| anyhow::anyhow!("Prover state not initialized"))?;
            let (preprocessing, trace, _, _) = prover_state_manager.get_prover_data();

            let committed_polys: Vec<_> = ALL_COMMITTED_POLYNOMIALS
                .par_iter()
                .map(|poly| poly.generate_witness(preprocessing, trace))
                .collect();

            let commitments: Vec<_> = committed_polys
                .par_iter()
                .map(|poly| PCS::commit(poly, &preprocessing.generators))
                .collect();

            let jolt_commitments = JoltCommitments {
                commitments: commitments.clone(),
            };

            prover_state_manager.set_commitments(jolt_commitments);
        }

        // Now work with the state manager
        let prover_state_manager = self.prover_state_manager.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Prover state not initialized"))?;

        // Append commitments to transcript
        let commitments = prover_state_manager.get_commitments();
        let transcript = prover_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Stage 1:
        let (_, trace, _, _) = prover_state_manager.get_prover_data();
        let padded_trace_length = trace.len().next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::<F>::default();
        let mut registers_dag = RegistersDag::default();
        spartan_dag.stage1_prove(prover_state_manager)?;

        // Stage 2:
        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_prover_instances(prover_state_manager))
            .chain(registers_dag.stage2_prover_instances(prover_state_manager))
            .chain(lookups_dag.stage2_prover_instances(prover_state_manager))
            .collect();
        let stage2_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
            .collect();

        let transcript = prover_state_manager.get_transcript();
        let (stage2_proof, _r_stage2) =
            BatchedSumcheck::prove(stage2_instances_mut, &mut *transcript.borrow_mut());

        prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2Sumcheck,
            ProofData::BatchableSumcheckData(stage2_proof),
        );

        let accumulator = prover_state_manager.get_prover_accumulator();
        for instance in stage2_instances.iter_mut() {
            instance.cache_openings_prover(Some(accumulator.clone()));
        }

        // Stage 3:
        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_prover_instances(prover_state_manager))
            .chain(registers_dag.stage3_prover_instances(prover_state_manager))
            .chain(lookups_dag.stage3_prover_instances(prover_state_manager))
            .collect();
        let stage3_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
            .collect();

        let (stage3_proof, _r_stage3) =
            BatchedSumcheck::prove(stage3_instances_mut, &mut *transcript.borrow_mut());

        prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::BatchableSumcheckData(stage3_proof),
        );

        let accumulator = prover_state_manager.get_prover_accumulator();
        for instance in stage3_instances.iter_mut() {
            instance.cache_openings_prover(Some(accumulator.clone()));
        }

        // Convert the state manager to a proof and return it
        Ok(JoltDagProof::from(&*prover_state_manager))
    }

    pub fn verify<const WORD_SIZE: usize>(&mut self, proof: JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>) -> Result<(), anyhow::Error> {
        // Convert proof to verifier state manager
        let verifier_preprocessing = self.verifier_state_manager.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Verifier state not initialized"))?
            .get_verifier_data().0;
        
        let verifier_state_manager_tuple = (proof, verifier_preprocessing);
        self.verifier_state_manager = Some(verifier_state_manager_tuple.into());
        let verifier_state_manager = self.verifier_state_manager.as_mut().unwrap();
        // Append commitments to transcript
        let commitments = verifier_state_manager.get_commitments();
        let transcript = verifier_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Stage 1:
        let (_, _, trace_length) = verifier_state_manager.get_verifier_data();
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::<F>::default();
        let mut registers_dag = RegistersDag::default();
        spartan_dag.stage1_verify(verifier_state_manager)?;

        // Stage 2:
        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(verifier_state_manager))
            .chain(registers_dag.stage2_verifier_instances(verifier_state_manager))
            .chain(lookups_dag.stage2_verifier_instances(verifier_state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn BatchableSumcheckInstance<F>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn BatchableSumcheckInstance<F>)
            .collect();

        let proofs = verifier_state_manager.proofs.borrow();
        let stage2_proof_data = proofs
            .get(&ProofKeys::Stage2Sumcheck)
            .expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = verifier_state_manager.get_transcript();
        let r_stage2 = BatchedSumcheck::verify(
            stage2_proof,
            stage2_instances_ref,
            &mut *transcript.borrow_mut(),
        )?;

        drop(proofs);

        let accumulator = verifier_state_manager.get_verifier_accumulator();
        for instance in stage2_instances.iter_mut() {
            instance.cache_openings_verifier(Some(accumulator.clone()), Some(&r_stage2));
        }

        // Stage 3:
        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(verifier_state_manager))
            .chain(registers_dag.stage3_verifier_instances(verifier_state_manager))
            .chain(lookups_dag.stage3_verifier_instances(verifier_state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn BatchableSumcheckInstance<F>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn BatchableSumcheckInstance<F>)
            .collect();

        let proofs = verifier_state_manager.proofs.borrow();
        let stage3_proof_data = proofs
            .get(&ProofKeys::Stage3Sumcheck)
            .expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let r_stage3 = BatchedSumcheck::verify(
            stage3_proof,
            stage3_instances_ref,
            &mut *transcript.borrow_mut(),
        )?;

        let accumulator = verifier_state_manager.get_verifier_accumulator();
        for instance in stage3_instances.iter_mut() {
            instance.cache_openings_verifier(Some(accumulator.clone()), Some(&r_stage3));
        }

        Ok(())
    }


}
