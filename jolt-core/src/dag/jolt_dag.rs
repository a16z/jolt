use crate::dag::stage::{StagedSumcheck, SumcheckStages};
use crate::dag::state_manager::{ProofData, ProofKeys, Proofs, StateManager};
use crate::field::JoltField;
use crate::jolt::vm::instruction_lookups::LookupsDag;
use crate::jolt::vm::ram::RamDag;
use crate::jolt::vm::registers::RegistersDag;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::ALL_COMMITTED_POLYNOMIALS;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::r1cs::spartan::SpartanDag;
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck};
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::sync::Arc;
#[derive(Default)]
pub struct JoltDAG;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltDagProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    pub verifier_preprocessing: Arc<crate::jolt::vm::JoltVerifierPreprocessing<F, PCS>>,
    pub program_io: tracer::JoltDevice,
    pub trace_length: usize,
    pub sumcheck_switch_index_registers: usize,
    pub sumcheck_switch_index_ram: usize,

    pub commitments: JoltCommitments<F, PCS>,
    pub dag_proofs: Proofs<F, ProofTranscript>,
    pub claims: crate::dag::state_manager::Claims<F>,
}

impl JoltDAG {
    pub fn prove<
        'a,
        const WORD_SIZE: usize,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        prover_state_manager: &mut StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>, anyhow::Error> {
        let _guard = {
            let (preprocessing, trace, _, _) = prover_state_manager.get_prover_data();
            let trace_length = trace.len();
            let padded_trace_length = trace_length.next_power_of_two();

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

            let _guard = DoryGlobals::initialize(K, padded_trace_length);

            // HACK
            prover_state_manager
                .proofs
                .borrow_mut()
                .insert(ProofKeys::RamK, ProofData::RamK(ram_K));

            _guard
        };

        // Generate and commit to all witness polynomials
        {
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
        let mut ram_dag = RamDag::new_prover(prover_state_manager);
        spartan_dag.stage1_prove(prover_state_manager)?;

        // Stage 2:
        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_prover_instances(prover_state_manager))
            .chain(registers_dag.stage2_prover_instances(prover_state_manager))
            .chain(lookups_dag.stage2_prover_instances(prover_state_manager))
            .chain(ram_dag.stage2_prover_instances(prover_state_manager))
            .collect();
        let stage2_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
            .collect();

        let transcript = prover_state_manager.get_transcript();
        let (stage2_proof, r_stage2) =
            BatchedSumcheck::prove(stage2_instances_mut, &mut *transcript.borrow_mut());

        prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2Sumcheck,
            ProofData::BatchableSumcheckData(stage2_proof),
        );

        let stage2_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        let accumulator = prover_state_manager.get_prover_accumulator();
        BatchedSumcheck::cache_openings(stage2_instances_mut, Some(accumulator.clone()), &r_stage2);

        // Stage 3:
        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_prover_instances(prover_state_manager))
            .chain(registers_dag.stage3_prover_instances(prover_state_manager))
            .chain(lookups_dag.stage3_prover_instances(prover_state_manager))
            .chain(ram_dag.stage3_prover_instances(prover_state_manager))
            .collect();
        let stage3_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
            .collect();

        let (stage3_proof, r_stage3) =
            BatchedSumcheck::prove(stage3_instances_mut, &mut *transcript.borrow_mut());

        prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::BatchableSumcheckData(stage3_proof),
        );

        let stage3_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        let accumulator = prover_state_manager.get_prover_accumulator();
        BatchedSumcheck::cache_openings(stage3_instances_mut, Some(accumulator.clone()), &r_stage3);

        // Stage 4:
        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_prover_instances(prover_state_manager))
            .collect();
        let stage4_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
            .collect();

        let (stage4_proof, r_stage4) =
            BatchedSumcheck::prove(stage4_instances_mut, &mut *transcript.borrow_mut());

        prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage4Sumcheck,
            ProofData::BatchableSumcheckData(stage4_proof),
        );

        let stage4_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        BatchedSumcheck::cache_openings(stage4_instances_mut, Some(accumulator.clone()), &r_stage4);

        // Convert the state manager to a proof
        Ok(JoltDagProof::from(&*prover_state_manager))
    }

    pub fn verify<
        const WORD_SIZE: usize,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        proof: JoltDagProof<WORD_SIZE, F, PCS, ProofTranscript>,
    ) -> Result<(), anyhow::Error> {
        let mut verifier_state_manager: StateManager<F, ProofTranscript, PCS> = proof.into();
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
        let mut ram_dag = RamDag::new_verifier(&verifier_state_manager);
        spartan_dag.stage1_verify(&mut verifier_state_manager)?;

        // Stage 2:
        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut verifier_state_manager))
            .chain(registers_dag.stage2_verifier_instances(&mut verifier_state_manager))
            .chain(lookups_dag.stage2_verifier_instances(&mut verifier_state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut verifier_state_manager))
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

        let stage2_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        let accumulator = verifier_state_manager.get_verifier_accumulator();
        BatchedSumcheck::cache_claims(stage2_instances_mut, Some(accumulator.clone()), &r_stage2);

        // Stage 3:
        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(&mut verifier_state_manager))
            .chain(registers_dag.stage3_verifier_instances(&mut verifier_state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut verifier_state_manager))
            .chain(ram_dag.stage3_verifier_instances(&mut verifier_state_manager))
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

        drop(proofs);

        let stage3_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        let accumulator = verifier_state_manager.get_verifier_accumulator();
        BatchedSumcheck::cache_claims(stage3_instances_mut, Some(accumulator.clone()), &r_stage3);

        // Stage 4:
        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_verifier_instances(&mut verifier_state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn BatchableSumcheckInstance<F>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn BatchableSumcheckInstance<F>)
            .collect();

        let proofs = verifier_state_manager.proofs.borrow();
        let stage4_proof_data = proofs
            .get(&ProofKeys::Stage4Sumcheck)
            .expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let r_stage4 = BatchedSumcheck::verify(
            stage4_proof,
            stage4_instances_ref,
            &mut *transcript.borrow_mut(),
        )?;

        let stage4_instances_mut: Vec<&mut dyn StagedSumcheck<F, PCS>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn StagedSumcheck<F, PCS>)
            .collect();
        let accumulator = verifier_state_manager.get_verifier_accumulator();
        BatchedSumcheck::cache_claims(stage4_instances_mut, Some(accumulator), &r_stage4);

        Ok(())
    }
}
