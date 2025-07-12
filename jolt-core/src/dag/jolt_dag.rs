use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::field::JoltField;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::ALL_COMMITTED_POLYNOMIALS;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::OpeningPoint;
use crate::poly::opening_proof::LITTLE_ENDIAN;
use crate::r1cs::spartan::SpartanDag;
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck};
use crate::utils::transcript::Transcript;
use rayon::prelude::*;
pub struct JoltDAG<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
{
    prover_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    verifier_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    JoltDAG<'a, F, ProofTranscript, PCS>
{
    pub fn new(
        prover_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        verifier_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Self {
        Self {
            prover_state_manager,
            verifier_state_manager,
        }
    }

    pub fn prove(&mut self) -> Result<(), anyhow::Error> {
        // Generate and commit to all witness polynomials
        self.generate_and_commit_polynomials()?;

        // Append commitments to transcript
        let commitments = self.prover_state_manager.get_commitments();
        let transcript = self.prover_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Stage 1:
        let spartan_dag = SpartanDag::default();
        spartan_dag.stage1_prove(&mut self.prover_state_manager)?;

        // // Stage 2: Collect instances and prove
        // let mut stage2_instances =
        //     spartan_dag.stage2_prover_instances(&mut self.prover_state_manager);
        // let stage2_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage2_instances
        //     .iter_mut()
        //     .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
        //     .collect();

        // let transcript = self.prover_state_manager.get_transcript();
        // let (stage2_proof, _r_stage2) =
        //     BatchedSumcheck::prove(stage2_instances_mut, &mut *transcript.borrow_mut());

        // self.prover_state_manager.proofs.borrow_mut().insert(
        //     ProofKeys::SpartanStage2Sumcheck,
        //     ProofData::SpartanSumcheck(stage2_proof),
        // );

        // // Cache openings for each stage 2 instance
        // let accumulator = self.prover_state_manager.get_prover_accumulator();
        // for instance in stage2_instances.iter_mut() {
        //     instance.cache_openings_prover(Some(accumulator.clone()));
        // }

        // // Stage 3: Collect instances and prove
        // let mut stage3_instances =
        //     spartan_dag.stage3_prover_instances(&mut self.prover_state_manager);
        // let stage3_instances_mut: Vec<&mut dyn BatchableSumcheckInstance<F>> = stage3_instances
        //     .iter_mut()
        //     .map(|instance| &mut **instance as &mut dyn BatchableSumcheckInstance<F>)
        //     .collect();

        // let (stage3_proof, _r_stage3) =
        //     BatchedSumcheck::prove(stage3_instances_mut, &mut *transcript.borrow_mut());

        // self.prover_state_manager.proofs.borrow_mut().insert(
        //     ProofKeys::SpartanStage3Sumcheck,
        //     ProofData::SpartanSumcheck(stage3_proof),
        // );

        // // Cache openings for each stage 3 instance
        // let accumulator = self.prover_state_manager.get_prover_accumulator();
        // for instance in stage3_instances.iter_mut() {
        //     instance.cache_openings_prover(Some(accumulator.clone()));
        // }

        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), anyhow::Error> {
        // Append commitments to transcript
        let commitments = self.verifier_state_manager.get_commitments();
        let transcript = self.verifier_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Receive openings from prover's accumulator
        self.receive_claims()?;

        let spartan_dag = SpartanDag::default();
        spartan_dag.stage1_verify(&mut self.verifier_state_manager)?;

        // // Stage 2: Collect instances and verify
        // let stage2_instances =
        //     spartan_dag.stage2_verifier_instances(&mut self.verifier_state_manager);
        // let stage2_instances_ref: Vec<&dyn BatchableSumcheckInstance<F>> = stage2_instances
        //     .iter()
        //     .map(|instance| &**instance as &dyn BatchableSumcheckInstance<F>)
        //     .collect();

        // // Retrieve stage 2 proof from state manager
        // let proofs = self.verifier_state_manager.proofs.borrow();
        // let stage2_proof_data = proofs
        //     .get(&ProofKeys::SpartanStage2Sumcheck)
        //     .expect("Stage 2 sumcheck proof not found");
        // let stage2_proof = match stage2_proof_data {
        //     ProofData::SpartanSumcheck(proof) => proof,
        //     _ => panic!("Invalid proof type for stage 2"),
        // };

        // let transcript = self.verifier_state_manager.get_transcript();
        // let _r_stage2 = BatchedSumcheck::verify(
        //     stage2_proof,
        //     stage2_instances_ref,
        //     &mut *transcript.borrow_mut(),
        // )?;

        // drop(proofs); // Drop the borrow before borrowing mutably again

        // // Stage 3: Collect instances and verify
        // let stage3_instances =
        //     spartan_dag.stage3_verifier_instances(&mut self.verifier_state_manager);
        // let stage3_instances_ref: Vec<&dyn BatchableSumcheckInstance<F>> = stage3_instances
        //     .iter()
        //     .map(|instance| &**instance as &dyn BatchableSumcheckInstance<F>)
        //     .collect();

        // // Retrieve stage 3 proof from state manager
        // let proofs = self.verifier_state_manager.proofs.borrow();
        // let stage3_proof_data = proofs
        //     .get(&ProofKeys::SpartanStage3Sumcheck)
        //     .expect("Stage 3 sumcheck proof not found");
        // let stage3_proof = match stage3_proof_data {
        //     ProofData::SpartanSumcheck(proof) => proof,
        //     _ => panic!("Invalid proof type for stage 3"),
        // };

        // let _r_stage3 = BatchedSumcheck::verify(
        //     stage3_proof,
        //     stage3_instances_ref,
        //     &mut *transcript.borrow_mut(),
        // )?;

        Ok(())
    }

    fn receive_claims(&mut self) -> Result<(), anyhow::Error> {
        let prover_accumulator = self.prover_state_manager.get_prover_accumulator();
        let prover_openings = prover_accumulator.borrow().evaluation_openings().clone();

        let verifier_accumulator = self.verifier_state_manager.get_verifier_accumulator();
        let mut verifier_accumulator_mut = verifier_accumulator.borrow_mut();

        // Copy all the opening evaluations from prover to verifier
        // Keep the F values but make the OpeningPoint just empty Vec
        for (key, (_, value)) in prover_openings.iter() {
            let empty_point = OpeningPoint::<{ LITTLE_ENDIAN }, F>::new(vec![]);
            verifier_accumulator_mut
                .evaluation_openings_mut()
                .insert(*key, (empty_point, *value));
        }

        Ok(())
    }

    fn generate_and_commit_polynomials(&mut self) -> Result<(), anyhow::Error> {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            self.prover_state_manager.get_prover_data();

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
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();

        let _guard = crate::poly::commitment::dory::DoryGlobals::initialize(K, padded_trace_length);

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

        self.prover_state_manager.set_commitments(jolt_commitments);

        Ok(())
    }
}
