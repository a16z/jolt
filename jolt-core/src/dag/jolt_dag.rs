use std::collections::HashMap;

use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::field::JoltField;
use crate::jolt::vm::bytecode::BytecodeDag;
use crate::jolt::vm::instruction_lookups::LookupsDag;
use crate::jolt::vm::ram::RamDag;
use crate::jolt::vm::registers::RegistersDag;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::AllCommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use crate::r1cs::spartan::SpartanDag;
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance};
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use anyhow::Context;
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
        // Initialize DoryGlobals at the beginning to keep it alive for the entire proof
        let (preprocessing, trace, _, _) = self.prover_state_manager.get_prover_data();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();

        // Calculate K for DoryGlobals initialization
        let ram_K = trace
            .par_iter()
            .map(|cycle| {
                crate::jolt::vm::ram::remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                ) as usize
            })
            .max()
            .unwrap()
            .next_power_of_two();
        let bytecode_d = preprocessing.shared.bytecode.d;

        // HACK
        self.prover_state_manager
            .proofs
            .borrow_mut()
            .insert(ProofKeys::RamK, ProofData::RamK(ram_K));

        println!("bytecode size: {}", preprocessing.shared.bytecode.code_size);
        let K = [
            preprocessing.shared.bytecode.code_size,
            // ram_K,
            1 << 8, // K for instruction lookups
        ]
        .into_iter()
        .max()
        .unwrap();

        let _guard = (
            DoryGlobals::initialize(K, padded_trace_length),
            AllCommittedPolynomials::initialize(ram_K, bytecode_d),
        );

        // Generate and commit to all witness polynomials
        self.generate_and_commit_polynomials()?;

        // Append commitments to transcript
        let commitments = self.prover_state_manager.get_commitments();
        let transcript = self.prover_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Stage 1:
        let span = tracing::span!(tracing::Level::INFO, "Stage 1 sumchecks");
        let _guard = span.enter();

        let (_, trace, _, _) = self.prover_state_manager.get_prover_data();
        let padded_trace_length = trace.len().next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_prover(&self.prover_state_manager);
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_prove(&mut self.prover_state_manager)
            .context("Stage 1")?;

        drop(_guard);
        drop(span);

        // Stage 2:
        let span = tracing::span!(tracing::Level::INFO, "Stage 2 sumchecks");
        let _guard = span.enter();

        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_prover_instances(&mut self.prover_state_manager))
            .chain(registers_dag.stage2_prover_instances(&mut self.prover_state_manager))
            .chain(ram_dag.stage2_prover_instances(&mut self.prover_state_manager))
            .collect();
        let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let transcript = self.prover_state_manager.get_transcript();
        let accumulator = self.prover_state_manager.get_prover_accumulator();
        let (stage2_proof, _r_stage2) = BatchedSumcheck::prove(
            stage2_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        self.prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2Sumcheck,
            ProofData::BatchableSumcheckData(stage2_proof),
        );

        drop(_guard);
        drop(span);

        // Stage 3:
        let span = tracing::span!(tracing::Level::INFO, "Stage 3 sumchecks");
        let _guard = span.enter();

        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_prover_instances(&mut self.prover_state_manager))
            .chain(registers_dag.stage3_prover_instances(&mut self.prover_state_manager))
            .chain(lookups_dag.stage3_prover_instances(&mut self.prover_state_manager))
            .chain(ram_dag.stage3_prover_instances(&mut self.prover_state_manager))
            .collect();
        let stage3_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let (stage3_proof, _r_stage3) = BatchedSumcheck::prove(
            stage3_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        self.prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::BatchableSumcheckData(stage3_proof),
        );

        drop(_guard);
        drop(span);

        // Stage 4:
        let span = tracing::span!(tracing::Level::INFO, "Stage 4 sumchecks");
        let _guard = span.enter();

        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_prover_instances(&mut self.prover_state_manager))
            .chain(bytecode_dag.stage4_prover_instances(&mut self.prover_state_manager))
            .collect();
        let stage4_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let (stage4_proof, _r_stage4) = BatchedSumcheck::prove(
            stage4_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        self.prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage4Sumcheck,
            ProofData::BatchableSumcheckData(stage4_proof),
        );

        drop(_guard);
        drop(span);

        // Batch-prove all openings
        let (_, trace, _, _) = self.prover_state_manager.get_prover_data();
        let mut polynomials_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            polynomials_map.insert(
                *polynomial,
                polynomial.generate_witness(preprocessing, trace),
            );
        }
        let opening_proof = accumulator.borrow_mut().reduce_and_prove(
            polynomials_map,
            &preprocessing.generators,
            &mut *transcript.borrow_mut(),
        );

        self.prover_state_manager.proofs.borrow_mut().insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(opening_proof),
        );

        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), anyhow::Error> {
        #[cfg(test)]
        {
            let prover_transcript = self.prover_state_manager.get_transcript().take();
            self.verifier_state_manager
                .transcript
                .borrow_mut()
                .compare_to(prover_transcript);

            let prover_opening_accumulator = self
                .prover_state_manager
                .get_prover_accumulator()
                .borrow()
                .clone();
            self.verifier_state_manager
                .get_verifier_accumulator()
                .borrow_mut()
                .compare_to(prover_opening_accumulator);
        }

        let ram_K = match self
            .verifier_state_manager
            .proofs
            .borrow()
            .get(&ProofKeys::RamK)
            .unwrap()
        {
            ProofData::RamK(ram_K) => *ram_K,
            _ => panic!("Unexpected ProofData"),
        };
        let bytecode_d = self
            .verifier_state_manager
            .get_verifier_data()
            .0
            .shared
            .bytecode
            .d;
        let _guard = AllCommittedPolynomials::initialize(ram_K, bytecode_d);

        // Append commitments to transcript
        let commitments = self.verifier_state_manager.get_commitments();
        let transcript = self.verifier_state_manager.get_transcript();
        for commitment in commitments.commitments.iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Receive opening claims from prover's accumulator
        self.receive_claims().context("Receive claims")?;

        // Stage 1:
        let (preprocessing, _, trace_length) = self.verifier_state_manager.get_verifier_data();
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_verifier(&self.verifier_state_manager);
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_verify(&mut self.verifier_state_manager)
            .context("Stage 1")?;

        // Stage 2:
        let stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut self.verifier_state_manager))
            .chain(registers_dag.stage2_verifier_instances(&mut self.verifier_state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut self.verifier_state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = self.verifier_state_manager.proofs.borrow();
        let stage2_proof_data = proofs
            .get(&ProofKeys::Stage2Sumcheck)
            .expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = self.verifier_state_manager.get_transcript();
        let opening_accumulator = self.verifier_state_manager.get_verifier_accumulator();
        let _r_stage2 = BatchedSumcheck::verify(
            stage2_proof,
            stage2_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 2")?;

        drop(proofs);

        // Stage 3:
        let stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(&mut self.verifier_state_manager))
            .chain(registers_dag.stage3_verifier_instances(&mut self.verifier_state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut self.verifier_state_manager))
            .chain(ram_dag.stage3_verifier_instances(&mut self.verifier_state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = self.verifier_state_manager.proofs.borrow();
        let stage3_proof_data = proofs
            .get(&ProofKeys::Stage3Sumcheck)
            .expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let _r_stage3 = BatchedSumcheck::verify(
            stage3_proof,
            stage3_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 3")?;

        drop(proofs);

        // Stage 4:
        let stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_verifier_instances(&mut self.verifier_state_manager))
            .chain(bytecode_dag.stage4_verifier_instances(&mut self.verifier_state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = self.verifier_state_manager.proofs.borrow();
        let stage4_proof_data = proofs
            .get(&ProofKeys::Stage4Sumcheck)
            .expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::BatchableSumcheckData(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let _r_stage4 = BatchedSumcheck::verify(
            stage4_proof,
            stage4_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 4")?;

        // Batch-prove all openings
        let batched_opening_proof = proofs
            .get(&ProofKeys::ReducedOpeningProof)
            .expect("Reduced opening proof not found");
        let batched_opening_proof = match batched_opening_proof {
            ProofData::ReducedOpeningProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let mut commitments_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            commitments_map.insert(
                *polynomial,
                commitments.commitments[polynomial.to_index()].clone(),
            );
        }
        let accumulator = self.verifier_state_manager.get_verifier_accumulator();
        accumulator
            .borrow_mut()
            .reduce_and_verify(
                &preprocessing.generators,
                &mut commitments_map,
                batched_opening_proof,
                &mut *transcript.borrow_mut(),
            )
            .context("Stage 5")?;

        Ok(())
    }

    fn receive_claims(&mut self) -> Result<(), anyhow::Error> {
        let prover_accumulator = self.prover_state_manager.get_prover_accumulator();
        let verifier_accumulator = self.verifier_state_manager.get_verifier_accumulator();

        // Copy only the claims from prover to verifier
        let prover_acc_borrow = prover_accumulator.borrow();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, F>::new(vec![]);
            verifier_acc_borrow
                .evaluation_openings_mut()
                .insert(*key, (empty_point, *value));
        }

        Ok(())
    }

    // Prover utility to commit to all the polynomials for the PCS
    #[tracing::instrument(skip_all)]
    fn generate_and_commit_polynomials(&mut self) -> Result<(), anyhow::Error> {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            self.prover_state_manager.get_prover_data();

        let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
            .par_bridge()
            .map(|poly| poly.generate_witness(preprocessing, trace))
            .collect();

        let commitments: Vec<_> = committed_polys
            .iter()
            .map(|poly| PCS::commit(poly, &preprocessing.generators))
            .collect();

        let jolt_commitments = JoltCommitments {
            commitments: commitments.clone(),
        };

        self.prover_state_manager.set_commitments(jolt_commitments);

        drop_in_background_thread(committed_polys);

        Ok(())
    }
}
