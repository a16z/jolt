use crate::jolt;

use super::constraints::R1CSBuilder; 

use common::{constants::RAM_START_ADDRESS, field_conversion::{ark_to_spartan_unsafe, spartan_to_ark_unsafe}, path::JoltPaths};
use itertools::Itertools;
use spartan2::{
  errors::SpartanError, provider::{
      bn256_grumpkin::bn256::{self, Point as SpartanG1, Scalar as Spartan2Fr},
      hyrax_pc::{HyraxCommitment as SpartanHyraxCommitment, HyraxCommitmentKey, HyraxEvaluationEngine as SpartanHyraxEE},
  }, r1cs::R1CSShape, spartan::upsnark::R1CSSNARK, traits::{
    commitment::CommitmentEngineTrait, upsnark::PrecommittedSNARKTrait, Group
  }, VerifierKey, SNARK
};
use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError
};
use ff::PrimeField;
use rayon::prelude::*;

/// Reorder and drop first element [[a1, b1, c1], [a2, b2, c2]] => [[a2], [b2], [c2]]
#[tracing::instrument(skip_all)]
fn reassemble_segments<F: PrimeField>(jolt_witnesses: Vec<Vec<F>>) -> Vec<Vec<F>> {
  let trace_len = jolt_witnesses.len();
  let num_variables = jolt_witnesses[0].len();
  let mut result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_variables - 1]; // ignore 1 

  result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][variable_idx]; // NOTE: 1 is at the end!
    }
  });

  result 
}

/// Reorder and drop first element [[a1, b1, c1], [a2, b2, c2]] => [[a2], [b2], [c2]]
/// Works 
#[tracing::instrument(skip_all)]
fn reassemble_segments_partial<F: PrimeField>(jolt_witnesses: Vec<Vec<F>>, num_front: usize, num_back: usize) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
  let trace_len = jolt_witnesses.len();
  let total_length = jolt_witnesses[0].len();
  let mut front_result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_front]; 
  let mut back_result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_back]; 

  // state starts at the beginning
  front_result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][variable_idx]; // NOTE: 1 is at the end!
    }
  });

  // [aux || 1] is the end, and we skip the 1
  back_result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][(total_length-1-num_back) + variable_idx]; 
    }
  });

  (front_result, back_result)
}

#[derive(Clone, Debug, Default)]
pub struct JoltCircuit<F: ff::PrimeField<Repr=[u8; 32]>> {
  num_steps: usize,
  inputs: Vec<Vec<F>>,
  // prog_a_rw: Vec<F>, 
  // prog_v_rw: Vec<F>, 
  // memreg_a_rw: Vec<F>, 
  // memreg_v_reads: Vec<F>, 
  // memreg_v_writes: Vec<F>, 
  // chunks_x: Vec<F>, 
  // chunks_y: Vec<F>, 
  // chunks_query: Vec<F>, 
  // lookup_outputs: Vec<F>,  
  // op_flags: Vec<F>,  
}

// This is a placeholder trait to satisfy Spartan's requirements. 
impl<F: ff::PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltCircuit<F> {
  #[tracing::instrument(skip_all, name = "JoltCircuit::synthesize")]
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<F: ff::PrimeField<Repr=[u8;32]>> JoltCircuit<F> {
  pub fn new_from_inputs(num_steps: usize, inputs: Vec<Vec<F>>) -> Self {
    JoltCircuit{
      num_steps: num_steps,
      inputs: inputs,
    }
  }

  #[tracing::instrument(name = "JoltCircuit::get_witnesses_by_step", skip_all)]
  fn synthesize_witnesses(&self, r1cs_builder: &R1CSBuilder<F>) -> Result<Vec<Vec<F>>, SynthesisError> {
    let TRACE_LEN = self.inputs[0].len();
    let NUM_STEPS = self.num_steps;

    let ABC_lens = r1cs_builder.ABCz_lens;
    let compute_witness_span = tracing::span!(tracing::Level::INFO, "compute_witness_loop");
    let _compute_witness_guard = compute_witness_span.enter();
    let jolt_witnesses: Vec<Vec<F>> = (0..NUM_STEPS).into_par_iter().map(|i| {
      let mut step_inputs: Vec<Vec<F>> = self.inputs.iter().map(|v| {
        v.iter()
         .skip(i)
         .step_by(TRACE_LEN)
         .cloned()
         .collect_vec()
      }).collect_vec();

      let program_counter = if i > 0 && self.inputs[0][i] == F::from(0) {
        F::from(0)
      } else {
          self.inputs[0][i] * F::from(4u64) + F::from(RAM_START_ADDRESS)
      };

      // For the non-circom version, we need to pre-prend the inputs.  
      step_inputs.insert(0, vec![F::from(i as u64), program_counter]);
      let step_inputs_flat = step_inputs.into_iter().flatten().collect::<Vec<_>>();

      let step_instance = R1CSBuilder::<F>::get_matrices(Some(step_inputs_flat), Some(ABC_lens)).unwrap(); 
      step_instance.z.unwrap()
    }).collect();
    drop(_compute_witness_guard);

    Ok(jolt_witnesses)
  }

  #[tracing::instrument(name = "synthesize_witness_segments", skip_all)]
  pub fn synthesize_state_aux_segments(&self, r1cs_builder: &R1CSBuilder<F>, num_state: usize, num_aux: usize) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>), SynthesisError> {
    let jolt_witnesses = self.synthesize_witnesses(&r1cs_builder)?;
    Ok(reassemble_segments_partial(jolt_witnesses, num_state, num_aux))
  }
}


pub struct R1CSProof  {
  proof: SNARK<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>, JoltCircuit<Spartan2Fr>>,
  vk: VerifierKey<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>>,
}

impl R1CSProof {
  #[tracing::instrument(skip_all, name = "R1CSProof::prove")]
  pub fn prove<ArkF: ark_ff::PrimeField> (
      _W: usize, 
      _C: usize, 
      TRACE_LEN: usize, 
      inputs_ark: Vec<Vec<ArkF>>, 
      generators: Vec<bn256::Affine>,
      jolt_commitments: &Vec<Vec<bn256::Affine>>,
  ) -> Result<Self, SpartanError> {
      type G1 = SpartanG1;
      type EE = SpartanHyraxEE<SpartanG1>;
      type S = spartan2::spartan::upsnark::R1CSSNARK<G1, EE>;
      type F = Spartan2Fr;

      let NUM_STEPS = TRACE_LEN;

      let span = tracing::span!(tracing::Level::TRACE, "convert_ark_to_spartan_fr");
      let _enter = span.enter();
      let inputs: Vec<Vec<Spartan2Fr>> = inputs_ark.into_par_iter().map(|vec| vec.into_par_iter().map(|ark_item| ark_to_spartan_unsafe::<ArkF, Spartan2Fr>(ark_item)).collect()).collect();
      drop(_enter);
      drop(span);

      let jolt_circuit = JoltCircuit::<F>::new_from_inputs(NUM_STEPS, inputs.clone());
      
      let jolt_shape = R1CSBuilder::<F>::get_matrices(None, None).unwrap(); 
      let constraints_F = jolt_shape.convert_to_field(); 
      let shape_single = R1CSShape::<G1> {
          A: constraints_F.0,
          B: constraints_F.1,
          C: constraints_F.2,
          num_cons: jolt_shape.num_constraints,
          num_vars: jolt_shape.num_aux, // shouldn't include 1 or IO 
          num_io: jolt_shape.num_inputs,
      };

      // Obtain public key 
      let hyrax_ck = HyraxCommitmentKey::<G1> {
          ck: spartan2::provider::pedersen::from_gens_bn256(generators)
      };

      // let w_segments_from_circuit = jolt_circuit.synthesize_witness_segments().unwrap();
      let (io_segments, aux_segments) = jolt_circuit.synthesize_state_aux_segments(&jolt_shape, 4, jolt_shape.num_internal).unwrap();

      let cloning_stuff_span = tracing::span!(tracing::Level::TRACE, "cloning_stuff");
      let _enter = cloning_stuff_span.enter();

      let inputs_segments: Vec<Vec<F>> = inputs.into_iter().flat_map(|input| {
        input.chunks(TRACE_LEN).map(|chunk| chunk.to_vec()).collect::<Vec<_>>()
      }).collect();

      let w_segments = io_segments.clone().into_iter()
        .chain(inputs_segments.iter().cloned())
        .chain(aux_segments.clone().into_iter())
        .collect::<Vec<_>>(); 

      drop(_enter);
      drop(cloning_stuff_span);

      // Commit to segments
      let commit_segments = |segments: Vec<Vec<F>>| -> Vec<_> {
        let span = tracing::span!(tracing::Level::TRACE, "commit_segments");
        let _g = span.enter();
        segments.into_par_iter().map(|segment| {
          <G1 as Group>::CE::commit(&hyrax_ck, &segment) 
        }).collect()
      };
      
      let io_comms = commit_segments(io_segments);
      let input_comms = jolt_commitments; 
      let aux_comms = commit_segments(aux_segments);

      let comm_w_vec = io_comms.into_iter()
      .chain(input_comms.iter().map(|comm| SpartanHyraxCommitment::from(comm.clone())))
      .chain(aux_comms.into_iter())
      .collect::<Vec<_>>();

      let (pk, vk) = SNARK::<G1, S, JoltCircuit<F>>::setup_precommitted(shape_single, NUM_STEPS, hyrax_ck).unwrap();

      SNARK::prove_precommitted(&pk, w_segments, comm_w_vec).map(|snark| Self {
        proof: snark,
        vk
      })
  }

  pub fn verify(&self) -> Result<(), SpartanError> {
    SNARK::verify_precommitted(&self.proof, &self.vk, &[])
  }
}
