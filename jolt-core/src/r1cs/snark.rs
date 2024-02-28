use std::collections::HashMap;

use common::{path::JoltPaths, field_conversion::{ark_to_spartan_unsafe, ff_to_ruint, ff_to_ruints, ruint_to_ff, spartan_to_ark_unsafe}};
use spartan2::{
  errors::SpartanError,
  VerifierKey,
  traits::{
    snark::RelaxedR1CSSNARKTrait,
      upsnark::{PrecommittedSNARKTrait, UniformSNARKTrait},
      Group,
  },
  SNARK,
  provider::{
      bn256_grumpkin::bn256::{self, Scalar as Spartan2Fr, Point as SpartanG1},
      hyrax_pc::HyraxEvaluationEngine as SpartanHyraxEE,
  },
  spartan::upsnark::R1CSSNARK,
};
use bellpepper_core::{
  Circuit, ConstraintSystem, LinearCombination, SynthesisError, Variable, Index, num::AllocatedNum,
};
use ff::PrimeField;
use ruint::aliases::U256;
use circom_scotia::r1cs::CircomConfig;
use rayon::prelude::*;

use ark_ff::PrimeField as arkPrimeField;
use ark_ec::CurveGroup as arkGroup;
use ark_std::{Zero, One};

use crate::{poly::pedersen::PedersenGenerators, utils};

use spartan2::traits::commitment::CommitmentEngineTrait;

use crate::jolt;

const WTNS_GRAPH_BYTES: &[u8] = include_bytes!("./graph.bin");

const NUM_CHUNKS: usize = 4;
const NUM_FLAGS: usize = 17;
const NUM_SEGMENTS: usize = 12; // 11 + 1 for aux variables 

#[tracing::instrument(skip_all, name = "JoltCircuit::assemble_by_segments")]
fn reassemble_by_segments<F: PrimeField>(mut jolt_witnesses: Vec<Vec<F>>) -> Vec<F> {
  get_segments(jolt_witnesses).into_iter().flatten().collect()
}

fn get_segments<F: PrimeField>(mut jolt_witnesses: Vec<Vec<F>>) -> Vec<Vec<F>> {
  let mut result: Vec<Vec<F>> = vec![Vec::new(); jolt_witnesses[0].len()-1]; // ignore 1 at the start 
  // result[0] = vec![F::from(1)]; // start with [1]

  for witness in &mut jolt_witnesses {
    witness.remove(0);  // ignore 1
    for (i, w) in witness.iter().enumerate() {
        result[i].push(*w);
    }
  }

  result 
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

impl<F: ff::PrimeField<Repr=[u8;32]>> JoltCircuit<F> {
  pub fn new_from_inputs(W: usize, c: usize, num_steps: usize, PC_START_ADDR: F, inputs: Vec<Vec<F>>) -> Self {
    JoltCircuit{
      num_steps: num_steps,
      inputs: inputs,
    }
  }

  fn get_witnesses_by_step(self) -> Result<Vec<Vec<F>>, SynthesisError> {
    let r1cs_path = JoltPaths::r1cs_path();
    let wtns_path = JoltPaths::witness_generator_path();

    let cfg: CircomConfig<F> = CircomConfig::new(wtns_path, r1cs_path).unwrap();

    let variable_names: Vec<String> = vec![
      "prog_a_rw".to_string(), 
      "prog_v_rw".to_string(), 
      "memreg_a_rw".to_string(), 
      "memreg_v_reads".to_string(), 
      "memreg_v_writes".to_string(), 
      "chunks_x".to_string(), 
      "chunks_y".to_string(), 
      "chunks_query".to_string(), 
      "lookup_output".to_string(), 
      "op_flags".to_string(),
      "input_state".to_string()
    ];

    let TRACE_LEN = self.inputs[0].len();
    let NUM_STEPS = self.num_steps;

    // TODO(sragss / arasuarun): Current chunking strategy is a mess and unnecessary. Can be handled with better indexing.
    // for variable [v], step_inputs[v][j] is the variable input for step j
    let inputs_chunked : Vec<Vec<_>> = self.inputs
      .into_par_iter()
      .map(|inner_vec| inner_vec.chunks(inner_vec.len()/TRACE_LEN).map(|chunk| chunk.to_vec()).collect())
      .collect();

    let graph = witness::init_graph(WTNS_GRAPH_BYTES).unwrap();
    let wtns_buffer_size = witness::get_inputs_size(&graph);
    let wtns_mapping = witness::get_input_mapping(&variable_names, &graph);

    let compute_witness_span = tracing::span!(tracing::Level::INFO, "compute_witness_loop");
    let _compute_witness_guard = compute_witness_span.enter();
    let jolt_witnesses: Vec<Vec<F>> = (0..NUM_STEPS).into_par_iter().map(|i| {
      let mut step_inputs: Vec<Vec<ark_bn254::Fr>> = inputs_chunked.iter().map(|v| v[i].iter().cloned().map(spartan_to_ark_unsafe).collect()).collect();

      step_inputs.push(vec![ark_bn254::Fr::from(i as u64), spartan_to_ark_unsafe(inputs_chunked[0][i][0])]); // [step_counter, program_counter]

      let input_map: HashMap<String, Vec<ark_bn254::Fr>> = variable_names
        .iter()
        .zip(step_inputs.into_iter())
        .map(|(name, input)| (name.to_owned(), input))
        .collect();

      // TODO(sragss): Could reuse the inputs buffer between parallel chunks
      let mut inputs_buffer = vec![ark_bn254::Fr::zero(); wtns_buffer_size];
      inputs_buffer[0] = ark_bn254::Fr::one();
      witness::populate_inputs_fr(&input_map, &wtns_mapping, &mut inputs_buffer);
      let ark_jolt_witness = witness::graph::evaluate_fr(&graph.nodes, &inputs_buffer, &graph.signals);

      let jolt_witnesses = ark_jolt_witness.into_iter().map(ark_to_spartan_unsafe).collect::<Vec<_>>();

      jolt_witnesses
    }).collect();
    drop(_compute_witness_guard);

    Ok(jolt_witnesses)
  }


}


impl<F: PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltCircuit<F> {
  #[tracing::instrument(skip_all, name = "JoltCircuit::synthesize")]
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let jolt_witnesses = self.get_witnesses_by_step()?;

    let witness_variable_wise = reassemble_by_segments(jolt_witnesses);

    let allocate_vars_span = tracing::span!(tracing::Level::INFO, "allocate_vars");
    let _allocate_vars_guard = allocate_vars_span.enter();
    (0..witness_variable_wise.len()).for_each(|i| {
        let f = witness_variable_wise[i];
        // let _ = AllocatedNum::alloc(cs.namespace(|| format!("{}_{}", if i < cfg.r1cs.num_inputs { "public" } else { "aux" }, i)), || Ok(f)).unwrap();
        let _ = AllocatedNum::alloc(cs.namespace(|| format!("{}_{}", "aux", i)), || Ok(f)).unwrap();
    });
    drop(_allocate_vars_guard);

    utils::thread::drop_in_background_thread(witness_variable_wise);

    Ok(())
  }
}

#[derive(Clone, Debug, Default)]
pub struct JoltSkeleton<F: ff::PrimeField<Repr = [u8; 32]>> {
  num_steps: usize,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: ff::PrimeField<Repr = [u8; 32]>> JoltSkeleton<F> {
  pub fn from_num_steps(num_steps: usize) -> Self {
    JoltSkeleton::<F>{
      num_steps: num_steps,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<F: ff::PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltSkeleton<F> {
  #[tracing::instrument(skip_all, name = "JoltSkeleton::synthesize")]
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let circuit_dir = JoltPaths::circuit_artifacts_path();
    let r1cs_path = JoltPaths::r1cs_path();
    let wtns_path = JoltPaths::witness_generator_path();

    let cfg = CircomConfig::new(wtns_path.clone(), r1cs_path.clone()).unwrap();

    let _ = circom_scotia::synthesize(
        &mut cs.namespace(|| "jolt_step_0"),
        cfg.r1cs.clone(),
        None,
    )
    .unwrap();

    Ok(())
  }
}

// pub fn precommit<G: Group>(jolt_circuit: JoltCircuit<Spartan2Fr>) -> Result<(), SpartanError> {
pub fn precommit<G: Group<Scalar = F>, S: PrecommittedSNARKTrait<G>, F: PrimeField<Repr = [u8; 32]>>(jolt_circuit: JoltCircuit<F>) -> Result<(<<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey, Vec<Vec<F>>, Vec<<G::CE as CommitmentEngineTrait<G>>::Commitment>), SynthesisError> {
  let jolt_witnesses = jolt_circuit.get_witnesses_by_step()?;
  let w_segments = get_segments(jolt_witnesses); 

  // // w_segments but the last three are combined into one 
  // let last_three_combined = old_w_segments[old_w_segments.len()-3..].concat();
  // let w_segments = [&old_w_segments[..old_w_segments.len()-3], &[last_three_combined]].concat();

  let N_SEGMENTS = w_segments.len();

  // find largest segment len
  let max_segment_len = w_segments.iter().map(|v| v.len()).max().unwrap();
  let ck = G::CE::setup(b"ck", max_segment_len);

  // for each segment, commit to it using CE::<G>::commit(ck, &self.W) 
  let commitments: Vec<<G::CE as CommitmentEngineTrait<G>>::Commitment> = (0..N_SEGMENTS)
    .into_par_iter()
    .map(|i| {
      G::CE::commit(&ck, &w_segments[i]) 
    }).collect();

  Ok((ck, w_segments, commitments))
}

pub struct R1CSProof  {
  proof: SNARK<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>, JoltCircuit<Spartan2Fr>>,
  vk: VerifierKey<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>>,
}

impl R1CSProof {
  #[tracing::instrument(skip_all, name = "R1CSProof::prove")]
  pub fn prove<ArkF: ark_ff::PrimeField, ArkG: ark_ec::CurveGroup>(
      W: usize, 
      C: usize, 
      TRACE_LEN: usize, 
      inputs: Vec<Vec<ArkF>>, 
      _generators: &PedersenGenerators<ArkG>,
  ) -> Result<Self, SpartanError> {
    Self::prove_precommitted(W, C, TRACE_LEN, inputs)
  }

  pub fn prove_uniform<ArkF: ark_ff::PrimeField>(
      W: usize, 
      C: usize, 
      TRACE_LEN: usize, 
      inputs: Vec<Vec<ArkF>>
  ) -> Result<Self, SpartanError> {

    type G1 = SpartanG1;
    type EE = SpartanHyraxEE<SpartanG1>;
    type S = spartan2::spartan::upsnark::R1CSSNARK<G1, EE>;
    type F = Spartan2Fr;

    let NUM_STEPS = TRACE_LEN;

    let span = tracing::span!(tracing::Level::TRACE, "convert_ark_to_spartan_fr");
    let _enter = span.enter();
    let inputs_ff = inputs
      .into_par_iter()
      .map(|input| input
          .into_par_iter()
          .map(|x| {
              ark_to_spartan_unsafe(x)
          })
          .collect::<Vec<Spartan2Fr>>()
      ).collect::<Vec<Vec<Spartan2Fr>>>();
      drop(_enter);

      let jolt_circuit = JoltCircuit::<F>::new_from_inputs(W, C, NUM_STEPS, inputs_ff[0][0], inputs_ff);
      let num_steps = jolt_circuit.num_steps;
      let skeleton_circuit = JoltSkeleton::<F>::from_num_steps(num_steps);

      let (pk, vk) = SNARK::<G1, S, JoltSkeleton<F>>::setup_uniform(skeleton_circuit, num_steps).unwrap();


      SNARK::prove(&pk, jolt_circuit).map(|snark| Self {
        proof: snark,
        vk
      })
  }

  pub fn prove_precommitted<ArkF: ark_ff::PrimeField>(
      W: usize, 
      C: usize, 
      TRACE_LEN: usize, 
      inputs: Vec<Vec<ArkF>>
  ) -> Result<Self, SpartanError> {

    type G1 = SpartanG1;
    type EE = SpartanHyraxEE<SpartanG1>;
    type S = spartan2::spartan::upsnark::R1CSSNARK<G1, EE>;
    type F = Spartan2Fr;

    let NUM_STEPS = TRACE_LEN;

    let span = tracing::span!(tracing::Level::TRACE, "convert_ark_to_spartan_fr");
    let _enter = span.enter();
    let inputs_ff = inputs
      .into_par_iter()
      .map(|input| input
          .into_par_iter()
          .map(|x| {
              ark_to_spartan_unsafe(x)
          })
          .collect::<Vec<Spartan2Fr>>()
      ).collect::<Vec<Vec<Spartan2Fr>>>();
      drop(_enter);

      let jolt_circuit = JoltCircuit::<F>::new_from_inputs(W, C, NUM_STEPS, inputs_ff[0][0], inputs_ff);
      let num_steps = jolt_circuit.num_steps;
      let skeleton_circuit = JoltSkeleton::<F>::from_num_steps(num_steps);

      let (ck, w_segments, comms) = precommit::<G1, S, F>(jolt_circuit.clone()).unwrap();
  
      let (pk, vk) = SNARK::<G1, S, JoltSkeleton<<G1 as Group>::Scalar>>::setup_precommitted(skeleton_circuit, num_steps, ck).unwrap();
    
      // produce a SNARK
      let proof = SNARK::prove_precommitted(&pk, jolt_circuit, w_segments, comms); 
      
      proof.map(|snark| Self {
        proof: snark,
        vk
      }) 
  }


  pub fn verify(&self) -> Result<(), SpartanError> {
    // SNARK::verify(&self.proof, &self.vk, &[])
    SNARK::verify_precommitted(&self.proof, &self.vk, &[])
  }
}

mod test {
  use spartan2::{
    provider::bn256_grumpkin::bn256,
    traits::Group,
    SNARK,
  };


  #[test]
  fn test_all_zeros() {
    type G1 = bn256::Point;
    type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
    type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;

    type F = <G1 as Group>::Scalar;

    let N = 1; 
    let W = 64;
    let c = 6;

    let prog_a_rw = vec![F::zero(); N * 6];
    let prog_v_rw = vec![F::zero(); N * 6];
    let prog_t_reads = vec![F::zero(); N * 6];
    let memreg_a_rw = vec![F::zero(); N * 3+(W/8)];
    let memreg_v_reads = vec![F::zero(); N * 3+(W/8)];
    let memreg_v_writes = vec![F::zero(); N * 3+(W/8)];
    let memreg_t_reads = vec![F::zero(); N * c];
    let chunks_x = vec![F::zero(); N * c];
    let chunks_y=  vec![F::zero(); N * c];
    let chunks_query = vec![F::zero(); N];
    let lookup_outputs = vec![F::zero(); N];
    let op_flags = vec![F::zero(); N * 15];

    let inputs = vec![
      prog_a_rw,
      prog_v_rw,
      prog_t_reads,
      memreg_a_rw,
      memreg_v_reads, 
      memreg_v_writes,
      memreg_t_reads, 
      chunks_x, 
      chunks_y,
      chunks_query,
      lookup_outputs,
      op_flags,
    ];

    // TODO(arasuarun): Need to switch to single step.
    unimplemented!("Test needs to be ported to jolt_single_step.circom");
    // let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(32, 3, inputs);
    // let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    // assert!(res_verifier.is_ok());
  }

  #[test]
  fn test_add() {
    type G1 = bn256::Point;
    type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
    type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;

    type F = <G1 as Group>::Scalar;

    let N = 1; 
    let W = 64;
    let c = 6;

    let mut prog_a_rw = vec![F::zero(); N * 6];
    let mut prog_v_rw = vec![F::zero(); N * 6];
    let mut prog_t_reads = vec![F::zero(); N * 6];
    let mut memreg_a_rw = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_v_reads = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_v_writes = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_t_reads = vec![F::zero(); N * c];
    let mut chunks_x = vec![F::zero(); N * c];
    let mut chunks_y=  vec![F::zero(); N * c];
    let mut chunks_query = vec![F::zero(); c];
    let mut lookup_outputs = vec![F::zero(); N];
    let mut op_flags = vec![F::zero(); N * 15];

    // prog_v_rw: (ADD, rs1= 1, rs2=2, rd=3, imm=0, op_flags_packed=12)
    prog_v_rw = vec![F::from(10), F::from(1), F::from(2), F::from(3), F::from(0), F::from(128)];

    // memreg_a_rw: (rs1=1, rs2=2, rd=3)
    memreg_a_rw[0] = F::from(1);
    memreg_a_rw[1] = F::from(2);
    memreg_a_rw[10] = F::from(3);

    // suppose rs1 stores 1, rs2 stores 6, rd has 0
    memreg_v_reads[0] = F::from(1);
    memreg_v_reads[1] = F::from(6);
    memreg_v_reads[10] = F::from(0);

    memreg_v_writes[0] = F::from(1);
    memreg_v_writes[1] = F::from(6);
    memreg_v_writes[10] = F::from(7);

    chunks_x[c-1] = F::from(0);
    chunks_y[c-1] = F::from(0);
    chunks_query[c-1] = F::from(0);

    op_flags[7] = F::from(1); // is_add_instr

    let inputs = vec![
      prog_a_rw,
      prog_v_rw,
      prog_t_reads,
      memreg_a_rw,
      memreg_v_reads, 
      memreg_v_writes,
      memreg_t_reads, 
      chunks_x, 
      chunks_y,
      chunks_query,
      lookup_outputs,
      op_flags,
    ];

    // TODO(arasuarun): Need to switch to single step.
    unimplemented!("Test needs to be ported to jolt_single_step.circom");
    // let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(64, 6, inputs);
    // let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    // assert!(res_verifier.is_ok());
  }

  #[test]
  fn test_add_should_fail() {
    type G1 = bn256::Point;
    type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
    type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;

    type F = <G1 as Group>::Scalar;

    let N = 1; 
    let W = 64;
    let c = 6;

    let mut prog_a_rw = vec![F::zero(); N * 6];
    let mut prog_v_rw = vec![F::zero(); N * 6];
    let mut prog_t_reads = vec![F::zero(); N * 6];
    let mut memreg_a_rw = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_v_reads = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_v_writes = vec![F::zero(); N * 3+(W/8)];
    let mut memreg_t_reads = vec![F::zero(); N * c];
    let mut chunks_x = vec![F::zero(); N * c];
    let mut chunks_y=  vec![F::zero(); N * c];
    let mut chunks_query = vec![F::zero(); c];
    let mut lookup_outputs = vec![F::zero(); N];
    let mut op_flags = vec![F::zero(); N * 15];

    prog_v_rw = vec![F::from(10), F::from(1), F::from(2), F::from(3), F::from(0), F::from(128)];

    // ERROR: this should be rs1 = 1
    memreg_a_rw[0] = F::from(16);
    memreg_a_rw[1] = F::from(2);
    memreg_a_rw[10] = F::from(3);

    memreg_v_reads[0] = F::from(1);
    memreg_v_reads[1] = F::from(6);
    memreg_v_reads[10] = F::from(0);

    memreg_v_writes[0] = F::from(1);
    memreg_v_writes[1] = F::from(6);
    memreg_v_writes[10] = F::from(7);

    chunks_x[c-1] = F::from(0);
    chunks_y[c-1] = F::from(0);
    chunks_query[c-1] = F::from(0);

    op_flags[7] = F::from(1); // is_add_instr

    let inputs = vec![
      prog_a_rw,
      prog_v_rw,
      prog_t_reads,
      memreg_a_rw,
      memreg_v_reads, 
      memreg_v_writes,
      memreg_t_reads, 
      chunks_x, 
      chunks_y,
      chunks_query,
      lookup_outputs,
      op_flags,
    ];

    // TODO(arasuarun): Need to switch to single step.
    unimplemented!("Test needs to be ported to jolt_single_step.circom");
    // let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(64, 6, inputs);
    // let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    // assert!(res_verifier.is_err());
  }

}