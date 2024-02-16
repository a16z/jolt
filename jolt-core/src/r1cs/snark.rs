use std::collections::HashMap;
use common::{path::JoltPaths, field_conversion::{ff_to_ruints, ruint_to_ff, ff_to_ruint}};
use spartan2::{
  traits::{snark::RelaxedR1CSSNARKTrait, Group, upsnark::{PrecommittedSNARKTrait, UniformSNARKTrait}},
  SNARK, errors::SpartanError, 
};
use bellpepper_core::{Circuit, ConstraintSystem, LinearCombination, SynthesisError, Variable, Index, num::AllocatedNum};
use ff::PrimeField;
use ruint::aliases::U256;
use circom_scotia::r1cs::CircomConfig;
use rayon::prelude::*;

use ark_ff::PrimeField as arkPrimeField;
use common::field_conversion::ark_to_ff; 
// Exact instantiation of the field used
use spartan2::provider::bn256_grumpkin::bn256;
use bn256::Scalar as Spartan2Fr;

const WTNS_GRAPH_BYTES: &[u8] = include_bytes!("./graph.bin");

const NUM_CHUNKS: usize = 4;
const NUM_FLAGS: usize = 17;
const SEGMENT_LENS: [usize; 11] = [4, 1, 6, 7, 7, 7, NUM_CHUNKS, NUM_CHUNKS, NUM_CHUNKS, 1, NUM_FLAGS];

#[tracing::instrument(skip_all, name = "JoltCircuit::assemble_by_segments")]
fn reassemble_by_segments<F: PrimeField>(mut jolt_witnesses: Vec<Vec<F>>) -> Vec<F> {
  let mut result: Vec<Vec<F>> = vec![Vec::new(); jolt_witnesses[0].len()];
  result[0] = vec![F::from(1)]; // start with [1]

  for witness in &mut jolt_witnesses {
    witness.remove(0); 
    for (i, w) in witness.iter().enumerate() {
        result[i + 1].push(*w);
    }
  }

  result.into_iter().flatten().collect()
}

#[derive(Clone, Debug, Default)]
pub struct JoltCircuit<F: PrimeField<Repr=[u8; 32]>> {
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

impl<F: PrimeField<Repr=[u8;32]>> JoltCircuit<F> {
  pub fn new_from_inputs(W: usize, c: usize, num_steps: usize, PC_START_ADDR: F, inputs: Vec<Vec<F>>) -> Self {
    JoltCircuit{
      num_steps: num_steps,
      inputs: inputs,
    }
  }
}

impl<F: PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltCircuit<F> {
  #[tracing::instrument(skip_all, name = "JoltCircuit::synthesize")]
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let r1cs_path = JoltPaths::r1cs_path();
    let wtns_path = JoltPaths::witness_generator_path();

    let cfg: CircomConfig<F> = CircomConfig::new(wtns_path.clone(), r1cs_path.clone()).unwrap();

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

    // for variable [v], step_inputs[v][j] is the variable input for step j
    let inputs_chunked : Vec<Vec<_>> = self.inputs
      .into_par_iter()
      .map(|inner_vec| inner_vec.chunks(inner_vec.len()/TRACE_LEN).map(|chunk| chunk.to_vec()).collect())
      .collect();

    let compute_witness_span = tracing::span!(tracing::Level::INFO, "compute_witness_loop");
    let _compute_witness_guard = compute_witness_span.enter();

    let graph = witness::init_graph(WTNS_GRAPH_BYTES).unwrap();
    let wtns_buffer_size = witness::get_inputs_size(&graph);
    let wtns_mapping = witness::get_input_mapping(&variable_names, &graph);

    let jolt_witnesses: Vec<Vec<F>> = (0..NUM_STEPS).into_par_iter().map(|i| {
      let mut step_inputs: Vec<Vec<U256>> = inputs_chunked.iter().map(|v| v[i].iter().map(|v| ff_to_ruint(v.clone())).collect()).collect::<Vec<_>>();
      step_inputs.push(vec![U256::from(i as u64), ff_to_ruint(inputs_chunked[0][i][0])]); // [step_counter, program_counter]

      let input_map: HashMap<String, Vec<U256>> = variable_names
        .iter()
        .zip(step_inputs.into_iter())
        .map(|(name, input)| (name.to_owned(), input))
        .collect();

      let mut inputs_buffer = witness::get_inputs_buffer(wtns_buffer_size);
      witness::populate_inputs(&input_map, &wtns_mapping, &mut inputs_buffer);
      let uint_jolt_witness = witness::graph::evaluate(&graph.nodes, &inputs_buffer, &graph.signals);

      uint_jolt_witness.into_iter().map(|x| ruint_to_ff(x)).collect::<Vec<_>>()
    }).collect();

    let witness_variable_wise = reassemble_by_segments(jolt_witnesses);

    (1..witness_variable_wise.len()).for_each(|i| {
        let f = witness_variable_wise[i];
        let _ = AllocatedNum::alloc(cs.namespace(|| format!("{}_{}", if i < cfg.r1cs.num_inputs { "public" } else { "aux" }, i)), || Ok(f)).unwrap();
    });

    Ok(())
  }
}

#[derive(Clone, Debug, Default)]
pub struct JoltSkeleton<F: PrimeField<Repr = [u8; 32]>> {
  num_steps: usize,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField<Repr = [u8; 32]>> JoltSkeleton<F> {
  pub fn from_num_steps(num_steps: usize) -> Self {
    JoltSkeleton::<F>{
      num_steps: num_steps,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<F: PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltSkeleton<F> {
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


#[tracing::instrument(skip_all, name = "JoltSkeleton::prove_jolt_circuit")]
pub fn prove_jolt_circuit<G: Group<Scalar = F>, S: PrecommittedSNARKTrait<G>, F: PrimeField<Repr = [u8; 32]>>(circuit: JoltCircuit<F>) -> Result<(), SpartanError> {
  let num_steps = circuit.num_steps; 
  let skeleton_circuit = JoltSkeleton::<G::Scalar>::from_num_steps(num_steps);

  let (pk, vk) = SNARK::<G, S, JoltSkeleton<<G as Group>::Scalar>>::setup_precommitted(skeleton_circuit, num_steps).unwrap();

  // produce a SNARK
  let proof = SNARK::prove(&pk, circuit);
  assert!(proof.is_ok());

  let res = SNARK::verify(&proof.unwrap(), &vk, &[]); 
  assert!(res.is_ok()); 

  Ok(())
}

pub fn prove_r1cs<ArkF: arkPrimeField>(
  W: usize, 
  C: usize, 
  TRACE_LEN: usize, 
  inputs: Vec<Vec<ArkF>>) -> Result<(), SpartanError> {

  type G1 = bn256::Point;
  type EE = spartan2::provider::hyrax_pc::HyraxEvaluationEngine<G1>;
  type S = spartan2::spartan::upsnark::R1CSSNARK<G1, EE>;

  let NUM_STEPS = TRACE_LEN; 
  // let NUM_STEPS = 2; 

  let inputs_ff = inputs
    .into_par_iter()
    .map(|input| input
        .into_par_iter()
        .map(|x| ark_to_ff(x))
        .collect::<Vec<Spartan2Fr>>()
    ).collect::<Vec<Vec<Spartan2Fr>>>();

  let jolt_circuit = JoltCircuit::<Spartan2Fr>::new_from_inputs(W, C, NUM_STEPS, inputs_ff[0][0], inputs_ff);
  prove_jolt_circuit::<G1, S, Spartan2Fr>(jolt_circuit)
}


mod test {
  use spartan2::{
    provider::bn256_grumpkin::bn256,
    traits::Group,
    SNARK,
  };

  use super::{JoltCircuit, prove_jolt_circuit}; 

  // #[test]
  // fn test_jolt_snark() {
  //   super::run_jolt_spartan();
  // }

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