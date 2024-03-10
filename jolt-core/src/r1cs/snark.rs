use std::collections::HashMap;

use common::{constants::RAM_START_ADDRESS, field_conversion::{ark_to_spartan_unsafe, ark_to_spartan_vec, ark_to_spartan_vecs, ff_to_ruint, ff_to_ruints, ruint_to_ff, spartan_to_ark_unsafe}, path::JoltPaths};
use spartan2::{
  errors::SpartanError, 
  provider::{
      bn256_grumpkin::bn256::{self, Point as SpartanG1, Scalar as Spartan2Fr},
      hyrax_pc::{HyraxEvaluationEngine as SpartanHyraxEE, HyraxCommitmentKey},
  }, 
  spartan::upsnark::R1CSSNARK, traits::{
    commitment::CommitmentEngineTrait, snark::RelaxedR1CSSNARKTrait, upsnark::{PrecommittedSNARKTrait, UniformSNARKTrait}, Group
  }, VerifierKey, SNARK
};
use bellpepper_core::{
  Circuit, ConstraintSystem, LinearCombination, SynthesisError, Variable, Index, num::AllocatedNum,
};
use crate::jolt::vm::{JoltCommitments, JoltPolynomials};
use ff::{derive::bitvec::mem, PrimeField};
use ruint::aliases::U256;
use circom_scotia::r1cs::CircomConfig;
use rayon::prelude::*;

use ark_ff::PrimeField as arkPrimeField;
use ark_std::{Zero, One};

use crate::utils;

const WTNS_GRAPH_BYTES: &[u8] = include_bytes!("./graph.bin");

const NUM_CHUNKS: usize = 4;
const NUM_FLAGS: usize = 17;
const SEGMENT_LENS: [usize; 11] = [4, 1, 6, 7, 7, 7, NUM_CHUNKS, NUM_CHUNKS, NUM_CHUNKS, 1, NUM_FLAGS];

type CommitmentKey<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;

/// Remap [[1, a1, a2, a3], [1, b1, b2, b3], ...]  -> [1, a1, b1, ..., a2, b2, ..., a3, b3, ...]
#[tracing::instrument(skip_all, name = "JoltCircuit::assemble_by_segments")]
fn reassemble_by_segments<F: PrimeField>(mut jolt_witnesses: Vec<Vec<F>>) -> Vec<F> {
  get_segments(jolt_witnesses).into_iter().flatten().collect()
}

fn get_segments<F: PrimeField>(mut jolt_witnesses: Vec<Vec<F>>) -> Vec<Vec<F>> {
  let mut result: Vec<Vec<F>> = vec![Vec::new(); jolt_witnesses[0].len()-1]; // ignore 1 at the start 

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
    // let inputs_chunked : Vec<Vec<_>> = self.inputs
    //   .into_par_iter()
    //   .map(|inner_vec| inner_vec.chunks(inner_vec.len()/TRACE_LEN).map(|chunk| chunk.to_vec()).collect())
    //   .collect();

    let graph = witness::init_graph(WTNS_GRAPH_BYTES).unwrap();
    let wtns_buffer_size = witness::get_inputs_size(&graph);
    let wtns_mapping = witness::get_input_mapping(&variable_names, &graph);

    let compute_witness_span = tracing::span!(tracing::Level::INFO, "compute_witness_loop");
    let _compute_witness_guard = compute_witness_span.enter();
    let jolt_witnesses: Vec<Vec<F>> = (0..NUM_STEPS).into_par_iter().map(|i| {
      // let mut step_inputs: Vec<Vec<ark_bn254::Fr>> = inputs_chunked.iter().map(|v| v[i].iter().cloned().map(spartan_to_ark_unsafe).collect()).collect();
      let mut step_inputs: Vec<Vec<ark_bn254::Fr>> = self.inputs.iter().map(|v| {
        v.iter()
         .skip(i)
         .step_by(TRACE_LEN)
         .cloned()
         .map(spartan_to_ark_unsafe)
         .collect()
      }).collect();

      let program_counter = if i > 0 && self.inputs[0][i] == F::from(0) {
        F::from(0)
      } else {
          self.inputs[0][i] * F::from(4u64) + F::from(RAM_START_ADDRESS)
      };
      step_inputs.push(vec![ark_bn254::Fr::from(i as u64), spartan_to_ark_unsafe(program_counter)]);

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

impl<F: ff::PrimeField<Repr = [u8; 32]>> Circuit<F> for JoltCircuit<F> {
  #[tracing::instrument(skip_all, name = "JoltCircuit::synthesize")]
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let jolt_witnesses = self.get_witnesses_by_step()?;

    let witness_variable_wise = reassemble_by_segments(jolt_witnesses);

    let allocate_vars_span = tracing::span!(tracing::Level::INFO, "allocate_vars");
    let _allocate_vars_guard = allocate_vars_span.enter();
    (0..witness_variable_wise.len()).for_each(|i| {
        let f = witness_variable_wise[i];
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

  pub fn get_w_segments<G: Group<Scalar = F>, S: PrecommittedSNARKTrait<G>, F: PrimeField<Repr = [u8; 32]>>(jolt_circuit: JoltCircuit<F>) -> Result<(Vec<Vec<F>>), SynthesisError> {
    let jolt_witnesses = jolt_circuit.get_witnesses_by_step()?;
    Ok(get_segments(jolt_witnesses))
  }
  
  pub fn precommit_with_ck<G: Group<Scalar = F>, S: PrecommittedSNARKTrait<G>, F: PrimeField<Repr = [u8; 32]>>(ck: &CommitmentKey<G>, w_segments: Vec<Vec<F>>) -> Result<(Vec<<G::CE as CommitmentEngineTrait<G>>::Commitment>), SynthesisError> {
    let N_SEGMENTS = w_segments.len();
  
    // for each segment, commit to it using CE::<G>::commit(ck, &self.W) 
    let commitments: Vec<<G::CE as CommitmentEngineTrait<G>>::Commitment> = (0..N_SEGMENTS)
      .into_par_iter()
      .map(|i| {
        G::CE::commit(&ck, &w_segments[i]) 
      }).collect();
  
    Ok(commitments)
  }

pub struct R1CSProof  {
  proof: SNARK<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>, JoltCircuit<Spartan2Fr>>,
  vk: VerifierKey<SpartanG1, R1CSSNARK<SpartanG1, SpartanHyraxEE<SpartanG1>>>,
}

impl R1CSProof {
  #[tracing::instrument(skip_all, name = "R1CSProof::prove")]
  pub fn prove<ArkF: ark_ff::PrimeField, ArkG: ark_ec::CurveGroup<ScalarField = ArkF>>(
      W: usize, 
      C: usize, 
      TRACE_LEN: usize, 
      inputs: Vec<Vec<ArkF>>, 
      generators: Vec<bn256::Affine>,
      jolt_polynomials: &JoltPolynomials<ArkF, ArkG>,
      jolt_commitments: &Vec<Vec<bn256::Affine>>,
  ) -> Result<Self, SpartanError> {
      type G1 = SpartanG1;
      type EE = SpartanHyraxEE<SpartanG1>;
      type S = spartan2::spartan::upsnark::R1CSSNARK<G1, EE>;
      type F = Spartan2Fr;

      let NUM_STEPS = TRACE_LEN;

      // All conversions *********************************************/
      let span = tracing::span!(tracing::Level::TRACE, "convert_ark_to_spartan_fr");
      let _enter = span.enter();

      let inputs_ff = ark_to_spartan_vecs(inputs);

      // bytecode polynomials 
      let bytecode_polys: Vec<Vec<F>> = ark_to_spartan_vecs(jolt_polynomials.bytecode.get_polys_r1cs().clone());

      let memreg_polys: Vec<Vec<F>> = ark_to_spartan_vecs(jolt_polynomials.read_write_memory.get_polys_r1cs());

      let lookup_polys: Vec<Vec<F>> = ark_to_spartan_vecs(jolt_polynomials.instruction_lookups.dim.iter().map(|poly| poly.evals()).collect());

      /**************************************************************/

      let jolt_circuit = JoltCircuit::<F>::new_from_inputs(W, C, NUM_STEPS, inputs_ff[0][0], inputs_ff);
      let num_steps = jolt_circuit.num_steps;
      let skeleton_circuit = JoltSkeleton::<F>::from_num_steps(num_steps);

      // Obtain public key 
      let hyrax_ck = HyraxCommitmentKey::<G1> {
          ck: spartan2::provider::pedersen::from_gens_bn256(generators)
      };

      // Obtain w_segments
      let mut w_segments = get_w_segments::<G1, S, F>(jolt_circuit.clone()).unwrap();
      w_segments[4..10].clone_from_slice(&bytecode_polys[0..6]); // including both bytecode a, v
      // w_segments[10] is circuit_flags_packed
      w_segments[11..14].clone_from_slice(&memreg_polys[0..3]); // memreg_a_rw[0,2,3] out of 7
      w_segments[14..18].clone_from_slice(&memreg_polys[3..7]); // remaining addresses 
      w_segments[18..32].clone_from_slice(&memreg_polys[7..21]); // all values 
      // 2 * C are for chunks_x and chunks_y
      w_segments[40..44].clone_from_slice(&lookup_polys[0..4]); // lookup query (dim)

      // Commit to segments
      let mut comm_w_vec = precommit_with_ck::<G1, S, F>(&hyrax_ck, w_segments.clone()).unwrap();
      for i in 0..6 {
        comm_w_vec[4 + i] = jolt_commitments[i].clone().into();
      }
      // comm_w_vec[10] is circuit_flags_packed
      for i in 0..3 {
        comm_w_vec[11+i] = jolt_commitments[6+i].clone().into(); // register addresses 
      }
      for i in 0..4 {
        comm_w_vec[14+i] = jolt_commitments[9+i].clone().into(); // RAM addresses 
      }
      for i in 0..14 {
        comm_w_vec[18+i] = jolt_commitments[13+i].clone().into(); // all memory v 
      }

      // C many chunks_x and chunks_y
      for i in 0..4 {
        comm_w_vec[40+i] = jolt_commitments[27+i].clone().into(); // lookup query (dim)
      }

      let (pk, vk) = SNARK::<G1, S, JoltSkeleton<F>>::setup_precommitted(skeleton_circuit, num_steps, hyrax_ck).unwrap();

      SNARK::prove_precommitted(&pk, jolt_circuit, w_segments, comm_w_vec).map(|snark| Self {
        proof: snark,
        vk
      })
  }

  pub fn verify(&self) -> Result<(), SpartanError> {
    SNARK::verify_precommitted(&self.proof, &self.vk, &[])
  }
}
