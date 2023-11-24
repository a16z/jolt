use std::env::current_dir;

use spartan2::{
  provider::bn256_grumpkin::bn256,
  traits::{snark::RelaxedR1CSSNARKTrait, Group},
  SNARK, errors::SpartanError,
};

use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

use circom_scotia::{calculate_witness, r1cs::CircomConfig};

#[derive(Clone, Debug, Default)]
pub struct JoltCircuit<F: PrimeField> {
  width: usize,
  c: usize,
  inputs: Vec<Vec<F>>,
  // prog_a_rw: Vec<F>, 
  // prog_v_rw: Vec<F>, 
  // prog_t_reads: Vec<F>, 
  // memreg_a_rw: Vec<F>, 
  // memreg_v_reads: Vec<F>, 
  // memreg_v_writes: Vec<F>, 
  // memreg_t_reads: Vec<F>,
  // chunks_x: Vec<F>, 
  // chunks_y: Vec<F>, 
  // chunks_query: Vec<F>, 
  // lookup_outputs: Vec<F>, 
  // op_flags: Vec<F>,
}

impl<F: PrimeField> JoltCircuit<F> {
  pub fn new_from_inputs(W: usize, c: usize, inputs: Vec<Vec<F>>) -> Self {
    JoltCircuit{
      width: 64,
      c: 6,
      inputs: inputs,
    }
  }

  pub fn all_zeros(W: usize, c: usize, N: usize) -> Self {
    JoltCircuit{
      width: W,
      c: c,
      inputs: vec![
        vec![F::ZERO; N * 6], // TODO: change this to just N * 1 as prog code address is de-duplicated
        vec![F::ZERO; N * 6],
        vec![F::ZERO; N * 6], // same for t_reads
        vec![F::ZERO; N * 3+(W/8)], 
        vec![F::ZERO; N * 3+(W/8)],
        vec![F::ZERO; N * 3+(W/8)],
        vec![F::ZERO; N * 3+(W/8)],
        vec![F::ZERO; N * c],
        vec![F::ZERO; N * c],
        vec![F::ZERO; N * c],
        vec![F::ZERO; N],
        vec![F::ZERO; N * 14],
      ]
    }
  }
}

impl<F: PrimeField> Circuit<F> for JoltCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let circuit_dir = std::env::var("CIRCUIT_DIR").expect("failed to read CICUIT_DIR, should've been set by build.rs");
    println!("CIRCUIT DIR IS: {}", circuit_dir);
    let circuit_dir = std::path::PathBuf::from(&circuit_dir);
    let r1cs_path = circuit_dir.join("jolt.r1cs");
    let wtns_path = circuit_dir.join("jolt_js/jolt.wasm");


    let cfg = CircomConfig::new(wtns_path, r1cs_path.clone()).unwrap();

    let variable_names = ["prog_a_rw", "prog_v_rw", "prog_t_reads", "memreg_a_rw", "memreg_v_reads", "memreg_v_writes", "memreg_t_reads", "chunks_x", "chunks_y", "lookup_outputs", "chunks_query", "op_flags"];

    let input: Vec<(String, Vec<F>)> = variable_names
      .iter()
      .zip(self.inputs.into_iter())
      .map(|(name, input)| (name.to_string(), input))
      .collect();

    let jolt_witness = calculate_witness(&cfg, input, true).expect("msg");

    let _ = circom_scotia::synthesize(
      cs, //.namespace(|| "jolt_circom"),
      cfg.r1cs.clone(),
      Some(jolt_witness),
    )
    .unwrap();

    Ok(())
  }
}

fn run_jolt_spartan() {
  // type G1 = pasta_curves::pallas::Point;
  type G1 = bn256::Point;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;
  run_jolt_spartan_with::<G1, S>();
}

fn run_jolt_spartan_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() {
  let circuit: JoltCircuit<<G as Group>::Scalar> = JoltCircuit::all_zeros(64, 6, 2);

  // produce keys
  let (pk, vk) = SNARK::<G, S, JoltCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let res = SNARK::prove(&pk, circuit);
  assert!(res.is_ok());
  let snark = res.unwrap();

  // verify the SNARK
  let res = snark.verify(&vk);
  assert!(res.is_ok());
}


pub fn run_jolt_spartan_with_circuit<G: Group, S: RelaxedR1CSSNARKTrait<G>>(circuit: JoltCircuit<<G as Group>::Scalar>) -> Result<Vec<<G as Group>::Scalar>, SpartanError> {
  // produce keys
  let (pk, vk) = SNARK::<G, S, JoltCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let res = SNARK::prove(&pk, circuit);
  assert!(res.is_ok());
  let snark = res.unwrap();

  // verify the SNARK
  let res = snark.verify(&vk);
  // assert!(res.is_ok());
  res 
}

mod test {
  use spartan2::{
    provider::bn256_grumpkin::bn256,
    traits::Group,
    SNARK,
  };
  use super::{JoltCircuit, run_jolt_spartan_with_circuit};

  #[test]
  fn test_jolt_snark() {
    super::run_jolt_spartan();
  }

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

    let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(32, 3, inputs);
    let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    assert!(res_verifier.is_ok());
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

    let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(64, 6, inputs);
    let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    assert!(res_verifier.is_ok());
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

    let jolt_circuit = JoltCircuit::<<G1 as Group>::Scalar>::new_from_inputs(64, 6, inputs);
    let res_verifier = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
    assert!(res_verifier.is_err());
  }
}
