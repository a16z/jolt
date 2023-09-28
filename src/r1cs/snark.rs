use ff::Field;
use pasta_curves::vesta::Base as Fr;
use std::env::current_dir;

use bellperson::{
  gadgets::num::AllocatedNum, util_cs::test_cs::TestConstraintSystem, Circuit, ConstraintSystem,
  LinearCombination, SynthesisError,
};
use spartan2::{
  provider::bn256_grumpkin::bn256,
  traits::{snark::RelaxedR1CSSNARKTrait, Group},
  SNARK,
};
// use bellpepper_core::{Circuit, ConstraintSystem, num::AllocatedNum, SynthesisError, LinearCombination};
// use bellpepper_core::ConstraintSystem;
// use spartan2::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

use circom_scotia::reader::load_r1cs;
use circom_scotia::{calculate_witness, r1cs::CircomConfig, synthesize, witness};

use nova_scotia::circom::circuit::CircomCircuit;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

// type G1 = spartan2::provider::bn256_grumpkin::bn256::Point;
// type G2 = spartan2::provider::bn256_grumpkin::grumpkin::Point;

#[derive(Clone, Debug, Default)]
struct JoltCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

// impl<Fr> Circuit<Fr> for JoltCircuit<Fr>
impl Circuit<<G1 as Group>::Scalar> for JoltCircuit<Fr> {
  fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let root = current_dir().unwrap().join("src/r1cs/circuits/jolt/");
    let r1cs_path = root.join("jolt.r1cs");
    let wtns_path = root.join("jolt_js/jolt.wasm");

    let cfg = CircomConfig::new(wtns_path, r1cs_path.clone()).unwrap();

    let arg_in = ("in".into(), vec![Fr::ZERO; 2]);
    let input = vec![arg_in];
    let jolt_witness = calculate_witness(&cfg, input, true).expect("msg");

    let jolt_r1cs = nova_scotia::circom::reader::load_r1cs::<G1, G2>(
      &nova_scotia::FileLocation::PathBuf(r1cs_path),
    );

    println!("NUMBER OF INPUTS IS : {}", jolt_r1cs.num_inputs);

    let jolt_circom_circuit = CircomCircuit {
      r1cs: jolt_r1cs,
      witness: Some(jolt_witness),
    };

    let mut z = vec![];
    for i in 0..2 {
      z.push(AllocatedNum::alloc(
        cs.namespace(|| format!("start_{}", i)),
        || Ok(Fr::ZERO),
      )?);
    }
    jolt_circom_circuit.vanilla_synthesize(cs, &z);

    // let output = circom_scotia::synthesize(
    //     cs, //.namespace(|| "jolt_circom"),
    //     cfg.r1cs.clone(),
    //     Some(jolt_witness),
    // ).unwrap();

    Ok(())
  }
}

fn run_jolt_spartan() {
  // type G = pasta_curves::pallas::Point;
  // type G = bn256::Point;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;
  type Spp = spartan2::spartan::ppsnark::RelaxedR1CSSNARK<G1, EE>;

  let circuit = JoltCircuit::default();

  // produce keys
  let (pk, vk) = SNARK::<G1, S, JoltCircuit<Fr>>::setup(circuit.clone()).unwrap();
  // SNARK::<G, S, JoltCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

  // produce a SNARK
  let res = SNARK::prove(&pk, circuit);
  assert!(res.is_ok());
  let snark = res.unwrap();

  // verify the SNARK
  let res = snark.verify(&vk);
  assert!(res.is_ok());

  let io = res.unwrap();

  // sanity: check the claimed output with a direct computation of the same
  assert_eq!(io, vec![<G1 as Group>::Scalar::from(15u64)]);
}

mod test {
  use super::*;

  #[test]
  fn test_spartan_full() {
    run_jolt_spartan();
  }
}
