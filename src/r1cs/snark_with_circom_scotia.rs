
use ff::Field; 
use pasta_curves::vesta::Base as Fr;
use std::env::current_dir;

use spartan2::{provider::bn256_grumpkin::bn256, SNARK, traits::{Group, snark::RelaxedR1CSSNARKTrait}};
// use bellperson::{gadgets::num::AllocatedNum, SynthesisError, LinearCombination, util_cs::test_cs::TestConstraintSystem};
use bellpepper_core::{Circuit, ConstraintSystem, num::AllocatedNum, SynthesisError, LinearCombination};
// use bellpepper_core::ConstraintSystem;
// use spartan2::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

use circom_scotia::{calculate_witness, r1cs::CircomConfig, synthesize, witness};
use circom_scotia::{reader::load_r1cs};

// type G1 = pasta_curves::pallas::Point;
// type G2 = pasta_curves::vesta::Point;

// type G1 = spartan2::provider::bn256_grumpkin::bn256::Point;
// type G2 = spartan2::provider::bn256_grumpkin::grumpkin::Point;

#[derive(Clone, Debug, Default)]
struct JoltCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

// impl<F> Circuit<F> for JoltCircuit<F>
impl<F: PrimeField> Circuit<F> for JoltCircuit<F>
{
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let root = current_dir().unwrap().join("src/r1cs/circuits/jolt/");
    let r1cs_path = root.join("jolt.r1cs");
    let wtns_path = root.join("jolt_js/jolt.wasm");

    let cfg = CircomConfig::new(wtns_path, r1cs_path.clone()).unwrap();

    let arg_in = ("inputs".into(), vec![F::ZERO; 2]);
    let input = vec![arg_in];
    let jolt_witness = calculate_witness(&cfg, input, true).expect("msg");

    let jolt_r1cs = load_r1cs::<F>(r1cs_path);

    println!("NUMBER OF INPUTS IS : {}", jolt_r1cs.num_inputs);

    let output = circom_scotia::synthesize(
        cs, //.namespace(|| "jolt_circom"),
        cfg.r1cs.clone(),
        Some(jolt_witness),
    ).unwrap();

    Ok(())
  }
}

fn run_jolt_spartan() {
  type G = pasta_curves::pallas::Point;
  // type G = bn256::Point;
  type EE = spartan2::provider::ipa_pc::EvaluationEngine<G>;
  type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;
  type Spp = spartan2::spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
  run_jolt_spartan_with::<G, S>();
  // run_jolt_spartan_with::<G, Spp>();
}

fn run_jolt_spartan_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() 
// where JoltCircuit: bellperson::Circuit<<G as spartan2::traits::Group>::Scalar>
{
    let circuit = JoltCircuit::default();
  
    // produce keys
    let (pk, vk) =
      SNARK::<G, S, JoltCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    // produce a SNARK
    let res = SNARK::prove(&pk, circuit);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());

    let io = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(io, vec![<G as Group>::Scalar::from(15u64)]);
}

mod test {
  use super::*;

  #[test]
  fn test_spartan_full() {
    run_jolt_spartan();
  }
}