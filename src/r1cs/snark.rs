use pasta_curves::vesta::Base as Fr;
use std::env::current_dir;

use spartan2::{
  provider::bn256_grumpkin::bn256,
  traits::{snark::RelaxedR1CSSNARKTrait, Group},
  SNARK,
};

use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
};
use core::marker::PhantomData;
use ff::PrimeField;

use circom_scotia::{calculate_witness, r1cs::CircomConfig};

#[derive(Clone, Debug, Default)]
struct JoltCircuit<F: PrimeField> {
  /* This is where the transcript should be.
  transcript: Transcript<F>
    - This can be easily fed to the circuit pipeline to generate a SNARK proof.
  */
  _p: PhantomData<F>,
}

impl<F: PrimeField> Circuit<F> for JoltCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let root = current_dir().unwrap().join("src/r1cs/circuits/jolt/");
    let r1cs_path = root.join("jolt.r1cs");
    let wtns_path = root.join("jolt_js/jolt.wasm");

    let cfg = CircomConfig::new(wtns_path, r1cs_path.clone()).unwrap();

    let arg_in: (String, Vec<F>) = ("in".into(), vec![F::from(0); 54]);
    let input = vec![arg_in];
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

fn run_jolt_spartan_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>()
{
  let circuit = JoltCircuit::default();

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

mod test {
    use std::process::Command;

  #[test]
  fn test_jolt_snark() {
    // compile the circom circuit first 
    let output = Command::new("bash")
        .arg("./src/r1cs/circuits/compile_jolt.sh")
        .output()
        .expect("Could not compile circom circuit file."); 

    // then run spartan
    super::run_jolt_spartan();
  }
}
