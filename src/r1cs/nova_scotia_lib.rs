use std::{
  collections::HashMap,
  env::current_dir,
  fs,
  path::{Path, PathBuf},
};

// use bellpepper_core::num::AllocatedNum;
use bellperson::gadgets::num::AllocatedNum;
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField};
use nova_scotia::circom::circuit::{CircomCircuit, R1CS};
use nova_scotia::circom::reader::generate_witness_from_bin;
use num_bigint::BigInt;
use num_traits::Num;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use spartan2::{
  traits::Group,
  SNARK,
  // TrivialTestCircuit,
};

use nova_scotia::circom::reader::generate_witness_from_wasm;

pub type F<G> = <G as Group>::Scalar;
pub type EE<G> = spartan2::provider::ipa_pc::EvaluationEngine<G>;
pub type S<G> = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE<G>>;
pub type C1<G> = CircomCircuit<<G as Group>::Scalar>;

#[derive(Clone)]
pub enum FileLocation {
  PathBuf(PathBuf),
  URL(String),
}

// pub fn create_public_params<G1, G2>(r1cs: R1CS<F<G1>>) -> PublicParams<G1, G2, C1<G1>, C2<G2>>
// where
//     G1: Group<Base = <G2 as Group>::Scalar>,
//     G2: Group<Base = <G1 as Group>::Scalar>,
// {
//     let circuit_primary = CircomCircuit {
//         r1cs,
//         witness: None,
//     };
//     let circuit_secondary = TrivialTestCircuit::default();

//     PublicParams::setup(circuit_primary.clone(), circuit_secondary.clone())
// }

#[derive(Serialize, Deserialize)]
struct CircomInput {
  step_in: Vec<String>,

  #[serde(flatten)]
  extra: HashMap<String, Value>,
}

#[cfg(not(target_family = "wasm"))]
fn compute_witness<G1, G2>(
  current_public_input: Vec<String>,
  private_input: HashMap<String, Value>,
  witness_generator_file: FileLocation,
  witness_generator_output: &Path,
) -> Vec<<G1 as Group>::Scalar>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  let decimal_stringified_input: Vec<String> = current_public_input
    .iter()
    .map(|x| BigInt::from_str_radix(x, 16).unwrap().to_str_radix(10))
    .collect();

  let input = CircomInput {
    step_in: decimal_stringified_input.clone(),
    extra: private_input.clone(),
  };

  let is_wasm = match &witness_generator_file {
    FileLocation::PathBuf(path) => path.extension().unwrap_or_default() == "wasm",
    FileLocation::URL(_) => true,
  };
  let input_json = serde_json::to_string(&input).unwrap();

  // if is_wasm {
  //     generate_witness_from_wasm::<F<G1>>(
  //         &witness_generator_file,
  //         &input_json,
  //         &witness_generator_output,
  //     )
  // } else {
  let witness_generator_file = match &witness_generator_file {
    FileLocation::PathBuf(path) => path,
    FileLocation::URL(_) => panic!("unreachable"),
  };
  generate_witness_from_bin::<F<G1>>(
    &witness_generator_file,
    &input_json,
    &witness_generator_output,
  )
  // }
}

#[derive(Clone)]
pub struct JoltCircuit<F: PrimeField> {
  z: Vec<AllocatedNum<F>>,
  circom_circuit: CircomCircuit<F>,
}

// impl<F: PrimeField> Circuit<F> for JoltCircuit<F> {
impl<F: PrimeField> Circuit<F> for JoltCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let _ = self
      .circom_circuit
      .vanilla_synthesize(cs, self.z.as_slice()).unwrap();
    Ok(())
  }
}

#[cfg(not(target_family = "wasm"))]
pub fn create_snark<G1, G2, S, C>(
  witness_generator_file: FileLocation,
  r1cs: R1CS<F<G1>>,
  private_inputs: HashMap<String, Value>,
  start_public_input: Vec<F<G1>>,
  // pp: &PublicParams<G1, G2, C1<G1>, C2<G2>>,
) -> Result<SNARK<G1, S, JoltCircuit<<G1 as Group>::Scalar>>, std::io::Error>
where
  G1: spartan2::traits::Group,
  S: spartan2::traits::snark::RelaxedR1CSSNARKTrait<G1>,
  // C: bellperson::Circuit<<G1 as spartan2::traits::Group>::Scalar>,
  C: bellperson::Circuit<<G1 as spartan2::traits::Group>::Scalar>,
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  let root = current_dir().unwrap();
  let witness_generator_output = root.join("circom_witness.wtns");
  let start_public_input_hex = start_public_input
    .iter()
    .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
    .collect::<Vec<String>>();
  let mut current_public_input = start_public_input_hex.clone();

  let witness_0 = compute_witness::<G1, G2>(
    current_public_input.clone(),
    private_inputs.clone(),
    witness_generator_file.clone(),
    &witness_generator_output,
  );

  let circuit_0 = CircomCircuit {
    r1cs: r1cs.clone(),
    witness: Some(witness_0),
  };

  // let mut recursive_snark = SNARK::<G1, S, C>::new(
  //     &pp,
  //     &circuit_0,
  //     start_public_input.clone(),
  //     z0_secondary.clone(),
  // );

  let jolt_circuit = JoltCircuit {
    circom_circuit: circuit_0,
    z: vec![],
  };

  let (pk, vk) = SNARK::<G1, S, JoltCircuit<<G1 as Group>::Scalar>>::setup(jolt_circuit.clone()).unwrap();

    let witness = compute_witness::<G1, G2>(
      current_public_input.clone(),
      private_inputs.clone(),
      witness_generator_file.clone(),
      &witness_generator_output,
    );

    let circuit = CircomCircuit {
      r1cs: r1cs.clone(),
      witness: Some(witness),
    };

    let current_public_output = circuit.get_public_outputs();
    current_public_input = current_public_output
      .iter()
      .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
      .collect();

    // let res = recursive_snark.prove_step(
    //     &pp,
    //     &circuit,
    //     &circuit_secondary,
    //     start_public_input.clone(),
    //     z0_secondary.clone(),
    // );
    let res = SNARK::prove(&pk, jolt_circuit);
    assert!(res.is_ok());
    let snark = res.unwrap();
        // verify the SNARK
        let res = snark.verify(&vk);
        assert!(res.is_ok());
    
  fs::remove_file(witness_generator_output)?;

  Ok(snark)
}

mod test {
    use std::hash::Hash;

    use serde_json::json;

    use super::*;

    #[test]
    fn test_nova_scotia_method() {
        type G1 = pasta_curves::pallas::Point;
        type G2 = pasta_curves::vesta::Point;

        type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
        type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;
        type Spp = spartan2::spartan::ppsnark::RelaxedR1CSSNARK<G1, EE>;

        let root = current_dir().unwrap().join("src/r1cs/circuits/jolt/");
        let r1cs_path = root.join("jolt.r1cs");
        let wtns_path = root.join("jolt_js/jolt.wasm");
    
        let jolt_r1cs = nova_scotia::circom::reader::load_r1cs::<G1, G2>(&nova_scotia::FileLocation::PathBuf(r1cs_path));

        let mut private_input = HashMap::new();
        private_input.insert("in1".to_string(), json!("0"));
        private_input.insert("in2".to_string(), json!("0"));

        let start_public_input = [F::<G1>::from(0), F::<G1>::from(0)];
    
        let _ = create_snark::<G1, G2, S, JoltCircuit<<G1 as Group>::Scalar>>(FileLocation::PathBuf(wtns_path), jolt_r1cs, private_input, start_public_input.to_vec());

    }
}