pub mod nova_scotia_lib;
pub mod snark;

// mod test {
//     // use bellpepper_core::ConstraintSystem;
//     // use circom_scotia::{calculate_witness, r1cs::CircomConfig, synthesize};
//     // use ff::Field;

//     // use pasta_curves::vesta::Base as Fr;
//     // use std::env::current_dir;

//     // use bellpepper_core::test_cs::TestConstraintSystem;
//     // use bellpepper_core::Comparable;

//     // #[test]
//     // fn test_circom_scotia() {
//     //     let root = current_dir().unwrap().join("src/r1cs/circuits/sha256");
//     //     let wtns = root.join("circom_sha256.wasm");
//     //     let r1cs = root.join("circom_sha256.r1cs");

//     //     let mut cs = TestConstraintSystem::<Fr>::new();
//     //     let cfg = CircomConfig::new(wtns, r1cs).unwrap();

//     //     let arg_in = ("arg_in".into(), vec![Fr::ZERO, Fr::ZERO]);
//     //     let input = vec![arg_in];
//     //     let witness = calculate_witness(&cfg, input, true).expect("msg");

//     //     let output = synthesize(
//     //         &mut cs.namespace(|| "sha256_circom"),
//     //         cfg.r1cs.clone(),
//     //         Some(witness),
//     //     );

//     //     let expected = "0x00000000008619b3767c057fdf8e6d99fde2680c5d8517eb06761c0878d40c40";
//     //     let output_num = format!("{:?}", output.unwrap().get_value().unwrap());
//     //     assert!(output_num == expected);

//     //     assert!(cs.is_satisfied());
//     //     assert_eq!(30134, cs.num_constraints());
//     //     assert_eq!(1, cs.num_inputs());
//     //     assert_eq!(29822, cs.aux().len());

//     //     println!("Congrats! You synthesized and satisfied a circom sha256 circuit in bellperson!");
//     // }

//     use bellperson::Circuit;
//     use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
//     use spartan2::{traits::{Group, snark::RelaxedR1CSSNARKTrait}, SNARK};
//     use core::marker::PhantomData;
//     use ff::PrimeField;

//     #[derive(Clone, Debug, Default)]
//     struct CubicCircuit<F: PrimeField> {
//       _p: PhantomData<F>,
//     }

//     impl<F> Circuit<F> for CubicCircuit<F>
//     where
//       F: PrimeField,
//     {
//       fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
//         // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
//         let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(F::ONE + F::ONE))?;
//         let x_sq = x.square(cs.namespace(|| "x_sq"))?;
//         let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
//         let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
//           Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
//         })?;

//         cs.enforce(
//           || "y = x^3 + x + 5",
//           |lc| {
//             lc + x_cu.get_variable()
//               + x.get_variable()
//               + CS::one()
//               + CS::one()
//               + CS::one()
//               + CS::one()
//               + CS::one()
//           },
//           |lc| lc + CS::one(),
//           |lc| lc + y.get_variable(),
//         );

//         let _ = y.inputize(cs.namespace(|| "output"));

//         Ok(())
//       }
//     }

//     #[test]
//     fn test_snark() {
//       type G = pasta_curves::pallas::Point;
//       type EE = spartan2::provider::ipa_pc::EvaluationEngine<G>;
//       type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G, EE>;
//       type Spp = spartan2::spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
//       test_snark_with::<G, S>();
//       test_snark_with::<G, Spp>();
//     }

//     fn test_snark_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() {
//       let circuit = CubicCircuit::default();

//       // produce keys
//       let (pk, vk) =
//         SNARK::<G, S, CubicCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

//       // produce a SNARK
//       let res = SNARK::prove(&pk, circuit);
//       assert!(res.is_ok());
//       let snark = res.unwrap();

//       // verify the SNARK
//       let res = snark.verify(&vk);
//       assert!(res.is_ok());

//       let io = res.unwrap();

//       // sanity: check the claimed output with a direct computation of the same
//       assert_eq!(io, vec![<G as Group>::Scalar::from(15u64)]);
//     }
// }
