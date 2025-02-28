// // use test_transcript::helper::Fqq;
// use std::fmt;
// use ark_bn254::{Fr as Scalar, Fq as Fp};
// use jolt_core::{subprotocols::sumcheck::SumcheckInstanceProof, utils::poseidon_transcript::PoseidonTranscript};
// use transcript::helper::convert_to_3_limbs;

// #[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
// pub struct SumcheckInstanceProofCircom{
//     pub uni_polys: Vec<UniPolyCircom>,
// }

// #[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
// pub struct UniPolyCircom{
//     pub coeffs: Vec<Fqq>
// }
// impl fmt::Debug for SumcheckInstanceProofCircom {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "{{\"uni_polys\": {:?}}}",
//             self.uni_polys
//         )
//     }
// }

// impl fmt::Debug for UniPolyCircom {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "{{\"coeffs\": {:?}}}",
//             self.coeffs
//         )
//     }
// }

// #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
// pub struct Fqq{
//     pub element: Scalar,
//     pub limbs: [Fp; 3],
// }

// impl fmt::Debug for Fqq {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             r#"{{
//             "element": "{}",
//             "limbs": ["{}", "{}", "{}"]
//             }}"#,
//             self.element, &self.limbs[0], &self.limbs[1].to_string(), &self.limbs[2].to_string()
//         )
//     }
// }
