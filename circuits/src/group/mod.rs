mod test_g1;
mod test_g2;
use core::fmt;
use ark_bn254::{G1Projective};
pub mod test_g1_grumpkin;
use ark_bn254::{Fq, Fr};

// #[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
// pub struct G1ProjectiveCircom{
//     pub x: Fq,
//     pub y: Fq,
//     pub z: Fq,
// }

// impl fmt::Debug for G1ProjectiveCircom {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             r#"{{
//                             "x": "{}",
//                             "y": "{}",
//                             "z": "{}"
//                             }}"#,
//             self.x,
//             self.y,
//             self.z
//         )
//     }
// }

// pub fn convert_g1_proj_to_circom(pt: G1Projective) -> G1ProjectiveCircom {
//     G1ProjectiveCircom{
//         x: pt.x,
//         y: pt.y,
//         z: pt.z,
//     }
// }