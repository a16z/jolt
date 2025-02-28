// use test_transcript::helper::Fqq;
use std::fmt;
use ark_bn254::{Fr as Scalar, Fq as Fp};
use jolt_core::{subprotocols::sumcheck::SumcheckInstanceProof, utils::poseidon_transcript::PoseidonTranscript};
use transcript::helper::convert_to_3_limbs;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SumcheckInstanceProofCircom{
    pub uni_polys: Vec<UniPolyCircom>,
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniPolyCircom{
    pub coeffs: Vec<Fqq>
}
impl fmt::Debug for SumcheckInstanceProofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\"uni_polys\": {:?}}}",
            self.uni_polys
        )
    }
}

impl fmt::Debug for UniPolyCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\"coeffs\": {:?}}}",
            self.coeffs
        )
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fqq{
    // pub element: Scalar,
    pub limbs: [Fp; 3],
}

impl fmt::Debug for Fqq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "limbs": ["{}", "{}", "{}"]
            }}"#,
             &self.limbs[0], &self.limbs[1].to_string(), &self.limbs[2].to_string()
        )
    }
}

pub fn convert_sum_check_proof_to_circom(
    sum_check_proof: &SumcheckInstanceProof<Scalar, PoseidonTranscript<Fp>>,
) -> SumcheckInstanceProofCircom {
    let mut uni_polys_circom = Vec::new();
    for poly in &sum_check_proof.uni_polys {
        let mut temp_coeffs = Vec::new();
        for coeff in &poly.coeffs {
            temp_coeffs.push(Fqq {
                // element: *coeff,
                limbs: convert_to_3_limbs(*coeff),
            });
        }
        uni_polys_circom.push(UniPolyCircom {
            coeffs: temp_coeffs,
        });
    }
    SumcheckInstanceProofCircom {
        uni_polys: uni_polys_circom,
    }
}
