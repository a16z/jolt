use std::fmt;

use super::hyrax::hyrax_commitment_to_circom;
use super::hyrax::HyraxCommitmentCircom;
use crate::{
    poly::commitment::hyrax::HyraxScheme, spartan::spartan_memory_checking::SpartanCommitments,
    utils::poseidon_transcript::PoseidonTranscript,
};
use ark_grumpkin::{Fq as Fp, Fr as Scalar, Projective};

pub struct SpartanCommitmentsHyraxCircom {
    pub witness: HyraxCommitmentCircom,
}

impl SpartanCommitmentsHyraxCircom {
    pub fn convert(
        commitments: &SpartanCommitments<
            HyraxScheme<Projective, PoseidonTranscript<Scalar, Fp>>,
            PoseidonTranscript<Scalar, Fp>,
        >,
    ) -> Self {
        Self {
            witness: hyrax_commitment_to_circom(&commitments.witness),
        }
    }
}

impl fmt::Debug for SpartanCommitmentsHyraxCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "w_commitment": {:?},
            }}"#,
            self.witness
        )
    }
}
