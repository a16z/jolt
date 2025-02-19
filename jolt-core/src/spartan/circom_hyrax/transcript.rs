use ark_grumpkin::{Fq as Fp, Fr as Scalar, Projective};
use core::fmt;

use crate::utils::poseidon_transcript::PoseidonTranscript;
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TestTranscript {
    pub state: Fp,
    pub nrounds: Fp,
}

impl fmt::Debug for TestTranscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "state": "{}",
            "nRounds": "{}"
            }}"#,
            self.state, self.nrounds
        )
    }
}

pub fn convert_transcript_to_circom(transcript: PoseidonTranscript<Scalar, Fp>) -> TestTranscript {
    TestTranscript {
        state: Fp::from(transcript.state.state[1]),
        nrounds: Fp::from(transcript.n_rounds),
    }
}
