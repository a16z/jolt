use jolt_field::Field;

use crate::stages::{
    stage6::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput},
    stage7::{Stage7ClearOutput, Stage7Output, Stage7ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage6: &'a Stage6ClearOutput<F>,
        stage7: &'a Stage7ClearOutput<F>,
    },
    Zk {
        stage6: &'a Stage6ZkOutput<F, C>,
        stage7: &'a Stage7ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage6: &'a Stage6Output<F, C>,
    stage7: &'a Stage7Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage6, stage7) {
        (Stage6Output::Clear(stage6), Stage7Output::Clear(stage7)) => {
            Ok(Deps::Clear { stage6, stage7 })
        }
        (Stage6Output::Zk(stage6), Stage7Output::Zk(stage7)) => Ok(Deps::Zk { stage6, stage7 }),
        (Stage6Output::Clear(_), Stage7Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage7" })
        }
        (Stage6Output::Zk(_), Stage7Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage7" })
        }
    }
}
