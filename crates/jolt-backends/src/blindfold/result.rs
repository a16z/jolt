use jolt_crypto::VectorCommitmentOpening;
use jolt_field::Field;

use crate::BackendValueSlot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldPrivateOpening<F: Field> {
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> BlindFoldPrivateOpening<F> {
    pub const fn new(slot: BackendValueSlot, value: F) -> Self {
        Self { slot, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldResult<F: Field, Proof> {
    pub proof: Proof,
    pub private_openings: Vec<BlindFoldPrivateOpening<F>>,
}

impl<F: Field, Proof> BlindFoldResult<F, Proof> {
    pub const fn new(proof: Proof, private_openings: Vec<BlindFoldPrivateOpening<F>>) -> Self {
        Self {
            proof,
            private_openings,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRowCommitmentResult<Commitment> {
    pub commitments: Vec<Commitment>,
}

impl<Commitment> BlindFoldRowCommitmentResult<Commitment> {
    pub const fn new(commitments: Vec<Commitment>) -> Self {
        Self { commitments }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRowOpeningResult<F: Field> {
    pub opening: VectorCommitmentOpening<F>,
    pub evaluation: F,
}

impl<F: Field> BlindFoldRowOpeningResult<F> {
    pub const fn new(opening: VectorCommitmentOpening<F>, evaluation: F) -> Self {
        Self {
            opening,
            evaluation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldErrorRowsResult<F: Field> {
    pub rows: Vec<Vec<F>>,
}

impl<F: Field> BlindFoldErrorRowsResult<F> {
    pub const fn new(rows: Vec<Vec<F>>) -> Self {
        Self { rows }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldRowsResult<F: Field> {
    pub rows: Vec<Vec<F>>,
}

impl<F: Field> BlindFoldFoldRowsResult<F> {
    pub const fn new(rows: Vec<Vec<F>>) -> Self {
        Self { rows }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldScalarsResult<F: Field> {
    pub scalars: Vec<F>,
}

impl<F: Field> BlindFoldFoldScalarsResult<F> {
    pub const fn new(scalars: Vec<F>) -> Self {
        Self { scalars }
    }
}
