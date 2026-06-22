use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::{BatchOpeningResult, BatchOpeningScheme, OpeningsError, PackedFamilyRef};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearBatch<PCS>(PhantomData<PCS>);

impl<PCS> PackedLinearBatch<PCS> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS> Default for PackedLinearBatch<PCS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearReductionProof {
    pub rounds: Vec<[Vec<u8>; 3]>,
    pub opening_eval: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearBatchProof<NativeProof> {
    pub reduction: Option<PackedLinearReductionProof>,
    pub native: NativeProof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLinearFamily {
    pub id: PackedFamilyRef,
    pub offset: usize,
    pub rows: usize,
    pub limbs: usize,
    pub alphabet_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLinearAddress {
    pub family: PackedFamilyRef,
    pub row: usize,
    pub limb: usize,
    pub symbol: usize,
}

pub trait PackedLinearLayout {
    fn digest(&self) -> [u8; 32];
    fn dimension(&self) -> usize;
    fn cells(&self) -> usize;
    fn family(&self, family: PackedFamilyRef) -> Result<Option<PackedLinearFamily>, OpeningsError>;
    fn rank(&self, address: PackedLinearAddress) -> Result<usize, OpeningsError>;
}

pub trait PackedLinearWitnessSource<F>
where
    F: Field,
{
    type Layout: PackedLinearLayout;

    fn layout(&self) -> &Self::Layout;

    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
}

pub trait PackedLinearBatchBackend: BatchOpeningScheme {
    type Layout: PackedLinearLayout;

    fn prover_layout(setup: &Self::ProverSetup) -> Option<&Self::Layout>;

    fn verifier_layout(setup: &Self::VerifierSetup) -> Option<&Self::Layout>;

    fn validate_packed_prover_inputs(
        _setup: &Self::ProverSetup,
        layout: &Self::Layout,
        _commitment: &Self::Output,
        polynomials: &[Self::Polynomial],
        hints: &[Self::OpeningHint],
    ) -> Result<(), OpeningsError> {
        if polynomials.len() != 1 {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed linear proof expects one packed polynomial, got {}",
                polynomials.len()
            )));
        }
        if polynomials[0].num_vars() != layout.dimension() {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed linear polynomial has {} variables but layout has {}",
                polynomials[0].num_vars(),
                layout.dimension()
            )));
        }
        if hints.len() != 1 {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed linear proof expects one opening hint, got {}",
                hints.len()
            )));
        }
        Ok(())
    }

    fn validate_packed_verifier_inputs(
        _setup: &Self::VerifierSetup,
        _layout: &Self::Layout,
        _commitment: &Self::Output,
    ) -> Result<(), OpeningsError> {
        Ok(())
    }

    fn bind_packed_prover_setup<T>(_setup: &Self::ProverSetup, _transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
    }

    fn bind_packed_verifier_setup<T>(_setup: &Self::VerifierSetup, _transcript: &mut T)
    where
        T: Transcript<Challenge = Self::Field>,
    {
    }
}

pub struct PackedLinearProverReduction<F> {
    pub proof: PackedLinearReductionProof,
    pub opening_point: Vec<F>,
    pub opening_eval: F,
}

pub struct PackedLinearVerifierReduction<F, C> {
    pub opening_point: Vec<F>,
    pub opening_eval: F,
    pub result: BatchOpeningResult<F, C>,
}
