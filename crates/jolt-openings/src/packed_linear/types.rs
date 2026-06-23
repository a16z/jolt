use std::{fmt, marker::PhantomData};

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::{BatchOpeningResult, OpeningsError, PackedFamilyRef};

pub struct PackedLinearBatch<PCS, L = crate::PackedWitnessLayout>(PhantomData<fn() -> (PCS, L)>);

impl<PCS, L> PackedLinearBatch<PCS, L> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS, L> Default for PackedLinearBatch<PCS, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<PCS, L> Clone for PackedLinearBatch<PCS, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<PCS, L> Copy for PackedLinearBatch<PCS, L> {}

impl<PCS, L> fmt::Debug for PackedLinearBatch<PCS, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedLinearBatch").finish()
    }
}

impl<PCS, L> PartialEq for PackedLinearBatch<PCS, L> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<PCS, L> Eq for PackedLinearBatch<PCS, L> {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearSetupParams<PCSParams, L = crate::PackedWitnessLayout> {
    pub pcs: PCSParams,
    pub layout: L,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearProverSetup<PCSSetup, L = crate::PackedWitnessLayout> {
    pub pcs: PCSSetup,
    pub layout: L,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedLinearVerifierSetup<PCSSetup, L = crate::PackedWitnessLayout> {
    pub pcs: PCSSetup,
    pub layout: L,
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
