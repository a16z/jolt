use std::{fmt, marker::PhantomData};

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::{BatchOpeningResult, OpeningsError, PackingFamilyRef};

pub struct PackingBatch<PCS, L = crate::PackingWitnessLayout>(PhantomData<fn() -> (PCS, L)>);

impl<PCS, L> PackingBatch<PCS, L> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS, L> Default for PackingBatch<PCS, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<PCS, L> Clone for PackingBatch<PCS, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<PCS, L> Copy for PackingBatch<PCS, L> {}

impl<PCS, L> fmt::Debug for PackingBatch<PCS, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackingBatch").finish()
    }
}

impl<PCS, L> PartialEq for PackingBatch<PCS, L> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<PCS, L> Eq for PackingBatch<PCS, L> {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingSetupParams<PCSParams, L = crate::PackingWitnessLayout> {
    pub pcs: PCSParams,
    pub layout: L,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackingProverSetup<PCSSetup, L = crate::PackingWitnessLayout> {
    pub pcs: PCSSetup,
    pub layout: L,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingVerifierSetup<PCSSetup, L = crate::PackingWitnessLayout> {
    pub pcs: PCSSetup,
    pub layout: L,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingReductionProof {
    pub rounds: Vec<[Vec<u8>; 3]>,
    pub opening_eval: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingBatchProof<NativeProof> {
    pub reduction: Option<PackingReductionProof>,
    pub native: NativeProof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingFamily {
    pub id: PackingFamilyRef,
    pub offset: usize,
    pub rows: usize,
    pub limbs: usize,
    pub alphabet_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingAddress {
    pub family: PackingFamilyRef,
    pub row: usize,
    pub limb: usize,
    pub symbol: usize,
}

pub trait PackingLayout {
    fn digest(&self) -> [u8; 32];
    fn dimension(&self) -> usize;
    fn cells(&self) -> usize;
    fn family(&self, family: PackingFamilyRef) -> Result<Option<PackingFamily>, OpeningsError>;
    fn rank(&self, address: PackingAddress) -> Result<usize, OpeningsError>;
}

pub trait PackingSource<F>
where
    F: Field,
{
    type Layout: PackingLayout;

    fn layout(&self) -> &Self::Layout;

    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
}

pub struct PackingProverReduction<F> {
    pub proof: PackingReductionProof,
    pub opening_point: Vec<F>,
    pub opening_eval: F,
}

pub struct PackingVerifierReduction<F, C> {
    pub opening_point: Vec<F>,
    pub opening_eval: F,
    pub result: BatchOpeningResult<F, C>,
}
