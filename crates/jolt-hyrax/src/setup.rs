use jolt_crypto::{DeriveSetup, VectorCommitment};
use serde::{Deserialize, Serialize};

use crate::{HyraxDimensions, HyraxError};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VC::Setup: Serialize",
    deserialize = "VC::Setup: for<'a> Deserialize<'a>"
))]
pub struct HyraxSetupParams<VC: VectorCommitment> {
    pub dimensions: HyraxDimensions,
    pub vc_setup: VC::Setup,
}

impl<VC: VectorCommitment> HyraxSetupParams<VC> {
    pub fn new(dimensions: HyraxDimensions, vc_setup: VC::Setup) -> Result<Self, HyraxError> {
        validate_capacity::<VC>(&dimensions, &vc_setup)?;
        Ok(Self {
            dimensions,
            vc_setup,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VC::Setup: Serialize",
    deserialize = "VC::Setup: for<'a> Deserialize<'a>"
))]
pub struct HyraxProverSetup<VC: VectorCommitment> {
    pub dimensions: HyraxDimensions,
    pub vc_setup: VC::Setup,
}

impl<VC: VectorCommitment> HyraxProverSetup<VC> {
    pub fn new(dimensions: HyraxDimensions, vc_setup: VC::Setup) -> Result<Self, HyraxError> {
        validate_capacity::<VC>(&dimensions, &vc_setup)?;
        Ok(Self {
            dimensions,
            vc_setup,
        })
    }

    pub fn derive_from<Source>(
        source: &Source,
        dimensions: HyraxDimensions,
    ) -> Result<Self, HyraxError>
    where
        VC::Setup: DeriveSetup<Source>,
    {
        dimensions.validate()?;
        let row_len = dimensions.row_len()?;
        let vc_setup = VC::Setup::derive(source, row_len);
        Self::new(dimensions, vc_setup)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "VC::Setup: Serialize",
    deserialize = "VC::Setup: for<'a> Deserialize<'a>"
))]
pub struct HyraxVerifierSetup<VC: VectorCommitment> {
    pub dimensions: HyraxDimensions,
    pub vc_setup: VC::Setup,
}

impl<VC: VectorCommitment> HyraxVerifierSetup<VC> {
    pub fn new(dimensions: HyraxDimensions, vc_setup: VC::Setup) -> Result<Self, HyraxError> {
        validate_capacity::<VC>(&dimensions, &vc_setup)?;
        Ok(Self {
            dimensions,
            vc_setup,
        })
    }

    pub fn derive_from<Source>(
        source: &Source,
        dimensions: HyraxDimensions,
    ) -> Result<Self, HyraxError>
    where
        VC::Setup: DeriveSetup<Source>,
    {
        dimensions.validate()?;
        let row_len = dimensions.row_len()?;
        let vc_setup = VC::Setup::derive(source, row_len);
        Self::new(dimensions, vc_setup)
    }
}

impl<VC: VectorCommitment> From<&HyraxProverSetup<VC>> for HyraxVerifierSetup<VC> {
    fn from(prover_setup: &HyraxProverSetup<VC>) -> Self {
        Self {
            dimensions: prover_setup.dimensions,
            vc_setup: prover_setup.vc_setup.clone(),
        }
    }
}

fn validate_capacity<VC: VectorCommitment>(
    dimensions: &HyraxDimensions,
    vc_setup: &VC::Setup,
) -> Result<(), HyraxError> {
    dimensions.validate()?;
    let row_len = dimensions.row_len()?;
    let capacity = VC::capacity(vc_setup);
    if row_len > capacity {
        return Err(HyraxError::CommitmentCapacityExceeded { capacity, row_len });
    }
    Ok(())
}
