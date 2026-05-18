//! Compatibility opening-claim containers.

use std::collections::BTreeMap;

use crate::compat::ids::OpeningId;
use jolt_field::Field;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LegacyOpeningClaims<F: Field>(pub BTreeMap<OpeningId, F>);
