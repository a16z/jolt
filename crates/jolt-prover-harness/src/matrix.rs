use serde::Serialize;

use crate::FeatureMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureMatrixRow {
    pub name: &'static str,
    pub mode: FeatureMode,
    pub cargo_features: &'static [&'static str],
}

pub const REQUIRED_FEATURE_MATRIX: &[FeatureMatrixRow] = &[
    FeatureMatrixRow {
        name: "default",
        mode: FeatureMode::Transparent,
        cargo_features: &[],
    },
    FeatureMatrixRow {
        name: "field-inline",
        mode: FeatureMode::FieldInline,
        cargo_features: &["field-inline"],
    },
    FeatureMatrixRow {
        name: "zk",
        mode: FeatureMode::Zk,
        cargo_features: &["zk"],
    },
    FeatureMatrixRow {
        name: "zk,field-inline",
        mode: FeatureMode::ZkFieldInline,
        cargo_features: &["zk", "field-inline"],
    },
    FeatureMatrixRow {
        name: "core-fixtures",
        mode: FeatureMode::Transparent,
        cargo_features: &["core-fixtures"],
    },
    FeatureMatrixRow {
        name: "core-fixtures,zk",
        mode: FeatureMode::Zk,
        cargo_features: &["core-fixtures", "zk"],
    },
    FeatureMatrixRow {
        name: "core-fixtures,field-inline",
        mode: FeatureMode::FieldInline,
        cargo_features: &["core-fixtures", "field-inline"],
    },
    FeatureMatrixRow {
        name: "core-fixtures,zk,field-inline",
        mode: FeatureMode::ZkFieldInline,
        cargo_features: &["core-fixtures", "zk", "field-inline"],
    },
];
