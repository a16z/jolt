use std::collections::{btree_map::Entry, BTreeMap};

use serde::{Deserialize, Serialize};

use crate::NamedValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonTarget {
    CoreCommitments,
    CoreStageOutput,
    CoreOpeningClaims,
    CoreProofShape,
    BackendReference,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityMismatch {
    pub path: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityReport {
    pub target: ComparisonTarget,
    pub mismatches: Vec<ParityMismatch>,
}

impl ParityReport {
    pub fn success(target: ComparisonTarget) -> Self {
        Self {
            target,
            mismatches: Vec::new(),
        }
    }

    pub fn is_success(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn compare_named_values(
    target: ComparisonTarget,
    expected: &[NamedValue],
    actual: &[NamedValue],
) -> ParityReport {
    let (expected, mut mismatches) = by_name("expected", expected);
    let (actual, actual_duplicates) = by_name("actual", actual);
    mismatches.extend(actual_duplicates);

    for (name, expected_value) in &expected {
        match actual.get(name) {
            Some(actual_value) if actual_value == expected_value => {}
            Some(actual_value) => mismatches.push(ParityMismatch {
                path: name.clone(),
                expected: Some(expected_value.clone()),
                actual: Some(actual_value.clone()),
            }),
            None => mismatches.push(ParityMismatch {
                path: name.clone(),
                expected: Some(expected_value.clone()),
                actual: None,
            }),
        }
    }

    for (name, actual_value) in &actual {
        if !expected.contains_key(name) {
            mismatches.push(ParityMismatch {
                path: name.clone(),
                expected: None,
                actual: Some(actual_value.clone()),
            });
        }
    }

    ParityReport { target, mismatches }
}

fn by_name(
    side: &'static str,
    values: &[NamedValue],
) -> (BTreeMap<String, String>, Vec<ParityMismatch>) {
    let mut map = BTreeMap::new();
    let mut duplicates = Vec::new();

    for value in values {
        match map.entry(value.name.clone()) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(value.value.clone());
            }
            Entry::Occupied(entry) => duplicates.push(ParityMismatch {
                path: format!("{side}.{}", value.name),
                expected: Some(format!("single value `{}`", entry.get())),
                actual: Some(format!("duplicate value `{}`", value.value)),
            }),
        }
    }

    (map, duplicates)
}
