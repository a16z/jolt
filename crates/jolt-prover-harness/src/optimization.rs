use std::collections::BTreeSet;

use crate::{FrontierSpec, HarnessError, HarnessResult};

pub const NON_PERFORMANCE_FRONTIER_ID: &str = "NON-PERF";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KnownOptimizationIds {
    ids: BTreeSet<String>,
}

impl KnownOptimizationIds {
    pub fn parse_inventory(markdown: &str) -> HarnessResult<Self> {
        let mut ids = BTreeSet::new();
        for line in markdown.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("| OPT-") {
                continue;
            }
            let mut columns = trimmed.split('|').map(str::trim);
            let _ = columns.next();
            let Some(id) = columns.next() else {
                continue;
            };
            if !ids.insert(id.to_owned()) {
                return Err(HarnessError::InvalidOptimizationInventory {
                    reason: format!("duplicate optimization ID `{id}`"),
                });
            }
        }
        if ids.is_empty() {
            return Err(HarnessError::InvalidOptimizationInventory {
                reason: "optimization inventory contains no OPT-* IDs".to_owned(),
            });
        }
        Ok(Self { ids })
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.ids.iter().map(String::as_str)
    }
}

pub fn validate_frontier_optimization_ids(
    frontier: FrontierSpec,
    known: &KnownOptimizationIds,
) -> HarnessResult<()> {
    frontier.validate()?;
    for id in frontier.optimization_ids {
        if *id == NON_PERFORMANCE_FRONTIER_ID || known.contains(id) {
            continue;
        }
        return Err(HarnessError::InvalidManifest {
            frontier: frontier.name,
            reason: format!("unknown optimization-inventory ID `{id}`"),
        });
    }
    Ok(())
}
