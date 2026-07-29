# Static Analysis Objective Template

Create `jolt-eval/src/objective/code_quality/<objective_name>.rs`:

```rust
use std::path::Path;
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const <UPPER_NAME>: OptimizationObjective =
    OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::<VariantName>(<Name>Objective {
        target_dir: "<target_directory>",
    }));

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct <Name>Objective {
    pub(crate) target_dir: &'static str,
}

impl <Name>Objective {
    pub fn collect_measurement_in(&self, repo_root: &Path) -> Result<f64, MeasurementError> {
        let src_dir = repo_root.join(self.target_dir);
        // Implement measurement logic
        todo!()
    }
}

impl Objective for <Name>Objective {
    type Setup = ();

    fn name(&self) -> &str { "<objective_name>" }

    fn description(&self) -> String {
        format!("Description of measurement in {}", self.target_dir)
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        self.collect_measurement_in(repo_root)
    }

    fn units(&self) -> Option<&str> { Some("units") }
}
```
