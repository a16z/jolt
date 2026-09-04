//! Jolt-owned external schedule catalog generation.
//!
//! Checked-in `.aks` artifacts are ordinary runtime data, not Rust modules.
//! Regenerate them with:
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
//! ```

/// Emit-spec construction shared by the generator and drift tests.
pub mod emit {
    use std::path::PathBuf;

    use akita_config::{policy_of, CommitmentConfig};
    use akita_pcs::AkitaError;
    use akita_planner::emit::GroupedGenerationRequest;
    use akita_planner::EmitSpec;
    use akita_types::{
        AkitaScheduleLookupKey, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout,
    };

    use crate::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
    use crate::planning::plan_schedule;

    /// Prefix packing produces one physical polynomial; two-polynomial rows
    /// cover adapter and tamper-test shapes.
    pub const ONE_HOT_TRACE_NUM_POLYS: &[usize] = &[1, 2];
    /// K=16 adds six selector variables to column arity `4 + log_T`.
    pub const K16_NUM_VARS: (usize, usize) = (12, 34);
    /// K=256 adds five selector variables to column arity `8 + log_T`.
    pub const K256_NUM_VARS: (usize, usize) = (12, 43);
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 34);

    /// Pure DP regeneration for `Cfg`; never consults an artifact.
    pub(crate) fn regen<Cfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        plan_schedule::<Cfg>(&AkitaScheduleLookupKey::single(key), &[])
    }

    fn reject_grouped(request: GroupedGenerationRequest) -> Result<FoldSchedule, AkitaError> {
        Err(AkitaError::InvalidSetup(format!(
            "jolt base families emit no grouped rows; refusing to plan {:?}",
            request.key()
        )))
    }

    /// Reachable scalar keys for one family grid.
    pub fn keys(
        num_polys: &[usize],
        (min_vars, max_vars): (usize, usize),
    ) -> Vec<PolynomialGroupLayout> {
        let mut keys = Vec::new();
        for &polys in num_polys {
            for num_vars in min_vars..=max_vars {
                let layout = OpeningClaimsLayout::new(num_vars, polys)
                    .and_then(|layout| layout.root_final_group_layout());
                if let Ok(key) = layout {
                    keys.push(key);
                }
            }
        }
        keys
    }

    fn spec<Cfg: CommitmentConfig>(
        family_name: &'static str,
        num_polys: &[usize],
        num_vars: (usize, usize),
        output_dir: PathBuf,
    ) -> Result<EmitSpec, AkitaError> {
        Ok(EmitSpec {
            family_name,
            policy: policy_of::<Cfg>(),
            source_contract: Cfg::committed_source_contract()?,
            keys: keys(num_polys, num_vars),
            grouped_requests: Vec::new(),
            preplanned_scalar: Vec::new(),
            output_dir,
            regen: regen::<Cfg>,
            regen_group_batch: reject_grouped,
            ring_challenge_config: Cfg::ring_challenge_config,
        })
    }

    /// All base family specs, in emission order.
    ///
    /// Instance-specific grouped advice/program rows are planned during setup
    /// and folded into the exact catalog serialized with that verifier setup.
    pub fn family_specs(output_dir: PathBuf) -> Result<[EmitSpec; 3], AkitaError> {
        Ok([
            spec::<JoltOneHotK16>(
                JoltOneHotK16::schedule_family_name(),
                ONE_HOT_TRACE_NUM_POLYS,
                K16_NUM_VARS,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256>(
                JoltOneHotK256::schedule_family_name(),
                ONE_HOT_TRACE_NUM_POLYS,
                K256_NUM_VARS,
                output_dir.clone(),
            )?,
            spec::<JoltDenseBounded>(
                JoltDenseBounded::schedule_family_name(),
                ONE_HOT_TRACE_NUM_POLYS,
                DENSE_NUM_VARS,
                output_dir,
            )?,
        ])
    }
}
