//! Jolt-owned external schedule catalog generation.
//!
//! Checked-in `.aks` artifacts are ordinary runtime data, not Rust modules.
//! Regenerate them with:
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
//! # Only the full-field dense family:
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules dense-full
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

    use crate::configs::{AkitaOneHotChunkProfile, JoltDenseBounded, JoltDenseFull};
    use crate::one_hot_family::{with_one_hot_family, OneHotFamily};
    use crate::planning::plan_schedule;
    use crate::AKITA_ONE_HOT_K16;

    /// Prefix packing produces one physical polynomial; two-polynomial rows
    /// cover adapter and tamper-test shapes.
    pub const ONE_HOT_TRACE_NUM_POLYS: &[usize] = &[1, 2];
    /// K=16 adds six selector variables to column arity `4 + log_T`;
    /// the catalog covers logical traces through `2^30`.
    pub const K16_NUM_VARS: (usize, usize) = (12, 40);
    /// K=256 adds five selector variables to column arity `8 + log_T`.
    pub const K256_NUM_VARS: (usize, usize) = (12, 43);
    /// Multi-chunk profiles require enough witness geometry for two chunked fold levels.
    pub const MULTI_CHUNK_MIN_NUM_VARS: usize = 16;
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 34);

    /// First Jolt trace exponent whose one-hot row uses setup offloading.
    ///
    /// K=16 has ten packing variables (`4 + log_T` column arity plus six
    /// selectors), while K=256 has thirteen (`8 + log_T` plus five). Keeping
    /// the cutover in logical trace space makes the two artifact families
    /// describe the same deployment policy. In the crossover sweep, `log_T=20`
    /// missed the 2x single-thread verifier gate and its proof-only phase
    /// exceeded 10% overhead; `log_T=21` was the first size to clear both.
    pub const RECURSIVE_TRACE_LOG_T_CUTOVER: usize = 21;
    /// Physical one-hot arity added to the logical trace exponent for K=16.
    pub const K16_PACKING_VARIABLES: usize = 10;
    /// Physical one-hot arity added to the logical trace exponent for K=256.
    pub const K256_PACKING_VARIABLES: usize = 13;

    /// Pure DP regeneration for `Cfg`; never consults an artifact.
    fn regen<Cfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        plan_schedule::<Cfg>(&AkitaScheduleLookupKey::single(key), &[])
    }

    fn regen_one_hot_k16<Cfg: CommitmentConfig, DirectCfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_PACKING_VARIABLES {
            regen::<Cfg>(key)
        } else {
            regen::<DirectCfg>(key)
        }
    }

    fn regen_one_hot_k256<Cfg: CommitmentConfig, DirectCfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_PACKING_VARIABLES {
            regen::<Cfg>(key)
        } else {
            regen::<DirectCfg>(key)
        }
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
        regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
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
            regen,
            regen_group_batch: reject_grouped,
            ring_challenge_config: Cfg::ring_challenge_config,
        })
    }

    /// All base family specs, in emission order.
    ///
    /// Instance-specific grouped advice/program rows are planned during setup
    /// and folded into the exact catalog serialized with that verifier setup.
    pub fn family_specs(output_dir: PathBuf) -> Result<Vec<EmitSpec>, AkitaError> {
        let mut specs = Vec::with_capacity(OneHotFamily::ALL.len() + 2);
        for family in OneHotFamily::ALL.iter().copied() {
            let family_num_vars = if family.k() == AKITA_ONE_HOT_K16 {
                K16_NUM_VARS
            } else {
                K256_NUM_VARS
            };
            let num_vars = if family.profile() == AkitaOneHotChunkProfile::Single {
                family_num_vars
            } else {
                (MULTI_CHUNK_MIN_NUM_VARS, family_num_vars.1)
            };
            specs.push(with_one_hot_family!(family, |Cfg, DirectCfg| {
                let regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> =
                    if family.k() == AKITA_ONE_HOT_K16 {
                        regen_one_hot_k16::<Cfg, DirectCfg>
                    } else {
                        regen_one_hot_k256::<Cfg, DirectCfg>
                    };
                spec::<Cfg>(
                    family.family_name(),
                    ONE_HOT_TRACE_NUM_POLYS,
                    num_vars,
                    regen,
                    output_dir.clone(),
                )
            })?);
        }
        specs.extend([
            spec::<JoltDenseBounded>(
                JoltDenseBounded::schedule_family_name(),
                ONE_HOT_TRACE_NUM_POLYS,
                DENSE_NUM_VARS,
                regen::<JoltDenseBounded>,
                output_dir.clone(),
            )?,
            spec::<JoltDenseFull>(
                JoltDenseFull::schedule_family_name(),
                &[1],
                DENSE_NUM_VARS,
                regen::<JoltDenseFull>,
                output_dir,
            )?,
        ]);
        Ok(specs)
    }
}
