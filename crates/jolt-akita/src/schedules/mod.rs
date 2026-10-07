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
    use akita_planner::{EmitSpec, RootShape};
    use akita_types::{
        AkitaScheduleLookupKey, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout,
    };

    use crate::configs::{
        JoltDenseBounded, JoltFieldDigits, JoltOneHotK16, JoltOneHotK16Direct, JoltOneHotK256,
        JoltOneHotK256Direct, JoltSignedBytes, JoltSignedBytesDirect,
    };
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
    /// Slot variables the signed-byte trace `Q` adds to the logical trace
    /// exponent.
    pub const SIGNED_BYTE_PACKING_VARIABLES: usize = 5;
    /// `Q` arities from the shortest padded Akita trace (`2^12`) through the
    /// longest K=256 trace (`2^30`).
    pub const SIGNED_BYTE_NUM_VARS: (usize, usize) = (
        12 + SIGNED_BYTE_PACKING_VARIABLES,
        30 + SIGNED_BYTE_PACKING_VARIABLES,
    );
    /// The T29 `Q` root: one D128 byte plane over `2^16` positions. Unpinned,
    /// the setup-first objective picks `2^15` positions, whose root witness is
    /// 27% larger.
    pub const SIGNED_BYTE_PINNED_ROOT: (usize, RootShape) = (
        29 + SIGNED_BYTE_PACKING_VARIABLES,
        RootShape {
            positions_per_block: 1 << 16,
            ring_dimension: 128,
        },
    );
    /// The byte link's histogram groups (jolt-claims `HistogramGroup::ALL`):
    /// six 24-variable triple tables and one 17-variable RAM table.
    pub const FIELD_DIGIT_GROUPS: [PolynomialGroupLayout; 2] = [
        PolynomialGroupLayout::new(24, 6),
        PolynomialGroupLayout::new(17, 1),
    ];

    /// Pure DP regeneration for `Cfg`; never consults an artifact.
    fn regen<Cfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        plan_schedule::<Cfg>(&AkitaScheduleLookupKey::single(key), &[], None)
    }

    fn regen_one_hot_k16(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_PACKING_VARIABLES {
            regen::<JoltOneHotK16>(key)
        } else {
            regen::<JoltOneHotK16Direct>(key)
        }
    }

    fn regen_one_hot_k256(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_PACKING_VARIABLES {
            regen::<JoltOneHotK256>(key)
        } else {
            regen::<JoltOneHotK256Direct>(key)
        }
    }

    fn regen_signed_bytes(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        let (pinned_num_vars, pinned_root) = SIGNED_BYTE_PINNED_ROOT;
        if key.num_vars() == pinned_num_vars {
            plan_schedule::<JoltSignedBytes>(
                &AkitaScheduleLookupKey::single(key),
                &[],
                Some(pinned_root),
            )
        } else if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + SIGNED_BYTE_PACKING_VARIABLES {
            regen::<JoltSignedBytes>(key)
        } else {
            regen::<JoltSignedBytesDirect>(key)
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
        keys: Vec<PolynomialGroupLayout>,
        regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
        output_dir: PathBuf,
    ) -> Result<EmitSpec, AkitaError> {
        Ok(EmitSpec {
            family_name: Cfg::schedule_family_name(),
            policy: policy_of::<Cfg>(),
            source_contract: Cfg::committed_source_contract()?,
            keys,
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
    pub fn family_specs(output_dir: PathBuf) -> Result<[EmitSpec; 5], AkitaError> {
        Ok([
            spec::<JoltOneHotK16>(
                keys(ONE_HOT_TRACE_NUM_POLYS, K16_NUM_VARS),
                regen_one_hot_k16,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256>(
                keys(ONE_HOT_TRACE_NUM_POLYS, K256_NUM_VARS),
                regen_one_hot_k256,
                output_dir.clone(),
            )?,
            spec::<JoltDenseBounded>(
                keys(ONE_HOT_TRACE_NUM_POLYS, DENSE_NUM_VARS),
                regen::<JoltDenseBounded>,
                output_dir.clone(),
            )?,
            spec::<JoltSignedBytes>(
                keys(&[1], SIGNED_BYTE_NUM_VARS),
                regen_signed_bytes,
                output_dir.clone(),
            )?,
            spec::<JoltFieldDigits>(
                FIELD_DIGIT_GROUPS.to_vec(),
                regen::<JoltFieldDigits>,
                output_dir,
            )?,
        ])
    }
}
