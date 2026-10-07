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
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use akita_config::{policy_of, CommitmentConfig};
    use akita_params::{
        FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout, ScheduleLookupKey,
    };
    use akita_pcs::AkitaError;
    use akita_planner::emit::GroupedGenerationRequest;
    use akita_planner::EmitSpec;
    use jolt_claims::protocols::jolt::lattice::strategy::MAX_ONE_HOT_TRACE_COLUMNS;
    use jolt_claims::protocols::jolt::lattice::{one_hot_trace_columns, OneHotTraceShape};
    use jolt_claims::protocols::jolt::{JoltFormulaDimensions, JoltOneHotDimensions};

    use crate::configs::{AkitaChunkProfile, JoltDenseBounded, JoltDenseFull};
    use crate::one_hot_family::{with_one_hot_family, OneHotFamily};
    use crate::planning::plan_schedule;
    use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

    /// Admitted production trace exponents for K=16.
    pub const K16_TRACE_LOG_T: (usize, usize) = (12, 30);
    pub const K16_NUM_VARS: (usize, usize) = (
        K16_TRACE_LOG_T.0 + K16_COLUMN_VARIABLES,
        K16_TRACE_LOG_T.1 + K16_COLUMN_VARIABLES,
    );
    /// One- and two-polynomial rows used by adapter and grouped-planner diagnostics.
    pub const ONE_HOT_SCALAR_NUM_POLYS: &[usize] = &[1, 2];
    pub const K16_SCALAR_NUM_VARS: (usize, usize) = (12, 40);
    pub const K256_SCALAR_NUM_VARS: (usize, usize) = (12, 43);
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 34);

    /// First logical trace exponent using setup offloading.
    pub const RECURSIVE_TRACE_LOG_T_CUTOVER: usize = 21;
    pub const K16_COLUMN_VARIABLES: usize = 4;
    pub const K256_COLUMN_VARIABLES: usize = 8;

    fn regen<Cfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        plan_schedule::<Cfg>(&ScheduleLookupKey::single(key), &[])
    }

    fn regen_one_hot_k16<Cfg: CommitmentConfig, DirectCfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES {
            regen::<Cfg>(key)
        } else {
            regen::<DirectCfg>(key)
        }
    }

    fn regen_one_hot_k256<Cfg: CommitmentConfig, DirectCfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_COLUMN_VARIABLES {
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

    /// Default production shapes plus the exact adapter, benchmark, and override fixtures.
    pub fn one_hot_keys(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        let mut admitted = ONE_HOT_SCALAR_NUM_POLYS
            .iter()
            .flat_map(|&num_polys| keys(&[num_polys], family.num_vars_range(num_polys)))
            .collect::<Vec<_>>();
        if one_hot_k == AKITA_ONE_HOT_K256 {
            // K=256 is exercised only by explicit adapter, benchmark, and override fixtures.
            let fixtures: &[(usize, usize)] = if profile == AkitaChunkProfile::Single {
                &[
                    (14, 1),
                    (15, 1),
                    (16, 1),
                    (20, 1),
                    (20, 29),
                    (25, 1),
                    (28, 27),
                    (29, 27),
                    (34, 27),
                ]
            } else {
                &[(16, 1)]
            };
            admitted.extend(
                fixtures
                    .iter()
                    .map(|&(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
            );
            admitted.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
            admitted.dedup();
            return Ok(admitted);
        }
        if one_hot_k != AKITA_ONE_HOT_K16 {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported one-hot K={one_hot_k}"
            )));
        }
        let column_variables = K16_COLUMN_VARIABLES;
        // RV64 lookup keys have two 64-bit operands. Bytecode PCs fit u32,
        // and remapped RAM addresses are u64 byte addresses divided by eight.
        let mut widths = BTreeSet::new();
        for bytecode_bits in 1..=u32::BITS {
            for ram_bits in 1..=u64::BITS - 3 {
                let dimensions = JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                    log_t: 0,
                    instruction_address_bits: 128,
                    bytecode_k: 1usize << bytecode_bits,
                    ram_k: 1usize << ram_bits,
                    committed_chunk_bits: column_variables,
                    lookup_virtual_chunk_bits: 32,
                })
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
                let columns = one_hot_trace_columns(&OneHotTraceShape {
                    ra_layout: dimensions.ra_layout,
                    log_t: 0,
                    log_k_chunk: column_variables,
                })
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
                if columns.len() <= MAX_ONE_HOT_TRACE_COLUMNS {
                    let _ = widths.insert(columns.len());
                }
            }
        }
        admitted.extend(keys(&widths.into_iter().collect::<Vec<_>>(), K16_NUM_VARS));
        let fixtures: &[(usize, usize)] = if profile == AkitaChunkProfile::Single {
            &[(12, 1), (12, 2), (16, 1), (25, 1)]
        } else {
            &[(16, 1)]
        };
        admitted.extend(
            fixtures
                .iter()
                .map(|&(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
        );
        admitted.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
        admitted.dedup();
        Ok(admitted)
    }

    fn spec<Cfg: CommitmentConfig>(
        family_name: &'static str,
        keys: Vec<PolynomialGroupLayout>,
        regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
        output_dir: PathBuf,
    ) -> Result<EmitSpec, AkitaError> {
        Ok(EmitSpec {
            family_name,
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
    pub fn family_specs(output_dir: PathBuf) -> Result<Vec<EmitSpec>, AkitaError> {
        let mut specs = Vec::with_capacity(OneHotFamily::ALL.len() + 2);
        for family in OneHotFamily::ALL.iter().copied() {
            specs.push(with_one_hot_family!(family, |Cfg, DirectCfg| {
                let regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> =
                    if family.k() == AKITA_ONE_HOT_K16 {
                        regen_one_hot_k16::<Cfg, DirectCfg>
                    } else {
                        regen_one_hot_k256::<Cfg, DirectCfg>
                    };
                spec::<Cfg>(
                    family.family_name(),
                    one_hot_keys(family.k(), family.profile())?,
                    regen,
                    output_dir.clone(),
                )
            })?);
        }
        specs.extend([
            spec::<JoltDenseBounded>(
                JoltDenseBounded::schedule_family_name(),
                keys(&[1, 2], DENSE_NUM_VARS),
                regen::<JoltDenseBounded>,
                output_dir.clone(),
            )?,
            spec::<JoltDenseFull>(
                JoltDenseFull::schedule_family_name(),
                keys(&[1], DENSE_NUM_VARS),
                regen::<JoltDenseFull>,
                output_dir,
            )?,
        ]);
        Ok(specs)
    }
}
