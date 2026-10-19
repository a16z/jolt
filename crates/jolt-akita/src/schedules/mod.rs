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
    use std::path::Path;

    use akita_config::{policy_of, CommitmentConfig};
    use akita_params::{OpeningClaimsLayout, PolynomialGroupLayout, ScheduleLookupKey};
    use akita_pcs::AkitaError;
    use akita_planner::emit::offline_planning_worker_count;
    use akita_planner::EmitSpec;
    use jolt_claims::protocols::jolt::lattice::strategy::MAX_ONE_HOT_TRACE_COLUMNS;
    use jolt_claims::protocols::jolt::lattice::{one_hot_trace_columns, OneHotTraceShape};
    use jolt_claims::protocols::jolt::{JoltFormulaDimensions, JoltOneHotDimensions};
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;

    use crate::configs::{AkitaChunkProfile, CatalogChunking, JoltDenseBounded, JoltDenseFull};
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
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 34);

    /// First logical trace exponent using setup offloading.
    pub const RECURSIVE_TRACE_LOG_T_CUTOVER: usize = 21;
    pub const K16_COLUMN_VARIABLES: usize = 4;
    pub const K256_COLUMN_VARIABLES: usize = 8;

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

    /// Default single-chunk shapes and the adapter's multi-chunk test fixtures.
    pub fn one_hot_keys(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        if profile != AkitaChunkProfile::Single {
            // These rows serve adapter roundtrips, grouped cutovers, and small guest
            // fixtures. Applications provision their own deployment coverage.
            let mut fixtures = ONE_HOT_SCALAR_NUM_POLYS
                .iter()
                .flat_map(|&polys| keys(&[polys], (family.num_vars_range(polys).0, 16)))
                .collect::<Vec<_>>();
            fixtures.extend([20, 22, 31, 32].map(PolynomialGroupLayout::singleton));
            if one_hot_k == AKITA_ONE_HOT_K16 {
                fixtures.extend((51..=64).map(|width| PolynomialGroupLayout::new(16, width)));
                fixtures.extend(
                    [(25, 51), (26, 51), (25, 64), (34, 51), (34, 64)]
                        .map(|(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
                );
            }
            fixtures.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
            fixtures.dedup();
            return Ok(fixtures);
        }
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

    fn spec<Cfg: CatalogChunking>(
        family_name: &'static str,
        keys: Vec<PolynomialGroupLayout>,
        output_dir: &Path,
    ) -> Result<EmitSpec, AkitaError> {
        Ok(EmitSpec {
            family_name,
            policy: policy_of::<Cfg>(Cfg::CHUNKING),
            source_contract: Cfg::committed_source_contract()?,
            keys,
            grouped_requests: Vec::new(),
            preplanned: Vec::new(),
            output_path: output_dir.join(format!("{family_name}.aks")),
            ring_challenge_config: Cfg::ring_challenge_config,
        })
    }

    /// All base family specs, in emission order.
    ///
    /// Instance-specific grouped advice/program rows are planned during setup
    /// and folded into the exact catalog serialized with that verifier setup.
    /// The emitter plans every key under the family's own policy; call
    /// [`preplan_direct_rows`] first to plan the rows below a one-hot family's
    /// setup-offloading cutover with its direct config.
    pub fn family_specs(output_dir: &Path) -> Result<Vec<EmitSpec>, AkitaError> {
        let mut specs = Vec::with_capacity(OneHotFamily::ALL.len() + 2);
        for family in OneHotFamily::ALL.iter().copied() {
            specs.push(with_one_hot_family!(family, |Cfg| {
                spec::<Cfg>(
                    family.family_name(),
                    one_hot_keys(family.k(), family.profile())?,
                    output_dir,
                )
            })?);
        }
        specs.extend([
            spec::<JoltDenseBounded>(
                JoltDenseBounded::schedule_family_name(),
                keys(&[1, 2], DENSE_NUM_VARS),
                output_dir,
            )?,
            spec::<JoltDenseFull>(
                JoltDenseFull::schedule_family_name(),
                keys(&[1], DENSE_NUM_VARS),
                output_dir,
            )?,
        ]);
        Ok(specs)
    }

    /// Plans, with each one-hot family's direct config, the keys below its
    /// first setup-offloading trace exponent, as preplanned rows of `specs`.
    pub fn preplan_direct_rows(specs: &mut [EmitSpec]) -> Result<(), AkitaError> {
        for family in OneHotFamily::ALL.iter().copied() {
            let Some(spec) = specs
                .iter_mut()
                .find(|spec| spec.family_name == family.family_name())
            else {
                continue;
            };
            let column_variables = if family.k() == AKITA_ONE_HOT_K16 {
                K16_COLUMN_VARIABLES
            } else {
                K256_COLUMN_VARIABLES
            };
            let direct_keys = spec
                .keys
                .iter()
                .copied()
                .filter(|key| key.num_vars() < RECURSIVE_TRACE_LOG_T_CUTOVER + column_variables)
                .map(ScheduleLookupKey::single)
                .collect::<Vec<_>>();
            let pool = ThreadPoolBuilder::new()
                .num_threads(offline_planning_worker_count(direct_keys.len()))
                .build()
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
            let planned = pool.install(|| {
                with_one_hot_family!(family, |_Cfg, DirectCfg| {
                    direct_keys
                        .into_par_iter()
                        .map(|key| {
                            plan_schedule::<DirectCfg>(&key, &[]).map(|schedule| (key, schedule))
                        })
                        .collect::<Result<Vec<_>, AkitaError>>()
                })
            })?;
            spec.preplanned.extend(planned);
        }
        Ok(())
    }
}
