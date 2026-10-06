//! Audit grouped provisioning without allocating backend matrices or proving traces.
//!
//! cargo run --release -p jolt-akita --bin check_grouped_schedules -- <catalog-dir> <report.csv> [full|boundary]

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

use akita_params::{PolynomialGroupLayout, ScheduleLookupKey};
use akita_planner::emit::{bounded_parallel_filter_map, offline_planning_worker_count};
use jolt_akita::schedule_registry::dense_group_profile;
use jolt_akita::{
    AkitaChunkProfile, AkitaError, AkitaScheduleArtifacts, GroupedScheduleParams,
    AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};

enum Outcome {
    Admitted,
    UnsupportedProducer(AkitaError),
    UnsupportedGroupedShape(AkitaError),
    Failed(AkitaError),
}

impl Outcome {
    fn grouped(
        result: Result<(), AkitaError>,
        k: usize,
        profile: AkitaChunkProfile,
        final_arity: usize,
    ) -> Self {
        match result {
            Ok(()) => Self::Admitted,
            Err(AkitaError::UnsupportedSchedule(message))
                if profile == AkitaChunkProfile::Single
                    && message.starts_with(&format!(
                        "one-hot K={k} profile Single final arity {final_arity} has no recursive child fold in its scalar guide; bounded grouped provisioning requires one; requested groups: "
                    )) =>
            {
                Self::UnsupportedGroupedShape(AkitaError::UnsupportedSchedule(message))
            }
            Err(error) => Self::Failed(error),
        }
    }
}

#[expect(clippy::print_stderr, reason = "offline diagnostic reports progress")]
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or("missing catalog directory")?;
    let output = args.next().ok_or("missing CSV report path")?;
    let mode = args.next().unwrap_or_else(|| "full".to_owned());
    if !matches!(mode.as_str(), "full" | "boundary") || args.next().is_some() {
        return Err(
            "usage: check_grouped_schedules <catalog-dir> <report.csv> [full|boundary]".into(),
        );
    }
    let artifacts = AkitaScheduleArtifacts::from_directory(directory)?;
    let dense = artifacts.dense_catalog()?;
    let full_dense = artifacts.full_dense_catalog()?;
    let mut report = BufWriter::new(File::create(output)?);
    writeln!(
        report,
        "k,profile,final_arity,untrusted_arity,trusted_arity,status,error"
    )?;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut unsupported = 0usize;
    for k in [AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256] {
        for profile in [
            AkitaChunkProfile::Single,
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            if mode == "boundary"
                && (k != AKITA_ONE_HOT_K16 || profile == AkitaChunkProfile::Single)
            {
                continue;
            }
            let base = artifacts.one_hot_catalog_for_profile(k, profile)?;
            let mut final_arities = base
                .rows()
                .filter(|row| row.profiles().final_group.group.num_polynomials() == 1)
                .map(|row| row.profiles().final_group.group.num_vars())
                .filter(|&arity| mode == "full" || [31, 32].contains(&arity))
                .collect::<Vec<_>>();
            final_arities.sort_unstable();
            final_arities.dedup();
            let advice_arities = if mode == "boundary" {
                vec![Some(21), Some(22)]
            } else {
                std::iter::once(None).chain((11..=22).map(Some)).collect()
            };
            for final_arity in final_arities {
                let mut catalog = base.clone();
                // Requests also provision singleton prefixes. Share those audited
                // rows so each worker searches only its distinct paired row.
                for arity in advice_arities.iter().copied().flatten() {
                    let params = GroupedScheduleParams::new(
                        None,
                        Some(arity),
                        Vec::new(),
                        PolynomialGroupLayout::new(final_arity, 1),
                    );
                    if let Ok(extended) =
                        params.extend_catalog(&dense, &full_dense, &catalog, k, profile)
                    {
                        catalog = extended;
                    }
                }
                let requests = advice_arities
                    .iter()
                    .copied()
                    .flat_map(|untrusted| {
                        advice_arities
                            .iter()
                            .copied()
                            .map(move |trusted| (untrusted, trusted))
                    })
                    .filter(|(untrusted, trusted)| untrusted.is_some() || trusted.is_some())
                    .collect::<Vec<_>>();
                let outcomes = bounded_parallel_filter_map(
                    &requests,
                    offline_planning_worker_count(requests.len()),
                    |&(untrusted, trusted)| {
                        let producers = [untrusted, trusted]
                            .into_iter()
                            .flatten()
                            .map(|arity| {
                                dense_group_profile(&dense, PolynomialGroupLayout::new(arity, 1))
                            })
                            .collect::<Result<Vec<_>, _>>();
                        let outcome = match producers {
                            Err(error) => Outcome::UnsupportedProducer(error),
                            Ok(precommitteds) => {
                                let params = GroupedScheduleParams::new(
                                    untrusted,
                                    trusted,
                                    Vec::new(),
                                    PolynomialGroupLayout::new(final_arity, 1),
                                );
                                let result = params.extend_catalog(&dense, &full_dense, &catalog, k, profile)
                                    .and_then(|extended| {
                                        let key = ScheduleLookupKey {
                                            final_group: PolynomialGroupLayout::new(final_arity, 1),
                                            precommitteds,
                                        };
                                        let row = extended.resolve_key(&key)?;
                                        if row.profiles().precommitteds != key.precommitteds
                                            || row.schedule().root.params.witness_chunk.num_chunks != profile.num_chunks()
                                        {
                                            return Err(AkitaError::InvalidSetup("provisioned row changed producer profiles or trace chunks".to_owned()));
                                        }
                                        Ok(())
                                    });
                                Outcome::grouped(result, k, profile, final_arity)
                            }
                        };
                        Ok(Some((untrusted, trusted, outcome)))
                    },
                )?;
                for (untrusted, trusted, outcome) in outcomes {
                    let (status, error) = match outcome {
                        Outcome::Admitted => {
                            passed += 1;
                            ("ok", String::new())
                        }
                        Outcome::UnsupportedProducer(error) => {
                            unsupported += 1;
                            ("unsupported_producer", error.to_string())
                        }
                        Outcome::UnsupportedGroupedShape(error) => {
                            unsupported += 1;
                            ("unsupported_grouped_shape", error.to_string())
                        }
                        Outcome::Failed(error) => {
                            failed += 1;
                            ("provisioning_failed", error.to_string())
                        }
                    };
                    let untrusted = untrusted.map(|arity| arity.to_string()).unwrap_or_default();
                    let trusted = trusted.map(|arity| arity.to_string()).unwrap_or_default();
                    writeln!(
                        report,
                        "{k},{profile:?},{final_arity},{untrusted},{trusted},{status},\"{}\"",
                        error.replace('"', "\"\"")
                    )?;
                }
                report.flush()?;
                eprintln!("K={k} {profile:?} final={final_arity}: {passed} passed, {failed} failed, {unsupported} unsupported inputs");
            }
        }
    }
    report.flush()?;
    if failed != 0 {
        return Err(
            format!("{failed} grouped requests failed provisioning; see CSV report").into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_known_scalar_guide_rejection_is_expected() -> Result<(), Box<dyn Error>> {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let params = GroupedScheduleParams::new(
            None,
            Some(14),
            Vec::new(),
            PolynomialGroupLayout::new(12, 1),
        );
        let result = params
            .extend_catalog(
                &artifacts.dense_catalog()?,
                &artifacts.full_dense_catalog()?,
                &artifacts.one_hot_catalog(AKITA_ONE_HOT_K16)?,
                AKITA_ONE_HOT_K16,
                AkitaChunkProfile::Single,
            )
            .map(|_| ());
        assert!(matches!(
            Outcome::grouped(result, AKITA_ONE_HOT_K16, AkitaChunkProfile::Single, 12),
            Outcome::UnsupportedGroupedShape(_)
        ));
        for profile in [
            AkitaChunkProfile::Single,
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            assert!(matches!(
                Outcome::grouped(
                    Err(AkitaError::UnsupportedSchedule(
                        "unexpected planner rejection".to_owned()
                    )),
                    AKITA_ONE_HOT_K16,
                    profile,
                    12,
                ),
                Outcome::Failed(_)
            ));
        }
        Ok(())
    }
}
