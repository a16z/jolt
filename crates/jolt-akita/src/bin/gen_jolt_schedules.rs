//! Offline generator for the Jolt-owned Akita schedule catalogs.
//!
//! Runs akita's planner DP over every `OneHotTrace` shape reachable from Jolt and
//! emits checked-in external `.aks` artifacts through the same
//! `akita_planner::emit` machinery that produces Akita's shipped catalogs.
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules [selector] [--check]
//! ```

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use akita_planner::emit::{
    publish_artifact_outputs, render_schedule_artifact_outputs_with_validation,
    MaterializationDiagnostics,
};
use jolt_akita::schedules::emit::family_specs;

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "offline generator: fail loud, narrate progress"
)]
fn main() {
    let mut args = std::env::args().skip(1);
    let output_dir = PathBuf::from(
        args.next()
            .expect("usage: gen_jolt_schedules <output-dir> [selector] [--check]"),
    );
    let mut only = None;
    let mut check = false;
    for argument in args {
        if argument == "--check" {
            check = true;
        } else {
            assert!(
                only.replace(argument).is_none(),
                "only one selector is supported"
            );
        }
    }
    let generation_dir = if check {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "jolt-akita-catalog-check-{}-{suffix}",
            std::process::id()
        ))
    } else {
        output_dir.clone()
    };
    std::fs::create_dir_all(&generation_dir).expect("create artifact output directory");

    // The documented selectors are family-name infixes, except the explicit
    // `*-single` selectors that distinguish base one-hot catalogs from their
    // multi-chunk companions.
    let specs = family_specs(generation_dir.clone())
        .expect("every family must declare a valid contract")
        .into_iter()
        .filter(|family| match only.as_deref() {
            None => true,
            Some("k16-single") => family.family_name == "jolt-fp128-onehot-k16",
            Some("k256-single") => family.family_name == "jolt-fp128-onehot-k256",
            Some(only) => family.family_name.contains(only),
        })
        .collect::<Vec<_>>();
    assert!(!specs.is_empty(), "no schedule family matches {only:?}");
    for family in &specs {
        println!(
            "generating {} ({} keys)…",
            family.family_name,
            family.keys.len()
        );
    }
    let result = (|| -> Result<(), String> {
        let outputs = render_schedule_artifact_outputs_with_validation(
            &specs,
            MaterializationDiagnostics { row_progress: true },
            |_, _| Ok(()),
        )?;
        let mut stale = Vec::new();
        for path in publish_artifact_outputs(outputs)? {
            if check {
                let name = path.file_name().ok_or("artifact has no filename")?;
                let checked_in = output_dir.join(name);
                let generated = std::fs::read(&path).map_err(|error| error.to_string())?;
                let existing = std::fs::read(&checked_in)
                    .map_err(|error| format!("{}: {error}", checked_in.display()))?;
                if generated != existing {
                    stale.push(checked_in.display().to_string());
                }
            } else {
                println!("wrote {}", path.display());
            }
        }
        if !stale.is_empty() {
            return Err(format!("stale schedule artifacts: {}", stale.join(", ")));
        }
        if check {
            println!(
                "all {} schedule artifacts match the pinned planner",
                specs.len()
            );
        }
        Ok(())
    })();
    if check {
        std::fs::remove_dir_all(&generation_dir).expect("remove temporary artifact directory");
    }
    result.expect("artifact generation or freshness check must succeed");
}
