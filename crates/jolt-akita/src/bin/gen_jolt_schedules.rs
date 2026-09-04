//! Offline generator for the Jolt-owned Akita schedule catalogs.
//!
//! Runs akita's planner DP over every `OneHotTrace` shape reachable from Jolt and
//! emits checked-in external `.aks` artifacts through the same
//! `akita_planner::emit` machinery that produces Akita's shipped catalogs.
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules [k16|k256|dense]
//! ```

use std::path::PathBuf;

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
            .expect("usage: gen_jolt_schedules <output-dir> [k16|k256|dense]"),
    );
    let only = args.next();
    std::fs::create_dir_all(&output_dir).expect("create artifact output directory");

    // Family names are `jolt-fp128-onehot-k16`, `jolt-fp128-onehot-k256`, and
    // `jolt-fp128-dense-bounded`, so the documented selectors are infixes,
    // not suffixes.
    let specs = family_specs(output_dir)
        .expect("every family must declare a valid contract")
        .into_iter()
        .filter(|family| {
            only.as_deref()
                .is_none_or(|only| family.family_name.contains(only))
        })
        .collect::<Vec<_>>();
    assert!(
        !specs.is_empty(),
        "no schedule family matches {only:?}; expected k16, k256, or dense"
    );
    for family in &specs {
        println!(
            "generating {} ({} keys)…",
            family.family_name,
            family.keys.len()
        );
    }
    let outputs = render_schedule_artifact_outputs_with_validation(
        &specs,
        MaterializationDiagnostics { row_progress: true },
        |_, _| Ok(()),
    )
    .expect("artifact generation must succeed");
    for path in publish_artifact_outputs(outputs).expect("publish generated artifacts") {
        println!("wrote {}", path.display());
    }
}
