//! Offline generator for the Jolt-owned Akita schedule catalogs.
//!
//! Runs akita's planner DP over every `OneHotTrace` shape reachable from Jolt and
//! emits the checked-in table modules under `src/schedules/` through the same
//! `akita_planner::emit` machinery that produces akita's shipped tables.
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/src/schedules [k16|k256]
//! ```

use std::path::PathBuf;

use akita_planner::write_family_module;
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
            .expect("usage: gen_jolt_schedules <output-dir> [k16|k256]"),
    );
    let only = args.next();

    for family in family_specs(output_dir) {
        if only
            .as_deref()
            .is_some_and(|only| !family.module_name.ends_with(only))
        {
            continue;
        }
        println!(
            "generating {} ({} keys)…",
            family.module_name,
            family.keys.len()
        );
        let path = write_family_module(&family).expect("table generation must succeed");
        println!("wrote {}", path.display());
    }
}
