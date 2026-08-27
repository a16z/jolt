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

use akita_planner::emit::emit_family_module;
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

    for family in family_specs(output_dir).expect("every family must declare a valid contract") {
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
        let path = family.output_dir.join(format!("{}.rs", family.module_name));
        let source = emit_family_module(&family).expect("table generation must succeed");
        std::fs::write(&path, source).expect("write generated table");
        // The emitter's fixed import header is not rustfmt-stable; format the
        // module so the checked-in file passes the workspace fmt check. The
        // drift oracle compares schedule data only, so formatting is free.
        let status = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2021")
            .arg(&path)
            .status()
            .expect("rustfmt must be installed to emit checked-in tables");
        assert!(status.success(), "rustfmt failed on {}", path.display());
        println!("wrote {}", path.display());
    }
}
