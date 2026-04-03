use std::collections::HashMap;
use std::process::Command;

use clap::Parser;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::JoltInvariants;
use jolt_eval::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use jolt_eval::objective::{Direction, Objective};

#[derive(Parser)]
#[command(name = "optimize")]
#[command(about = "AI-driven optimization of Jolt codebase objectives")]
struct Cli {
    /// Objectives to optimize (comma-separated). Default: all.
    #[arg(long)]
    objectives: Option<String>,

    /// Number of optimization iterations
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// AI model to use
    #[arg(long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Maximum number of Claude agentic turns per iteration
    #[arg(long, default_value = "30")]
    max_turns: usize,

    /// Extra context to include in the optimization prompt
    #[arg(long)]
    hint: Option<String>,
}

struct RealEnv {
    objectives: Vec<Objective>,
    invariants: Vec<JoltInvariants>,
    repo_dir: std::path::PathBuf,
}

impl OptimizeEnv for RealEnv {
    fn measure(&mut self) -> HashMap<String, f64> {
        self.objectives
            .iter()
            .filter_map(|o| {
                let name = o.name().to_string();
                o.collect_measurement().ok().map(|v| (name, v))
            })
            .collect()
    }

    fn check_invariants(&mut self) -> bool {
        self.invariants.iter().all(|inv| {
            let results = inv.run_checks(0);
            results.iter().all(|r| r.is_ok())
        })
    }

    fn directions(&self) -> HashMap<String, Direction> {
        self.objectives
            .iter()
            .map(|o| (o.name().to_string(), o.direction()))
            .collect()
    }

    fn apply_diff(&mut self, diff: &str) {
        if let Err(e) = jolt_eval::agent::apply_diff(&self.repo_dir, diff) {
            tracing::warn!("Failed to apply diff: {e}");
        }
    }

    fn accept(&mut self, iteration: usize) {
        println!("  Improvement found -- keeping changes.");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["add", "-A"])
            .status();
        let msg = format!("perf(auto-optimize): iteration {iteration}");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["commit", "-m", &msg, "--allow-empty"])
            .status();
    }

    fn reject(&mut self) {
        println!("  Reverting changes.");
        let _ = Command::new("git")
            .current_dir(&self.repo_dir)
            .args(["checkout", "."])
            .status();
    }
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let repo_dir = std::env::current_dir()?;
    let all_objectives = Objective::all(&repo_dir);
    let all_names: Vec<String> = all_objectives
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let filter_names: Option<Vec<String>> = cli
        .objectives
        .as_ref()
        .map(|s| s.split(',').map(|n| n.trim().to_string()).collect());

    let objectives: Vec<Objective> = if let Some(names) = &filter_names {
        all_objectives
            .into_iter()
            .filter(|o| names.contains(&o.name().to_string()))
            .collect()
    } else {
        all_objectives
    };

    if objectives.is_empty() {
        eprintln!(
            "No matching objectives. Available: {}",
            all_names.join(", ")
        );
        std::process::exit(1);
    }

    let invariants = JoltInvariants::all();

    let mut env = RealEnv {
        objectives,
        invariants,
        repo_dir: repo_dir.clone(),
    };

    println!("=== Baseline measurements ===");
    let baseline = env.measure();
    print_measurements(&env.directions(), &baseline);
    println!();

    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let config = OptimizeConfig {
        num_iterations: cli.iterations,
        hint: cli.hint.clone(),
    };

    let result = auto_optimize(&agent, &mut env, &config, &repo_dir);

    println!("=== Optimization summary ===");
    println!(
        "{}/{} iterations produced improvements.",
        result
            .attempts
            .iter()
            .filter(|a| a.invariants_passed
                && a.measurements
                    .iter()
                    .any(|(name, &val)| { result.baseline.get(name).is_some_and(|&b| val < b) }))
            .count(),
        result.attempts.len()
    );
    println!();
    println!("Final measurements:");
    print_measurements(&env.directions(), &result.best);

    Ok(())
}

fn print_measurements(
    directions: &HashMap<String, Direction>,
    measurements: &HashMap<String, f64>,
) {
    let mut names: Vec<_> = directions.keys().collect();
    names.sort();
    for name in names {
        let val = measurements
            .get(name)
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "N/A".to_string());
        let dir = match directions[name] {
            Direction::Minimize => "min",
            Direction::Maximize => "max",
        };
        println!("  {:<30} {:>15} {:>6}", name, val, dir);
    }
}
