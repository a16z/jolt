use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use clap::Parser;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::JoltInvariants;
use jolt_eval::objective::optimize::{
    auto_optimize, ObjectiveFunction, OptimizeConfig, OptimizeEnv, SingleObjective,
};
use jolt_eval::objective::{perf_objective_names, Objective};

#[derive(Parser)]
#[command(name = "optimize")]
#[command(about = "AI-driven optimization of Jolt codebase objectives")]
struct Cli {
    /// Objective to minimize (e.g. "lloc", "prover_time_fibonacci_100").
    /// Default: all measurements are taken but you must specify which to optimize.
    #[arg(long)]
    objective: String,

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
    bench_perf: bool,
}

impl OptimizeEnv for RealEnv {
    fn measure(&mut self) -> HashMap<String, f64> {
        let mut results: HashMap<String, f64> = self
            .objectives
            .iter()
            .filter_map(|o| {
                let name = o.name().to_string();
                o.collect_measurement().ok().map(|v| (name, v))
            })
            .collect();

        if self.bench_perf {
            for &name in perf_objective_names() {
                let status = Command::new("cargo")
                    .current_dir(&self.repo_dir)
                    .args([
                        "bench",
                        "-p",
                        "jolt-eval",
                        "--bench",
                        name,
                        "--",
                        "--quick",
                        "--save-baseline",
                        "optimize",
                    ])
                    .status();

                if matches!(status, Ok(s) if s.success()) {
                    if let Some(secs) = read_criterion_estimate(name) {
                        results.insert(name.to_string(), secs);
                    }
                }
            }
        }

        results
    }

    fn check_invariants(&mut self) -> bool {
        self.invariants.iter().all(|inv| {
            let results = inv.run_checks(0);
            results.iter().all(|r| r.is_ok())
        })
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
    let objectives = Objective::all(&repo_dir);

    let bench_perf = perf_objective_names().contains(&cli.objective.as_str());

    let invariants = JoltInvariants::all();

    let objective_fn = SingleObjective {
        name: cli.objective.clone(),
    };

    let mut env = RealEnv {
        objectives,
        invariants,
        repo_dir: repo_dir.clone(),
        bench_perf,
    };

    println!("=== Baseline ===");
    let baseline = env.measure();
    let baseline_score = objective_fn.evaluate(&baseline);
    print_measurements(&baseline);
    println!("Objective: {} = {:.6}\n", cli.objective, baseline_score);

    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let config = OptimizeConfig {
        num_iterations: cli.iterations,
        hint: cli.hint.clone(),
    };

    let result = auto_optimize(&agent, &mut env, &objective_fn, &config, &repo_dir);

    println!("=== Summary ===");
    println!(
        "{}/{} iterations improved the objective.",
        result
            .attempts
            .iter()
            .filter(|a| a.invariants_passed && a.score < result.baseline_score)
            .count(),
        result.attempts.len()
    );
    println!(
        "Score: {:.6} -> {:.6}",
        result.baseline_score, result.best_score
    );
    println!("\nFinal measurements:");
    print_measurements(&result.best_measurements);

    Ok(())
}

fn print_measurements(measurements: &HashMap<String, f64>) {
    let mut names: Vec<_> = measurements.keys().collect();
    names.sort();
    for name in names {
        println!("  {:<35} {:>15.6}", name, measurements[name]);
    }
}

fn read_criterion_estimate(bench_name: &str) -> Option<f64> {
    let path = Path::new("target/criterion")
        .join(bench_name)
        .join("optimize")
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}
