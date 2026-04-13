use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use clap::Parser;

use jolt_eval::agent::ClaudeCodeAgent;
use jolt_eval::invariant::JoltInvariants;
use jolt_eval::objective::objective_fn::ObjectiveFunction;
use jolt_eval::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use jolt_eval::objective::performance::read_criterion_estimate;
use jolt_eval::objective::{OptimizationObjective, PerformanceObjective, StaticAnalysisObjective};

#[derive(Parser)]
#[command(name = "optimize")]
#[command(about = "AI-driven optimization of Jolt codebase objectives")]
struct Cli {
    /// Objective function to minimize (mutually exclusive with --test).
    #[arg(long, conflicts_with = "test")]
    objective: Option<String>,

    /// Run the built-in e2e sort optimization test.
    #[arg(long, conflicts_with = "objective")]
    test: bool,

    /// List all available objective functions and exit.
    #[arg(long)]
    list: bool,

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

    /// Print agent prompts and responses to stderr.
    #[arg(long)]
    verbose: bool,
}

struct RealEnv {
    repo_dir: std::path::PathBuf,
    work_dir: std::path::PathBuf,
    base_commit: String,
    invariants: Vec<JoltInvariants>,
    bench_perf: bool,
}

impl RealEnv {
    fn new(
        repo_dir: std::path::PathBuf,
        invariants: Vec<JoltInvariants>,
        bench_perf: bool,
    ) -> eyre::Result<Self> {
        // Fail fast if the working tree is dirty.
        let status_out = Command::new("git")
            .current_dir(&repo_dir)
            .args(["status", "--porcelain"])
            .output()?;
        let status_str = String::from_utf8_lossy(&status_out.stdout);
        if !status_str.trim().is_empty() {
            eyre::bail!(
                "Repository has uncommitted changes — commit or stash before optimizing.\n\
                 Dirty files:\n{status_str}"
            );
        }

        // Record the base commit so we can export a cumulative patch later.
        let head_out = Command::new("git")
            .current_dir(&repo_dir)
            .args(["rev-parse", "HEAD"])
            .output()?;
        eyre::ensure!(head_out.status.success(), "failed to resolve HEAD");
        let base_commit = String::from_utf8_lossy(&head_out.stdout).trim().to_string();

        // Create an isolated worktree for the optimization run.
        let work_dir = jolt_eval::agent::claude::create_worktree(&repo_dir)
            .map_err(|e| eyre::eyre!("Failed to create optimization worktree: {e}"))?;
        eprintln!("Created optimization worktree at {}", work_dir.display());

        Ok(Self {
            repo_dir,
            work_dir,
            base_commit,
            invariants,
            bench_perf,
        })
    }
}

impl Drop for RealEnv {
    fn drop(&mut self) {
        jolt_eval::agent::claude::remove_worktree(&self.repo_dir, &self.work_dir);
        let _ = std::fs::remove_dir_all(&self.work_dir);
    }
}

impl OptimizeEnv for RealEnv {
    fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    fn measure(&mut self) -> HashMap<OptimizationObjective, f64> {
        let mut results = HashMap::new();

        for sa in StaticAnalysisObjective::all() {
            if let Ok(v) = sa.collect_measurement_in(&self.work_dir) {
                results.insert(OptimizationObjective::StaticAnalysis(sa), v);
            }
        }

        if self.bench_perf {
            for p in PerformanceObjective::all() {
                let status = Command::new("cargo")
                    .current_dir(&self.work_dir)
                    .args([
                        "bench",
                        "-p",
                        "jolt-eval",
                        "--bench",
                        p.name(),
                        "--",
                        "--quick",
                        "--save-baseline",
                        "optimize",
                    ])
                    .status();

                if matches!(status, Ok(s) if s.success()) {
                    if let Some(secs) = read_criterion_estimate(p.name(), "optimize") {
                        results.insert(OptimizationObjective::Performance(p), secs);
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
        if let Err(e) = jolt_eval::agent::apply_diff(&self.work_dir, diff) {
            tracing::warn!("Failed to apply diff: {e}");
        }
    }

    fn accept(&mut self, iteration: usize, commit_msg: &str) {
        println!("  Improvement found -- keeping changes (iteration {iteration}).");
        let _ = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["add", "-A"])
            .status();
        let _ = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["commit", "-m", commit_msg])
            .status();
    }

    fn reject(&mut self) {
        println!("  Reverting changes.");
        let _ = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["checkout", "."])
            .status();
        let _ = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["clean", "-fd"])
            .status();
    }

    fn finish(&mut self) -> Option<String> {
        let output = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["diff", &self.base_commit, "HEAD"])
            .output()
            .ok()?;
        let diff = String::from_utf8_lossy(&output.stdout).to_string();
        if diff.trim().is_empty() {
            None
        } else {
            Some(diff)
        }
    }
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list {
        println!("Available objective functions:\n");
        for f in ObjectiveFunction::all() {
            let inputs: Vec<_> = f.inputs.iter().map(|i| i.name().to_string()).collect();
            println!("  {:<35} inputs: {}", f.name, inputs.join(", "));
        }
        println!("\nBuilt-in e2e targets (use --test):");
        println!("  naive_sort");
        return Ok(());
    }

    if cli.test {
        const SORT_TARGETS_PATH: &str = "jolt-eval/src/sort_targets.rs";
        let objective = ObjectiveFunction::by_name("minimize_naive_sort_time").unwrap();
        let repo_dir = std::env::current_dir()?;
        let invariants = JoltInvariants::all();
        let mut env = RealEnv::new(repo_dir.clone(), invariants, true)?;
        let baseline = env.measure();
        let baseline_score = (objective.evaluate)(&baseline, &baseline);
        let hint = cli.hint.unwrap_or_else(|| {
            format!(
                "The target is the `naive_sort` function in {SORT_TARGETS_PATH}. \
                 Replace it with a faster sorting algorithm. \
                 You MAY modify that file for this task."
            )
        });
        let config = OptimizeConfig {
            num_iterations: cli.iterations,
            hint: Some(hint),
            verbose: cli.verbose,
        };
        println!("=== Optimize e2e: naive bubble sort ===");
        println!(
            "model={}, max_turns={}, iterations={}",
            cli.model, cli.max_turns, cli.iterations
        );
        println!("Baseline sort time: {baseline_score:.6}s");
        println!();
        let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
        let result = auto_optimize(&agent, &mut env, objective, &config, &repo_dir);
        println!("Best score: {:.6}s", result.best_score);
        println!(
            "Improvement: {:.1}%",
            (1.0 - result.best_score / baseline_score) * 100.0
        );
        for (i, a) in result.attempts.iter().enumerate() {
            println!(
                "  attempt {}: score={:.6}, invariants={}",
                i + 1,
                a.score,
                a.invariants_passed
            );
        }
        save_best_patch(&result, &repo_dir, "minimize_naive_sort_time");
        return Ok(());
    }

    let objective_name = cli
        .objective
        .as_deref()
        .expect("--objective or --test is required (use --list to see options)");

    let objective = ObjectiveFunction::by_name(objective_name).unwrap_or_else(|| {
        eprintln!("Unknown objective function: {objective_name}");
        eprintln!("Available:");
        for f in ObjectiveFunction::all() {
            eprintln!("  {}", f.name);
        }
        std::process::exit(1);
    });

    let repo_dir = std::env::current_dir()?;

    let bench_perf = objective.inputs.iter().any(|i| i.is_perf());
    let invariants = JoltInvariants::all();

    let mut env = RealEnv::new(repo_dir.clone(), invariants, bench_perf)?;

    let baseline = env.measure();

    println!("=== Baseline ===");
    print_measurements(&baseline);
    let baseline_score = (objective.evaluate)(&baseline, &baseline);
    println!("Objective: {} = {:.6}\n", objective.name, baseline_score);

    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let config = OptimizeConfig {
        num_iterations: cli.iterations,
        hint: cli.hint.clone(),
        verbose: cli.verbose,
    };

    let result = auto_optimize(&agent, &mut env, objective, &config, &repo_dir);

    println!("=== Summary ===");
    println!(
        "{}/{} iterations improved the objective.",
        result.attempts.iter().filter(|a| a.accepted).count(),
        result.attempts.len()
    );
    println!(
        "Score: {:.6} -> {:.6}",
        result.baseline_score, result.best_score
    );
    println!("\nFinal measurements:");
    print_measurements(&result.best_measurements);

    save_best_patch(&result, &repo_dir, objective_name);

    Ok(())
}

fn save_best_patch(
    result: &jolt_eval::objective::optimize::OptimizeResult,
    repo_dir: &Path,
    objective_name: &str,
) {
    if let Some(ref patch) = result.best_patch {
        let history_dir = repo_dir
            .join("jolt-eval/optimize-history")
            .join(objective_name);
        let _ = std::fs::create_dir_all(&history_dir);
        let patch_path = history_dir.join("best.patch");
        match std::fs::write(&patch_path, patch) {
            Ok(()) => {
                println!(
                    "\nBest patch saved to: {}\nApply with: git apply {}",
                    patch_path.display(),
                    patch_path.display()
                );
            }
            Err(e) => eprintln!("\nWarning: failed to save patch: {e}"),
        }
    }
}

fn print_measurements(measurements: &HashMap<OptimizationObjective, f64>) {
    let mut entries: Vec<_> = measurements.iter().collect();
    entries.sort_by_key(|(k, _)| k.name());
    for (key, val) in entries {
        println!("  {:<35} {:>15.6}", key.name(), val);
    }
}
