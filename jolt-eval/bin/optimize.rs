use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use clap::Parser;

use jolt_eval::agent::ClaudeCodeAgent;
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
    #[arg(long, default_value = "sonnet")]
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

    /// Name of the result branch (default: auto-optimize/{objective}-{timestamp}).
    #[arg(long)]
    branch: Option<String>,
}

struct RealEnv {
    repo_dir: std::path::PathBuf,
    work_dir: std::path::PathBuf,
    base_commit: String,
}

impl RealEnv {
    fn new(repo_dir: std::path::PathBuf) -> eyre::Result<Self> {
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

    fn measure(
        &mut self,
        objectives: &[OptimizationObjective],
    ) -> HashMap<OptimizationObjective, f64> {
        let mut results = HashMap::new();

        for sa in StaticAnalysisObjective::all() {
            let obj = OptimizationObjective::StaticAnalysis(sa);
            if objectives.contains(&obj) {
                if let Ok(v) = sa.collect_measurement_in(&self.work_dir) {
                    results.insert(obj, v);
                }
            }
        }

        for p in PerformanceObjective::all() {
            let obj = OptimizationObjective::Performance(p);
            if !objectives.contains(&obj) {
                continue;
            }
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
                if let Some(secs) = read_criterion_estimate(&self.work_dir, p.name(), "optimize") {
                    results.insert(obj, secs);
                }
            }
        }

        results
    }

    fn check_invariants(&mut self) -> bool {
        // Run invariant tests as a subprocess in the worktree so they are
        // compiled against the *modified* code, not the binary we're
        // running in.
        let status = Command::new("cargo")
            .current_dir(&self.work_dir)
            .args([
                "nextest",
                "run",
                "-p",
                "jolt-eval",
                "--cargo-quiet",
                "-E",
                "test(/_synthesized::/)",
            ])
            .status();
        matches!(status, Ok(s) if s.success())
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

    fn finish(&mut self, branch_name: &str) -> Option<String> {
        // Check whether any commits were added beyond the base.
        let head_out = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()?;
        let head = String::from_utf8_lossy(&head_out.stdout).trim().to_string();
        if head == self.base_commit {
            return None;
        }

        // Create (or move) the branch at the worktree HEAD.  Because
        // worktrees share refs with the main repo, this branch is
        // immediately visible from repo_dir.
        let status = Command::new("git")
            .current_dir(&self.work_dir)
            .args(["branch", "-f", branch_name, "HEAD"])
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        Some(branch_name.to_string())
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
        let mut env = RealEnv::new(repo_dir.clone())?;
        let baseline = env.measure(objective.inputs);
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
            branch: cli.branch.clone(),
        };
        println!("=== Optimize e2e: naive bubble sort ===");
        println!(
            "model={}, max_turns={}, iterations={}",
            cli.model, cli.max_turns, cli.iterations
        );
        println!("Baseline sort time: {baseline_score:.8}s");
        println!();
        let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
        let result = auto_optimize(&agent, &mut env, objective, &config, &repo_dir);
        println!("Best score: {:.8}s", result.best_score);
        println!(
            "Improvement: {:.1}%",
            (1.0 - result.best_score / baseline_score) * 100.0
        );
        for (i, a) in result.attempts.iter().enumerate() {
            println!(
                "  attempt {}: score={:.8}, invariants={}",
                i + 1,
                a.score,
                a.invariants_passed
            );
        }
        print_branch_info(&result, &env.base_commit);
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

    let mut env = RealEnv::new(repo_dir.clone())?;

    let baseline = env.measure(objective.inputs);

    println!("=== Baseline ===");
    print_measurements(&baseline);
    let baseline_score = (objective.evaluate)(&baseline, &baseline);
    println!("Objective: {} = {:.8}\n", objective.name, baseline_score);

    let agent = ClaudeCodeAgent::new(&cli.model, cli.max_turns);
    let config = OptimizeConfig {
        num_iterations: cli.iterations,
        hint: cli.hint.clone(),
        verbose: cli.verbose,
        branch: cli.branch.clone(),
    };

    let result = auto_optimize(&agent, &mut env, objective, &config, &repo_dir);

    println!("=== Summary ===");
    println!(
        "{}/{} iterations improved the objective.",
        result.attempts.iter().filter(|a| a.accepted).count(),
        result.attempts.len()
    );
    println!(
        "Score: {:.8} -> {:.8}",
        result.baseline_score, result.best_score
    );
    println!("\nFinal measurements:");
    print_measurements(&result.best_measurements);

    print_branch_info(&result, &env.base_commit);

    Ok(())
}

fn print_branch_info(result: &jolt_eval::objective::optimize::OptimizeResult, base_commit: &str) {
    if let Some(ref branch) = result.branch {
        println!("\nResult branch: {branch}");
        println!("  Inspect: git log {base_commit}..{branch}");
        println!("  Apply:   git cherry-pick {base_commit}..{branch}");
    }
}

fn print_measurements(measurements: &HashMap<OptimizationObjective, f64>) {
    let mut entries: Vec<_> = measurements.iter().collect();
    entries.sort_by_key(|(k, _)| k.name());
    for (key, val) in entries {
        println!("  {:<35} {:>15.6}", key.name(), val);
    }
}
