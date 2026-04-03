use clap::Parser;

use jolt_eval::objective::Objective;

#[derive(Parser)]
#[command(name = "measure-objectives")]
#[command(about = "Measure Jolt code quality and performance objectives")]
struct Cli {
    /// Only measure the named objective (default: all)
    #[arg(long)]
    objective: Option<String>,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let repo_root = std::env::current_dir()?;
    let objectives = Objective::all(&repo_root);

    let filtered: Vec<_> = if let Some(name) = &cli.objective {
        objectives
            .into_iter()
            .filter(|o| o.name() == name.as_str())
            .collect()
    } else {
        objectives
    };

    if filtered.is_empty() {
        let all_names: Vec<_> = Objective::all(&repo_root)
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        eprintln!(
            "No matching objectives. Available: {}",
            all_names.join(", ")
        );
        std::process::exit(1);
    }

    println!("{:<30} {:>15} {:>10}", "Objective", "Value", "Direction");
    println!("{}", "-".repeat(57));

    for obj in &filtered {
        match obj.collect_measurement() {
            Ok(val) => {
                let dir = match obj.direction() {
                    jolt_eval::Direction::Minimize => "min",
                    jolt_eval::Direction::Maximize => "max",
                };
                println!("{:<30} {:>15.2} {:>10}", obj.name(), val, dir);
            }
            Err(e) => {
                println!("{:<30} {:>15}", obj.name(), format!("ERROR: {e}"));
            }
        }
    }

    Ok(())
}
