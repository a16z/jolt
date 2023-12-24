use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    example: String,
}

fn main() {
    let opts: Opts = Opts::parse();
    println!("Example: {}", opts.example);

    let output = std::process::Command::new("cargo")
        .args(&["build", "-p", &opts.example, "--release"])
        .output()
        .expect("Failed to execute command");

    if !output.status.success() {
        println!("Failed to build example: {}", opts.example);
        std::process::exit(1);
    }

    println!("Successfully built example: {}", opts.example);

    let output = std::process::Command::new("cargo")
        .args(&["run", "-p", "tracer", &opts.example])
        .output()
        .expect("Failed to execute command");

    if !output.status.success() {
        println!("Failed to run tracer on example: {}", opts.example);
        std::process::exit(1);
    }

    println!("Successfully ran tracer on example: {}", opts.example);
}


