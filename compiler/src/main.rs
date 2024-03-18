use clap::Parser;
use compiler::compile_example;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    example: String,
}

fn main() {
    let opts: Opts = Opts::parse();
    println!("Example: {}", opts.example);
    compile_example(&opts.example);
}
