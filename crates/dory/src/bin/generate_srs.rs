//! Generates a transparent prover SRS
//! We can derive the verifier side from the prover SRS
//! stores to disk, can be reused.
use dory::curve::{test_rng, ArkBn254Pairing};
use dory::generate_srs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <max_log_n>", args[0]);
        eprintln!("Example: {} 18", args[0]);
        std::process::exit(1);
    }

    let max_log_n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Error: max_log_n must be a valid number");
        std::process::exit(1);
    });

    if max_log_n > 25 {
        eprintln!("Warning: max_log_n > 25 may take a very long time and use lots of memory");
        eprintln!(
            "Polynomial size will be 2^{} = {} elements",
            max_log_n,
            1usize << max_log_n
        );
        print!("Continue? (y/N): ");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        if !input.trim().to_lowercase().starts_with('y') {
            println!("Aborted.");
            std::process::exit(0);
        }
    }

    println!(
        "Generating SRS for max_log_n = {} (polynomial size: 2^{} = {})",
        max_log_n,
        max_log_n,
        1usize << max_log_n
    );

    let mut rng = test_rng();
    let start = std::time::Instant::now();

    match generate_srs::<ArkBn254Pairing, _>(&mut rng, max_log_n) {
        Ok(filename) => {
            let elapsed = start.elapsed();
            println!("✓ Successfully generated SRS in {:?}", elapsed);
            println!("✓ Saved to: {}", filename);

            // Print file size
            if let Ok(metadata) = std::fs::metadata(&filename) {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                println!("✓ File size: {:.2} MB", size_mb);
            }
        }
        Err(e) => {
            eprintln!("✗ Error generating SRS: {}", e);
            std::process::exit(1);
        }
    }
}
