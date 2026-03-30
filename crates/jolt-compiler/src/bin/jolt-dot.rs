//! CLI tool to render a serialized L0 Protocol as Graphviz DOT.
//!
//! ```text
//! # From file (auto-detects JSON vs RON):
//! jolt-dot protocol.json | dot -Tsvg -o protocol.svg
//!
//! # From stdin:
//! cat protocol.json | jolt-dot
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, unused_results)]

use std::io::Read;
use std::{env, fs, process};

use jolt_compiler::dot::protocol_to_dot;
use jolt_compiler::Protocol;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "\
Usage: jolt-dot [OPTIONS] [FILE]

Reads a serialized L0 Protocol and writes Graphviz DOT to stdout.
Auto-detects JSON vs RON format. Reads stdin if no file given.

Options:
  --json     Force JSON input
  --ron      Force RON input
  -h, --help Show this help"
        );
        process::exit(0);
    }

    let force_json = args.iter().any(|a| a == "--json");
    let force_ron = args.iter().any(|a| a == "--ron");
    let file_args: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .map(|a| a.as_str())
        .collect();

    let input = if let Some(&path) = file_args.first() {
        fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error: cannot read {path}: {e}");
            process::exit(1);
        })
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .unwrap_or_else(|e| {
                eprintln!("error: cannot read stdin: {e}");
                process::exit(1);
            });
        buf
    };

    let trimmed = input.trim_start();

    let protocol: Protocol = if force_json || (!force_ron && trimmed.starts_with('{')) {
        serde_json::from_str(&input).unwrap_or_else(|e| {
            eprintln!("error: invalid JSON: {e}");
            process::exit(1);
        })
    } else {
        ron::from_str(&input).unwrap_or_else(|e| {
            eprintln!("error: invalid RON: {e}");
            process::exit(1);
        })
    };

    print!("{}", protocol_to_dot(&protocol));
}
