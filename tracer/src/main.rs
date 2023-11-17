extern crate tracer;

use tracer::trace;

pub fn main() {
    let rows = trace("../jolt-compiler/target/riscv32i-unknown-none-elf/release/program".into());
    for row in &rows {
        println!("{:?}\n", row);
    }

    println!("trace lenth: {}", rows.len());
}
