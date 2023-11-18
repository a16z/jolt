extern crate tracer;

use tracer::{trace, decode};

pub fn main() {
    let rows = trace("./target/riscv32i-unknown-none-elf/release/fibonacci".into());
    println!("{:?}", rows);

    let instructions = decode("./target/riscv32i-unknown-none-elf/release/fibonacci".into());
    println!("{:?}", instructions);
}
