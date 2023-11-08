extern crate tracer;

use tracer::trace::trace;

pub fn main() {
    trace("../jolt-compiler/target/riscv32i-unknown-none-elf/debug/program".into())
}
