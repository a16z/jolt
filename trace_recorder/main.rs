use jolt_core::host;

// Default (64-bit): cargo run --release <guest_name>
// 32-bit mode: cargo run --release --features rv32 --no-default-features <guest_name>
// Explicit 64-bit: cargo run --release --features rv64 --no-default-features <guest_name>

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <guest_name>", args[0]);
        println!("Example: {} sha3-chain-guest", args[0]);
        return;
    }

    let guest_name = &args[1];

    let mut program = host::Program::new(guest_name);
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&100000u32).unwrap());

    let (trace, _final_memory_state, _io_device) = program.trace(&inputs);
    let (bytecode, _init_memory_state) = program.decode();
    println!(
        "Trace length: {}, Bytecode length: {}",
        trace.len(),
        bytecode.len()
    );
    println!("Output is: {:x?}", _io_device.inputs);
    // let result = program.trace_analyze::<Fr>(&inputs);

    // #[cfg(feature = "rv32")]
    // let filename = format!("{}_RV32_trace.txt", guest_name);
    // #[cfg(feature = "rv64")]
    // let filename = format!("{}_RV64_trace.txt", guest_name);

    // Write trace analysis to file
    // match result.write_trace_analysis::<Fr>(&filename) {
    //     Ok(_) => println!("✅ Saved complete trace analysis to: {}", filename),
    //     Err(e) => println!("❌ Failed to save trace analysis: {}", e),
    // }
}
