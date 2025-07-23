use jolt_sdk::host;
use jolt_sdk::postcard;

fn main() {
    let mut program = host::Program::new("blake2-inline-guest");
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&100000u32).unwrap());

    let (trace, final_memory_state, io_device) = program.trace(&inputs);
    let (bytecode, init_memory_state) = program.decode();
    println!(
        "Trace length: {}  ----  Bytecode Length: {}",
        trace.len(),
        bytecode.len()
    );

    let result = guest::blake2_inline([5u8; 32], 1);
    println!("result: {:x?}", &result);
    println!("result: {:x?}", &io_device.outputs);
}
