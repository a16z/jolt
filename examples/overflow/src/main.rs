use std::any::Any;
use std::panic;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    // An overflowing stack should fail to prove.
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_overflow_stack(target_dir);
    let prover_preprocessing = guest::preprocess_overflow_stack(&mut program);
    let prove_overflow_stack = guest::build_prover_overflow_stack(program, prover_preprocessing);

    let res = panic::catch_unwind(|| {
        // trying to allocate 1024 elems array and sum it up
        // with stack_size=1024, should panic
        let (_, _, _) = prove_overflow_stack();
    });
    handle_result(res);

    // now lets try to overflow the heap, should also panic
    let mut program = guest::compile_overflow_heap(target_dir);
    let prover_preprocessing = guest::preprocess_overflow_heap(&mut program);
    let prove_overflow_heap = guest::build_prover_overflow_heap(program, prover_preprocessing);

    let res = panic::catch_unwind(|| {
        let (_, _, _) = prove_overflow_heap();
    });
    handle_result(res);

    // valid case for stack allocation, calls overflow_stack() under the hood
    // but with stack_size=8192
    let mut program = guest::compile_allocate_stack_with_increased_size(target_dir);
    let prover_preprocessing = guest::preprocess_allocate_stack_with_increased_size(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_allocate_stack_with_increased_size(
            &prover_preprocessing,
        );

    let prove_allocate_stack_with_increased_size =
        guest::build_prover_allocate_stack_with_increased_size(program, prover_preprocessing);
    let verify_allocate_stack_with_increased_size =
        guest::build_verifier_allocate_stack_with_increased_size(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_allocate_stack_with_increased_size();
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_allocate_stack_with_increased_size(output, program_io.panic, proof);

    info!("output: {output}");
    info!("valid: {is_valid}");
}

fn handle_result(res: Result<(), Box<dyn Any + Send>>) {
    if let Err(e) = &res {
        if let Some(msg) = e.downcast_ref::<String>() {
            info!("--> Panic occurred with message: {msg}\n");
        }
    }
}
