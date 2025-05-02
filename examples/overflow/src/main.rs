use std::any::Any;
use std::panic;
use std::time::Instant;

pub fn main() {
    // An overflowing stack should fail to prove.
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_overflow_stack(target_dir);
    let prover_preprocessing = guest::preprocess_prover_overflow_stack(&program);
    let prove_overflow_stack = guest::build_prover_overflow_stack(program, prover_preprocessing);

    let res = panic::catch_unwind(|| {
        // trying to allocate 1024 elems array and sum it up
        // with stack_size=1024, should panic
        let (_, _) = prove_overflow_stack();
    });
    handle_result(res);

    // now lets try to overflow the heap, should also panic
    let program = guest::compile_overflow_heap(target_dir);
    let prover_preprocessing = guest::preprocess_prover_overflow_heap(&program);
    let prove_overflow_heap = guest::build_prover_overflow_heap(program, prover_preprocessing);

    let res = panic::catch_unwind(|| {
        let (_, _) = prove_overflow_heap();
    });
    handle_result(res);

    // valid case for stack allocation, calls overflow_stack() under the hood
    // but with stack_size=8192
    let program = guest::compile_allocate_stack_with_increased_size(target_dir);

    let prover_preprocessing =
        guest::preprocess_prover_allocate_stack_with_increased_size(&program);
    let verifier_preprocessing =
        guest::preprocess_verifier_allocate_stack_with_increased_size(&program);

    let prove_allocate_stack_with_increased_size =
        guest::build_prover_allocate_stack_with_increased_size(program, prover_preprocessing);
    let verify_allocate_stack_with_increased_size =
        guest::build_verifier_allocate_stack_with_increased_size(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof) = prove_allocate_stack_with_increased_size();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_allocate_stack_with_increased_size(output, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}

fn handle_result(res: Result<(), Box<dyn Any + Send>>) {
    if let Err(e) = &res {
        if let Some(msg) = e.downcast_ref::<String>() {
            println!("--> Panic occurred with message: {msg}\n");
        }
    }
}
