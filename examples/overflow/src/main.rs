use std::any::Any;
use std::panic;
use std::time::Instant;

pub fn main() {
    let (prove_overflow_stack, _) = guest::build_overflow_stack();

    let res = panic::catch_unwind(|| {
        // trying to allocate 1024 elems array and sum it up
        // with stack_size=1024, should panic
        let (_, _) = prove_overflow_stack();
    });
    handle_result(res);

    // now lets try to overflow the heap, should also panic
    let (prove_overflow_heap, _) = guest::build_overflow_heap();

    let res = panic::catch_unwind(|| {
        let (_, _) = prove_overflow_heap();
    });
    handle_result(res);

    // valid case for stack allocation, calls overflow_stack() under the hood
    // but with stack_size=8192
    let (prove_allocate_stack_with_increased_size, verify_allocate_stack_with_increased_size) =
        guest::build_allocate_stack_with_increased_size();

    let now = Instant::now();
    let (output, proof) = prove_allocate_stack_with_increased_size();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_allocate_stack_with_increased_size(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}

fn handle_result(res: Result<(), Box<dyn Any + Send>>) {
    if let Err(e) = &res {
        if let Some(msg) = e.downcast_ref::<String>() {
            println!("--> Panic occurred with message: {}\n", msg);
        }
    }
}
