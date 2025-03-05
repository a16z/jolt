use std::any::Any;
use std::panic;
use std::time::Instant;

pub fn main() {
    let (prove_overflow_stack, _) = guest::build_overflow_stack();

    let res = panic::catch_unwind(|| {
        // Trying to allocate a 1024-element array and sum it up.
        // With stack_size=1024, this should panic.
        prove_overflow_stack(); 
    });
    handle_result(res);

    // Now let's try to overflow the heap, this should also panic.
    let (prove_overflow_heap, _) = guest::build_overflow_heap();

    let res = panic::catch_unwind(|| {
        prove_overflow_heap();
    });
    handle_result(res);

    // Valid case for stack allocation. Calls overflow_stack() under the hood
    // but with stack_size=8192.
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
