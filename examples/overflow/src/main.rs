use std::any::Any;
use std::panic;

pub fn main() {
    let (prove_allocate_stack, _) = guest::build_allocate_stack();

    let res = panic::catch_unwind(|| {
        // trying to allocate 1024 elems array and sum it up
        // with stack_size=1024, should panic
        let (_, _) = prove_allocate_stack();
    });
    handle_result(res);

    // now lets try to overflow the heap, should also panic
    let (prove_allocate_heap, _) = guest::build_allocate_heap();

    let res = panic::catch_unwind(|| {
        let (_, _) = prove_allocate_heap();
    });
    handle_result(res);

    // valid case for stack allocation, calls allocate_stack() under the hood
    // but with stack_size=8192
    let (prove_allocate_stack_with_increased_size, verfiy_allocate_stack_with_increased_size) =
        guest::build_allocate_stack_with_increased_size();

    let (output, proof) = prove_allocate_stack_with_increased_size();
    let is_valid = verfiy_allocate_stack_with_increased_size(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}

fn handle_result(res: Result<(), Box<dyn Any + Send>>) {
    match &res {
        Err(e) => match e.downcast_ref::<String>() {
            Some(msg) => println!("Panic occurred with message: {}\n", msg),
            _ => (),
        },
        _ => (),
    }
}
