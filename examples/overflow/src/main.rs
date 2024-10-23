use std::panic;

pub fn main() {
    let (prove_overflow_atack, verify_overflow_stack) = guest::build_overflow_stack();

    let result = panic::catch_unwind(|| {
        let (_, proof) = prove_overflow_atack();
        verify_overflow_stack(proof);
    });

    match &result {
        Err(e) => match e.downcast_ref::<String>() {
            Some(msg) => println!("Panic occurred with message: {}", msg),
            None => println!("Panic occurred with unknown message type"),
        },
        Ok(_) => println!("No panic occurred"),
    }

    // println!("output: {}", output);
    // println!("valid: {}", is_valid);
}
