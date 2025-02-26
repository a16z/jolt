use std::time::Instant;

pub fn main() {
    let (prove, verify) = guest::build_int_to_string();

    let (output, proof) = prove(81);
    let is_valid = verify(proof);

    println!("int to string output: {:?}", output);
    println!("int to string valid: {}", is_valid);

    let (prove, verify) = guest::build_string_concat();

    let now = Instant::now();
    let (output, proof) = prove(20);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(proof);

    println!("string concat output: {:?}", output);
    println!("string concat valid: {}", is_valid);
}
