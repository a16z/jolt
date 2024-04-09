pub fn main() {
    let (prove_mandelbrot, verify_mandelbrot) = guest::build_mandelbrot();

    let (output, proof) = prove_mandelbrot(2.0, 1.0, 50);
    let is_valid = verify_mandelbrot(proof);

    println!("output: {:?}", output);
    println!("valid: {}", is_valid);
}
