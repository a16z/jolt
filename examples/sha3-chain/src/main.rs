use std::any::Any;
use tracing_subscriber::prelude::*;
use tracing_chrome::ChromeLayerBuilder;

pub fn main() {

    let mut layers = Vec::new();
    let mut guards: Vec<Box<dyn Any>> = vec![];
    
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    layers.push(chrome_layer.boxed());
    guards.push(Box::new(guard));

    let _subscriber = tracing_subscriber::registry().with(layers).init();

    let (prove_sha3_chain, verify_sha3_chain) = guest::build_sha3_chain();

    let input  = [5u8; 32];
    let iters = 100;
    let (output, proof) = prove_sha3_chain(input, iters);
    let is_valid = verify_sha3_chain(proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {}", is_valid);
}
