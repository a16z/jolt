#![expect(
    clippy::print_stdout,
    reason = "the compile probe emits one resource-admission report"
)]

use std::error::Error;

use jolt_kernels::metal::solinas::hamming_weight_claim_reduction::compile_hamming_weight_claim_reduction_probe;

fn main() -> Result<(), Box<dyn Error>> {
    let report = compile_hamming_weight_claim_reduction_probe()?;
    println!("{report:#?}");
    if !report.admitted() {
        return Err("fixed-29 Hamming-weight pipeline was not admitted".into());
    }
    Ok(())
}
