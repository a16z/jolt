use common::constants::RAM_START_ADDRESS;
use tracer::JoltDevice;

use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, PolynomialEvaluation},
    zkvm::ram::remap_address,
};

pub struct ProgramIOPolynomial<F: JoltField> {
    poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> ProgramIOPolynomial<F> {
    pub fn new(program_io: &JoltDevice) -> Self {
        let range_end = remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap();

        // TODO(moodlezoup) avoid next_power_of_two
        let mut coeffs: Vec<u64> = vec![0; range_end.next_power_of_two() as usize];

        let mut input_index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        // Convert input bytes into words and populate `coeffs`
        for chunk in program_io.inputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            coeffs[input_index] = word;
            input_index += 1;
        }

        let mut output_index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        // Convert output bytes into words and populate `coeffs`
        for chunk in program_io.outputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            coeffs[output_index] = word;
            output_index += 1;
        }

        // Copy panic bit
        let panic_index = remap_address(program_io.memory_layout.panic, &program_io.memory_layout)
            .unwrap() as usize;
        coeffs[panic_index] = program_io.panic as u64;

        if !program_io.panic {
            // Set termination bit
            let termination_index = remap_address(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            )
            .unwrap() as usize;
            coeffs[termination_index] = 1;
        }

        Self {
            poly: coeffs.into(),
        }
    }

    pub fn evaluate(&self, r_address: &[F::Challenge]) -> F {
        let (r_hi, r_lo) = r_address.split_at(r_address.len() - self.poly.get_num_vars());
        debug_assert_eq!(r_lo.len(), self.poly.get_num_vars());

        let mut result = self.poly.evaluate(r_lo);
        for r_i in r_hi.iter() {
            result *= F::one() - r_i;
        }

        result
    }
}
