use std::marker::PhantomData;

use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use tracer::JoltDevice;

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        range_mask_polynomial::is_valid_range,
    },
    utils::thread::unsafe_allocate_zero_vec,
};

pub struct ProgramIOPolynomial<F: JoltField> {
    range_start: u64,
    range_end: u64,
    poly: MultilinearPolynomial<F>,
    _field: PhantomData<F>,
}

// Differs from the `remap_address` in `ram.rs` because we want to map
// input_start to the 0 index
fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4
    } else {
        panic!("Unexpected address {address}")
    }
}

impl<F: JoltField> ProgramIOPolynomial<F> {
    pub fn new(program_io: &JoltDevice) -> Self {
        let range_start = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        let range_end = remap_address(RAM_START_ADDRESS, &program_io.memory_layout);

        println!("Range: [{range_start:b}, {range_end:b})");
        assert!(is_valid_range(range_start, range_end));

        let mut coeffs: Vec<u32> = vec![0; (range_end - range_start) as usize];

        let mut input_index = range_start as usize;
        // Convert input bytes into words and populate `coeffs`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            coeffs[input_index] = word;
            input_index += 1;
        }

        let mut output_index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert output bytes into words and populate `v_io`
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            coeffs[output_index] = word;
            output_index += 1;
        }

        // Copy panic bit
        let panic_index =
            remap_address(program_io.memory_layout.panic, &program_io.memory_layout) as usize;
        coeffs[panic_index] = program_io.panic as u32;

        if !program_io.panic {
            // Set termination bit
            let termination_index = remap_address(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            ) as usize;
            coeffs[termination_index] = 1;
        }

        Self {
            range_start,
            range_end,
            _field: PhantomData,
            poly: coeffs.into(),
        }
    }

    pub fn evaluate(&self, r_address: &[F]) -> F {
        let (r_hi, r_lo) = r_address.split_at(r_address.len() - self.poly.get_num_vars());
        debug_assert_eq!(r_lo.len(), self.poly.get_num_vars());

        let mut result = self.poly.evaluate(r_lo);

        let num_leading_zeros = r_address.len() - self.range_end.trailing_zeros() as usize;
        let num_ones = (self.range_start >> self.range_start.trailing_zeros()).trailing_ones();
        debug_assert_eq!(r_hi.len(), num_leading_zeros + num_ones as usize);

        for r_i in r_hi[..num_leading_zeros].iter() {
            result *= F::one() - r_i;
        }

        for r_i in r_hi[num_leading_zeros..num_leading_zeros + num_ones as usize].iter() {
            result *= *r_i;
        }

        result
    }
}
