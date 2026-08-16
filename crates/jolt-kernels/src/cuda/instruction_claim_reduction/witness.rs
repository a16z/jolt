use jolt_witness::witnesses::{
    LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
    RightLookupOperand,
};
use jolt_witness::WitnessBundle;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionClaimReductionWitness {
    #[opening(LookupOutput)]
    pub output: LookupOutput,
    #[opening(LeftLookupOperand)]
    pub left_lookup: LeftLookupOperand,
    #[opening(RightLookupOperand)]
    pub right_lookup: RightLookupOperand,
    #[opening(LeftInstructionInput)]
    pub left_input: LeftInstructionInput,
    #[opening(RightInstructionInput)]
    pub right_input: RightInstructionInput,
}

pub struct Packed {
    pub output: Vec<u64>,
    pub left_lookup: Vec<u64>,
    pub right_lookup: Vec<u128>,
    pub left_input: Vec<u64>,
    pub right_input: Vec<i128>,
}

const PACK_CHUNK: usize = 1 << 14;

pub fn device_columns(
    context: &CudaKernelContext,
    rows: &[InstructionClaimReductionWitness],
) -> Result<Vec<DeviceFrVec>, CudaError> {
    let packed = packed_columns(rows);
    Ok(vec![
        context.u64_to_montgomery(&packed.output)?,
        context.u64_to_montgomery(&packed.left_lookup)?,
        context.u128_to_montgomery(&packed.right_lookup)?,
        context.u64_to_montgomery(&packed.left_input)?,
        context.i128_to_montgomery(&packed.right_input)?,
    ])
}

pub fn packed_columns(rows: &[InstructionClaimReductionWitness]) -> Packed {
    let mut output = vec![0u64; rows.len()];
    let mut left_lookup = vec![0u64; rows.len()];
    let mut right_lookup = vec![0u128; rows.len()];
    let mut left_input = vec![0u64; rows.len()];
    let mut right_input = vec![0i128; rows.len()];

    #[cfg(feature = "parallel")]
    output
        .par_chunks_mut(PACK_CHUNK)
        .zip(left_lookup.par_chunks_mut(PACK_CHUNK))
        .zip(right_lookup.par_chunks_mut(PACK_CHUNK))
        .zip(left_input.par_chunks_mut(PACK_CHUNK))
        .zip(right_input.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|(((((out, ll), rl), li), ri), rows)| fill(out, ll, rl, li, ri, rows));
    #[cfg(not(feature = "parallel"))]
    output
        .chunks_mut(PACK_CHUNK)
        .zip(left_lookup.chunks_mut(PACK_CHUNK))
        .zip(right_lookup.chunks_mut(PACK_CHUNK))
        .zip(left_input.chunks_mut(PACK_CHUNK))
        .zip(right_input.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|(((((out, ll), rl), li), ri), rows)| fill(out, ll, rl, li, ri, rows));

    Packed {
        output,
        left_lookup,
        right_lookup,
        left_input,
        right_input,
    }
}

fn fill(
    out: &mut [u64],
    left_lookup: &mut [u64],
    right_lookup: &mut [u128],
    left_input: &mut [u64],
    right_input: &mut [i128],
    rows: &[InstructionClaimReductionWitness],
) {
    for (index, row) in rows.iter().enumerate() {
        out[index] = row.output.0;
        left_lookup[index] = row.left_lookup.0;
        right_lookup[index] = row.right_lookup.0;
        left_input[index] = row.left_input.0;
        right_input[index] = row.right_input.0;
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{
        LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
        RightLookupOperand,
    };

    use super::{device_columns, InstructionClaimReductionWitness};
    use crate::cuda::common::context::shared_context;

    fn sample_rows() -> Vec<InstructionClaimReductionWitness> {
        let right_inputs: [i128; 6] = [0, -1, 1, -(1i128 << 100), 1i128 << 100, -(i128::MAX - 3)];
        let right_lookups: [u128; 6] = [
            0,
            1,
            u64::MAX as u128,
            (u64::MAX as u128) + 1,
            1u128 << 127,
            u128::MAX,
        ];
        right_inputs
            .into_iter()
            .zip(right_lookups)
            .enumerate()
            .map(
                |(index, (right_input, right_lookup))| InstructionClaimReductionWitness {
                    output: LookupOutput(index as u64 * 7 + 1),
                    left_lookup: LeftLookupOperand(u64::MAX - index as u64),
                    right_lookup: RightLookupOperand(right_lookup),
                    left_input: LeftInstructionInput(index as u64 * 11),
                    right_input: RightInstructionInput(right_input),
                },
            )
            .collect()
    }

    #[test]
    fn sample_rows_exercise_the_sign_and_wide_paths() {
        let rows = sample_rows();
        assert!(
            rows.iter().any(|row| row.right_input.0 < 0),
            "no synthetic row carries a negative right instruction input",
        );
        assert!(
            rows.iter().any(|row| row.right_input.0 > 0),
            "no synthetic row carries a positive right instruction input",
        );
        assert!(
            rows.iter().any(|row| row.right_input.0 == 0),
            "no synthetic row carries a zero right instruction input",
        );
        assert!(
            rows.iter()
                .any(|row| row.right_input.0.unsigned_abs() > u64::MAX as u128),
            "no synthetic row needs the high magnitude word of the right instruction input",
        );
        assert!(
            rows.iter()
                .any(|row| row.right_lookup.0 > i128::MAX as u128),
            "no synthetic row exceeds the signed range of the right lookup operand, so a \
             conversion that treats it as signed would still pass",
        );
    }

    #[test]
    fn synthetic_device_columns_match_the_host_conversion() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = sample_rows();
        let expected: Vec<Vec<Fr>> = vec![
            rows.iter().map(|row| Fr::from_u64(row.output.0)).collect(),
            rows.iter()
                .map(|row| Fr::from_u64(row.left_lookup.0))
                .collect(),
            rows.iter()
                .map(|row| Fr::from_u128(row.right_lookup.0))
                .collect(),
            rows.iter()
                .map(|row| Fr::from_u64(row.left_input.0))
                .collect(),
            rows.iter()
                .map(|row| Fr::from_i128(row.right_input.0))
                .collect(),
        ];
        let got: Vec<Vec<Fr>> = device_columns(context, &rows)
            .expect("device columns")
            .iter()
            .map(|column| column.to_host().expect("download"))
            .collect();
        assert_eq!(got, expected, "packed device columns diverged");
    }
}
