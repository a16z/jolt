#![expect(
    dead_code,
    reason = "implementation target: the instruction read-RAF kernel wires this once it lands"
)]

use jolt_field::{Field, Fr};

use super::context::CudaKernelContext;
use super::dense_product::DeviceDenseProduct;
use super::error::CudaError;

pub struct DeviceCycleRounds {
    product: DeviceDenseProduct,
    ra_count: usize,
}

impl DeviceCycleRounds {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        eq_reduction: &[F],
        combined_val: &[F],
        ra: &[Vec<F>],
        rounds: usize,
    ) -> Result<Self, CudaError> {
        let mut factors = Vec::with_capacity(ra.len() + 2);
        factors.push(eq_reduction.to_vec());
        factors.push(combined_val.to_vec());
        factors.extend(ra.iter().cloned());
        let degree = ra.len() + 2;
        let product = DeviceDenseProduct::new(context, &[], &factors, None, None, rounds, degree)?;
        Ok(Self {
            product,
            ra_count: ra.len(),
        })
    }

    pub fn round_message<F: Field>(
        &self,
        context: &CudaKernelContext,
    ) -> Result<Vec<F>, CudaError> {
        self.product.round_evals(context)
    }

    pub fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        self.product.bind(context, challenge)
    }

    pub fn ra_finals<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        let finals: Vec<F> = self.product.factor_finals(context)?;
        if finals.len() != self.ra_count + 2 {
            return Err(CudaError::LengthMismatch {
                expected: self.ra_count + 2,
                got: finals.len(),
            });
        }
        Ok(finals[2..].to_vec())
    }

    pub const fn rounds_bound(&self) -> usize {
        self.product.rounds_bound()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::Fr;
    use jolt_lookup_tables::tables::LookupTableKind;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::DeviceCycleRounds;
    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };

    const ADDRESS_BITS: usize = 128;
    const RA_COUNT: usize = 8;

    fn rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> =
            LookupTableKind::<RISCV_XLEN>::iter().collect();
        (0..1usize << log_t)
            .map(|j| {
                let mixed = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let index = (u128::from(mixed) << 61) | u128::from(mixed.rotate_left(17));
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(index),
                    table_index: TableIndex(if mixed.is_multiple_of(11) {
                        None
                    } else {
                        Some(tables[(mixed % tables.len() as u64) as usize].index())
                    }),
                    raf_flag: InstructionRafFlag(mixed.is_multiple_of(3)),
                }
            })
            .collect()
    }

    fn reference_at_cycle_rounds(log_t: usize, seed: u64) -> InstructionReadRafKernel<Fr> {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            ADDRESS_BITS,
            NonZeroUsize::new(RA_COUNT).unwrap(),
        );
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
        let mut kernel = InstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            rows(log_t, seed),
            fr(seed + 1),
        )
        .expect("reference kernel");
        for round in 0..ADDRESS_BITS {
            kernel
                .bind(fr(seed + round as u64 + 71))
                .expect("reference bind");
        }
        kernel
    }

    proptest! {
        #[test]
        fn cycle_rounds_match_the_reference_round_for_round(
            log_t in 4usize..=8,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let mut host = reference_at_cycle_rounds(log_t, seed);
            let tables = host.cycle_tables.as_ref().expect("cycle tables");
            let ra: Vec<Vec<Fr>> = tables.ra.iter().map(|p| p.evals().to_vec()).collect();
            let mut device = DeviceCycleRounds::new(
                context,
                tables.eq_reduction.evals(),
                tables.combined_val.evals(),
                &ra,
                log_t,
            )
            .expect("device cycle rounds");

            for round in 0..log_t {
                let expected = host.cycle_message().expect("reference cycle message");
                let got: Vec<Fr> = device.round_message(context).expect("device message");
                prop_assert_eq!(
                    got,
                    expected,
                    "cycle round {} message diverged",
                    round
                );

                let challenge = fr(seed + round as u64 + 211);
                host.bind(challenge).expect("reference bind");
                device.bind(context, challenge).expect("device bind");
            }

            let expected: Vec<Fr> = host
                .cycle_tables
                .as_ref()
                .expect("cycle tables")
                .ra
                .iter()
                .map(|ra| ra.evals()[0])
                .collect();
            let got: Vec<Fr> = device.ra_finals(context).expect("device ra finals");
            prop_assert_eq!(got, expected, "ra final claims diverged");
        }
    }
}
