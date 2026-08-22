use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr_slice};
use crate::cuda::common::error::CudaError;

pub fn rs2_ra_claim<F: Field>(
    context: &CudaKernelContext,
    indices: &CudaSlice<u32>,
    cycles: usize,
    r_address: &[F],
    r_cycle: &[F],
) -> Result<F, CudaError> {
    if cycles != 1usize << r_cycle.len() {
        return Err(CudaError::LengthMismatch {
            expected: 1usize << r_cycle.len(),
            got: cycles,
        });
    }
    let addresses = 1usize << r_address.len();
    let eq_cycle = context.eq_evals(require_fr_slice(r_cycle)?)?;
    let eq_address = context.eq_evals(require_fr_slice(r_address)?)?;
    let mut terms = context.alloc(cycles)?;
    let cycle_count = CudaKernelContext::count_of(cycles)?;
    let address_count = CudaKernelContext::count_of(addresses)?;

    let mut builder = context.stream().launch_builder(context.rs2_claim());
    let _ = builder.arg(indices);
    let _ = builder.arg(eq_cycle.limbs());
    let _ = builder.arg(eq_address.limbs());
    let _ = builder.arg(&cycle_count);
    let _ = builder.arg(&address_count);
    let _ = builder.arg(terms.limbs_mut());
    // SAFETY: thread `j < cycles` reads `indices[j]` of `cycles` u32s and, only
    // when `indices[j] < addresses`, `eq_cycle[j]` and `eq_address[indices[j]]`
    // — both in range, the latter by that guard and the former because
    // `eq_cycle` holds `2^r_cycle.len() == cycles` elements (checked above). It
    // writes only `terms[j]` of `cycles`. All buffers are distinct allocations.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycle_count)) }?;
    context.stream().synchronize()?;

    let total = context.sum(&terms)?;
    fr_into(total).ok_or(CudaError::NotImplemented {
        kernel: "CUDA kernels support only the BN254 scalar field",
    })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::ra_poly::COLD;

    use super::rs2_ra_claim;

    const LOG_T: usize = 6;
    const ADDRESS_BITS: usize = 7;

    fn random_cycle(rng: &mut StdRng) -> Cycle {
        let variants: Vec<Cycle> = Cycle::iter().collect();
        for _ in 0..10_000 {
            let index = rng.next_u64() as usize % variants.len();
            let candidate = variants[index].random(rng);
            if candidate.instruction().try_jolt_instruction_row().is_ok() {
                return candidate;
            }
        }
        panic!("no convertible cycle variant found");
    }

    fn trace(seed: u64) -> Vec<Cycle> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..1usize << LOG_T)
            .map(|_| random_cycle(&mut rng))
            .collect()
    }

    fn one_hot_grid_mle(indices: &[u32], r_address: &[Fr], r_cycle: &[Fr]) -> Fr {
        let eq_address = EqPolynomial::new(r_address.to_vec()).evaluations();
        let eq_cycle = EqPolynomial::new(r_cycle.to_vec()).evaluations();
        let mut total = Fr::from_u64(0);
        for (cycle, &index) in indices.iter().enumerate() {
            if (index as usize) < eq_address.len() {
                total += eq_address[index as usize] * eq_cycle[cycle];
            }
        }
        total
    }

    #[test]
    fn device_rs2_claim_matches_the_one_hot_grid_mle() {
        let Some(context) = shared_context() else {
            return;
        };
        let trace = trace(11);
        let r_address: Vec<Fr> = (0..ADDRESS_BITS)
            .map(|i| Fr::from_u64(23 + i as u64))
            .collect();
        let r_cycle: Vec<Fr> = (0..LOG_T).map(|i| Fr::from_u64(71 + i as u64)).collect();

        let encoded: Vec<u32> = trace
            .iter()
            .map(|cycle| cycle.rs2_read().map_or(COLD, |(rs2, _)| u32::from(rs2)))
            .collect();
        assert!(
            encoded.iter().any(|&index| index != COLD),
            "no cycle in the fixture reads rs2, so the hot path is unexercised",
        );

        let expected = one_hot_grid_mle(&encoded, &r_address, &r_cycle);
        let indices = context
            .upload_u32_slice(&encoded)
            .expect("upload rs2 addresses");
        let got: Fr = rs2_ra_claim(context, &indices, encoded.len(), &r_address, &r_cycle)
            .expect("device rs2 claim");

        assert_eq!(got, expected);
    }
}
