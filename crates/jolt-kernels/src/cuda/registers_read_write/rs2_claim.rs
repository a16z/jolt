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
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::Fr;
    use jolt_prover_legacy::field::challenge::MontU128Challenge as LegacyChallenge;
    use jolt_prover_legacy::zkvm::registers::read_write_checking::RegistersReadWriteCheckingProver;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use crate::cuda::common::context::shared_context;

    use super::rs2_ra_claim;

    const LOG_T: usize = 6;
    const ADDRESS_BITS: usize = 7;

    fn random_cycle(rng: &mut StdRng) -> Cycle {
        let variants: Vec<Cycle> = Cycle::iter().collect();
        for _ in 0..10_000 {
            let index = rng.next_u64() as usize % variants.len();
            let candidate = variants[index].random(rng);
            if jolt_prover_legacy::zkvm::instruction::JoltTraceCycle::try_new(&candidate).is_ok() {
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

    #[test]
    fn device_rs2_claim_matches_legacy() {
        let Some(context) = shared_context() else {
            return;
        };
        let trace = trace(11);
        let r_address: Vec<LegacyChallenge<LegacyFr>> = (0..ADDRESS_BITS)
            .map(|i| LegacyChallenge::new(23 + i as u128))
            .collect();
        let r_cycle: Vec<LegacyChallenge<LegacyFr>> = (0..LOG_T)
            .map(|i| LegacyChallenge::new(71 + i as u128))
            .collect();

        let expected = Fr::from(
            RegistersReadWriteCheckingProver::<LegacyFr>::compute_rs2_ra_claim(
                &trace, &r_address, &r_cycle,
            ),
        );

        let encoded: Vec<u32> = trace
            .iter()
            .map(|cycle| {
                cycle
                    .rs2_read()
                    .map_or(crate::cuda::common::ra_poly::COLD, |(rs2, _)| {
                        u32::from(rs2)
                    })
            })
            .collect();
        let indices = context
            .upload_u32_slice(&encoded)
            .expect("upload rs2 addresses");
        let to_fr = |c: &LegacyChallenge<LegacyFr>| Fr::from(LegacyFr::from(*c));
        let got: Fr = rs2_ra_claim(
            context,
            &indices,
            encoded.len(),
            &r_address.iter().map(to_fr).collect::<Vec<_>>(),
            &r_cycle.iter().map(to_fr).collect::<Vec<_>>(),
        )
        .expect("device rs2 claim");

        assert_eq!(got, expected);
    }
}
