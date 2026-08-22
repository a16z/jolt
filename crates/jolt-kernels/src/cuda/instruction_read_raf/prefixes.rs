use cudarc::driver::PushKernelArg;
use jolt_field::Fr;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub const NUM_PREFIXES: usize = 46;

pub fn update_checkpoints(
    context: &CudaKernelContext,
    checkpoints: &DeviceFrVec,
    r_x: Fr,
    r_y: Fr,
    round: usize,
    suffix_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    if checkpoints.len() != NUM_PREFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES,
            got: checkpoints.len(),
        });
    }
    let challenge_x = context.upload(&[r_x])?;
    let challenge_y = context.upload(&[r_y])?;
    let mut out = context.alloc(NUM_PREFIXES)?;
    let count = CudaKernelContext::count_of(NUM_PREFIXES)?;
    let round_arg = CudaKernelContext::count_of(round)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;

    let mut builder = context
        .stream()
        .launch_builder(context.pfx_update_checkpoints());
    let _ = builder.arg(checkpoints.limbs());
    let _ = builder.arg(challenge_x.limbs());
    let _ = builder.arg(challenge_y.limbs());
    let _ = builder.arg(&round_arg);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < NUM_PREFIXES` reads any of the `NUM_PREFIXES` input
    // checkpoints (length-checked above) plus the two challenge elements, and
    // writes only `out[i]`. `out` is a fresh allocation distinct from
    // `checkpoints`, which is what lets a prefix read another's PRE-update value.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

pub fn default_checkpoints(context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(NUM_PREFIXES)?;
    let count = CudaKernelContext::count_of(NUM_PREFIXES)?;
    let mut builder = context
        .stream()
        .launch_builder(context.pfx_default_checkpoints());
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < NUM_PREFIXES` writes only `out[i]` of `NUM_PREFIXES`
    // elements and reads nothing but its own index. `out` is a fresh allocation.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::prefixes::{PrefixEval, Prefixes, ALL_PREFIXES};
    use jolt_poly::{BindingOrder, Polynomial};
    use strum::EnumCount;

    use super::{default_checkpoints, update_checkpoints, NUM_PREFIXES};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;

    const CHUNK_LEN: usize = 8;
    const CHUNK_SIZE: usize = 1 << CHUNK_LEN;
    const ADDRESS_BITS: usize = 128;

    #[test]
    fn prefix_count_matches_the_rust_enum() {
        assert_eq!(
            NUM_PREFIXES,
            Prefixes::COUNT,
            "device prefix count is out of sync with jolt-lookup-tables",
        );
        assert_eq!(ALL_PREFIXES.len(), NUM_PREFIXES);
    }

    fn modular_defaults() -> Vec<Fr> {
        ALL_PREFIXES
            .iter()
            .map(|prefix| prefix.default_checkpoint::<Fr>().value())
            .collect()
    }

    #[test]
    fn default_checkpoints_match_the_rust_defaults() {
        let Some(context) = shared_context() else {
            return;
        };
        let expected = modular_defaults();
        let got = default_checkpoints(context)
            .expect("device default_checkpoints")
            .to_host()
            .expect("download");
        assert_eq!(got, expected);
        assert_ne!(
            expected[Prefixes::Eq as usize],
            Fr::from_u64(0),
            "the Eq checkpoint seeds a product family and must not default to zero",
        );
    }

    fn bound_chunk_checkpoints(
        checkpoints: &[Fr],
        suffix_len: usize,
        round_challenges: &[Fr],
    ) -> Vec<Fr> {
        let wrapped: Vec<PrefixEval<Fr>> =
            checkpoints.iter().copied().map(PrefixEval::from).collect();
        ALL_PREFIXES
            .iter()
            .map(|prefix| {
                let mut table = Polynomial::new(
                    (0..CHUNK_SIZE)
                        .map(|x| {
                            prefix
                                .evaluate::<Fr>(
                                    &wrapped,
                                    LookupBits::new(x as u128, CHUNK_LEN),
                                    suffix_len,
                                )
                                .value()
                        })
                        .collect(),
                );
                for &challenge in round_challenges {
                    table.bind_with_order(challenge, BindingOrder::HighToLow);
                }
                table.evals()[0]
            })
            .collect()
    }

    fn divergent_prefixes(got: &[Fr], expected: &[Fr]) -> Vec<String> {
        ALL_PREFIXES
            .iter()
            .enumerate()
            .filter(|&(index, _)| got[index] != expected[index])
            .map(|(index, prefix)| format!("{prefix:?} (index {index})"))
            .collect()
    }

    #[test]
    fn checkpoint_updates_match_optimized_composition() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut expected = modular_defaults();
        let mut device = context.upload(&expected).expect("upload defaults");

        for chunk in 0..ADDRESS_BITS / CHUNK_LEN {
            let suffix_len = ADDRESS_BITS - (chunk + 1) * CHUNK_LEN;
            let round_challenges: Vec<Fr> = (0..CHUNK_LEN)
                .map(|i| fr(17 + (chunk * CHUNK_LEN + i) as u64))
                .collect();

            expected = bound_chunk_checkpoints(&expected, suffix_len, &round_challenges);
            for pair in 0..CHUNK_LEN / 2 {
                let round = chunk * CHUNK_LEN + 2 * pair + 1;
                device = update_checkpoints(
                    context,
                    &device,
                    round_challenges[2 * pair],
                    round_challenges[2 * pair + 1],
                    round,
                    suffix_len,
                )
                .expect("device update_checkpoints");
            }

            let got = device.to_host().expect("download");
            assert_eq!(
                divergent_prefixes(&got, &expected),
                Vec::<String>::new(),
                "chunk {chunk}: the device's closed-form per-round-pair checkpoint update \
                 disagrees with optimized's materialize-the-chunk-then-bind composition. The \
                 two tiers must reach the same checkpoints because the device's are consumed by \
                 the same lookup-table combine; adopting optimized's structure is the agreed \
                 fix. KNOWN FAILING on the two sign-extension prefixes until that lands.",
            );
        }
    }
}
