use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::Fr;

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub const NUM_PREFIXES: usize = 46;

pub const HINT_POINTS: usize = 2;

pub fn prefix_mle_round(
    context: &CudaKernelContext,
    checkpoints: &DeviceFrVec,
    challenge: Fr,
    has_r_x: bool,
    round: usize,
    b_len: usize,
    half: usize,
) -> Result<DeviceFrVec, CudaError> {
    if checkpoints.len() != NUM_PREFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES,
            got: checkpoints.len(),
        });
    }
    let mut out = context.alloc(HINT_POINTS * NUM_PREFIXES * half)?;
    let has_r_x = u32::from(has_r_x);
    let challenge = context.upload(&[challenge])?;
    let round = CudaKernelContext::count_of(round)?;
    let b_len_arg = CudaKernelContext::count_of(b_len)?;
    let half_arg = CudaKernelContext::count_of(half)?;
    let prefix_count = CudaKernelContext::count_of(NUM_PREFIXES)?;
    let points = CudaKernelContext::count_of(HINT_POINTS)?;

    let mut builder = context.stream().launch_builder(context.pfx_mle_round());
    let _ = builder.arg(checkpoints.limbs());
    let _ = builder.arg(challenge.limbs());
    let _ = builder.arg(&has_r_x);
    let _ = builder.arg(&round);
    let _ = builder.arg(&b_len_arg);
    let _ = builder.arg(&half_arg);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: block `(b_block, prefix < NUM_PREFIXES, point < HINT_POINTS)`
    // with thread `b < half` reads the `NUM_PREFIXES` checkpoints (length-checked
    // above) and the single-element challenge, and writes
    // `out[(point * NUM_PREFIXES + prefix) * half + b]`, one slot per
    // (point, prefix, b) of the `HINT_POINTS * NUM_PREFIXES * half` allocated.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (half_arg.div_ceil(BLOCK), prefix_count, points),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    context.stream().synchronize()?;
    Ok(out)
}

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
    use core::ops::Not as _;

    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::prefixes::{PrefixEval, Prefixes, ALL_PREFIXES};
    use jolt_poly::{BindingOrder, Polynomial};
    use strum::EnumCount;

    use super::{
        default_checkpoints, prefix_mle_round, update_checkpoints, HINT_POINTS, NUM_PREFIXES,
    };
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;
    use crate::optimized::instruction_read_raf::extension_pair;

    const CHUNK_LEN: usize = 8;
    const CHUNK_SIZE: usize = 1 << CHUNK_LEN;
    const ADDRESS_BITS: usize = 128;

    const WINDOW_RESTRICTED: [Prefixes; 6] = [
        Prefixes::RightShift,
        Prefixes::LeftShift,
        Prefixes::LeftShiftHelper,
        Prefixes::RightShiftW,
        Prefixes::LeftShiftWHelper,
        Prefixes::LeftShiftW,
    ];

    fn window_restricted(index: usize) -> bool {
        WINDOW_RESTRICTED
            .iter()
            .any(|prefix| *prefix as usize == index)
    }

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

    fn chunk_tables(checkpoints: &[Fr], suffix_len: usize) -> Vec<Polynomial<Fr>> {
        let wrapped: Vec<PrefixEval<Fr>> =
            checkpoints.iter().copied().map(PrefixEval::from).collect();
        ALL_PREFIXES
            .iter()
            .map(|prefix| {
                Polynomial::new(
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
                )
            })
            .collect()
    }

    fn bound_chunk_checkpoints(
        checkpoints: &[Fr],
        suffix_len: usize,
        round_challenges: &[Fr],
    ) -> Vec<Fr> {
        chunk_tables(checkpoints, suffix_len)
            .into_iter()
            .map(|mut table| {
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
    fn prefix_round_evaluations_match_optimized_bound_tables() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut host_checkpoints = modular_defaults();
        let mut device_checkpoints = context.upload(&host_checkpoints).expect("upload defaults");
        let mut restricted_diverged = std::collections::BTreeSet::new();

        for chunk in 0..ADDRESS_BITS / CHUNK_LEN {
            let suffix_len = ADDRESS_BITS - (chunk + 1) * CHUNK_LEN;
            let round_challenges: Vec<Fr> = (0..CHUNK_LEN)
                .map(|i| fr(17 + (chunk * CHUNK_LEN + i) as u64))
                .collect();
            let mut tables = chunk_tables(&host_checkpoints, suffix_len);

            for step in 0..CHUNK_LEN {
                let round = chunk * CHUNK_LEN + step;
                let half = (CHUNK_SIZE >> step) / 2;
                let has_r_x = round.is_multiple_of(2).not();
                let challenge = if has_r_x {
                    round_challenges[step - 1]
                } else {
                    Fr::from_u64(0)
                };
                let got = prefix_mle_round(
                    context,
                    &device_checkpoints,
                    challenge,
                    has_r_x,
                    round,
                    half.ilog2() as usize,
                    half,
                )
                .expect("device prefix_mle_round")
                .to_host()
                .expect("download");

                let mut divergent: Vec<String> = Vec::new();
                for (index, prefix) in ALL_PREFIXES.iter().enumerate() {
                    for b in 0..half {
                        let (at_zero, at_two) = extension_pair(tables[index].evals(), b, half);
                        for (point, expected) in [at_zero, at_two].into_iter().enumerate() {
                            if got[(point * NUM_PREFIXES + index) * half + b] != expected {
                                if window_restricted(index) {
                                    let _ = restricted_diverged.insert(index);
                                } else {
                                    divergent.push(format!("{prefix:?} (b {b}, point {point})"));
                                }
                            }
                        }
                    }
                }
                assert_eq!(
                    divergent,
                    Vec::<String>::new(),
                    "chunk {chunk} round {round}: the device's closed-form prefix round \
                     evaluation disagrees with optimized's materialized-then-bound chunk table",
                );

                for table in &mut tables {
                    table.bind_with_order(round_challenges[step], BindingOrder::HighToLow);
                }
                if round.is_multiple_of(2) {
                    continue;
                }
                device_checkpoints = update_checkpoints(
                    context,
                    &device_checkpoints,
                    round_challenges[step - 1],
                    round_challenges[step],
                    round,
                    suffix_len,
                )
                .expect("device update_checkpoints");
            }

            host_checkpoints = tables.iter().map(|table| table.evals()[0]).collect();
        }
        assert_eq!(HINT_POINTS, 2, "the oracle pairs c=0 with c=2");
        assert_eq!(
            restricted_diverged
                .iter()
                .map(|index| format!("{:?}", ALL_PREFIXES[*index]))
                .collect::<Vec<_>>(),
            WINDOW_RESTRICTED
                .iter()
                .map(|prefix| format!("{prefix:?}"))
                .collect::<Vec<_>>(),
            "WINDOW_RESTRICTED must name exactly the prefixes whose device closed form is valid \
             only where the y window is a contiguous run of leading ones — the domain \
             jolt_lookup_tables::tables::test_utils::gen_bitmask_lookup_index generates, and the \
             only domain the four shift/rotate tables that declare it ever reach. A prefix that \
             no longer diverges belongs in the checked set; a new divergence is a defect, not an \
             exclusion.",
        );
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
