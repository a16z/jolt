//! Initial-RAM advice evaluation uploads the oracle table and reduces it on the
//! GPU, while program-image evaluation uses the canonical host helper.
//!
//! CUDA builds the equality weights and sums their products with the advice
//! values on device; optimized performs the same computation on the host.

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_field::JoltField;
use jolt_witness::JoltWitnessPlane;

use super::common::device::{fr_into, require_fr_slice};
use super::{require_context, CudaBackend};
use crate::opening::{RamInitialOpening, RamInitialOpeningEvaluation};
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

#[cfg(test)]
pub(crate) const NO_HOT: u32 = u32::MAX;

impl<F: JoltField> RamInitialOpeningEvaluation<F> for CudaBackend {
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        openings: &[RamInitialOpening<'_, F>],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<Vec<F>, KernelError<F>> {
        let context = require_context::<F>()?;
        openings
            .iter()
            .map(|opening| {
                let (table, point, scale) = match opening {
                    RamInitialOpening::Advice { kind, point } => (
                        dense_view(witness, ram_val_check_advice_opening(*kind))?,
                        *point,
                        F::one(),
                    ),
                    RamInitialOpening::ProgramImage { layout, point } => {
                        // Main owns sparse segment geometry, including non-aligned image starts.
                        return Ok(crate::opening::evaluate_program_image(
                            layout, point, witness,
                        ));
                    }
                };
                if table.len() != 1usize << point.len() {
                    return Err(KernelError::TableSizeMismatch {
                        table: "initial RAM opening".to_owned(),
                        expected: 1usize << point.len(),
                        got: table.len(),
                    });
                }
                let device = context.upload(require_fr_slice(&table)?)?;
                let eq = context.eq_evals(require_fr_slice(point)?)?;
                let products = context.mul(&device, &eq)?;
                let value = context.sum(&products)?;
                fr_into::<F>(value)
                    .map(|value| value * scale)
                    .ok_or(KernelError::Unsupported {
                        reason: "CUDA kernels support only the BN254 scalar field",
                    })
            })
            .collect()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::JoltAdviceKind;
    use jolt_field::{Fr, Ring};
    use jolt_witness::JoltWitnessPlane;
    use proptest::prelude::*;

    use super::{CudaBackend, NO_HOT};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{advice_plane, fr};
    use crate::opening::{RamInitialOpening, RamInitialOpeningEvaluation};
    use crate::reference::ReferenceBackend;
    use crate::ProofSession;

    const ADVICE_BYTES: usize = 4096;

    const ADVICE_KINDS: [JoltAdviceKind; 2] = [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted];

    const FOLD_GEOMETRIES: [(usize, usize, usize); 4] =
        [(8, 16, 8), (10, 18, 8), (12, 18, 6), (6, 16, 4)];

    fn synthetic_hot(cycles: usize, one_hot_k: usize, seed: u64) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                let mixed = (cycle as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                if mixed.is_multiple_of(7) {
                    NO_HOT
                } else {
                    (mixed >> 13) as usize as u32 % one_hot_k as u32
                }
            })
            .collect()
    }

    fn dense_one_hot_grid(hot: &[u32], cycles: usize, domain: usize) -> Vec<Fr> {
        let mut table = vec![Fr::from_u64(0); domain];
        for (cycle, &address) in hot.iter().enumerate() {
            if address != NO_HOT {
                table[address as usize * cycles + cycle] = Fr::from_u64(1);
            }
        }
        table
    }

    #[test]
    fn prefix_fold_matches_the_zero_extended_fold() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut exercised = 0usize;
        for (index, &(log_t, total_vars, _)) in FOLD_GEOMETRIES.iter().enumerate() {
            let cycles = 1usize << log_t;
            let domain = 1usize << total_vars;
            let sigma = total_vars.div_ceil(2);
            let columns = 1usize << sigma;
            if cycles >= domain || !cycles.is_multiple_of(columns) {
                continue;
            }
            exercised += 1;
            let column: Vec<Fr> = (0..cycles)
                .map(|cycle| fr((index as u64 + 1) * 7_919 + cycle as u64 + 1))
                .collect();
            let left: Vec<Fr> = (0..domain >> sigma)
                .map(|row| fr((index as u64 + 1) * 1_000 + row as u64 + 3))
                .collect();

            let device_column = context.upload(&column).expect("upload the real column");
            let extended = context
                .zero_extend(&device_column, domain)
                .expect("zero-extend the column");
            let expected = context
                .fold_rows(&extended, &left, sigma)
                .expect("zero-extended fold");
            let rows = cycles / columns;
            let got = context
                .fold_rows(&device_column, &left[..rows], sigma)
                .expect("prefix fold");

            assert_eq!(
                got, expected,
                "geometry {index} (log_T {log_t}, total_vars {total_vars}, sigma {sigma}): folding \
                 the real column against the head of `left` diverged from folding its \
                 zero-extended embedding",
            );
            assert!(
                expected.iter().any(|value| *value != Fr::from_u64(0)),
                "geometry {index}: the fold is identically zero, so the comparison proves nothing",
            );
        }
        assert!(
            exercised > 0,
            "no fold geometry has a column that tiles whole fold rows, so the prefix fold this \
             test exists to check was never run",
        );
    }

    #[test]
    fn one_hot_fold_matches_the_dense_fold() {
        let Some(context) = shared_context() else {
            return;
        };
        for (index, &(log_t, total_vars, log_k_chunk)) in FOLD_GEOMETRIES.iter().enumerate() {
            let cycles = 1usize << log_t;
            let one_hot_k = 1usize << log_k_chunk;
            let domain = 1usize << total_vars;
            assert!(
                one_hot_k * cycles <= domain,
                "geometry {index}: the one-hot span exceeds the grid",
            );
            let sigma = total_vars.div_ceil(2);
            let rows = domain >> sigma;
            let hot = synthetic_hot(cycles, one_hot_k, 17 + index as u64);
            assert!(
                hot.contains(&NO_HOT) && hot.iter().any(|&address| address != NO_HOT),
                "geometry {index}: the synthetic column is all hot or all cold",
            );
            let left: Vec<Fr> = (0..rows)
                .map(|row| fr((index as u64 + 1) * 1_000 + row as u64 + 3))
                .collect();

            let table = dense_one_hot_grid(&hot, cycles, domain);
            let device_table = context.upload(&table).expect("upload dense grid");
            let expected = context
                .fold_rows(&device_table, &left, sigma)
                .expect("dense fold");
            let device_hot = context.upload_u32_slice(&hot).expect("upload hot column");
            if sigma > log_t {
                assert!(
                    context
                        .one_hot_fold_window(&device_hot, cycles, 0, cycles, &left, sigma)
                        .is_err(),
                    "geometry {index} (sigma {sigma} > log_T {log_t}): the sparse fold accepted a \
                     geometry whose column index depends on the address, where its \
                     thread-per-column scheme does not hold",
                );
            } else {
                let got = context
                    .one_hot_fold_window(&device_hot, cycles, 0, cycles, &left, sigma)
                    .expect("sparse fold");
                assert_eq!(
                    got, expected,
                    "geometry {index} (log_T {log_t}, total_vars {total_vars}, log_k \
                     {log_k_chunk}, sigma {sigma}): the sparse fold diverged from the dense fold",
                );
            }

            let embedded = context
                .one_hot_embed_device(&device_hot, cycles, domain)
                .expect("sparse embed")
                .to_host()
                .expect("download embedded grid");
            assert_eq!(
                embedded, table,
                "geometry {index}: the sparse embed diverged from the dense grid",
            );
        }
    }

    #[test]
    fn prefix_fold_windows_sum_to_the_whole_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut exercised = 0usize;
        for (index, &(log_t, total_vars, _)) in FOLD_GEOMETRIES.iter().enumerate() {
            let cycles = 1usize << log_t;
            let domain = 1usize << total_vars;
            let sigma = total_vars.div_ceil(2);
            let columns = 1usize << sigma;
            if cycles >= domain || !cycles.is_multiple_of(columns) {
                continue;
            }
            let rows = cycles / columns;
            let column: Vec<Fr> = (0..cycles)
                .map(|cycle| fr((index as u64 + 3) * 6_151 + cycle as u64 + 5))
                .collect();
            let left: Vec<Fr> = (0..rows)
                .map(|row| fr((index as u64 + 2) * 1_301 + row as u64 + 13))
                .collect();
            let device_column = context.upload(&column).expect("upload the whole column");
            let expected = context
                .fold_rows(&device_column, &left, sigma)
                .expect("whole-domain prefix fold");
            assert!(
                expected.iter().any(|value| *value != Fr::from_u64(0)),
                "geometry {index}: the whole-domain fold is identically zero",
            );

            for windows in [2usize, 4, 8] {
                if cycles < windows {
                    continue;
                }
                exercised += 1;
                let len = cycles / windows;
                let mut got = vec![Fr::from_u64(0); columns];
                for window in 0..windows {
                    let base = window * len;
                    let shard = context
                        .upload(&column[base..base + len])
                        .expect("upload window column");
                    let part = context
                        .fold_rows_window(&shard, base, len, &left, sigma)
                        .expect("window prefix fold");
                    for (total, value) in got.iter_mut().zip(&part) {
                        *total += *value;
                    }
                }
                assert_eq!(
                    got, expected,
                    "geometry {index} (log_T {log_t}, sigma {sigma}, {windows} windows of {len} \
                     cycles): the windowed folds do not sum to the whole-domain fold",
                );
            }
        }
        assert!(
            exercised > 0,
            "no fold geometry was split, so the windowed prefix fold was never run",
        );
    }

    #[test]
    fn one_hot_fold_windows_sum_to_the_whole_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut exercised = 0usize;
        for (index, &(log_t, total_vars, log_k_chunk)) in FOLD_GEOMETRIES.iter().enumerate() {
            let cycles = 1usize << log_t;
            let sigma = total_vars.div_ceil(2);
            if sigma > log_t {
                continue;
            }
            let one_hot_k = 1usize << log_k_chunk;
            let rows = (1usize << total_vars) >> sigma;
            let hot = synthetic_hot(cycles, one_hot_k, 41 + index as u64);
            let left: Vec<Fr> = (0..rows)
                .map(|row| fr((index as u64 + 5) * 977 + row as u64 + 11))
                .collect();
            let device_hot = context.upload_u32_slice(&hot).expect("upload hot column");
            let expected = context
                .one_hot_fold_window(&device_hot, cycles, 0, cycles, &left, sigma)
                .expect("whole-domain sparse fold");
            assert!(
                expected.iter().any(|value| *value != Fr::from_u64(0)),
                "geometry {index}: the whole-domain fold is identically zero",
            );

            for windows in [2usize, 4, 8] {
                if cycles < windows {
                    continue;
                }
                exercised += 1;
                let len = cycles / windows;
                let mut got = vec![Fr::from_u64(0); 1usize << sigma];
                for window in 0..windows {
                    let base = window * len;
                    let shard = context
                        .upload_u32_slice(&hot[base..base + len])
                        .expect("upload window column");
                    let part = context
                        .one_hot_fold_window(&shard, cycles, base, len, &left, sigma)
                        .expect("window sparse fold");
                    for (total, value) in got.iter_mut().zip(&part) {
                        *total += *value;
                    }
                }
                assert_eq!(
                    got, expected,
                    "geometry {index} (log_T {log_t}, sigma {sigma}, {windows} windows of {len} \
                     cycles): the windowed folds do not sum to the whole-domain fold",
                );
            }
        }
        assert!(
            exercised > 0,
            "no fold geometry was split, so the windowed fold was never run",
        );
    }

    #[test]
    fn fixture_fold_geometries_cover_the_production_regime() {
        let strict = FOLD_GEOMETRIES
            .iter()
            .filter(|&&(log_t, total_vars, _)| total_vars.div_ceil(2) < log_t)
            .count();
        assert!(
            strict >= 2,
            "fewer than two geometries have sigma < log_T, so the multi-iteration fold loop that \
             production runs (sigma 13, log_T 22) is barely covered",
        );
        assert!(
            FOLD_GEOMETRIES
                .iter()
                .any(|&(log_t, total_vars, _)| total_vars.div_ceil(2) == log_t),
            "no geometry has sigma == log_T, the single-iteration boundary case",
        );
        assert!(
            FOLD_GEOMETRIES
                .iter()
                .any(|&(log_t, total_vars, _)| total_vars.div_ceil(2) > log_t),
            "no geometry has sigma > log_T, where the fold must fall back to the dense path",
        );
    }

    fn advice_opening(
        backend: &dyn RamInitialOpeningEvaluation<Fr>,
        witness: &dyn JoltWitnessPlane<Fr>,
        kind: JoltAdviceKind,
        point: &[Fr],
    ) -> Fr {
        backend
            .evaluate(
                &mut ProofSession::default(),
                &[RamInitialOpening::Advice { kind, point }],
                witness,
            )
            .expect("advice opening")[0]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn advice_opening_matches_reference(seed in any::<u64>()) {
            let Some(_) = shared_context() else { return Ok(()); };
            let fixture = advice_plane(ADVICE_BYTES, seed);
            let vars = fixture.trusted.len().ilog2() as usize;
            let witness = &fixture.plane as &dyn JoltWitnessPlane<Fr>;
            let point: Vec<Fr> = (0..vars)
                .map(|index| fr(seed ^ (index as u64).wrapping_mul(17).wrapping_add(5)))
                .collect();
            let expected: Vec<Fr> = ADVICE_KINDS
                .iter()
                .map(|&kind| advice_opening(&ReferenceBackend, witness, kind, &point))
                .collect();
            for (&kind, &expected) in ADVICE_KINDS.iter().zip(&expected) {
                let got = advice_opening(&CudaBackend, witness, kind, &point);
                prop_assert_eq!(got, expected, "{:?} advice opening diverged", kind);
            }
        }
    }

    #[test]
    fn fixture_advice_columns_discriminate() {
        let fixture = advice_plane(ADVICE_BYTES, 41);
        assert_eq!(
            fixture.trusted.len(),
            fixture.untrusted.len(),
            "the two advice columns must share a width for one point to serve both",
        );
        assert_ne!(
            fixture.trusted, fixture.untrusted,
            "the two advice columns are identical, so a kernel that ignored the kind would pass",
        );
        assert!(
            fixture
                .trusted
                .iter()
                .any(|value| *value != Fr::from_u64(0)),
            "the trusted advice column is all zeros",
        );
        assert!(
            fixture.trusted.len().is_power_of_two() && fixture.trusted.len() > 1,
            "the advice column must span at least one variable",
        );
    }
}
