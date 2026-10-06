//! Streamed trace decomposition benchmark; setup is untimed and rows are generated lazily.
//!
//! See specs/akita-native-trace-batching.md for the comparison procedure.

use std::{env::VarError, error::Error, hint::black_box, sync::Arc, time::Instant};

use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_params::SetupMatrixCapacity;
use akita_pcs::custom_source::{
    DecomposeFoldBatchPlan, OpeningBatchKernel, RootOpeningSource, RootPolyShape,
};
use akita_pcs::{AkitaProverSetup, CpuBackend};
use jolt_akita::{AkitaField, TraceOneHotColumn, TraceOneHotRows};

struct SyntheticTrace<const ACTIVE_EVERY: usize> {
    rows: usize,
    columns: usize,
    k: usize,
}

impl<const ACTIVE_EVERY: usize> TraceOneHotRows for SyntheticTrace<ACTIVE_EVERY> {
    fn num_rows(&self) -> usize {
        self.rows
    }

    fn num_columns(&self) -> usize {
        self.columns
    }

    fn fill_row(&self, row: usize, selected: &mut [u8]) {
        for (column, value) in selected.iter_mut().enumerate() {
            *value = if ACTIVE_EVERY == 1 || (row + column).is_multiple_of(ACTIVE_EVERY) {
                ((row * (2 * column + 1) + column) % self.k) as u8
            } else {
                0
            };
        }
    }
}

#[expect(clippy::print_stdout, reason = "Benchmark emits a CSV record")]
fn measure<const D: usize>(args: &[usize]) -> Result<(), Box<dyn Error>> {
    let [log_rows, samples, _, columns, k, positions, digits, chunks] = *args else {
        return Err("expected eight benchmark arguments".into());
    };
    let rows = 1usize << log_rows;
    let (activity, trace): (&str, Arc<dyn TraceOneHotRows>) =
        match std::env::var("JOLT_AKITA_BENCH_ACTIVITY").as_deref() {
            Ok("all") | Err(VarError::NotPresent) => {
                ("all", Arc::new(SyntheticTrace::<1> { rows, columns, k }))
            }
            Ok("quarter") => (
                "quarter",
                Arc::new(SyntheticTrace::<4> { rows, columns, k }),
            ),
            Ok(_) => return Err("JOLT_AKITA_BENCH_ACTIVITY must be all or quarter".into()),
            Err(error) => return Err(error.to_string().into()),
        };
    let source_columns = TraceOneHotColumn::new(k, D, trace)?;
    let source = &source_columns[0];
    let num_blocks = RootPolyShape::<AkitaField, D>::num_ring_elems(source).div_ceil(positions);
    let synthetic_challenges = match std::env::var("JOLT_AKITA_BENCH_CHALLENGES").as_deref() {
        Ok("synthetic") => true,
        Ok("production") | Err(VarError::NotPresent) => false,
        Ok(_) => return Err("JOLT_AKITA_BENCH_CHALLENGES must be production or synthetic".into()),
        Err(error) => return Err(error.to_string().into()),
    };
    let production = SparseChallengeConfig::production_for_ring_dim(D)
        .ok_or("unsupported production challenge dimension")?;
    let challenges = (0..num_blocks * source_columns.len())
        .map(|block| {
            let weight = if synthetic_challenges {
                D / 4
            } else {
                production.weight()
            };
            SparseChallenge {
                positions: (0..weight)
                    .map(|index| {
                        if synthetic_challenges {
                            (index * 4) as u32
                        } else {
                            ((index * (2 * block + 1) + block) & (D - 1)) as u32
                        }
                    })
                    .collect::<Vec<_>>()
                    .into(),
                coeffs: (0..weight)
                    .map(|index| {
                        let magnitude = if synthetic_challenges || index < production.count_pm1 {
                            1
                        } else {
                            2
                        };
                        if (index + block) % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        }
                    })
                    .collect::<Vec<_>>()
                    .into(),
            }
        })
        .collect::<Vec<_>>();
    let challenges = Challenges::from_sparse(challenges, num_blocks, source_columns.len())?;
    let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, chunks)?;
    let plan = if chunks == 1 {
        DecomposeFoldBatchPlan::Sparse {
            challenges: challenges.as_slice(),
            num_positions_per_block: positions,
            num_digits: digits,
            log_basis: 3,
        }
    } else {
        DecomposeFoldBatchPlan::SparseChunked {
            challenges: &challenges,
            chunk_ranges: &chunk_ranges,
            num_positions_per_block: positions,
            num_digits: digits,
            log_basis: 3,
        }
    };
    let setup = AkitaProverSetup::<AkitaField>::generate_with_capacity(
        1,
        1,
        SetupMatrixCapacity {
            num_field_elements: 1,
        },
    )?;
    let backend = CpuBackend::<AkitaField, AkitaField>::new(setup.expanded)?;
    let sources = source_columns.iter().collect::<Vec<_>>();
    let warmups =
        std::env::var("JOLT_AKITA_BENCH_WARMUPS").map_or(Ok(2), |value| value.parse::<usize>())?;
    let mut timings = Vec::with_capacity(samples);
    for sample in 0..samples + warmups {
        let view = RootOpeningSource::<AkitaField, D>::opening_batch(&sources)?;
        let start = Instant::now();
        let witness = <CpuBackend<AkitaField, AkitaField> as OpeningBatchKernel<
            _,
            AkitaField,
            D,
        >>::decompose_fold_batch(&backend, None, black_box(view), plan)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let _ = black_box(&witness);
        if sample >= warmups {
            timings.push(elapsed);
        }
    }
    timings.sort_by(f64::total_cmp);
    let median = timings
        .get(samples / 2)
        .ok_or("sample count must be positive")?;
    let challenge_profile = if synthetic_challenges {
        "synthetic"
    } else {
        "production"
    };
    println!("{log_rows},{D},{columns},{k},{positions},{digits},{chunks},{challenge_profile},{activity},{median:.6}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args()
        .skip(1)
        .map(|arg| arg.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let [log_rows, samples, d, _, _, positions, digits, chunks] = args.as_slice() else {
        return Err(
            "usage: trace_decompose log2_rows samples D columns K positions digits chunks".into(),
        );
    };
    if !(12..=28).contains(log_rows)
        || *samples == 0
        || !positions.is_power_of_two()
        || *digits == 0
        || ![1, 2, 4, 8].contains(chunks)
    {
        return Err("require log2_rows=12..28, positive samples/digits, power-of-two positions, chunks=1/2/4/8".into());
    }
    match d {
        64 => measure::<64>(&args),
        128 => measure::<128>(&args),
        256 => measure::<256>(&args),
        512 => measure::<512>(&args),
        _ => Err("D must be 64, 128, 256, or 512".into()),
    }
}
