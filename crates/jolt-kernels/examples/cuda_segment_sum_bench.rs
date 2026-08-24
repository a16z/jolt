#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use jolt_kernels::cuda::{shared_context, AffineLimbs, DeviceSegments, JacobianLimbs, SegmentMode};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const SAMPLES: usize = 3;

const SMALL_WIDEST_CEILING: usize = 1_024;

const CHUNK_LEN: usize = 16_384;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Spread {
    Uniform,
    Skew25,
    Spiked,
    Skew70,
    Skew90,
    Concentrated,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Ascending,
    Random,
}

struct Shape {
    chunks: usize,
    one_hot_k: usize,
    spread: Spread,
    order: Order,
}

impl Spread {
    const fn label(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Skew25 => "spike25",
            Self::Spiked => "spike50",
            Self::Skew70 => "spike70",
            Self::Skew90 => "spike90",
            Self::Concentrated => "onehot",
        }
    }

    fn counts(self, one_hot_k: usize) -> Vec<u32> {
        let total = CHUNK_LEN as u32;
        match self {
            Self::Uniform => vec![total / one_hot_k as u32; one_hot_k],
            Self::Skew25 | Self::Spiked | Self::Skew70 | Self::Skew90 => {
                let head = match self {
                    Self::Skew25 => total / 4,
                    Self::Skew70 => total * 7 / 10,
                    Self::Skew90 => total * 9 / 10,
                    _ => total / 2,
                };
                let rest = (total - head) / (one_hot_k as u32 - 1).max(1);
                (0..one_hot_k)
                    .map(|address| if address == 0 { head } else { rest })
                    .collect()
            }
            Self::Concentrated => (0..one_hot_k)
                .map(|address| if address == 0 { total } else { 0 })
                .collect(),
        }
    }
}

impl Order {
    const fn label(self) -> &'static str {
        match self {
            Self::Ascending => "asc",
            Self::Random => "rand",
        }
    }
}

fn plan_for(shape: &Shape) -> (Vec<u32>, Vec<u32>) {
    let per_chunk = shape.spread.counts(shape.one_hot_k);
    let mut counts = Vec::with_capacity(shape.chunks * shape.one_hot_k);
    for _ in 0..shape.chunks {
        counts.extend_from_slice(&per_chunk);
    }
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut indices = Vec::with_capacity(shape.chunks * CHUNK_LEN);
    for _ in 0..shape.chunks {
        let mut cycle = 0u32;
        for &count in &per_chunk {
            match shape.order {
                Order::Ascending => {
                    for step in 0..count {
                        indices.push(cycle + step);
                    }
                    cycle += count;
                }
                Order::Random => {
                    for _ in 0..count {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        indices.push((state % CHUNK_LEN as u64) as u32);
                    }
                }
            }
        }
    }
    (indices, counts)
}

fn affine_limbs(point: &ark_bn254::G1Projective) -> AffineLimbs {
    let affine = point.into_affine();
    if affine.infinity {
        return AffineLimbs::IDENTITY;
    }
    AffineLimbs {
        x: affine.x.0 .0,
        y: affine.y.0 .0,
        infinity: false,
    }
}

fn time<T>(mut run: impl FnMut() -> T) -> (f64, T) {
    let warm = run();
    let mut best = f64::MAX;
    for _ in 0..SAMPLES {
        let now = Instant::now();
        let _ = run();
        best = best.min(now.elapsed().as_secs_f64() * 1e3);
    }
    (best, warm)
}

fn same(left: &[JacobianLimbs], right: &[JacobianLimbs]) -> bool {
    let point = |limbs: &JacobianLimbs| {
        ark_bn254::G1Projective::new_unchecked(
            ark_bn254::Fq::new_unchecked(ark_ff::BigInt(limbs.x)),
            ark_bn254::Fq::new_unchecked(ark_ff::BigInt(limbs.y)),
            ark_bn254::Fq::new_unchecked(ark_ff::BigInt(limbs.z)),
        )
        .into_affine()
    };
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| point(left) == point(right))
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device");
        return;
    };
    let mut rng = ChaCha20Rng::seed_from_u64(20_260_825);
    let points: Vec<ark_bn254::G1Projective> = (0..CHUNK_LEN)
        .map(|_| ark_bn254::G1Projective::rand(&mut rng))
        .collect();
    let bases = context
        .upload_g1_bases(&points.iter().map(affine_limbs).collect::<Vec<_>>())
        .expect("upload bases");

    let single: Vec<String> = std::env::args().skip(1).collect();
    let chunk_counts: Vec<usize> = if single.is_empty() {
        vec![32, 128, 512]
    } else {
        vec![single[0].parse().expect("chunks")]
    };

    println!(
        "{:>6}  {:>5}  {:>7}  {:>5}  {:>8}  {:>10}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}           {:>9}",
        "chunks",
        "k",
        "spread",
        "order",
        "segments",
        "entries",
        "widest",
        "small ms",
        "heavy ms",
        "warp ms",
        "class ms",
        "heavy Ma/s",
        "class Ma/s"
    );
    for chunks in chunk_counts {
        for one_hot_k in [4usize, 16, 64] {
            for spread in [
                Spread::Uniform,
                Spread::Skew25,
                Spread::Spiked,
                Spread::Skew70,
                Spread::Skew90,
                Spread::Concentrated,
            ] {
                for order in [Order::Ascending, Order::Random] {
                    let shape = Shape {
                        chunks,
                        one_hot_k,
                        spread,
                        order,
                    };
                    let (indices, counts) = plan_for(&shape);
                    let entries = indices.len();
                    let plan: DeviceSegments = context
                        .upload_segments(&bases, &indices, &counts)
                        .expect("upload segment plan");
                    let widest = plan.widest();

                    let (heavy_ms, heavy) = time(|| {
                        context
                            .segment_sums(&bases, &plan, SegmentMode::Heavy)
                            .expect("heavy segment sums")
                    });
                    let small = if widest <= SMALL_WIDEST_CEILING {
                        let (small_ms, small) = time(|| {
                            context
                                .segment_sums(&bases, &plan, SegmentMode::Small)
                                .expect("small segment sums")
                        });
                        assert!(
                            same(&small, &heavy),
                            "the small and heavy kernels disagree at k={one_hot_k} {} {}",
                            spread.label(),
                            order.label(),
                        );
                        format!("{small_ms:9.3}")
                    } else {
                        format!("{:>9}", "-")
                    };
                    let (class_ms, class_rate) = {
                        let (ms, piece) = time(|| {
                            context
                                .segment_sums(&bases, &plan, SegmentMode::Classes)
                                .expect("classed segment sums")
                        });
                        assert!(
                            same(&piece, &heavy),
                            "the classed and heavy kernels disagree at k={one_hot_k} {} {}",
                            spread.label(),
                            order.label(),
                        );
                        (
                            format!("{ms:9.3}"),
                            format!("{:9.1}", entries as f64 / ms / 1e3),
                        )
                    };
                    let warp = context.segment_sums(&bases, &plan, SegmentMode::Warp);
                    let (warp_ms, warp_rate) = match warp {
                        Ok(warp) => {
                            assert!(
                                same(&warp, &heavy),
                                "the warp and heavy kernels disagree at k={one_hot_k} {} {}",
                                spread.label(),
                                order.label(),
                            );
                            let (warp_ms, _) = time(|| {
                                context
                                    .segment_sums(&bases, &plan, SegmentMode::Warp)
                                    .expect("warp segment sums")
                            });
                            (
                                format!("{warp_ms:9.3}"),
                                format!("{:9.1}", entries as f64 / warp_ms / 1e3),
                            )
                        }
                        Err(_) => (format!("{:>9}", "-"), format!("{:>9}", "-")),
                    };

                    println!(
                        "{chunks:>6}  {one_hot_k:>5}  {:>7}  {:>5}  {:>8}  {entries:>10}  \
                         {widest:>7}  {small}  {heavy_ms:>9.3}  {warp_ms}  {class_ms}  {:>9.1}  \
                         {class_rate}",
                        spread.label(),
                        order.label(),
                        counts.len(),
                        entries as f64 / heavy_ms / 1e3,
                    );
                    let _ = warp_rate;
                }
            }
        }
    }
}
