#![expect(
    clippy::unwrap_used,
    reason = "test assertions report construction failures"
)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jolt_field::{Fr, JoltField, F128};
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_kernels::optimized::SplitLt;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

#[derive(Clone)]
struct Indices {
    cycles: usize,
    digits: Vec<Vec<Option<usize>>>,
    bounds: Vec<Option<usize>>,
    calls: Arc<AtomicUsize>,
}

impl Indices {
    fn new(cycles: usize, digits: Vec<Vec<Option<usize>>>, bounds: Vec<Option<usize>>) -> Self {
        Self {
            cycles,
            digits,
            bounds,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ChunkIndexSource for Indices {
    fn num_polys(&self) -> usize {
        self.digits.len()
    }
    fn cycles(&self) -> usize {
        self.cycles
    }
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        assert!(i < self.num_polys());
        assert!(j < self.cycles);
        let _ = self.calls.fetch_add(1, Ordering::Relaxed);
        self.digits[i][j]
    }
    fn index_bound(&self, i: usize) -> Option<usize> {
        self.bounds[i]
    }
}

fn bound_weight<F: JoltField>(challenges: &[F], vertex: usize) -> F {
    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bit, &rho)| {
            weight
                * if vertex >> bit & 1 == 1 {
                    rho
                } else {
                    F::one() - rho
                }
        })
}

fn partial_evaluation<F: JoltField>(table: &[F], challenges: &[F], j: usize) -> F {
    let width = 1 << challenges.len();
    (0..width).fold(F::zero(), |sum, u| {
        sum + bound_weight(challenges, u) * table[j * width + u]
    })
}

fn lazy_columns<F: JoltField>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6b_128);
    for cycles in [1usize, 2, 8, 16, 64] {
        let lengths = [2, 5, 16, 0];
        let tables: Vec<Vec<F>> = lengths
            .iter()
            .map(|&len| (0..len).map(|_| F::random(&mut rng)).collect())
            .collect();
        let digits: Vec<Vec<Option<usize>>> = lengths
            .iter()
            .map(|&len| {
                (0..cycles)
                    .map(|j| {
                        if len == 0 || j % 3 == 0 {
                            None
                        } else {
                            Some(rng.next_u64() as usize % len)
                        }
                    })
                    .collect()
            })
            .collect();
        let original: Vec<Vec<F>> = tables
            .iter()
            .zip(&digits)
            .map(|(table, column)| {
                column
                    .iter()
                    .map(|digit| digit.map_or_else(F::zero, |index| table[index]))
                    .collect()
            })
            .collect();
        let rounds = cycles.ilog2() as usize;
        let challenges: Vec<F> = (0..rounds).map(|_| F::random(&mut rng)).collect();
        let mut lazy =
            LazyFoldedRa::try_new(tables, Indices::new(cycles, digits, vec![None; 4])).unwrap();
        assert_eq!(lazy.num_polys(), 4);
        for bound in 0..=rounds {
            let point = &challenges[..bound];
            let current_len = cycles >> bound;
            let expected: Vec<Vec<F>> = original
                .iter()
                .map(|column| {
                    (0..current_len)
                        .map(|j| partial_evaluation(column, point, j))
                        .collect()
                })
                .collect();
            for (i, column) in expected.iter().enumerate() {
                for (j, &value) in column.iter().enumerate() {
                    assert_eq!(
                        lazy.value(i, j),
                        value,
                        "cycles={cycles}, bound={bound}, i={i}, j={j}"
                    );
                }
                for row in 0..current_len / 2 {
                    assert_eq!(lazy.lo_hi(i, row), (column[2 * row], column[2 * row + 1]));
                }
            }
            for row in 0..current_len / 2 {
                let sentinel = (F::one(), F::one());
                for len in [3, 4, 5] {
                    let mut out = vec![sentinel; len];
                    lazy.lo_hi_all(row, &mut out);
                    for (pair, column) in out.iter().zip(&expected) {
                        assert_eq!(*pair, (column[2 * row], column[2 * row + 1]));
                    }
                    if len == 5 {
                        assert_eq!(out[4], sentinel);
                    }
                }
            }
            assert_eq!(
                lazy.final_values(),
                expected.iter().map(|column| column[0]).collect::<Vec<_>>()
            );
            if bound < rounds {
                lazy.bind(challenges[bound]);
            }
        }
        assert!(catch_unwind(AssertUnwindSafe(|| lazy.bind(F::one()))).is_err());
    }
}

fn lazy_construction<F: JoltField>() {
    let table = || vec![vec![F::one(); 2]];
    let source = |cycles, digits, bounds| Indices::new(cycles, digits, bounds);
    assert_eq!(
        LazyFoldedRa::try_new(
            Vec::<Vec<F>>::new(),
            source(16, vec![vec![None; 16]], vec![None])
        )
        .err(),
        Some(LazyRaError::TableCount {
            tables: 0,
            polys: 1
        })
    );
    assert_eq!(
        LazyFoldedRa::try_new(Vec::<Vec<F>>::new(), source(16, vec![], vec![])).err(),
        Some(LazyRaError::NoColumns)
    );
    for cycles in [12, 0] {
        assert_eq!(
            LazyFoldedRa::try_new(
                table(),
                source(cycles, vec![vec![None; cycles]], vec![None])
            )
            .err(),
            Some(LazyRaError::CyclesNotPowerOfTwo { cycles })
        );
    }
    assert_eq!(
        LazyFoldedRa::try_new(table(), source(16, vec![vec![Some(2); 16]], vec![Some(3)])).err(),
        Some(LazyRaError::IndexBoundExceedsTable {
            poly: 0,
            bound: 3,
            len: 2
        })
    );
    let mut invalid = vec![None; 16];
    invalid[11] = Some(3);
    invalid[3] = Some(2);
    assert_eq!(
        LazyFoldedRa::try_new(table(), source(16, vec![invalid], vec![None])).err(),
        Some(LazyRaError::IndexOutOfRange {
            poly: 0,
            cycle: 3,
            index: 2,
            len: 2
        })
    );
    assert_eq!(
        LazyFoldedRa::try_new(Vec::<Vec<F>>::new(), source(0, vec![vec![]], vec![Some(3)])).err(),
        Some(LazyRaError::TableCount {
            tables: 0,
            polys: 1
        })
    );
    assert_eq!(
        LazyFoldedRa::try_new(Vec::<Vec<F>>::new(), source(0, vec![], vec![])).err(),
        Some(LazyRaError::NoColumns)
    );
    assert_eq!(
        LazyFoldedRa::try_new(table(), source(12, vec![vec![Some(2); 12]], vec![Some(3)])).err(),
        Some(LazyRaError::CyclesNotPowerOfTwo { cycles: 12 })
    );
    assert_eq!(
        LazyFoldedRa::try_new(
            vec![vec![F::one(); 2]; 2],
            source(
                16,
                vec![vec![Some(2); 16], vec![Some(3); 16]],
                vec![None, Some(4)]
            )
        )
        .err(),
        Some(LazyRaError::IndexOutOfRange {
            poly: 0,
            cycle: 0,
            index: 2,
            len: 2
        })
    );
    let bounded = source(
        16,
        vec![vec![Some(1); 16], vec![None; 16]],
        vec![Some(2), Some(0)],
    );
    let calls = Arc::clone(&bounded.calls);
    let lazy = LazyFoldedRa::try_new(vec![vec![F::one(); 2], vec![]], bounded).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(lazy.value(0, 0), F::one());
    assert_eq!(lazy.value(1, 0), F::zero());
}

#[test]
fn lazy_columns_prime() {
    lazy_columns::<Fr>();
}
#[test]
fn lazy_columns_binary() {
    lazy_columns::<F128>();
}
#[test]
fn lazy_construction_prime() {
    lazy_construction::<Fr>();
}
#[test]
fn lazy_construction_binary() {
    lazy_construction::<F128>();
}

fn split_less_than<F: JoltField>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x17_128);
    for n in [0, 1, 2, 3, 6, 7] {
        let point: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let challenges: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        for with_constant in [false, true] {
            let constant = if with_constant {
                F::random(&mut rng)
            } else {
                F::zero()
            };
            let mut lt = if with_constant {
                SplitLt::new_plus_constant(&point, constant)
            } else {
                SplitLt::new(&point)
            };
            let original: Vec<F> = (0..1 << n)
                .map(|j| {
                    (j + 1..1 << n).fold(constant, |sum, k| {
                        let equality =
                            point
                                .iter()
                                .enumerate()
                                .fold(F::one(), |weight, (bit, &r)| {
                                    weight
                                        * if k >> (n - bit - 1) & 1 == 1 {
                                            r
                                        } else {
                                            F::one() - r
                                        }
                                });
                        sum + equality
                    })
                })
                .collect();
            for bound in 0..=n {
                let current_len = original.len() >> bound;
                let expected: Vec<F> = (0..current_len)
                    .map(|j| partial_evaluation(&original, &challenges[..bound], j))
                    .collect();
                for y in 0..current_len / 2 {
                    assert_eq!(
                        lt.pair(y),
                        (expected[2 * y], expected[2 * y + 1]),
                        "n={n}, bound={bound}, y={y}, with_constant={with_constant}"
                    );
                }
                assert_eq!(
                    lt.bound_value(),
                    if bound == n { Some(expected[0]) } else { None }
                );
                assert!(catch_unwind(AssertUnwindSafe(|| lt.pair(current_len / 2))).is_err());
                if bound < n {
                    lt.bind(challenges[bound]);
                }
            }
            lt.bind(F::one());
            assert_eq!(lt.bound_value(), None);
            assert!(catch_unwind(AssertUnwindSafe(|| lt.pair(0))).is_err());
        }
    }
}

#[test]
fn split_less_than_prime() {
    split_less_than::<Fr>();
}

#[test]
fn split_less_than_binary() {
    split_less_than::<F128>();
}
