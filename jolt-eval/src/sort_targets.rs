//! Sorting functions used as targets for e2e optimization and red-team tests.

use crate::objective::Objective;

/// Naive bubble sort — the optimization target.
/// Intentionally O(n²) so a "smarter" sort is measurably faster.
pub fn naive_sort(data: &mut [i32]) {
    let n = data.len();
    for i in 0..n {
        for j in 0..n.saturating_sub(1 + i) {
            if data[j] > data[j + 1] {
                data.swap(j, j + 1);
            }
        }
    }
}

/// A sorting routine used as a red-team target.
pub fn candidate_sort(data: &mut [i32]) {
    if data.len() <= 16 {
        // Small-array path: insertion sort.
        for i in 1..data.len() {
            let key = data[i];
            let mut j = i;
            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
    } else {
        // Large-array path: delegate to an optimized routine.
        let last = data.len() - 1;
        data[..last].sort();
    }
}

const SORT_DATA_SIZE: usize = 5000;

#[derive(Clone, Copy, Default)]
pub struct NaiveSortObjective;

impl Objective for NaiveSortObjective {
    type Setup = Vec<i32>;

    fn name(&self) -> &str {
        "naive_sort_time"
    }

    fn description(&self) -> &str {
        "Wall-clock time of the naive_sort function in jolt-eval/src/sort_targets.rs"
    }

    fn setup(&self) -> Vec<i32> {
        (0..SORT_DATA_SIZE as i32).rev().collect()
    }

    fn run(&self, mut setup: Vec<i32>) {
        naive_sort(&mut setup);
        std::hint::black_box(&setup);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
