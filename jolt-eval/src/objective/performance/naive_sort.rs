use crate::objective::Objective;
use crate::sort_targets::naive_sort;

const SORT_DATA_SIZE: usize = 5000;

#[derive(Clone, Copy, Default)]
pub struct NaiveSortObjective;

impl Objective for NaiveSortObjective {
    type Setup = Vec<i32>;

    fn name(&self) -> &str {
        "naive_sort_time"
    }

    fn description(&self) -> String {
        "Wall-clock time of the naive_sort function in jolt-eval/src/sort_targets.rs".to_string()
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
