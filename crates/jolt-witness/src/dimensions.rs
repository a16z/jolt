#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WitnessDimensions {
    pub rows: usize,
    pub log_rows: usize,
}

impl WitnessDimensions {
    pub const fn new(rows: usize, log_rows: usize) -> Self {
        Self { rows, log_rows }
    }
}
