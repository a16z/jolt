#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U64AndSign {
    pub magnitude: u64,
    pub is_positive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U128AndSign {
    pub magnitude: u128,
    pub is_positive: bool,
}

impl From<i128> for U64AndSign {
    fn from(value: i128) -> Self {
        Self {
            magnitude: value.abs() as u64,
            is_positive: value >= 0,
        }
    }
}

impl From<i128> for U128AndSign {
    fn from(value: i128) -> Self {
        Self {
            magnitude: value.abs() as u128,
            is_positive: value >= 0,
        }
    }
}
