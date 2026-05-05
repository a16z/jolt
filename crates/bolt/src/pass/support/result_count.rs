#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::pass) enum LoweredResultCount {
    None,
    One,
    Two,
    Three,
    Four,
}

impl LoweredResultCount {
    pub(in crate::pass) const fn as_usize(self) -> usize {
        match self {
            Self::None => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}
