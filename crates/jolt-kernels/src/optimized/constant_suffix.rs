use std::ops::Index;

/// A logical table retaining one representative of its constant suffix.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct ConstantSuffix<T> {
    representatives: Vec<T>,
    len: usize,
}

impl<T: PartialEq> ConstantSuffix<T> {
    pub(crate) fn new(mut representatives: Vec<T>, len: usize) -> Self {
        assert!(representatives.len() <= len);
        assert_eq!(representatives.is_empty(), len == 0);
        if let Some(last) = representatives.last() {
            let prefix = representatives
                .iter()
                .rposition(|row| row != last)
                .map_or(0, |i| i + 1);
            representatives.truncate(prefix + 1);
            representatives.shrink_to_fit();
        }
        Self {
            representatives,
            len,
        }
    }
}

impl<T> ConstantSuffix<T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn prefix_len(&self) -> usize {
        self.representatives.len().saturating_sub(1)
    }

    pub(crate) fn representatives(&self) -> &[T] {
        &self.representatives
    }
}

impl<T: PartialEq> From<Vec<T>> for ConstantSuffix<T> {
    fn from(rows: Vec<T>) -> Self {
        let len = rows.len();
        Self::new(rows, len)
    }
}

impl<T> Index<usize> for ConstantSuffix<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        assert!(index < self.len);
        &self.representatives[index.min(self.prefix_len())]
    }
}
