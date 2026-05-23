use std::ops::Deref;

use serde::{Deserialize, Serialize};

pub type Endianness = bool;
pub const HIGH_TO_LOW: Endianness = false;
pub const LOW_TO_HIGH: Endianness = true;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point<const E: Endianness, F> {
    coords: Vec<F>,
}

impl<const E: Endianness, F> Point<E, F> {
    pub fn new(coords: impl Into<Vec<F>>) -> Self {
        Self {
            coords: coords.into(),
        }
    }

    pub fn match_endianness<const TARGET: Endianness>(&self) -> Point<TARGET, F>
    where
        F: Clone,
    {
        let mut coords = self.coords.clone();
        if E != TARGET {
            coords.reverse();
        }
        Point::<TARGET, F>::new(coords)
    }

    pub fn concat(points: impl IntoIterator<Item = Self>) -> Self {
        let mut coords = Vec::new();
        for point in points {
            coords.extend(point.coords);
        }
        Self { coords }
    }

    pub fn as_slice(&self) -> &[F] {
        &self.coords
    }

    pub fn into_vec(self) -> Vec<F> {
        self.coords
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

impl<F> Point<HIGH_TO_LOW, F> {
    pub fn high_to_low(coords: impl Into<Vec<F>>) -> Self {
        Self::new(coords)
    }
}

impl<F> Point<LOW_TO_HIGH, F> {
    pub fn low_to_high(coords: impl Into<Vec<F>>) -> Self {
        Self::new(coords)
    }
}

impl<const E: Endianness, F> From<Vec<F>> for Point<E, F> {
    fn from(coords: Vec<F>) -> Self {
        Self::new(coords)
    }
}

impl<const E: Endianness, F> From<Point<E, F>> for Vec<F> {
    fn from(point: Point<E, F>) -> Self {
        point.coords
    }
}

impl<const E: Endianness, F> AsRef<[F]> for Point<E, F> {
    fn as_ref(&self) -> &[F] {
        self.as_slice()
    }
}

impl<const E: Endianness, F> Deref for Point<E, F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<const E: Endianness, F: PartialEq> PartialEq<Vec<F>> for Point<E, F> {
    fn eq(&self, other: &Vec<F>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::{Point, HIGH_TO_LOW, LOW_TO_HIGH};

    #[test]
    fn match_endianness_reverses_when_order_changes() {
        let point = Point::<LOW_TO_HIGH, _>::low_to_high(vec![1, 2, 3]);
        let canonical = point.match_endianness::<HIGH_TO_LOW>();

        assert_eq!(canonical.into_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn match_endianness_preserves_matching_order() {
        let point = Point::<HIGH_TO_LOW, _>::high_to_low(vec![1, 2, 3]);
        let canonical = point.match_endianness::<HIGH_TO_LOW>();

        assert_eq!(canonical.into_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn concat_preserves_point_order() {
        let point = Point::<HIGH_TO_LOW, _>::concat([
            Point::<HIGH_TO_LOW, _>::high_to_low(vec![1, 2]),
            Point::<LOW_TO_HIGH, _>::low_to_high(vec![3, 4]).match_endianness::<HIGH_TO_LOW>(),
        ]);

        assert_eq!(point.into_vec(), vec![1, 2, 4, 3]);
    }
}
