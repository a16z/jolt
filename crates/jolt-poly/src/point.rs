use std::ops::Deref;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point<F> {
    coords: Vec<F>,
}

impl<F> Point<F> {
    pub fn high_to_low(coords: impl Into<Vec<F>>) -> Self {
        Self {
            coords: coords.into(),
        }
    }

    pub fn low_to_high(coords: impl Into<Vec<F>>) -> Self {
        let mut coords = coords.into();
        coords.reverse();
        Self { coords }
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

impl<F> From<Vec<F>> for Point<F> {
    fn from(coords: Vec<F>) -> Self {
        Self::high_to_low(coords)
    }
}

impl<F> From<Point<F>> for Vec<F> {
    fn from(point: Point<F>) -> Self {
        point.coords
    }
}

impl<F> AsRef<[F]> for Point<F> {
    fn as_ref(&self) -> &[F] {
        self.as_slice()
    }
}

impl<F> Deref for Point<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<F: PartialEq> PartialEq<Vec<F>> for Point<F> {
    fn eq(&self, other: &Vec<F>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::Point;

    #[test]
    fn low_to_high_reverses_to_canonical_order() {
        assert_eq!(Point::low_to_high(vec![1, 2, 3]).into_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn concat_preserves_canonical_part_order() {
        let point = Point::concat([
            Point::high_to_low(vec![1, 2]),
            Point::low_to_high(vec![3, 4]),
        ]);

        assert_eq!(point.into_vec(), vec![1, 2, 4, 3]);
    }
}
