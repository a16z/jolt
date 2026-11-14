use std::ops::{Deref, Range};

use serde::{Deserialize, Serialize};

pub type Endianness = bool;
pub const HIGH_TO_LOW: Endianness = false;
pub const LOW_TO_HIGH: Endianness = true;

/// Coordinates in a declared significance order. Bare slices, arrays and vectors
/// use high-to-low order; references preserve the referenced point's order.
pub trait PointLike {
    type Coord: Copy;
    const ENDIANNESS: Endianness;
    /// A borrowed range with the same endianness as this point.
    type View<'a>: PointLike<Coord = Self::Coord> + Copy
    where
        Self: 'a;

    fn coords(&self) -> &[Self::Coord];

    /// Borrows significance ranks `range`, where rank zero is the MSB.
    /// Panics if the range is unordered or outside the point.
    fn msb_range(&self, range: Range<usize>) -> Self::View<'_>;

    #[inline(always)]
    fn num_vars(&self) -> usize {
        self.coords().len()
    }

    /// Returns the coordinate of significance rank `k`, where zero is the MSB.
    #[inline(always)]
    fn msb(&self, k: usize) -> Self::Coord {
        let coords = self.coords();
        coords[msb_index(Self::ENDIANNESS, coords.len(), k)]
    }
}

#[inline(always)]
pub(crate) fn msb_index(endianness: Endianness, num_vars: usize, k: usize) -> usize {
    if endianness == HIGH_TO_LOW {
        k
    } else {
        num_vars - 1 - k
    }
}

/// A zero-copy point whose coordinates are stored in order `E`.
#[derive(Clone, Copy, Debug)]
pub struct PointRef<'a, const E: Endianness, C>(&'a [C]);

impl<'a, const E: Endianness, C> PointRef<'a, E, C> {
    pub fn new(coords: &'a [C]) -> Self {
        Self(coords)
    }

    fn range(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.0.len());
        let range = if E == HIGH_TO_LOW {
            range
        } else {
            self.0.len() - range.end..self.0.len() - range.start
        };
        Self::new(&self.0[range])
    }
}

impl<'a, C> PointRef<'a, HIGH_TO_LOW, C> {
    pub fn high_to_low(coords: &'a [C]) -> Self {
        Self::new(coords)
    }
}

impl<'a, C> PointRef<'a, LOW_TO_HIGH, C> {
    pub fn low_to_high(coords: &'a [C]) -> Self {
        Self::new(coords)
    }
}

impl<const E: Endianness, C: Copy> PointLike for PointRef<'_, E, C> {
    type Coord = C;
    const ENDIANNESS: Endianness = E;
    type View<'a>
        = PointRef<'a, E, C>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[C] {
        self.0
    }

    #[inline]
    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        self.range(range)
    }
}

impl<const E: Endianness, C: Copy> PointLike for Point<E, C> {
    type Coord = C;
    const ENDIANNESS: Endianness = E;
    type View<'a>
        = PointRef<'a, E, C>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[C] {
        self.as_slice()
    }

    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        self.as_point_ref().range(range)
    }
}

impl<C: Copy> PointLike for [C] {
    type Coord = C;
    const ENDIANNESS: Endianness = HIGH_TO_LOW;
    type View<'a>
        = PointRef<'a, HIGH_TO_LOW, C>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[C] {
        self
    }

    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        PointRef::high_to_low(&self[range])
    }
}

impl<C: Copy> PointLike for Vec<C> {
    type Coord = C;
    const ENDIANNESS: Endianness = HIGH_TO_LOW;
    type View<'a>
        = PointRef<'a, HIGH_TO_LOW, C>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[C] {
        self
    }

    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        self.as_slice().msb_range(range)
    }
}

impl<C: Copy, const N: usize> PointLike for [C; N] {
    type Coord = C;
    const ENDIANNESS: Endianness = HIGH_TO_LOW;
    type View<'a>
        = PointRef<'a, HIGH_TO_LOW, C>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[C] {
        self
    }

    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        self.as_slice().msb_range(range)
    }
}

impl<T: PointLike + ?Sized> PointLike for &T {
    type Coord = T::Coord;
    const ENDIANNESS: Endianness = T::ENDIANNESS;
    type View<'a>
        = T::View<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn coords(&self) -> &[Self::Coord] {
        T::coords(self)
    }

    #[inline(always)]
    fn msb(&self, k: usize) -> Self::Coord {
        T::msb(self, k)
    }

    fn msb_range(&self, range: Range<usize>) -> Self::View<'_> {
        T::msb_range(self, range)
    }
}

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

    pub fn as_point_ref(&self) -> PointRef<'_, E, F> {
        PointRef::new(&self.coords)
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
    use super::{Point, PointLike, PointRef, HIGH_TO_LOW, LOW_TO_HIGH};

    #[test]
    fn range_views_preserve_order_and_borrow_storage() {
        fn check<P: PointLike<Coord = usize> + ?Sized>(point: &P) {
            for start in 0..=point.num_vars() {
                for end in start..=point.num_vars() {
                    let view = point.msb_range(start..end);
                    assert_eq!(view.num_vars(), end - start);
                    assert_eq!(P::ENDIANNESS, <P::View<'_> as PointLike>::ENDIANNESS);
                    for k in 0..end - start {
                        assert_eq!(view.msb(k), point.msb(start + k));
                    }
                    let offset = if P::ENDIANNESS == HIGH_TO_LOW {
                        start
                    } else {
                        point.num_vars() - end
                    };
                    assert_eq!(view.coords().as_ptr(), point.coords()[offset..].as_ptr());
                }
            }
        }
        for n in 0..=20 {
            let coords: Vec<_> = (0..n).collect();
            check(&coords);
            check(coords.as_slice());
            check(&Point::<HIGH_TO_LOW, _>::new(coords.clone()));
            let point = Point::<LOW_TO_HIGH, _>::new(coords.clone());
            check(&point);
            let view = PointRef::low_to_high(&coords);
            check(&view);
            check(&&view);
        }
        check(&[1, 2, 3]);
    }

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
