// An iterator that maps over the values of `iter` with `f`, which modifies its accumulated state.
pub struct MapState<I, F> {
    iter: std::iter::Cycle<I>,
    f: F,
}

pub fn map_state<B, I, F>(iter: I, f: F) -> MapState<I, F>
where
    I: Iterator + Clone,
    F: FnMut(I::Item) -> B,
{
    MapState::new(iter, f)
}

impl<I, F> MapState<I, F> {
    fn new<B>(iter: I, f: F) -> Self
    where
        I: Iterator + Clone,
        F: FnMut(I::Item) -> B,
    {
        MapState {
            iter: iter.cycle(),
            f,
        }
    }
}

impl<B, I, F> Iterator for MapState<I, F>
where
    I: Iterator + Clone,
    F: FnMut(I::Item) -> B,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| (self.f)(x))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
