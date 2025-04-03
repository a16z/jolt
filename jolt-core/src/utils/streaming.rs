pub trait Oracle {
        type Item;

        fn next_eval(&mut self) -> Self::Item;

        fn reset_oracle(&mut self);
    }
    

// // An iterator that maps over the values of `iter` with `f`, which modifies its accumulated state.
// pub struct MapState<I, F> {
//     iter: std::iter::Cycle<I>,
//     f: F,
// }

// pub fn map_state<B, I, F>(iter: I, f: F) -> MapState<I, F>
// where
//     I: Iterator + Clone,
//     F: FnMut(I::Item) -> B,
// {
//     MapState::new(iter, f)
// }

// impl<I, F> MapState<I, F> {
//     fn new<B>(iter: I, f: F) -> Self
//     where
//         I: Iterator + Clone,
//         F: FnMut(I::Item) -> B,
//     {
//         MapState {
//             iter: iter.cycle(),
//             f,
//         }
//     }
// }

// impl<B, I, F> Iterator for MapState<I, F>
// where
//     I: Iterator + Clone,
//     F: FnMut(I::Item) -> B,
// {
//     type Item = B;

//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next().map(|x| (self.f)(x))
//     }

//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

// // An iterator that maps over the values of `iter` with `f`, which modifies its accumulated state.
// pub struct TestMapState<O, F>
// where
//     O: Oracle,
// {
//     iter: O,
//     f: F,
//     index: usize,
// }

// pub fn test_map_state<B, O, F>(iter: O, f: F) -> TestMapState<O, F>
// where
//     O: Oracle,
//     F: FnMut(O::Item) -> B,
// {
//     TestMapState::new(iter, f)
// }

// impl<O, F> TestMapState<O, F>
// where
//     O: Oracle,
// {
//     fn new<B>(iter: O, f: F) -> Self
//     where
//         O: Oracle,
//         F: FnMut(O::Item) -> B,
//     {
//         TestMapState { iter, f, index: 0 }
//     }
//     // fn update(&mut self) {
//     //     self.iter = self.intitial_iter.clone()
//     // }
// }

// //impl Oracle.
// //impl next for Trace

// pub trait Oracle {
//     type Item;
//     fn next(&mut self) -> Option<Self::Item>;
// }
// struct Trace2<'a> {
//     pub length: usize,
//     pub counter: usize,
//     pub trace: &'a Vec<usize>,
// }
// impl<O, F, B> Oracle for TestMapState<O, F>
// where
//     O: Oracle,
//     F: FnMut(O::Item) -> B,
// {
//     type Item = B;
//     fn next(&mut self) -> Option<Self::Item> {
//         Some((self.f)(self.iter.next().unwrap()))
//     }
// }




// #[cfg(test)]
// mod tests {
//     use super::*;

//     struct DummyTrace<'a> {
//         pub length: usize,
//         pub counter: usize,
//         pub trace: &'a Vec<usize>,
//     }
//     impl<'a> Oracle for DummyTrace<'a> {
//         type Item = usize;
//         fn next(&mut self) -> Self::Item {
//             let step = self.trace[self.counter].clone();
//             self.counter = (self.counter + 1) % self.length;
//             step
//         }
//     }

//     #[test]
//     fn test_testmapstate_oracle() {
//         let data = vec![1, 2, 3, 4];
//         let trace = DummyTrace {
//             length: data.len(),
//             counter: 0,
//             trace: &data,
//         };
//         let mapped = test_map_state(trace, |x| x * 2);
//         let mut mapped = test_map_state(mapped, |x| x * 2);

//         assert_eq!(mapped.next(), 4);
//         assert_eq!(mapped.next(), 8);
//         assert_eq!(mapped.next(), 12);
//         assert_eq!(mapped.next(), 16);
//         assert_eq!(mapped.next(), 4);
//         assert_eq!(mapped.next(), 8);
//     }
// }
