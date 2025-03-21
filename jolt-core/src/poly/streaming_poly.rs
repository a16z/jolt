use crate::field::JoltField;

pub struct StreamingPolyinomial<I: Iterator, F> {
    func: fn(&I::Item) -> F,
}

impl<I, F> StreamingPolyinomial<I, F>
where
    I: Iterator,
    F: JoltField,
{
    pub fn new(func: fn(&I::Item) -> F) -> Self {
        Self { func }
    }
}

pub struct Oracle<I: Iterator, F> {
    trace_iter: I,
    polys: Vec<StreamingPolyinomial<I, F>>,
}

impl<I, F> Oracle<I, F>
where
    I: Iterator,
    F: JoltField,
{
    pub fn new(trace_iter: I, polys: Vec<StreamingPolyinomial<I, F>>) -> Self {
        Self { trace_iter, polys }
    }

    pub fn stream_next_evals(&mut self) -> Vec<F> {
        let trace = self.trace_iter.next().unwrap();
        let mut evals = Vec::new();
        for poly in &self.polys {
            evals.push((poly.func)(&trace));
        }
        evals
    }

    pub fn stream_next_shards(&mut self, shard_len: usize) -> Vec<Vec<F>> {
        (0..shard_len).map(|_| self.stream_next_evals()).collect()
    }
}

mod test {
    use ark_bn254::Fr;
    use rand::{thread_rng, Rng};
    use rayon::vec;
    use std::slice::Iter;

    use super::*;

    struct Trace {
        elem_1: u64,
        elem_2: u64,
    }

    #[test]
    fn test_oracle() {
        let mut rng = thread_rng();
        let trace_1 = (0..10).map(|_| rng.gen()).collect::<Vec<u64>>();
        let trace_2 = (0..10).map(|_| rng.gen()).collect::<Vec<u64>>();
        let mut trace = Vec::new();
        for i in 0..10 {
            trace.push(Trace {
                elem_1: trace_1[i],
                elem_2: trace_2[i],
            });
        }

        let poly_1 = StreamingPolyinomial::<Iter<Trace>, Fr>::new(|elem: &&Trace| {
            Fr::from_u64(elem.elem_1) * Fr::from_u64(elem.elem_2)
        });
        let poly_2 = StreamingPolyinomial::<Iter<Trace>, Fr>::new(|elem: &&Trace| {
            Fr::from_u64(elem.elem_1) + Fr::from_u64(elem.elem_2)
        });

        let mut oracle = Oracle::new(trace.iter(), vec![poly_1, poly_2]);

        for i in 0..10 {
            let temp = oracle.stream_next_evals();
            println!("The {}th evals are {}, {}", i, temp[0], temp[1]);
        }
    }
}

// use crate::field::JoltField;

// pub struct StreamingPoly<I: Iterator, F> {
//     trace_iter: I,
//     func: fn(I::Item) -> Vec<F>,
// }

// impl<I, F> StreamingPoly<I, F>
// where
//     I: Iterator,
//     F: JoltField,
// {
//     pub fn new(trace_iter: I, func: fn(I::Item) -> Vec<F>) -> Self {
//         Self { trace_iter, func }
//     }

//     pub fn stream_next(&mut self) -> Vec<F> {
//         (self.func)(self.trace_iter.next().unwrap())
//     }
// }

// mod test {
//     use ark_bn254::Fr;
//     use rand::{thread_rng, Rng};

//     use super::*;

//     // struct Trace {
//     //     elem_1: u64,
//     //     elem_2: u64,
//     // }

//     #[test]
//     fn test_oracle() {
//         let mut rng = thread_rng();
//         let trace = (0..10).map(|_| rng.gen()).collect::<Vec<u64>>();
//         // let trace_2 = (0..10).map(|_| rng.gen()).collect::<Vec<u64>>();
//         // let mut trace = Vec::new();

//         // for i in 0..10 {
//         //     trace.push(Trace{elem_1: trace_1[i], elem_2: trace_2[i]});
//         // }

//         let mut oracle_1 = StreamingPoly::new(trace.iter().take(5), |x| {
//             for idx in 0..x.len() {
//                 let result = Vec::new();
//                 result.push(Fr::from_u64(*x) * Fr::from_u64(*x))
//             }
//         });

//         let mut oracle_2 =
//             StreamingPoly::new(trace.iter(), |x| Fr::from_u64(*x) + Fr::from_u64(*x));

//         for i in 0..10 {
//             println!("The {}th element is {}", i, oracle_1.stream_next());
//         }

//         for i in 0..10 {
//             println!("The {}th element is {}", i, oracle_2.stream_next());
//         }
//     }
// }
