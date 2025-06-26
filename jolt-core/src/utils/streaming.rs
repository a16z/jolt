pub trait Oracle: Send + Sync {
    type Shard;
    fn next_shard(&mut self) -> Self::Shard;

    // TODO: We want to run Jolt on programs with upto 2^40 instructions. On a 32 bit machine, the
    // length won't fit in a usize.

    // TODO: Remove.
    fn reset(&mut self) {
        unimplemented!("Not required for all impl")
    }

    // TODO: We want to run Jolt on programs with upto 2^40 instructions. On a 32 bit machine, the
    // step won't fit in a usize.
    fn get_len(&self) -> usize {
        unimplemented!("Not required for all impl")
    }

    // TODO: Remove
    fn peek(&self) -> Option<Self::Shard> {
        unimplemented!("Not required for all impl")
    }

    fn get_step(&self) -> usize {
        unimplemented!("Not required for all impl")
    }
}

// pub trait Oracle {
//     type Shard;

//     type Evals;

//     fn next_shard(&mut self, shard_len: usize) -> Self::Shard;

//     fn reset(&mut self);

//     /// Returns one evaluation for all polynomials being streamed by the oracle without advancing the oracle.
//     /// This is required for Offset constraints in Spartan.
//     fn peek(&mut self) -> Option<Self::Evals> {
//         unimplemented!("Not required for all impl")
//     }

//     fn get_len(&self) -> usize {
//         unimplemented!("Not required for all impl")
//     }

//     fn get_step(&self) -> usize {
//         unimplemented!("Not required for all impl")
//     }
// }

// pub trait Oracle {
//     type Item;

//     fn next_eval(&mut self, shard_len: usize) -> Self::Item;

//     fn next_shard(&mut self, shard_len: usize) -> Vec<Self::Item>;

//     fn reset(&mut self);

//     fn peek(&mut self) -> Option<Self::Item> {
//         unimplemented!("Not required for all impl")
//     }

//     fn get_len(&self) -> usize {
//         unimplemented!("Not required for all impl")
//     }

//     fn get_step(&self) -> usize {
//         unimplemented!("Not required for all impl")
//     }
// }
