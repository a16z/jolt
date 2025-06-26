pub trait Oracle: Send + Sync {
    type Shard;
    fn next_shard(&mut self) -> Self::Shard;

    // TODO: We want to run Jolt on programs with upto 2^40 instructions. On a 32 bit machine, the
    // length won't fit in a usize.

    // TODO: We want to run Jolt on programs with upto 2^40 instructions. On a 32 bit machine, the
    // step won't fit in a usize.
    fn get_len(&self) -> usize {
        unimplemented!("Not required for all impl")
    }

    fn get_step(&self) -> usize {
        unimplemented!("Not required for all impl")
    }
}
