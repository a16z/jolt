pub trait Oracle {
    type Item;

    fn next_shard(&mut self, shard_len: usize) -> Self::Item;

    fn reset(&mut self);

    fn peek(&mut self) -> Option<Self::Item> {
        unimplemented!("Not required for all impl")
    }

    fn get_len(&self) -> usize {
        unimplemented!("Not required for all impl")
    }

    fn get_step(&self) -> usize {
        unimplemented!("Not required for all impl")
    }
}
