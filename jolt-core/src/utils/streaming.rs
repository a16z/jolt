pub trait Oracle {
    type Item;

    fn next_shard(&mut self, shard_len: usize) -> Self::Item;

    fn reset(&mut self);

    fn peek(&mut self) -> Option<Self::Item>;

    fn get_length(&self) -> usize;

    fn get_step(&self) -> usize;
}
