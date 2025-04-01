pub trait StreamingOracle<I: Iterator> {
    fn stream_next_shard(&mut self, shard_len: usize);
}
