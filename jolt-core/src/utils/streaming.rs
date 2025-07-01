pub trait Oracle: Send + Sync {
    type Shard;
    fn next_shard(&mut self) -> Self::Shard;
    
    fn get_len(&self) -> usize {
        unimplemented!("Not required for all impl")
    }

    fn get_step(&self) -> usize {
        unimplemented!("Not required for all impl")
    }
}
