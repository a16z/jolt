use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    fn prefix_mle(
        &self,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F;
    fn update_prefix_checkpoint(&mut self, checkpoints: &[Option<F>], r_x: F, r_y: F, j: usize);
}

pub enum Prefixes {
    And,
    Xor,
    Or,
    TruncateUpper,
    TruncateLower,
    LessThan,
    Eq,
}
