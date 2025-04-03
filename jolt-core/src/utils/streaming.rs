pub trait Oracle {
    type Item;

    fn next_eval(&mut self) -> Self::Item;

    fn reset_oracle(&mut self);
}
