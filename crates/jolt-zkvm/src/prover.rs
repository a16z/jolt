//! Graph-driven prover loop.

#[derive(Debug)]
pub enum ProveError {}

impl std::fmt::Display for ProveError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::error::Error for ProveError {}
