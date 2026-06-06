use jolt_claims::protocols::jolt::formulas::spartan::SpartanOuterDimensions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1ProverConfig {
    pub log_t: usize,
    #[cfg(feature = "zk")]
    pub committed_rounds: bool,
}

impl Stage1ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self {
            log_t,
            #[cfg(feature = "zk")]
            committed_rounds: false,
        }
    }

    #[cfg(feature = "zk")]
    pub const fn with_committed_rounds(mut self, committed_rounds: bool) -> Self {
        self.committed_rounds = committed_rounds;
        self
    }

    pub fn dimensions(self) -> SpartanOuterDimensions {
        SpartanOuterDimensions::rv64(self.log_t)
    }
}

#[cfg(not(feature = "field-inline"))]
#[derive(Clone, Copy, Debug)]
pub struct Stage1ProverInput<'a, W> {
    pub config: Stage1ProverConfig,
    pub witness: &'a W,
}

#[cfg(not(feature = "field-inline"))]
impl<'a, W> Stage1ProverInput<'a, W> {
    pub const fn new(config: Stage1ProverConfig, witness: &'a W) -> Self {
        Self { config, witness }
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
pub struct Stage1ProverInput<'a, W, FI> {
    pub config: Stage1ProverConfig,
    pub witness: &'a W,
    pub field_inline_witness: &'a FI,
}

#[cfg(feature = "field-inline")]
impl<'a, W, FI> Stage1ProverInput<'a, W, FI> {
    pub const fn new(
        config: Stage1ProverConfig,
        witness: &'a W,
        field_inline_witness: &'a FI,
    ) -> Self {
        Self {
            config,
            witness,
            field_inline_witness,
        }
    }
}
