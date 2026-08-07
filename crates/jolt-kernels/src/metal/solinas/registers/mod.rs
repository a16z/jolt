//! Backend-neutral register event ownership.
//!
//! [`RegisterCsr256`] checks the shader-facing CSR contract.
//! [`CertifiedRegisterOwner::build`] additionally checks raw read values and
//! write pre-values against one carried register state. The test-only
//! dense/BTree oracle derives the same planes by scanning every register in
//! every block, so it does not share the CSR construction algorithm.

mod owner;

pub use owner::{
    CertifiedRegisterOwner, RdIncrement, RdIncrementActivity, RegisterCsr256, RegisterCsr256Parts,
    RegisterCsrCensus, RegisterEventCounts, RegisterOwnerError, RegisterOwnerRead,
    RegisterOwnerRow, RegisterOwnerWrite, RegisterStateFlowCertificate, REGISTER_CSR_BLOCK_CYCLES,
    REGISTER_CSR_COLUMNS, REGISTER_CSR_NON_AUTHORITATIVE_LOG_T_26_CENSUS,
};

#[cfg(test)]
mod oracle;
#[cfg(test)]
#[expect(clippy::panic)]
mod tests;
