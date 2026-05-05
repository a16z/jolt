use super::super::super::support::{rust_crate_ident, snake_case};
use super::ProtocolArtifactConfig;

impl ProtocolArtifactConfig {
    pub(in crate::emit::rust::artifacts) fn protocol_snake(&self) -> String {
        snake_case(&self.protocol_name)
    }

    pub(in crate::emit::rust::artifacts) fn verifier_crate_import(&self) -> String {
        rust_crate_ident(&self.verifier_crate_name)
    }
}
