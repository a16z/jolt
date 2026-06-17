use crate::NamespaceId;

pub const RV64_NAMESPACE: NamespaceId = NamespaceId::new("jolt_vm.rv64");

#[derive(Clone, Debug, Default)]
pub struct Rv64Witness;
