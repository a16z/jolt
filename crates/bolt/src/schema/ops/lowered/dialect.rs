pub(in crate::schema::ops) trait LoweredDialect {
    const PREFIX: &'static str;
    const PRIMARY_SUMCHECK_REFERENCE_ATTR: &'static str;
    const CAPABILITIES: LoweredDialectCapabilities;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::schema::ops) enum LoweredDialectCapabilities {
    Compute,
    Cpu,
}

impl LoweredDialectCapabilities {
    pub(in crate::schema::ops) const fn has_relation_op(self) -> bool {
        matches!(self, Self::Compute)
    }

    pub(in crate::schema::ops) const fn has_kernel_sumcheck_ops(self) -> bool {
        matches!(self, Self::Compute)
    }
}

pub(in crate::schema::ops) enum ComputeDialect {}

impl LoweredDialect for ComputeDialect {
    const PREFIX: &'static str = "compute";
    const PRIMARY_SUMCHECK_REFERENCE_ATTR: &'static str = "relation";
    const CAPABILITIES: LoweredDialectCapabilities = LoweredDialectCapabilities::Compute;
}

pub(in crate::schema::ops) enum CpuDialect {}

impl LoweredDialect for CpuDialect {
    const PREFIX: &'static str = "cpu";
    const PRIMARY_SUMCHECK_REFERENCE_ATTR: &'static str = "kernel";
    const CAPABILITIES: LoweredDialectCapabilities = LoweredDialectCapabilities::Cpu;
}

pub(in crate::schema::ops) fn is_verifier_forbidden(name: &str) -> bool {
    matches!(
        name,
        "compute.kernel"
            | "compute.sumcheck_claim"
            | "compute.sumcheck_driver"
            | "compute.sumcheck_kernel_claim"
            | "compute.sumcheck_kernel_driver"
            | "compute.generate_oracle"
            | "compute.generate_oracle_family"
            | "cpu.kernel"
            | "cpu.sumcheck_claim"
            | "cpu.sumcheck_driver"
    )
}
