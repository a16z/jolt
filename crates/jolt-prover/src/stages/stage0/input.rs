use jolt_claims::protocols::jolt::formulas::ra::JoltRaPolynomialLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentStageConfig {
    pub ra_layout: JoltRaPolynomialLayout,
    pub include_trusted_advice: bool,
    pub include_untrusted_advice: bool,
}

impl CommitmentStageConfig {
    pub const fn new(
        ra_layout: JoltRaPolynomialLayout,
        include_trusted_advice: bool,
        include_untrusted_advice: bool,
    ) -> Self {
        Self {
            ra_layout,
            include_trusted_advice,
            include_untrusted_advice,
        }
    }
}
