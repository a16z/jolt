#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::protocols::jolt::emit::rust) struct StageVerifierErrorShape {
    has_missing_ram: bool,
}

impl StageVerifierErrorShape {
    pub(in crate::protocols::jolt::emit::rust) const STANDARD: Self = Self {
        has_missing_ram: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const RAM: Self = Self {
        has_missing_ram: true,
    };

    pub(in crate::protocols::jolt::emit::rust) fn has_missing_ram(self) -> bool {
        self.has_missing_ram
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::protocols::jolt::emit::rust) struct Stage23VerifierTypeShape {
    has_opening_equalities: bool,
}

impl Stage23VerifierTypeShape {
    pub(in crate::protocols::jolt::emit::rust) const STAGE2: Self = Self {
        has_opening_equalities: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const STAGE3: Self = Self {
        has_opening_equalities: true,
    };

    pub(in crate::protocols::jolt::emit::rust) fn has_opening_equalities(self) -> bool {
        self.has_opening_equalities
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::protocols::jolt::emit::rust) struct StageRuntimeVerifierTypeShape {
    has_point_zeros: bool,
}

impl StageRuntimeVerifierTypeShape {
    pub(in crate::protocols::jolt::emit::rust) const STAGE4_OR_5: Self = Self {
        has_point_zeros: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const STAGE6_OR_7: Self = Self {
        has_point_zeros: true,
    };

    pub(in crate::protocols::jolt::emit::rust) fn has_point_zeros(self) -> bool {
        self.has_point_zeros
    }
}
