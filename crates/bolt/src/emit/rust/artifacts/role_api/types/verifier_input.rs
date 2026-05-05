#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::emit::rust::artifacts::role_api) enum VerifierStageInputKind {
    Openings,
    Ram,
    Data,
}

impl VerifierStageInputKind {
    pub(in crate::emit::rust::artifacts::role_api) fn field_suffix(self) -> &'static str {
        match self {
            Self::Openings => "openings",
            Self::Ram => "ram",
            Self::Data => "data",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::emit::rust::artifacts::role_api) struct VerifierStageInput<'a> {
    pub(in crate::emit::rust::artifacts::role_api) kind: VerifierStageInputKind,
    pub(in crate::emit::rust::artifacts::role_api) type_name: &'a str,
}
