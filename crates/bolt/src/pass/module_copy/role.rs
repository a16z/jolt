use crate::ir::Role;

#[derive(Clone, Copy, Debug)]
pub(in crate::pass) enum PhaseCopyRole<'a> {
    Absent,
    Present(&'a Role),
}

impl<'a> PhaseCopyRole<'a> {
    pub(in crate::pass) const fn absent() -> Self {
        Self::Absent
    }

    pub(in crate::pass) const fn present(role: &'a Role) -> Self {
        Self::Present(role)
    }

    pub(super) fn append_attr(self, source: &mut String) {
        if let Self::Present(role) = self {
            source.push_str(", bolt.role = \"");
            source.push_str(role.as_str());
            source.push('"');
        }
    }
}
