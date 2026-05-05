#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::pass) struct LoweredAttr {
    pub(super) name: &'static str,
    pub(super) kind: LoweredAttrKind,
}

impl LoweredAttr {
    pub(in crate::pass) const fn symbol_ref(name: &'static str) -> Self {
        Self {
            name,
            kind: LoweredAttrKind::SymbolRef,
        }
    }

    pub(in crate::pass) const fn symbol_array(name: &'static str) -> Self {
        Self {
            name,
            kind: LoweredAttrKind::SymbolArray,
        }
    }

    pub(in crate::pass) const fn string(name: &'static str) -> Self {
        Self {
            name,
            kind: LoweredAttrKind::String,
        }
    }

    pub(in crate::pass) const fn int(name: &'static str) -> Self {
        Self {
            name,
            kind: LoweredAttrKind::Int,
        }
    }

    pub(in crate::pass) const fn bool(name: &'static str) -> Self {
        Self {
            name,
            kind: LoweredAttrKind::Bool,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LoweredAttrKind {
    SymbolRef,
    SymbolArray,
    String,
    Int,
    Bool,
}
