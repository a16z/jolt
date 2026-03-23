//! Symbolic integer expressions for config-dependent quantities.
//!
//! [`SymbolicExpr`] represents values like `log_T`, `3 * D_total`,
//! `log_T + log_k` that depend on protocol configuration. The graph stays
//! parameterized until instantiation via [`SymbolicExpr::resolve`].

use std::collections::HashMap;
use std::fmt;

/// A named configuration symbol (e.g., `log_T`, `D_instr`).
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Symbol(pub &'static str);

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Well-known configuration symbols.
impl Symbol {
    pub const LOG_T: Self = Self("log_T");
    pub const LOG_K: Self = Self("log_k");
    pub const LOG_ROWS: Self = Self("log_rows");
    pub const LOG_COLS: Self = Self("log_cols");
    pub const D_INSTR: Self = Self("D_instr");
    pub const D_BC: Self = Self("D_bc");
    pub const D_RAM: Self = Self("D_ram");
    pub const D_TOTAL: Self = Self("D_total");
}

/// Symbolic integer expression — resolved from config at graph build time.
///
/// Used for `num_vars`, gamma power counts, and any other config-dependent
/// quantity. Keeps the graph parameterized until instantiation.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum SymbolicExpr {
    Concrete(usize),
    Symbol(Symbol),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
}

impl SymbolicExpr {
    pub fn concrete(n: usize) -> Self {
        Self::Concrete(n)
    }

    pub fn symbol(s: Symbol) -> Self {
        Self::Symbol(s)
    }

    /// Resolve to a concrete value given a symbol table.
    ///
    /// Returns `None` if any referenced symbol is missing from the table.
    pub fn resolve(&self, symbols: &HashMap<Symbol, usize>) -> Option<usize> {
        match self {
            Self::Concrete(n) => Some(*n),
            Self::Symbol(s) => symbols.get(s).copied(),
            Self::Add(a, b) => Some(a.resolve(symbols)? + b.resolve(symbols)?),
            Self::Mul(a, b) => Some(a.resolve(symbols)? * b.resolve(symbols)?),
        }
    }

    /// Returns `true` if this expression is a concrete value.
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    /// Returns the concrete value if this is `Concrete`, else `None`.
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            Self::Concrete(n) => Some(*n),
            _ => None,
        }
    }
}

impl From<usize> for SymbolicExpr {
    fn from(n: usize) -> Self {
        Self::Concrete(n)
    }
}

impl std::ops::Add for SymbolicExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (Self::Concrete(a), Self::Concrete(b)) => Self::Concrete(a + b),
            _ => Self::Add(Box::new(self), Box::new(rhs)),
        }
    }
}

impl std::ops::Mul for SymbolicExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (Self::Concrete(a), Self::Concrete(b)) => Self::Concrete(a * b),
            _ => Self::Mul(Box::new(self), Box::new(rhs)),
        }
    }
}

impl fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(n) => write!(f, "{n}"),
            Self::Symbol(s) => write!(f, "{s}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
        }
    }
}

/// Number of variables in a polynomial or sumcheck instance.
pub type NumVars = SymbolicExpr;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_concrete() {
        let e = SymbolicExpr::Concrete(42);
        assert_eq!(e.resolve(&HashMap::new()), Some(42));
    }

    #[test]
    fn resolve_symbol() {
        let e = SymbolicExpr::Symbol(Symbol::LOG_T);
        let mut syms = HashMap::new();
        let _ = syms.insert(Symbol::LOG_T, 20);
        assert_eq!(e.resolve(&syms), Some(20));
    }

    #[test]
    fn resolve_missing_symbol() {
        let e = SymbolicExpr::Symbol(Symbol::LOG_T);
        assert_eq!(e.resolve(&HashMap::new()), None);
    }

    #[test]
    fn resolve_arithmetic() {
        // 3 * D_total + log_k
        let e = SymbolicExpr::Concrete(3) * SymbolicExpr::Symbol(Symbol::D_TOTAL)
            + SymbolicExpr::Symbol(Symbol::LOG_K);
        let mut syms = HashMap::new();
        let _ = syms.insert(Symbol::D_TOTAL, 16);
        let _ = syms.insert(Symbol::LOG_K, 8);
        assert_eq!(e.resolve(&syms), Some(56));
    }

    #[test]
    fn concrete_arithmetic_eagerly_folds() {
        let a = SymbolicExpr::Concrete(3);
        let b = SymbolicExpr::Concrete(5);
        let result = a + b;
        assert_eq!(result, SymbolicExpr::Concrete(8));
    }

    #[test]
    fn display() {
        let e = SymbolicExpr::Symbol(Symbol::LOG_T) + SymbolicExpr::Symbol(Symbol::LOG_K);
        assert_eq!(e.to_string(), "(log_T + log_k)");
    }
}
