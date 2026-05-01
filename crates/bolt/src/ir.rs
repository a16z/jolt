use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use melior::ir::operation::OperationLike;
use melior::ir::{Attribute, Module};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Protocol;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Concrete;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Party;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compute;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cpu;

pub trait Phase {
    const NAME: &'static str;
}

impl Phase for Protocol {
    const NAME: &'static str = "protocol";
}

impl Phase for Concrete {
    const NAME: &'static str = "concrete";
}

impl Phase for Party {
    const NAME: &'static str = "party";
}

impl Phase for Compute {
    const NAME: &'static str = "compute";
}

impl Phase for Cpu {
    const NAME: &'static str = "cpu";
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    Prover,
    Verifier,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prover => "prover",
            Self::Verifier => "verifier",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "prover" => Some(Self::Prover),
            "verifier" => Some(Self::Verifier),
            _ => None,
        }
    }
}

impl Display for Role {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct BoltModule<'c, P: Phase> {
    module: Module<'c>,
    phase: PhantomData<P>,
}

impl<'c, P: Phase> BoltModule<'c, P> {
    pub(crate) fn from_mlir(module: Module<'c>) -> Self {
        Self {
            module,
            phase: PhantomData,
        }
    }

    pub fn as_mlir_module(&self) -> &Module<'c> {
        &self.module
    }

    pub fn as_mlir_module_mut(&mut self) -> &mut Module<'c> {
        &mut self.module
    }

    pub fn into_mlir_module(self) -> Module<'c> {
        self.module
    }

    pub fn name(&self) -> String {
        self.string_attr("sym_name")
            .unwrap_or_else(|| "anonymous".to_owned())
    }

    pub fn role(&self) -> Option<Role> {
        self.string_attr("bolt.role")
            .and_then(|value| Role::parse(&value))
    }

    pub fn verify(&self) -> bool {
        self.module.as_operation().verify()
    }

    fn string_attr(&self, name: &str) -> Option<String> {
        self.module
            .as_operation()
            .attribute(name)
            .ok()
            .and_then(string_attribute_value)
    }
}

pub trait TextMlir {
    fn to_text_mlir(&self) -> String;
}

impl<P: Phase> TextMlir for BoltModule<'_, P> {
    fn to_text_mlir(&self) -> String {
        self.module.as_operation().to_string()
    }
}

pub(crate) fn string_attribute_value(attribute: Attribute<'_>) -> Option<String> {
    let value = attribute.to_string();
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .map(ToOwned::to_owned)
}

pub(crate) fn symbol_attribute_value(attribute: Attribute<'_>) -> Option<String> {
    attribute
        .to_string()
        .strip_prefix('@')
        .map(ToOwned::to_owned)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub message: String,
}

impl Diagnostic {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
