use std::error::Error;
use std::fmt::{self, Display, Formatter};

use melior::dialect::DialectRegistry;
use melior::ir::attribute::StringAttribute;
use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationBuilder, OperationMutLike};
use melior::ir::{Attribute, Identifier, Location, Module, OperationRef, Type, Value};
use melior::utility::register_all_dialects;
use melior::Context;

use crate::dialects::load_bolt_dialects;
use crate::ir::{BoltModule, Phase, Role, TextMlir};

#[derive(Debug)]
pub struct MeliorContext {
    context: Context,
}

impl MeliorContext {
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(Self::abort_init_error)
    }

    pub fn try_new() -> Result<Self, MlirError> {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new_with_registry(&registry, false);
        context.load_all_available_dialects();
        load_bolt_dialects(&context)
            .map_err(|message| MlirError::DialectRegistration { message })?;
        context.set_allow_unregistered_dialects(false);
        Ok(Self { context })
    }

    fn abort_init_error(error: MlirError) -> Self {
        drop(error);
        std::process::abort();
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn new_module<'c, P: Phase>(&'c self, name: &str, role: Option<Role>) -> BoltModule<'c, P> {
        let mut module = Module::new(Location::unknown(&self.context));
        module
            .as_operation_mut()
            .set_attribute("sym_name", StringAttribute::new(&self.context, name).into());
        module.as_operation_mut().set_attribute(
            "bolt.phase",
            StringAttribute::new(&self.context, P::NAME).into(),
        );
        if let Some(role) = role {
            module.as_operation_mut().set_attribute(
                "bolt.role",
                StringAttribute::new(&self.context, role.as_str()).into(),
            );
        }
        BoltModule::from_mlir(module)
    }

    pub fn parse_module<'c, P: Phase>(
        &'c self,
        source: &str,
    ) -> Result<BoltModule<'c, P>, MlirError> {
        Module::parse(&self.context, source)
            .map(BoltModule::from_mlir)
            .ok_or_else(|| MlirError::ParseFailed {
                source: source.to_owned(),
            })
    }

    pub fn append_op<'c, P: Phase>(
        &'c self,
        module: &BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: &[(&str, &str)],
    ) -> Result<(), MlirError> {
        self.append_op_from_iter(module, name, symbol, attrs.iter().copied())
    }

    pub fn append_op_with_owned_attrs<'c, P: Phase>(
        &'c self,
        module: &BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: &[(String, String)],
    ) -> Result<(), MlirError> {
        self.append_op_from_iter(
            module,
            name,
            symbol,
            attrs
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
        )
    }

    fn append_op_from_iter<'c, P: Phase, I, K, V>(
        &'c self,
        module: &BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: I,
    ) -> Result<(), MlirError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut attributes = Vec::new();
        if let Some(symbol) = symbol {
            attributes.push((
                Identifier::new(&self.context, "sym_name"),
                StringAttribute::new(&self.context, symbol).into(),
            ));
        }
        for (name, source) in attrs {
            let name = name.as_ref();
            let source = source.as_ref();
            attributes.push((
                Identifier::new(&self.context, name),
                self.parse_attr(name, source)?,
            ));
        }

        let operation = OperationBuilder::new(name, Location::unknown(&self.context))
            .add_attributes(&attributes)
            .build()
            .map_err(|source| MlirError::OperationBuild {
                op: name.to_owned(),
                source,
            })?;
        let _operation = module.as_mlir_module().body().append_operation(operation);
        Ok(())
    }

    pub(crate) fn append_typed_op<'c, 'a, P: Phase>(
        &'c self,
        module: &'a BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: &[(&str, &str)],
        operands: &[Value<'c, 'a>],
        result_types: &[&str],
    ) -> Result<OperationRef<'c, 'a>, MlirError> {
        self.append_typed_op_from_iter(
            module,
            name,
            symbol,
            attrs.iter().copied(),
            operands,
            result_types,
        )
    }

    pub(crate) fn append_typed_op_with_owned_attrs<'c, 'a, P: Phase>(
        &'c self,
        module: &'a BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: &[(String, String)],
        operands: &[Value<'c, 'a>],
        result_types: &[&str],
    ) -> Result<OperationRef<'c, 'a>, MlirError> {
        self.append_typed_op_from_iter(
            module,
            name,
            symbol,
            attrs
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
            operands,
            result_types,
        )
    }

    fn append_typed_op_from_iter<'c, 'a, P: Phase, I, K, V>(
        &'c self,
        module: &'a BoltModule<'c, P>,
        name: &str,
        symbol: Option<&str>,
        attrs: I,
        operands: &[Value<'c, 'a>],
        result_types: &[&str],
    ) -> Result<OperationRef<'c, 'a>, MlirError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut attributes = Vec::new();
        if let Some(symbol) = symbol {
            attributes.push((
                Identifier::new(&self.context, "sym_name"),
                StringAttribute::new(&self.context, symbol).into(),
            ));
        }
        for (name, source) in attrs {
            let name = name.as_ref();
            let source = source.as_ref();
            attributes.push((
                Identifier::new(&self.context, name),
                self.parse_attr(name, source)?,
            ));
        }
        let result_types = result_types
            .iter()
            .map(|source| {
                Type::parse(&self.context, source).ok_or_else(|| MlirError::TypeParse {
                    source: (*source).to_owned(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let operation = OperationBuilder::new(name, Location::unknown(&self.context))
            .add_operands(operands)
            .add_results(&result_types)
            .add_attributes(&attributes)
            .build()
            .map_err(|source| MlirError::OperationBuild {
                op: name.to_owned(),
                source,
            })?;
        Ok(module.as_mlir_module().body().append_operation(operation))
    }

    fn parse_attr<'c>(&'c self, name: &str, source: &str) -> Result<Attribute<'c>, MlirError> {
        Attribute::parse(&self.context, source).ok_or_else(|| MlirError::AttributeParse {
            name: name.to_owned(),
            source: source.to_owned(),
        })
    }
}

impl Default for MeliorContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum MlirError {
    AttributeParse { name: String, source: String },
    TypeParse { source: String },
    OperationBuild { op: String, source: melior::Error },
    ParseFailed { source: String },
    Schema { message: String },
    DialectRegistration { message: String },
    VerificationFailed { source: String },
}

impl Display for MlirError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AttributeParse { name, source } => {
                write!(
                    formatter,
                    "failed to parse MLIR attribute `{name}` from `{source}`"
                )
            }
            Self::TypeParse { source } => {
                write!(formatter, "failed to parse MLIR type `{source}`")
            }
            Self::OperationBuild { op, source } => {
                write!(formatter, "failed to build MLIR operation `{op}`: {source}")
            }
            Self::ParseFailed { source } => {
                write!(formatter, "failed to parse MLIR module:\n{source}")
            }
            Self::Schema { message } => formatter.write_str(message),
            Self::DialectRegistration { message } => formatter.write_str(message),
            Self::VerificationFailed { source } => {
                write!(formatter, "MLIR module verification failed:\n{source}")
            }
        }
    }
}

impl Error for MlirError {}

pub(crate) fn verify_module<P: Phase>(module: &BoltModule<'_, P>) -> Result<(), MlirError> {
    if module.verify() {
        Ok(())
    } else {
        Err(MlirError::VerificationFailed {
            source: module.to_text_mlir(),
        })
    }
}
