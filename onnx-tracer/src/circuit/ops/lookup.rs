use super::Op;
use super::*;
use crate::{
    circuit::utils,
    graph::utilities::{multiplier_to_scale, scale_to_multiplier},
    tensor::{self, Tensor, TensorError, TensorType},
    trace_types::ONNXOpcode,
};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex
/// operations via accumulation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Abs,
    Div { denom: utils::F32 },
    Cast { scale: utils::F32 },
    ReLU,
    Max { scale: utils::F32, a: utils::F32 },
    Min { scale: utils::F32, a: utils::F32 },
    Ceil { scale: utils::F32 },
    Floor { scale: utils::F32 },
    Round { scale: utils::F32 },
    RoundHalfToEven { scale: utils::F32 },
    Sqrt { scale: utils::F32 },
    Rsqrt { scale: utils::F32 },
    Recip { scale: utils::F32 },
    LeakyReLU { slope: utils::F32 },
    Sigmoid { scale: utils::F32 },
    Ln { scale: utils::F32 },
    Exp { scale: utils::F32 },
    Cos { scale: utils::F32 },
    ACos { scale: utils::F32 },
    Cosh { scale: utils::F32 },
    ACosh { scale: utils::F32 },
    Sin { scale: utils::F32 },
    ASin { scale: utils::F32 },
    Sinh { scale: utils::F32 },
    ASinh { scale: utils::F32 },
    Tan { scale: utils::F32 },
    ATan { scale: utils::F32 },
    Tanh { scale: utils::F32 },
    ATanh { scale: utils::F32 },
    Erf { scale: utils::F32 },
    GreaterThan { a: utils::F32 },
    LessThan { a: utils::F32 },
    GreaterThanEqual { a: utils::F32 },
    LessThanEqual { a: utils::F32 },
    Sign,
    KroneckerDelta,
    Pow { scale: utils::F32, a: utils::F32 },
}

impl From<&LookupOp> for ONNXOpcode {
    fn from(value: &LookupOp) -> Self {
        match value {
            LookupOp::ReLU => ONNXOpcode::Relu,
            LookupOp::Sigmoid { .. } => ONNXOpcode::Sigmoid,
            LookupOp::Rsqrt { .. } => ONNXOpcode::Sqrt,
            LookupOp::Div { .. } => ONNXOpcode::Div,
            _ => {
                panic!("LookupOp {value:?} cannot be converted to ONNXOpcode",);
            }
        }
    }
}

impl LookupOp {
    /// Returns the range of values that can be represented by the table
    pub fn bit_range(max_len: usize) -> (i128, i128) {
        let range = (max_len - 1) as f64 / 2_f64;
        let range = range as i128;
        (-range, range)
    }
}

impl<F: TensorType + PartialOrd> Op<F> for LookupOp
where
    i128: std::convert::From<F>,
    F: From<i128>,
{
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = x[0].clone();
        let x = x.map(|x| i128::from(x));
        let res = match &self {
            LookupOp::Abs => Ok(tensor::ops::abs(&x)?),
            LookupOp::Ceil { scale } => Ok(tensor::ops::nonlinearities::ceil(&x, scale.into())),
            LookupOp::Floor { scale } => Ok(tensor::ops::nonlinearities::floor(&x, scale.into())),
            LookupOp::Round { scale } => Ok(tensor::ops::nonlinearities::round(&x, scale.into())),
            LookupOp::RoundHalfToEven { scale } => Ok(
                tensor::ops::nonlinearities::round_half_to_even(&x, scale.into()),
            ),
            LookupOp::Pow { scale, a } => Ok(tensor::ops::nonlinearities::pow(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::KroneckerDelta => Ok(tensor::ops::nonlinearities::kronecker_delta(&x)),
            LookupOp::Max { scale, a } => Ok(tensor::ops::nonlinearities::max(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::Min { scale, a } => Ok(tensor::ops::nonlinearities::min(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::Sign => Ok(tensor::ops::nonlinearities::sign(&x)),
            LookupOp::LessThan { a } => Ok(tensor::ops::nonlinearities::less_than(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::LessThanEqual { a } => Ok(tensor::ops::nonlinearities::less_than_equal(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::GreaterThan { a } => Ok(tensor::ops::nonlinearities::greater_than(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::GreaterThanEqual { a } => Ok(
                tensor::ops::nonlinearities::greater_than_equal(&x, f32::from(*a).into()),
            ),
            LookupOp::Div { denom } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*denom).into(),
            )),
            LookupOp::Cast { scale } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*scale).into(),
            )),
            LookupOp::Recip { scale } => Ok(tensor::ops::nonlinearities::recip(&x, scale.into())),
            LookupOp::ReLU => Ok(tensor::ops::nonlinearities::leakyrelu(&x, 0_f64)),

            LookupOp::LeakyReLU { slope: a } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, a.0.into()))
            }
            LookupOp::Sigmoid { scale } => {
                Ok(tensor::ops::nonlinearities::sigmoid(&x, scale.into()))
            }
            LookupOp::Sqrt { scale } => Ok(tensor::ops::nonlinearities::sqrt(&x, scale.into())),
            LookupOp::Rsqrt { scale } => Ok(tensor::ops::nonlinearities::rsqrt(&x, scale.into())),
            LookupOp::Erf { scale } => Ok(tensor::ops::nonlinearities::erffunc(&x, scale.into())),
            LookupOp::Exp { scale } => Ok(tensor::ops::nonlinearities::exp(&x, scale.into())),
            LookupOp::Ln { scale } => Ok(tensor::ops::nonlinearities::ln(&x, scale.into())),
            LookupOp::Cos { scale } => Ok(tensor::ops::nonlinearities::cos(&x, scale.into())),
            LookupOp::ACos { scale } => Ok(tensor::ops::nonlinearities::acos(&x, scale.into())),
            LookupOp::Cosh { scale } => Ok(tensor::ops::nonlinearities::cosh(&x, scale.into())),
            LookupOp::ACosh { scale } => Ok(tensor::ops::nonlinearities::acosh(&x, scale.into())),
            LookupOp::Sin { scale } => Ok(tensor::ops::nonlinearities::sin(&x, scale.into())),
            LookupOp::ASin { scale } => Ok(tensor::ops::nonlinearities::asin(&x, scale.into())),
            LookupOp::Sinh { scale } => Ok(tensor::ops::nonlinearities::sinh(&x, scale.into())),
            LookupOp::ASinh { scale } => Ok(tensor::ops::nonlinearities::asinh(&x, scale.into())),
            LookupOp::Tan { scale } => Ok(tensor::ops::nonlinearities::tan(&x, scale.into())),
            LookupOp::ATan { scale } => Ok(tensor::ops::nonlinearities::atan(&x, scale.into())),
            LookupOp::ATanh { scale } => Ok(tensor::ops::nonlinearities::atanh(&x, scale.into())),
            LookupOp::Tanh { scale } => Ok(tensor::ops::nonlinearities::tanh(&x, scale.into())),
        }?;

        let output = res.map(|x| F::from(x));

        Ok(ForwardResult {
            output,
            intermediate_lookups: vec![],
        })
    }

    /// Returns the name of the operation
    fn as_string(&self) -> String {
        match self {
            LookupOp::Abs => "ABS".into(),
            LookupOp::Ceil { scale } => format!("CEIL(scale={scale})"),
            LookupOp::Floor { scale } => format!("FLOOR(scale={scale})"),
            LookupOp::Round { scale } => format!("ROUND(scale={scale})"),
            LookupOp::RoundHalfToEven { scale } => format!("ROUND_HALF_TO_EVEN(scale={scale})"),
            LookupOp::Pow { a, scale } => format!("POW(scale={scale}, exponent={a})"),
            LookupOp::KroneckerDelta => "K_DELTA".into(),
            LookupOp::Max { scale, a } => format!("MAX(scale={scale}, a={a})"),
            LookupOp::Min { scale, a } => format!("MIN(scale={scale}, a={a})"),
            LookupOp::Sign => "SIGN".into(),
            LookupOp::GreaterThan { .. } => "GREATER_THAN".into(),
            LookupOp::GreaterThanEqual { .. } => "GREATER_THAN_EQUAL".into(),
            LookupOp::LessThan { .. } => "LESS_THAN".into(),
            LookupOp::LessThanEqual { .. } => "LESS_THAN_EQUAL".into(),
            LookupOp::Recip { scale, .. } => format!("RECIP(scale={scale})"),
            LookupOp::Div { denom, .. } => format!("DIV(denom={denom})"),
            LookupOp::Cast { scale } => format!("CAST(scale={scale})"),
            LookupOp::Ln { scale } => format!("LN(scale={scale})"),
            LookupOp::ReLU => "RELU".to_string(),
            LookupOp::LeakyReLU { slope: a } => format!("L_RELU(slope={a})"),
            LookupOp::Sigmoid { scale } => format!("SIGMOID(scale={scale})"),
            LookupOp::Sqrt { scale } => format!("SQRT(scale={scale})"),
            LookupOp::Erf { scale } => format!("ERF(scale={scale})"),
            LookupOp::Rsqrt { scale } => format!("RSQRT(scale={scale})"),
            LookupOp::Exp { scale } => format!("EXP(scale={scale})"),
            LookupOp::Tan { scale } => format!("TAN(scale={scale})"),
            LookupOp::ATan { scale } => format!("ATAN(scale={scale})"),
            LookupOp::Tanh { scale } => format!("TANH(scale={scale})"),
            LookupOp::ATanh { scale } => format!("ATANH(scale={scale})"),
            LookupOp::Cos { scale } => format!("COS(scale={scale})"),
            LookupOp::ACos { scale } => format!("ACOS(scale={scale})"),
            LookupOp::Cosh { scale } => format!("COSH(scale={scale})"),
            LookupOp::ACosh { scale } => format!("ACOSH(scale={scale})"),
            LookupOp::Sin { scale } => format!("SIN(scale={scale})"),
            LookupOp::ASin { scale } => format!("ASIN(scale={scale})"),
            LookupOp::Sinh { scale } => format!("SINH(scale={scale})"),
            LookupOp::ASinh { scale } => format!("ASINH(scale={scale})"),
        }
    }

    // fn layout(
    //     &self,
    //     config: &mut crate::circuit::BaseConfig<F>,
    //     region: &mut RegionCtx<F>,
    //     values: &[ValTensor<F>],
    // ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
    //     Ok(Some(layouts::nonlinearity(
    //         config,
    //         region,
    //         values[..].try_into()?,
    //         self,
    //     )?))
    // }

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, inputs_scale: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
            LookupOp::Cast { scale } => {
                let in_scale = inputs_scale[0];
                in_scale + multiplier_to_scale(1. / scale.0 as f64)
            }
            LookupOp::Recip { scale } => {
                let mut out_scale = inputs_scale[0];
                out_scale +=
                    multiplier_to_scale(scale.0 as f64 / scale_to_multiplier(out_scale).powf(2.0));
                out_scale
            }
            LookupOp::Sign
            | LookupOp::GreaterThan { .. }
            | LookupOp::LessThan { .. }
            | LookupOp::GreaterThanEqual { .. }
            | LookupOp::LessThanEqual { .. }
            | LookupOp::KroneckerDelta
            | LookupOp::Round { .. }
            | LookupOp::RoundHalfToEven { .. }
            | LookupOp::Ceil { .. }
            | LookupOp::Floor { .. } => 0,
            _ => inputs_scale[0],
        };
        Ok(scale)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        vec![*self]
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(*self) // Forward to the derive(Clone) impl
    }
}
