use std::ops::{Add, Mul, Neg, Sub};

use super::*;
use crate::{
    circuit::{self, ops::chip::Tolerance, utils},
    tensor::{self, Tensor, TensorError, TensorType},
    trace_types::ONNXOpcode,
};
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::tract_itertools::Itertools;

// import run args from model

#[allow(missing_docs)]
/// An enum representing the operations that consist of both lookups and arithmetic
/// operations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HybridOp {
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceArgMax {
        dim: usize,
    },
    SumPool {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        kernel_shape: (usize, usize),
        normalized: bool,
    },
    MaxPool2d {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    ReduceArgMin {
        dim: usize,
    },
    Softmax {
        scale: utils::F32,
        axes: Vec<usize>,
    },
    RangeCheck(Tolerance),
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equals,
    Gather {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    TopK {
        dim: usize,
        k: usize,
        largest: bool,
    },
    OneHot {
        dim: usize,
        num_classes: usize,
    },
    GatherElements {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    ScatterElements {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
}

impl From<&HybridOp> for ONNXOpcode {
    fn from(value: &HybridOp) -> Self {
        match value {
            HybridOp::Softmax { .. } => ONNXOpcode::Softmax,
            HybridOp::Gather { .. } => ONNXOpcode::Gather,
            HybridOp::GreaterEqual => ONNXOpcode::Gte,
            _ => {
                panic!("HybridOp {value:?} cannot be converted to ONNXOpcode",);
            }
        }
    }
}

impl<
        F: TensorType
            + PartialOrd
            + Add<Output = F>
            + Ord
            + Send
            + Sync
            + Sub
            + From<u32>
            + From<i128>
            + Mul<Output = F>
            + Sub<Output = F>
            + Neg<Output = F>
            + std::iter::Sum,
    > Op<F> for HybridOp
where
    i128: std::convert::From<F>,
{
    ///
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        match self {
            HybridOp::Greater | HybridOp::Less | HybridOp::Equals => vec![0, 1],
            HybridOp::ScatterElements { .. } => vec![0, 2],
            _ => vec![],
        }
    }

    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = inputs[0].clone();
        let x = x.map(|x| i128::from(x));
        let (res, intermediate_lookups) = match &self {
            HybridOp::ReduceMax { axes, .. } => {
                let res = tensor::ops::max_axes(&x, axes)?;
                let max_minus_one =
                    Tensor::from(vec![x.clone().into_iter().max().unwrap() - 1].into_iter());
                let unit = Tensor::from(vec![1].into_iter());
                // relu(x - max(x - 1)
                let inter_1 = (x.clone() - max_minus_one)?;
                // relu(1 - sum(relu(inter_1)))
                let inter_2 = (unit
                    - tensor::ops::sum(&tensor::ops::nonlinearities::leakyrelu(&inter_1, 0.0))?)?;

                (res.clone(), vec![inter_1, inter_2])
            }
            HybridOp::ReduceMin { axes, .. } => {
                let res = tensor::ops::min_axes(&x, axes)?;
                let min_plus_one =
                    Tensor::from(vec![x.clone().into_iter().min().unwrap() + 1].into_iter());
                let unit = Tensor::from(vec![1].into_iter());
                // relu(min(x + 1) - x)
                let inter_1 = (min_plus_one - x.clone())?;
                // relu(1 - sum(relu(inter_1)))
                let inter_2 = (unit
                    - tensor::ops::sum(&tensor::ops::nonlinearities::leakyrelu(&inter_1, 0.0))?)?;
                (res.clone(), vec![inter_1, inter_2])
            }
            HybridOp::ReduceArgMax { dim } => {
                let res = tensor::ops::argmax_axes(&x, *dim)?;
                let indices = Tensor::from(0..x.dims()[*dim] as i128);
                let mut inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                let inter =
                    Op::f(&HybridOp::ReduceMax { axes: vec![*dim] }, inputs)?.intermediate_lookups;
                inter_equals.extend(inter);

                (res.clone(), inter_equals)
            }
            HybridOp::ReduceArgMin { dim } => {
                let res = tensor::ops::argmin_axes(&x, *dim)?;
                let indices = Tensor::from(0..x.dims()[*dim] as i128);
                let mut inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                let inter =
                    Op::f(&HybridOp::ReduceMin { axes: vec![*dim] }, inputs)?.intermediate_lookups;
                inter_equals.extend(inter);

                (res.clone(), inter_equals)
            }
            HybridOp::Gather { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    log::debug!("idx: {}", idx.show());
                    let res = tensor::ops::gather(&x, idx, *dim)?;
                    (res.clone(), vec![])
                } else {
                    let y = inputs[1].clone().map(|x| i128::from(x));
                    let indices = Tensor::from(0..x.dims()[*dim] as i128);
                    let inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                    let res = tensor::ops::gather(&x, &y.map(|x| x as usize), *dim)?;
                    (res.clone(), inter_equals)
                }
            }
            HybridOp::OneHot { dim, num_classes } => {
                let indices = Tensor::from(0..x.dims()[*dim] as i128);
                let inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                let res = tensor::ops::one_hot(&x, *num_classes, *dim)?;
                (res.clone(), inter_equals)
            }
            HybridOp::TopK { dim, k, largest } => {
                let res = tensor::ops::topk_axes(&x, *k, *dim, *largest)?;

                let mut inter_equals = x
                    .clone()
                    .into_iter()
                    .flat_map(|elem| {
                        tensor::ops::equals(&res, &vec![elem].into_iter().into())
                            .unwrap()
                            .1
                    })
                    .collect::<Vec<_>>();

                // sort in descending order and take pairwise differences
                inter_equals.push(
                    x.into_iter()
                        .sorted()
                        .tuple_windows()
                        .map(|(a, b)| b - a)
                        .into(),
                );

                (res.clone(), inter_equals)
            }
            HybridOp::GatherElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    log::debug!("idx: {}", idx.show());
                    let res = tensor::ops::gather_elements(&x, idx, *dim)?;
                    (res.clone(), vec![])
                } else {
                    let y = inputs[1].clone().map(|x| i128::from(x));
                    let indices = Tensor::from(0..x.dims()[*dim] as i128);
                    let inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                    let res = tensor::ops::gather_elements(&x, &y.map(|x| x as usize), *dim)?;
                    (res.clone(), inter_equals)
                }
            }
            HybridOp::ScatterElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    log::debug!("idx: {}", idx.show());
                    let src = inputs[1].clone().map(|x| i128::from(x));
                    let res = tensor::ops::scatter(&x, idx, &src, *dim)?;
                    (res.clone(), vec![])
                } else {
                    let idx = inputs[1].clone().map(|x| i128::from(x) as usize);
                    let src = inputs[2].clone().map(|x| i128::from(x));
                    let indices = Tensor::from(0..x.dims()[*dim] as i128);
                    let inter_equals: Vec<Tensor<i128>> = vec![indices.clone(), -indices];
                    let res = tensor::ops::scatter(&x, &idx, &src, *dim)?;
                    (res.clone(), inter_equals)
                }
            }
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => {
                let max_minus_one =
                    Tensor::from(vec![x.clone().into_iter().max().unwrap() - 1].into_iter());
                let unit = Tensor::from(vec![1].into_iter());
                // relu(x - max(x - 1)
                let inter_1 = (x.clone() - max_minus_one)?;
                // relu(1 - sum(relu(inter_1)))
                let inter_2 = (unit
                    - tensor::ops::sum(&tensor::ops::nonlinearities::leakyrelu(&inter_1, 0.0))?)?;
                (
                    tensor::ops::max_pool2d(&x, padding, stride, pool_dims)?,
                    vec![inter_1, inter_2],
                )
            }
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
            } => tensor::ops::sumpool(&x, *padding, *stride, *kernel_shape, *normalized)?,
            HybridOp::Softmax { scale, axes } => {
                tensor::ops::nonlinearities::softmax_axes(&x, scale.into(), axes)
            }
            HybridOp::RangeCheck(tol) => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                (
                    tensor::ops::nonlinearities::range_check_percent(&[x, y], 128, 128, tol.val),
                    vec![],
                )
            }
            HybridOp::Greater => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                tensor::ops::greater(&x, &y)?
            }
            HybridOp::GreaterEqual => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                tensor::ops::greater_equal(&x, &y)?
            }
            HybridOp::Less => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                tensor::ops::less(&x, &y)?
            }
            HybridOp::LessEqual => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                tensor::ops::less_equal(&x, &y)?
            }
            HybridOp::Equals => {
                let y = inputs[1].clone().map(|x| i128::from(x));
                tensor::ops::equals(&x, &y)?
            }
        };

        // convert back to felt
        let output = res.map(|x| F::from(x));

        Ok(ForwardResult {
            output,
            intermediate_lookups,
        })
    }

    fn as_string(&self) -> String {
        match self {
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
            } => format!(
                "SUMPOOL (padding={padding:?}, stride={stride:?}, kernel_shape={kernel_shape:?}, normalized={normalized})"
            ),
            HybridOp::ReduceMax { axes } => format!("REDUCEMAX (axes={axes:?})"),
            HybridOp::ReduceArgMax { dim } => format!("REDUCEARGMAX (dim={dim})"),
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => format!(
                "MAXPOOL2D (padding={padding:?}, stride={stride:?}, pool_dims={pool_dims:?})"
            ),
            HybridOp::ReduceMin { axes } => format!("REDUCEMIN (axes={axes:?})"),
            HybridOp::ReduceArgMin { dim } => format!("REDUCEARGMIN (dim={dim})"),
            HybridOp::Softmax { scale, axes } => {
                format!("SOFTMAX (scale={scale}, axes={axes:?})")
            }
            HybridOp::RangeCheck(p) => format!("RANGECHECK (tol={p:?})"),
            HybridOp::Greater => "GREATER".into(),
            HybridOp::GreaterEqual => "GREATEREQUAL".into(),
            HybridOp::Less => "LESS".into(),
            HybridOp::LessEqual => "LESSEQUAL".into(),
            HybridOp::Equals => "EQUALS".into(),
            HybridOp::Gather { dim, .. } => format!("GATHER (dim={dim})"),
            HybridOp::TopK { k, dim, largest } => {
                format!("TOPK (k={k}, dim={dim}, largest={largest})")
            }
            HybridOp::GatherElements { dim, .. } => format!("GATHERELEMENTS (dim={dim})"),
            HybridOp::ScatterElements { dim, .. } => format!("SCATTERELEMENTS (dim={dim})"),
            HybridOp::OneHot { dim, num_classes } => {
                format!("ONEHOT (dim={dim}, num_classes={num_classes})")
            }
        }
    }

    //   fn layout(
    //     &self,
    //     config: &mut crate::circuit::BaseConfig<F>,
    //     region: &mut RegionCtx<F>,
    //     values: &[ValTensor<F>],
    //   ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
    //     Ok(Some(match self {
    //       HybridOp::SumPool {
    //         padding,
    //         stride,
    //         kernel_shape,
    //         normalized,
    //       } => layouts::sumpool(
    //         config,
    //         region,
    //         values[..].try_into()?,
    //         *padding,
    //         *stride,
    //         *kernel_shape,
    //         *normalized,
    //       )?,
    //       HybridOp::Gather { dim, constant_idx } => {
    //         if let Some(idx) = constant_idx {
    //           tensor::ops::gather(values[0].get_inner_tensor()?, idx, *dim)?.into()
    //         } else {
    //           layouts::gather(config, region, values[..].try_into()?, *dim)?
    //         }
    //       }
    //       HybridOp::GatherElements { dim, constant_idx } => {
    //         if let Some(idx) = constant_idx {
    //           tensor::ops::gather_elements(values[0].get_inner_tensor()?, idx,
    // *dim)?.into()         } else {
    //           layouts::gather_elements(config, region, values[..].try_into()?, *dim)?
    //         }
    //       }
    //       HybridOp::ScatterElements { dim, constant_idx } => {
    //         if let Some(idx) = constant_idx {
    //           tensor::ops::scatter(
    //             values[0].get_inner_tensor()?,
    //             idx,
    //             values[1].get_inner_tensor()?,
    //             *dim,
    //           )?
    //           .into()
    //         } else {
    //           layouts::scatter_elements(config, region, values[..].try_into()?, *dim)?
    //         }
    //       }
    //       HybridOp::MaxPool2d {
    //         padding,
    //         stride,
    //         pool_dims,
    //       } => layouts::max_pool2d(
    //         config,
    //         region,
    //         values[..].try_into()?,
    //         *padding,
    //         *stride,
    //         *pool_dims,
    //       )?,
    //       HybridOp::ReduceMax { axes } => {
    //         layouts::max_axes(config, region, values[..].try_into()?, axes)?
    //       }
    //       HybridOp::ReduceArgMax { dim } => {
    //         layouts::argmax_axes(config, region, values[..].try_into()?, *dim)?
    //       }
    //       HybridOp::ReduceMin { axes } => {
    //         layouts::min_axes(config, region, values[..].try_into()?, axes)?
    //       }
    //       HybridOp::ReduceArgMin { dim } => {
    //         layouts::argmin_axes(config, region, values[..].try_into()?, *dim)?
    //       }
    //       HybridOp::Softmax { scale, axes } => {
    //         layouts::softmax_axes(config, region, values[..].try_into()?, *scale,
    // axes)?       }
    //       HybridOp::RangeCheck(tol) => {
    //         layouts::range_check_percent(config, region, values[..].try_into()?,
    // tol.scale, tol.val)?       }
    //       HybridOp::Greater => layouts::greater(config, region,
    // values[..].try_into()?)?,       HybridOp::GreaterEqual =>
    // layouts::greater_equal(config, region, values[..].try_into()?)?,
    //       HybridOp::Less => layouts::less(config, region, values[..].try_into()?)?,
    //       HybridOp::LessEqual => layouts::less_equal(config, region,
    // values[..].try_into()?)?,       HybridOp::Equals => layouts::equals(config,
    // region, values[..].try_into()?)?,       HybridOp::TopK { dim, k, largest } => {
    //         layouts::topk_axes(config, region, values[..].try_into()?, *k, *dim,
    // *largest)?       }
    //       HybridOp::OneHot { dim, num_classes } => {
    //         layouts::one_hot_axis(config, region, values[..].try_into()?,
    // *num_classes, *dim)?       }
    //     }))
    //   }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
            HybridOp::Greater
            | HybridOp::GreaterEqual
            | HybridOp::Less
            | HybridOp::LessEqual
            | HybridOp::ReduceArgMax { .. }
            | HybridOp::OneHot { .. }
            | HybridOp::ReduceArgMin { .. } => 0,
            HybridOp::Softmax { .. } => 2 * in_scales[0],
            _ => in_scales[0],
        };
        Ok(scale)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            HybridOp::ReduceMax { .. }
            | HybridOp::ReduceMin { .. }
            | HybridOp::MaxPool2d { .. } => Op::<F>::required_lookups(&LookupOp::ReLU),
            HybridOp::Softmax { scale, .. } => {
                vec![
                    LookupOp::Exp { scale: *scale },
                    LookupOp::Recip {
                        scale: scale.0.powf(2.0).into(),
                    },
                ]
            }
            HybridOp::RangeCheck(tol) => {
                let mut lookups = vec![];
                if tol.val > 0.0 {
                    let scale_squared = tol.scale.0.powf(2.0);
                    lookups.extend([
                        LookupOp::Recip {
                            scale: scale_squared.into(),
                        },
                        LookupOp::GreaterThan {
                            a: circuit::utils::F32((tol.val * scale_squared) / 100.0),
                        },
                    ]);
                }
                lookups
            }
            HybridOp::Greater | HybridOp::Less => {
                vec![LookupOp::GreaterThan {
                    a: circuit::utils::F32(0.),
                }]
            }
            HybridOp::GreaterEqual | HybridOp::LessEqual => {
                vec![LookupOp::GreaterThanEqual {
                    a: circuit::utils::F32(0.),
                }]
            }
            HybridOp::TopK { .. } => {
                vec![
                    LookupOp::GreaterThan {
                        a: circuit::utils::F32(0.),
                    },
                    LookupOp::KroneckerDelta,
                ]
            }
            HybridOp::Gather {
                constant_idx: None, ..
            }
            | HybridOp::OneHot { .. }
            | HybridOp::GatherElements {
                constant_idx: None, ..
            }
            | HybridOp::ScatterElements {
                constant_idx: None, ..
            }
            | HybridOp::Equals => {
                vec![LookupOp::KroneckerDelta]
            }
            HybridOp::ReduceArgMax { .. } | HybridOp::ReduceArgMin { .. } => {
                vec![LookupOp::ReLU, LookupOp::KroneckerDelta]
            }
            HybridOp::SumPool {
                kernel_shape,
                normalized: true,
                ..
            } => {
                vec![LookupOp::Div {
                    denom: utils::F32((kernel_shape.0 * kernel_shape.1) as f32),
                }]
            }
            _ => vec![],
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
