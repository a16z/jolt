use crate::graph::{model::NodeType, vars::VarScales};
use crate::trace_types::{ONNXInstr, ONNXOpcode};
use crate::{
    circuit::ops::{
        hybrid::HybridOp, lookup::LookupOp, poly::PolyOp, Constant, ForwardResult, Input, Op,
        Unknown,
    },
    fieldutils::{felt_to_i128, i128_to_felt},
    graph::utilities::{
        multiplier_to_scale, new_op_from_onnx, node_output_shapes, quantize_tensor,
        scale_to_multiplier,
    },
    tensor::{Tensor, TensorError},
};
use halo2curves::bn256::Fr as Fp;
use log::{trace, warn};
use std::fmt::Debug;
use std::{collections::BTreeMap, error::Error, fmt};
use tabled::Tabled;
use tract_onnx::{
    self,
    prelude::{tract_itertools::Itertools, Node as OnnxNode, SymbolValues, TypedFact, TypedOp},
};

// /// A node's input is a tensor from another node's output.
pub type Outlet = (usize, usize);

/// A single operation in a [crate::graph::Model].
#[derive(Clone, Debug)]
pub struct Node {
    /// [Op] i.e what operation this node represents.
    pub opkind: SupportedOp,
    /// The denominator in the fixed point representation for the node's output.
    /// Tensors of differing scales should not be combined.
    pub out_scale: i32,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine
    // node has three input_shapes (one for the input, weight, and bias), but in_dim is [in],
    // out_dim is [out]
    /// The indices of the node's inputs.
    pub inputs: Vec<Outlet>,
    /// Dimensions of output.
    pub out_dims: Vec<usize>,
    /// The node's unique identifier.
    pub idx: usize,
    /// The node's num of uses
    pub num_uses: usize,
}

impl Node {
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the
    ///   computational graph.
    ///  * `public_params` - flag if parameters of model are public
    ///  * `idx` - The node's unique identifier.
    pub fn new(
        node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, NodeType>,
        scales: &VarScales,
        idx: usize,
        symbol_values: &SymbolValues,
    ) -> Self {
        trace!("Create {node:?}",);
        trace!("Create op {:?}", node.op);
        let num_uses = std::cmp::max(
            node.outputs
                .iter()
                .map(|outlet| outlet.successors.len())
                .sum::<usize>(),
            // cmp to 1 for outputs
            1,
        );
        // load the node inputs
        let mut inputs = vec![];
        // we can only take the inputs as mutable once -- so we need to collect them first
        let mut input_ids = node
            .inputs
            .iter()
            .map(|i| (i.node, i.slot))
            .collect::<Vec<_>>();
        input_ids.iter().for_each(|(i, _)| {
            inputs.push(other_nodes.get(i).unwrap().clone());
        });
        let (mut opkind, deleted_indices) =
            new_op_from_onnx(idx, scales, node.clone(), &mut inputs, symbol_values).unwrap(); // parses the op name
                                                                                              // we can only take the inputs as mutable once -- so we need to collect them first
        other_nodes.extend(
            inputs
                .iter()
                .map(|i| (i.idx(), i.clone()))
                .collect::<BTreeMap<_, _>>(),
        );
        input_ids.iter_mut().enumerate().for_each(|(i, (idx, _))| {
            if deleted_indices.contains(&i) {
                // this input is not used
                *idx = usize::MAX;
            }
        });

        // remove the inputs that are not used
        input_ids.retain(|(idx, _)| *idx != usize::MAX);

        // rescale the inputs if necessary to get consistent fixed points
        let mut in_scales: Vec<crate::Scale> = input_ids
            .iter()
            .map(|(idx, outlet)| {
                let idx = inputs.iter().position(|x| *idx == x.idx()).unwrap();
                inputs[idx].out_scales()[*outlet]
            })
            .collect::<Vec<_>>();

        let homogenous_inputs = opkind.requires_homogenous_input_scales();
        // autoamtically increases a constant's scale if it is only used once and
        for input in homogenous_inputs
            .into_iter()
            .filter(|i| !deleted_indices.contains(i))
        {
            if inputs.len() > input {
                let input_node = other_nodes
                    .get_mut(&inputs[input].idx())
                    .ok_or("input not found")
                    .unwrap();
                let input_opkind = &mut input_node.opkind();
                if let Some(constant) = input_opkind.get_mutable_constant() {
                    rescale_const_with_single_use(
                        constant,
                        in_scales.clone(),
                        input_node.num_uses(),
                    )
                    .unwrap();
                    input_node.replace_opkind(constant.clone_dyn().into());
                    let out_scale = input_opkind.out_scale(vec![]).unwrap();
                    input_node.bump_scale(out_scale);
                    in_scales[input] = out_scale;
                }
            } else {
                warn!("input {input} not found for rescaling, skipping ...",);
            }
        }

        opkind = opkind.homogenous_rescale(in_scales.clone()).unwrap().into();
        let mut out_scale = opkind.out_scale(in_scales.clone()).unwrap();
        // rescale the inputs if necessary to get consistent fixed points, we
        // select the largest scale (highest precision)
        let global_scale = scales.get_max();
        opkind = RebaseScale::rebase(opkind, global_scale, out_scale, scales.rebase_multiplier);
        out_scale = opkind.out_scale(in_scales).unwrap();

        // get the output shape
        let out_dims = node_output_shapes(&node, symbol_values).unwrap();
        // nodes vs subgraphs always have a single output
        let mut out_dims = out_dims[0].clone();

        if out_dims.is_empty() {
            out_dims = vec![1];
        }

        Node {
            idx,
            opkind,
            inputs: input_ids,
            out_dims,
            out_scale,
            num_uses,
        }
    }
}

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{v:?}",)
    } else {
        String::new()
    }
}

impl Node {
    /// Decodes the current [Node] into an [ONNXInstr] at the specified `address`.
    ///
    /// This method is typically used during preprocessing to transform the ONNX binary into the zkVM program code.
    ///
    /// # Arguments
    ///
    /// * `address` - The memory address or program counter where the decoded instruction will be placed.
    ///
    /// # Returns
    ///
    /// An [ONNXInstr] representing the decoded instruction for this node.
    ///
    /// # Panics
    ///
    /// This method will panic if there is an unsupported operator
    pub fn decode(&self, address: usize) -> ONNXInstr {
        match &self.opkind {
            SupportedOp::Linear(poly_op) => self.decode_with_opcode(poly_op, address),
            SupportedOp::Nonlinear(lookup_op) => self.decode_with_opcode(lookup_op, address),
            SupportedOp::Hybrid(hybrid_op) => self.decode_with_opcode(hybrid_op, address),
            SupportedOp::Constant(constant) => self.decode_with_opcode(constant, address),
            _ => panic!("Opkind {:?} not supported", self.opkind),
        }
    }

    /// Helper function to decode the node with a specific opcode.
    ///
    /// # Arguments
    /// * `op` - The operation to decode.
    /// * `address` - The address in the bytecode where this instruction will be placed.
    ///
    /// # Returns
    /// An [ONNXInstr] representing the decoded instruction.
    ///
    /// # Panics
    /// Panics if the operation does not have exactly two operands, as this is expected for the current implementation.
    fn decode_with_opcode<T>(&self, op: &T, address: usize) -> ONNXInstr
    where
        for<'a> &'a T: Into<ONNXOpcode> + Debug,
    {
        // FIXME: It is not guaranteed that there will be two operands and secondly need to understand Vec<Outlet> to see if this is even correct for two operands
        let input = self.inputs.first();
        // ONNXInstr {
        //     address,
        //     opcode: op.into(),
        //     ts1: input.map(|(node, slot)| *node),
        //     ts2,
        // }
        todo!()
    }
}

fn display_opkind(v: &SupportedOp) -> String {
    v.as_string()
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, PartialEq)]
pub struct Rescaled {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// The scale of the operation's inputs.
    pub scale: Vec<(usize, u128)>,
}

impl Op<Fp> for Rescaled {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<Fp>]) -> Result<ForwardResult<Fp>, TensorError> {
        if self.scale.len() != x.len() {
            return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
        }

        let mut rescaled_inputs = vec![];
        let inputs = &mut x.to_vec();
        for (i, ri) in inputs.iter_mut().enumerate() {
            let mult_tensor = Tensor::from([Fp::from(self.scale[i].1 as u64)].into_iter());
            let res = (ri.clone() * mult_tensor)?;
            rescaled_inputs.push(res);
        }
        Op::<Fp>::f(&*self.inner, &rescaled_inputs)
    }

    fn as_string(&self) -> String {
        format!("RESCALED INPUT ({})", self.inner.as_string())
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let in_scales = in_scales
            .into_iter()
            .zip(self.scale.iter())
            .map(|(a, b)| a + multiplier_to_scale(b.1 as f64))
            .collect();

        Op::<Fp>::out_scale(&*self.inner, in_scales)
    }

    // fn required_lookups(&self) -> Vec<LookupOp> {
    //   self.inner.required_lookups()
    // }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, PartialEq)]
pub struct RebaseScale {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// the multiplier applied to the node output
    pub multiplier: f64,
    /// scale being rebased to
    pub target_scale: i32,
    /// The original scale of the operation's inputs.
    pub original_scale: i32,
}

impl RebaseScale {
    ///
    pub fn rebase(
        inner: SupportedOp,
        global_scale: crate::Scale,
        op_out_scale: crate::Scale,
        scale_rebase_multiplier: u32,
    ) -> SupportedOp {
        if (op_out_scale > (global_scale * scale_rebase_multiplier as i32))
            && !inner.is_constant()
            && !inner.is_input()
        {
            let multiplier =
                scale_to_multiplier(op_out_scale - global_scale * scale_rebase_multiplier as i32);
            if let Some(op) = inner.get_rebased() {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier: op.multiplier * multiplier,
                    original_scale: op.original_scale,
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale: global_scale * scale_rebase_multiplier as i32,
                    multiplier,
                    original_scale: op_out_scale,
                })
            }
        } else {
            inner
        }
    }

    ///
    pub fn rebase_up(
        inner: SupportedOp,
        target_scale: crate::Scale,
        op_out_scale: crate::Scale,
    ) -> SupportedOp {
        if (op_out_scale < (target_scale)) && !inner.is_constant() && !inner.is_input() {
            let multiplier = scale_to_multiplier(op_out_scale - target_scale);
            if let Some(op) = inner.get_rebased() {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier: op.multiplier * multiplier,
                    original_scale: op.original_scale,
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale,
                    multiplier,
                    original_scale: op_out_scale,
                })
            }
        } else {
            inner
        }
    }
}

impl Op<Fp> for RebaseScale {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<Fp>]) -> Result<ForwardResult<Fp>, TensorError> {
        let mut res = Op::<Fp>::f(&*self.inner, x)?;
        let ri = res.output.map(felt_to_i128);
        let rescaled = crate::tensor::ops::nonlinearities::const_div(&ri, self.multiplier);
        res.output = rescaled.map(i128_to_felt);

        res.intermediate_lookups.push(ri);

        Ok(res)
    }

    fn as_string(&self) -> String {
        format!(
            "REBASED (div={:?}) ({})",
            self.multiplier,
            self.inner.as_string()
        )
    }

    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        Ok(self.target_scale)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        let mut lookups = self.inner.required_lookups();
        lookups.push(LookupOp::Div {
            denom: crate::circuit::utils::F32(self.multiplier as f32),
        });
        lookups
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// A single operation in a [crate::graph::Model].
#[derive(Clone, Debug, PartialEq)]
pub enum SupportedOp {
    /// A linear operation.
    Linear(PolyOp<Fp>),
    /// A nonlinear operation.
    Nonlinear(LookupOp),
    /// A hybrid operation.
    Hybrid(HybridOp),
    ///
    Input(Input),
    ///
    Constant(Constant<Fp>),
    ///
    Unknown(Unknown),
    ///
    Rescaled(Rescaled),
    ///
    RebaseScale(RebaseScale),
}

impl SupportedOp {
    ///
    pub fn is_lookup(&self) -> bool {
        match self {
            SupportedOp::Nonlinear(_) => true,
            SupportedOp::RebaseScale(op) => op.inner.is_lookup(),
            _ => false,
        }
    }
    ///
    pub fn get_input(&self) -> Option<Input> {
        match self {
            SupportedOp::Input(op) => Some(op.clone()),
            _ => None,
        }
    }

    ///
    pub fn get_rebased(&self) -> Option<&RebaseScale> {
        match self {
            SupportedOp::RebaseScale(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_lookup(&self) -> Option<&LookupOp> {
        match self {
            SupportedOp::Nonlinear(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_constant(&self) -> Option<&Constant<Fp>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_mutable_constant(&mut self) -> Option<&mut Constant<Fp>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn homogenous_rescale(
        &self,
        in_scales: Vec<crate::Scale>,
    ) -> Result<Box<dyn Op<Fp>>, Box<dyn Error>> {
        use crate::graph::utilities::homogenize_input_scales;

        let inputs_to_scale = self.requires_homogenous_input_scales();
        // creates a rescaled op if the inputs are not homogenous
        let op = self.clone_dyn();
        homogenize_input_scales(op, in_scales, inputs_to_scale)
    }

    /// Since each associated value of `SupportedOp` implements `Op`, let's define a
    /// helper method to retrieve it.
    fn as_op(&self) -> &dyn Op<Fp> {
        match self {
            SupportedOp::Linear(op) => op,
            SupportedOp::Nonlinear(op) => op,
            SupportedOp::Hybrid(op) => op,
            SupportedOp::Input(op) => op,
            SupportedOp::Constant(op) => op,
            SupportedOp::Unknown(op) => op,
            SupportedOp::Rescaled(op) => op,
            SupportedOp::RebaseScale(op) => op,
        }
    }
}

impl From<Box<dyn Op<Fp>>> for SupportedOp {
    fn from(value: Box<dyn Op<Fp>>) -> Self {
        if let Some(op) = value.as_any().downcast_ref::<PolyOp<Fp>>() {
            return SupportedOp::Linear(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<LookupOp>() {
            return SupportedOp::Nonlinear(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<HybridOp>() {
            return SupportedOp::Hybrid(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Input>() {
            return SupportedOp::Input(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Constant<Fp>>() {
            return SupportedOp::Constant(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Unknown>() {
            return SupportedOp::Unknown(op.clone());
        };
        if let Some(op) = value.as_any().downcast_ref::<Rescaled>() {
            return SupportedOp::Rescaled(op.clone());
        };
        if let Some(op) = value.as_any().downcast_ref::<RebaseScale>() {
            return SupportedOp::RebaseScale(op.clone());
        };

        log::error!("Unsupported op type");
        log::warn!("defaulting to Unknown");
        SupportedOp::Unknown(Unknown {})
    }
}

impl Op<Fp> for SupportedOp {
    fn f(&self, inputs: &[Tensor<Fp>]) -> Result<ForwardResult<Fp>, crate::tensor::TensorError> {
        self.as_op().f(inputs)
    }

    fn is_input(&self) -> bool {
        self.as_op().is_input()
    }

    fn is_constant(&self) -> bool {
        self.as_op().is_constant()
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        self.as_op().requires_homogenous_input_scales()
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        self.as_op().clone_dyn()
    }

    fn as_string(&self) -> String {
        self.as_op().as_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        self.as_op().required_lookups()
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        self.as_op().out_scale(in_scales)
    }
}

impl Tabled for Node {
    const LENGTH: usize = 6;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        let mut headers = Vec::with_capacity(Self::LENGTH);
        for i in [
            "idx",
            "opkind",
            "out_scale",
            "inputs",
            "out_dims",
            "required_lookups",
        ] {
            headers.push(std::borrow::Cow::Borrowed(i));
        }
        headers
    }

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        let mut fields = Vec::with_capacity(Self::LENGTH);
        fields.push(std::borrow::Cow::Owned(self.idx.to_string()));
        fields.push(std::borrow::Cow::Owned(display_opkind(&self.opkind)));
        fields.push(std::borrow::Cow::Owned(self.out_scale.to_string()));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.inputs)));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.out_dims)));
        fields.push(std::borrow::Cow::Owned(format!(
            "{:?}",
            self.opkind
                .required_lookups()
                .iter()
                .map(<LookupOp as Op<Fp>>::as_string)
                .collect_vec()
        )));
        fields
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        (self.out_scale == other.out_scale)
            && (self.inputs == other.inputs)
            && (self.out_dims == other.out_dims)
            && (self.idx == other.idx)
            && (self.opkind.as_string() == other.opkind.as_string())
    }
}

fn rescale_const_with_single_use(
    constant: &mut Constant<Fp>,
    in_scales: Vec<crate::Scale>,
    num_uses: usize,
) -> Result<(), Box<dyn Error>> {
    if num_uses == 1 {
        let current_scale = constant.out_scale(vec![])?;
        let scale_max = in_scales.iter().max().ok_or("no scales")?;
        if scale_max > &current_scale {
            let raw_values = constant.raw_values.clone();
            constant.quantized_values = quantize_tensor(raw_values, *scale_max)?;
        }
    }
    Ok(())
}
