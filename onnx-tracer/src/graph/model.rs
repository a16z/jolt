use super::node::*;
use crate::{
    circuit::ops::{Input, Op, Unknown},
    fieldutils::felt_to_i128,
    graph::{
        input::GraphData,
        utilities::{node_output_shapes, scale_to_multiplier},
        vars::VarScales,
        GraphError,
    },
    tensor::Tensor,
    RunArgs,
};
use halo2curves::bn256::Fr as Fp;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
};
use tabled::Table;
use tract_onnx::{
    prelude::{
        tract_itertools::Itertools, Framework, Graph, InferenceFact, InferenceModelExt,
        SymbolValues, TypedFact, TypedOp,
    },
    tract_core::internal::DatumType,
    tract_hir::ops::scan::Scan,
};

/// A struct for loading from an Onnx file and converting a computational graph to a
/// circuit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub graph: ParsedNodes,
}

impl Model {
    /// Creates a `Model` from a specified path to an Onnx file.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn new(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> Self {
        let graph = Self::load_onnx_model(reader, run_args);
        let om = Model { graph };
        debug!("\n {}", om.table_nodes());
        om
    }

    /// Runs a forward pass on sample data !
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `model_inputs` - A vector of [Tensor]s to use as inputs to the model.
    /// * `run_args` - [RunArgs]
    pub fn forward(&self, model_inputs: &[Tensor<Fp>]) -> Result<ForwardResult, Box<dyn Error>> {
        let mut results: BTreeMap<&usize, Vec<Tensor<Fp>>> = BTreeMap::new();
        let mut max_lookup_inputs = 0;
        let mut min_lookup_inputs = 0;
        let input_shapes = self.graph.input_shapes()?;
        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            let mut input = model_inputs[i].clone();
            input.reshape(&input_shapes[i])?;
            results.insert(input_idx, vec![input]);
        }
        for (idx, n) in self.graph.nodes.iter() {
            let mut inputs = vec![];
            if n.is_input() {
                let t = results.get(idx).ok_or(GraphError::MissingResults)?[0].clone();
                inputs.push(t);
            } else {
                for (idx, outlet) in n.inputs().iter() {
                    match results.get(&idx) {
                        Some(value) => inputs.push(value[*outlet].clone()),
                        None => return Err(Box::new(GraphError::MissingNode(*idx))),
                    }
                }
            };
            debug!("executing {}: {}", idx, n.as_str());
            debug!("dims: {:?}", n.out_dims());
            debug!(
                "input_dims: {:?}",
                inputs.iter().map(|x| x.dims()).collect::<Vec<_>>()
            );
            if n.is_lookup() {
                let (mut min, mut max) = (0, 0);
                for i in &inputs {
                    max = max.max(
                        i.iter()
                            .map(|x| felt_to_i128(*x))
                            .max()
                            .ok_or("missing max")?,
                    );
                    min = min.min(
                        i.iter()
                            .map(|x| felt_to_i128(*x))
                            .min()
                            .ok_or("missing min")?,
                    );
                }
                max_lookup_inputs = max_lookup_inputs.max(max);
                min_lookup_inputs = min_lookup_inputs.min(min);
                debug!("max lookup inputs: {max}");
                debug!("min lookup inputs: {min}");
            }
            match n {
                NodeType::Node(n) => {
                    // execute the op
                    let start = instant::Instant::now();
                    let res = Op::<Fp>::f(&n.opkind, &inputs)?;
                    let elapsed = start.elapsed();
                    trace!("op took: {elapsed:?}",);
                    // see if any of the intermediate lookup calcs are the max
                    if !res.intermediate_lookups.is_empty() {
                        let (mut min, mut max) = (0, 0);
                        for i in &res.intermediate_lookups {
                            max = max.max(i.clone().into_iter().max().ok_or("missing max")?);
                            min = min.min(i.clone().into_iter().min().ok_or("missing min")?);
                        }
                        max_lookup_inputs = max_lookup_inputs.max(max);
                        min_lookup_inputs = min_lookup_inputs.min(min);
                        debug!("intermediate max lookup inputs: {max}",);
                        debug!("intermediate min lookup inputs: {min}",);
                    }
                    debug!(
                        "------------ output node int {}: {} \n ------------ float: {} \n ------------ max: {} \n ------------ min: {} ------------ scale: {}",
                        idx,
                        res.output.map(crate::fieldutils::felt_to_i32).show(),
                        res.output
                            .map(|x| crate::fieldutils::felt_to_f64(x)
                                / scale_to_multiplier(n.out_scale))
                            .show(),
                        res.output.clone().into_iter().map(crate::fieldutils::felt_to_i128).max().unwrap_or(0),
                        res.output.clone().into_iter().map(crate::fieldutils::felt_to_i128).min().unwrap_or(0),
                        n.out_scale
                    );
                    results.insert(idx, vec![res.output]);
                }
                NodeType::SubGraph {
                    model,
                    output_mappings,
                    input_mappings,
                    inputs: input_tuple,
                    ..
                } => {
                    let orig_inputs = inputs.clone();
                    let input_mappings = input_mappings.clone();

                    let input_dims = inputs.iter().map(|inp| inp.dims());
                    let num_iter = number_of_iterations(&input_mappings, input_dims.collect());
                    debug!(
                        "{} iteration(s) in a subgraph with inputs {:?} and sources {:?}",
                        num_iter, input_tuple, model.graph.inputs
                    );
                    debug!("input_mappings: {input_mappings:?}",);
                    let mut full_results: Vec<Tensor<Fp>> = vec![];
                    for i in 0..num_iter {
                        // replace the Stacked input with the current chunk iter
                        for ((mapping, inp), og_input) in
                            input_mappings.iter().zip(&mut inputs).zip(&orig_inputs)
                        {
                            if let InputMapping::Stacked { axis, chunk } = mapping {
                                let start = i * chunk;
                                let end = (i + 1) * chunk;
                                let t = crate::tensor::ops::slice(og_input, axis, &start, &end)?;
                                *inp = t;
                            }
                        }
                        let res = model.forward(&inputs)?;
                        // recursively get the max lookup inputs for subgraphs
                        max_lookup_inputs = max_lookup_inputs.max(res.max_lookup_inputs);
                        min_lookup_inputs = min_lookup_inputs.min(res.min_lookup_inputs);
                        let mut outlets = BTreeMap::new();
                        for (mappings, outlet_res) in output_mappings.iter().zip(res.outputs) {
                            for mapping in mappings {
                                match mapping {
                                    OutputMapping::Single { outlet, .. } => {
                                        outlets.insert(outlet, outlet_res.clone());
                                    }
                                    OutputMapping::Stacked { outlet, axis, .. } => {
                                        if !full_results.is_empty() {
                                            let stacked_res = crate::tensor::ops::concat(
                                                &[&full_results[*outlet], &outlet_res],
                                                *axis,
                                            )?;

                                            outlets.insert(outlet, stacked_res);
                                        } else {
                                            outlets.insert(outlet, outlet_res.clone());
                                        }
                                    }
                                }
                            }
                        }
                        full_results = outlets.into_values().collect_vec();
                        let output_states = output_state_idx(output_mappings);
                        let input_states = input_state_idx(&input_mappings);
                        assert_eq!(input_states.len(), output_states.len());
                        for (input_idx, output_idx) in input_states.iter().zip(output_states) {
                            inputs[*input_idx] = full_results[output_idx].clone();
                        }
                    }
                    trace!(
                        "------------ output subgraph node {}: {:?}",
                        idx,
                        full_results
                            .iter()
                            .map(|x|
                            // convert to tensor i32
                            x.map(crate::fieldutils::felt_to_i32).show())
                            .collect_vec()
                    );
                    results.insert(idx, full_results);
                }
            }
        }
        let output_nodes = self.graph.outputs.iter();
        debug!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let outputs = output_nodes
            .map(|(idx, outlet)| {
                Ok(results.get(&idx).ok_or(GraphError::MissingResults)?[*outlet].clone())
            })
            .collect::<Result<Vec<_>, GraphError>>()?;
        let res = ForwardResult {
            outputs,
            max_lookup_inputs,
            min_lookup_inputs,
        };
        Ok(res)
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    fn load_onnx_model(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> ParsedNodes {
        let start_time = instant::Instant::now();
        let (model, symbol_values) = Self::load_onnx_using_tract(reader, run_args);
        let scales = VarScales::from_args(run_args);
        let nodes = Self::nodes_from_graph(&model, run_args, &scales, &symbol_values, None, None);
        debug!("\n {model}",);
        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };
        let duration = start_time.elapsed();
        trace!("model loading took: {duration:?}",);
        parsed_nodes
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    fn load_onnx_using_tract(
        reader: &mut dyn std::io::Read,
        run_args: &RunArgs,
    ) -> (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues) {
        let mut model = tract_onnx::onnx().model_for_read(reader).unwrap();
        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());
        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();
            for (i, x) in fact.clone().shape.dims().enumerate() {
                use tract_onnx::tract_hir::infer::GenericFactoid;
                if matches!(x, GenericFactoid::Any) {
                    let batch_size = variables.get("batch_size").unwrap();
                    fact.shape
                        .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
                }
            }
            model.set_input_fact(i, fact).unwrap();
        }
        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default()).unwrap();
        }
        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            use log::info;
            let symbol = model.symbols.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            info!("set {symbol} to {value}");
            println!("set {symbol} to {value}");
        }

        // Note: do not optimize the model, as the layout will depend on
        // underlying hardware
        let mut typed_model = model
            .into_typed()
            .unwrap()
            .concretize_dims(&symbol_values)
            .unwrap()
            .into_decluttered()
            .unwrap();
        // concretize constants
        for node in typed_model.eval_order().unwrap() {
            let node = typed_model.node_mut(node);
            if node.op_is::<tract_onnx::tract_hir::ops::konst::Const>() {
                // map option to err
                let op = node
                    .op_as_mut::<tract_onnx::tract_hir::ops::konst::Const>()
                    .unwrap();
                // get inner value to Arc<Tensor>
                let constant = op.0.as_ref();
                if constant.datum_type() == DatumType::TDim {
                    // Generally a shape or hyperparam
                    use tract_onnx::prelude::TDim;
                    let vec = constant
                        .as_slice::<tract_onnx::prelude::TDim>()
                        .unwrap()
                        .to_vec();
                    let data: Vec<TDim> = vec.into_iter().map(|x| x.eval(&symbol_values)).collect();
                    unsafe {
                        let bytes = std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * DatumType::TDim.size_of(),
                        );
                        op.0 = std::sync::Arc::new(
                            tract_onnx::prelude::Tensor::from_raw_dt(
                                DatumType::TDim,
                                constant.shape(),
                                bytes,
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
        (typed_model, symbol_values)
    }

    /// Creates ezkl nodes from a tract graph
    /// # Arguments
    /// * `graph` - A tract graph.
    /// * `run_args` - [RunArgs]
    /// * `visibility` - Which inputs to the model are public and private (params,
    // inputs, outputs) using [VarVisibility].
    /// * `input_scales` - The scales of
    // the model's inputs.
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        _run_args: &RunArgs,
        scales: &VarScales,
        symbol_values: &SymbolValues,
        override_input_scales: Option<Vec<crate::Scale>>,
        override_output_scales: Option<HashMap<usize, crate::Scale>>,
    ) -> BTreeMap<usize, NodeType> {
        // use crate::graph::node_output_shapes;
        let mut nodes = BTreeMap::<usize, NodeType>::new();
        let mut input_idx = 0;
        for (i, n) in graph.nodes.iter().enumerate() {
            // Extract the slope layer hyperparams
            match n.op().downcast_ref::<Scan>() {
                Some(b) => {
                    let model = b.body.clone();
                    let input_scales = n
                        .inputs
                        .iter()
                        .map(|i| nodes.get(&i.node).unwrap().out_scales()[0])
                        .collect::<Vec<_>>();
                    let mut input_mappings = vec![];
                    for mapping in &b.input_mapping {
                        match mapping {
                            tract_onnx::tract_hir::ops::scan::InputMapping::Scan(info) => {
                                input_mappings.push(InputMapping::Stacked {
                                    axis: info.axis,
                                    chunk: info.chunk as usize,
                                });
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::State => {
                                input_mappings.push(InputMapping::State);
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::Full => {
                                input_mappings.push(InputMapping::Full);
                            }
                        }
                    }
                    let input_state_idx = input_state_idx(&input_mappings);
                    let mut output_mappings = vec![];
                    for mapping in b.output_mapping.iter() {
                        let mut mappings = vec![];
                        if let Some(outlet) = mapping.last_value_slot {
                            mappings.push(OutputMapping::Single {
                                outlet,
                                is_state: mapping.state,
                            });
                        }
                        if let Some(last) = mapping.scan {
                            mappings.push(OutputMapping::Stacked {
                                outlet: last.0,
                                axis: last.1.axis,
                                is_state: false,
                            });
                        }
                        output_mappings.push(mappings);
                    }
                    let output_state_idx = output_state_idx(&output_mappings);
                    let mut output_scale_override = HashMap::new();
                    // if input_state_idx and output_state_idx have mismatched
                    // scales we need to rebase the scale of the output node
                    for (input_idx, output_idx) in input_state_idx.iter().zip(output_state_idx) {
                        let input_scale = input_scales[*input_idx]; // output mappings is a vec of vec. we need to find
                                                                    // the outer index of the output node  we want to rebase.
                        let mut traversed_len = 0;
                        for (outer_idx, mappings) in output_mappings.iter().enumerate() {
                            let mapping_len = mappings.len();
                            if traversed_len + mapping_len > output_idx {
                                let output_node_idx = b.body.outputs[outer_idx].node;
                                output_scale_override.insert(output_node_idx, input_scale);
                            }
                            traversed_len += mapping_len;
                        }
                    }
                    let subgraph_nodes = Self::nodes_from_graph(
                        &model,
                        _run_args,
                        scales,
                        symbol_values,
                        Some(input_scales.clone()),
                        Some(output_scale_override),
                    );
                    let subgraph = ParsedNodes {
                        nodes: subgraph_nodes,
                        inputs: model.inputs.iter().map(|o| o.node).collect(),
                        outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
                    };
                    let om = Model { graph: subgraph };
                    let out_dims = node_output_shapes(n, symbol_values).unwrap();
                    let mut output_scales = BTreeMap::new();
                    for (i, _mapping) in b.output_mapping.iter().enumerate() {
                        for mapping in b.output_mapping.iter() {
                            if let Some(outlet) = mapping.last_value_slot {
                                output_scales.insert(outlet, om.graph.get_output_scales()[i]);
                            }
                            if let Some(last) = mapping.scan {
                                output_scales.insert(last.0, om.graph.get_output_scales()[i]);
                            }
                        }
                    }
                    let out_scales = output_scales.into_values().collect_vec();
                    nodes.insert(
                        i,
                        NodeType::SubGraph {
                            model: om,
                            inputs: n.inputs.iter().map(|i| (i.node, i.slot)).collect_vec(),
                            idx: i,
                            output_mappings,
                            input_mappings,
                            out_dims,
                            out_scales,
                        },
                    );
                }
                None => {
                    let mut n = Node::new(n.clone(), &mut nodes, scales, i, symbol_values);
                    if let Some(ref scales) = override_input_scales {
                        if let Some(inp) = n.opkind.get_input() {
                            let scale = scales[input_idx];
                            n.opkind = SupportedOp::Input(Input {
                                scale,
                                datum_type: inp.datum_type,
                            });
                            input_idx += 1;
                            n.out_scale = scale;
                        }
                    }
                    if let Some(ref scales) = override_output_scales {
                        if scales.contains_key(&i) {
                            let scale_diff = n.out_scale - scales[&i];
                            n.opkind = if scale_diff > 0 {
                                RebaseScale::rebase(n.opkind, scales[&i], n.out_scale, 1)
                            } else {
                                RebaseScale::rebase_up(n.opkind, scales[&i], n.out_scale)
                            };
                            n.out_scale = scales[&i];
                        }
                    }
                    nodes.insert(i, NodeType::Node(n));
                }
            }
        }
        Self::remove_unused_nodes(&mut nodes);
        nodes
    }

    /// Run tract onnx model on sample data !
    pub fn run_onnx_predictions(
        run_args: &RunArgs,
        model_path: &std::path::Path,
        data_chunks: &[GraphData],
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<Tensor<f32>>>, Box<dyn Error>> {
        use tract_onnx::tract_core::internal::IntoArcTensor;
        let (model, _) = Model::load_onnx_using_tract(
            &mut std::fs::File::open(model_path)
                .map_err(|_| format!("failed to load model at {}", model_path.display()))?,
            run_args,
        );
        let datum_types: Vec<DatumType> = model
            .input_outlets()?
            .iter()
            .map(|o| model.node(o.node).outputs[o.slot].fact.datum_type)
            .collect();
        let runnable_model = model.into_runnable()?;
        let mut outputs = vec![];
        for chunk in data_chunks {
            let result = runnable_model.run(chunk.to_tract_data(&input_shapes, &datum_types)?)?;
            outputs.push(
                result
                    .into_iter()
                    .map(|t| {
                        crate::graph::utilities::extract_tensor_value(t.into_arc_tensor()).unwrap()
                    })
                    .collect(),
            );
        }
        Ok(outputs)
    }

    /// Removes all nodes that are consts with 0 uses
    fn remove_unused_nodes(nodes: &mut BTreeMap<usize, NodeType>) {
        // remove all nodes that are consts with 0 uses now
        nodes.retain(|_, n| match n {
            NodeType::Node(n) => match &mut n.opkind {
                SupportedOp::Constant(c) => {
                    c.empty_raw_value();
                    n.num_uses > 0
                }
                _ => n.num_uses > 0,
            },
            NodeType::SubGraph { model, .. } => {
                Self::remove_unused_nodes(&mut model.graph.nodes);
                true
            }
        });
    }

    pub fn table_nodes(&self) -> String {
        let mut node_accumulator = vec![];
        let mut string = String::new();
        for (idx, node) in &self.graph.nodes {
            match node {
                NodeType::Node(n) => {
                    node_accumulator.push(n);
                }
                NodeType::SubGraph { model, inputs, .. } => {
                    let mut table = Table::new(node_accumulator.iter());
                    table.with(tabled::settings::Style::modern());
                    table.with(tabled::settings::Shadow::new(1));
                    table.with(
                        tabled::settings::style::BorderColor::default()
                            .top(tabled::settings::Color::BG_YELLOW),
                    );
                    string = format!("{} \n\n  MAIN GRAPH \n\n{}", string, table);
                    node_accumulator = vec![];
                    string = format!(
                        "{}\n\n SUBGRAPH AT IDX {} WITH INPUTS {:?}\n{}",
                        string,
                        idx,
                        inputs,
                        model.table_nodes(),
                    );
                }
            }
        }
        let mut table = Table::new(node_accumulator.iter());
        table.with(tabled::settings::Style::modern());
        format!("{} \n{}", string, table)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
/// A set of ONNX nodes that represent a computational graph.
pub struct ParsedNodes {
    /// The nodes in the graph.
    pub nodes: BTreeMap<usize, NodeType>,
    inputs: Vec<usize>,
    outputs: Vec<Outlet>,
}

impl ParsedNodes {
    /// Returns the fixed point scale of the computational graph's inputs
    pub fn get_input_scales(&self) -> Vec<crate::Scale> {
        let input_nodes = self.inputs.iter();
        input_nodes
            .flat_map(|idx| {
                self.nodes
                    .get(idx)
                    .ok_or(GraphError::MissingNode(*idx))
                    .map(|n| n.out_scales())
                    .unwrap_or_default()
            })
            .collect()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let mut inputs = vec![];

        for input in self.inputs.iter() {
            let node = self
                .nodes
                .get(input)
                .ok_or(GraphError::MissingNode(*input))?;
            let input_dims = node.out_dims();
            let input_dim = input_dims.first().unwrap();
            inputs.push(input_dim.clone());
        }

        Ok(inputs)
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<crate::Scale> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .map(|(idx, outlet)| self.nodes.get(idx).unwrap().out_scales()[*outlet])
            .collect::<Vec<_>>()
    }
}

// /// Enables model as subnode of other models
#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    /// A node in the model
    Node(Node),
    /// A submodel
    SubGraph {
        /// The subgraph
        model: Model,
        /// The subgraph's inputs
        inputs: Vec<Outlet>,
        /// the subgraph's idx within the parent graph
        idx: usize,
        /// output mappings
        output_mappings: Vec<Vec<OutputMapping>>,
        /// input mappings
        input_mappings: Vec<InputMapping>,
        ///
        out_dims: Vec<Vec<usize>>,
        ///
        out_scales: Vec<crate::Scale>,
    },
}

impl NodeType {
    pub fn is_lookup(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_lookup(),
            NodeType::SubGraph { .. } => false,
        }
    }

    pub fn num_uses(&self) -> usize {
        match self {
            NodeType::Node(n) => n.num_uses,
            NodeType::SubGraph { .. } => 0,
        }
    }

    /// Returns the indices of the node's inputs.
    pub fn inputs(&self) -> Vec<Outlet> {
        match self {
            NodeType::Node(n) => n.inputs.clone(),
            NodeType::SubGraph { inputs, .. } => inputs.clone(),
        }
    }

    /// Returns the dimensions of the node's output.
    pub fn out_dims(&self) -> Vec<Vec<usize>> {
        match self {
            NodeType::Node(n) => vec![n.out_dims.clone()],
            NodeType::SubGraph { out_dims, .. } => out_dims.clone(),
        }
    }

    /// Returns the scales of the node's output.
    pub fn out_scales(&self) -> Vec<crate::Scale> {
        match self {
            NodeType::Node(n) => vec![n.out_scale],
            NodeType::SubGraph { out_scales, .. } => out_scales.clone(),
        }
    }

    /// Returns a string representation of the operation.
    pub fn as_str(&self) -> String {
        match self {
            NodeType::Node(n) => n.opkind.as_string(),
            NodeType::SubGraph { .. } => "SUBGRAPH".into(),
        }
    }

    /// Returns true if the operation is a rebase
    pub fn is_rebase(&self) -> bool {
        match self {
            NodeType::Node(n) => matches!(n.opkind, SupportedOp::RebaseScale { .. }),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns true if the operation is an input.
    pub fn is_input(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_input(),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns true if the operation is a const.
    pub fn is_constant(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_constant(),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns the node's unique identifier.
    pub fn idx(&self) -> usize {
        match self {
            NodeType::Node(n) => n.idx,
            NodeType::SubGraph { idx, .. } => *idx,
        }
    }

    /// decrement const num times used
    pub fn decrement_use(&mut self) {
        match self {
            NodeType::Node(n) => n.num_uses -= 1,
            NodeType::SubGraph { .. } => log::warn!(
                "Cannot decrement const of
subgraph"
            ),
        }
    }

    /// bunp scale of node
    pub fn bump_scale(&mut self, scale: crate::Scale) {
        match self {
            NodeType::Node(n) => n.out_scale = scale,
            NodeType::SubGraph { .. } => log::warn!("Cannot bump scale of subgraph"),
        }
    }

    /// Replace the operation kind of the node.
    pub fn replace_opkind(&mut self, opkind: SupportedOp) {
        match self {
            NodeType::Node(n) => n.opkind = opkind,
            NodeType::SubGraph { .. } => log::warn!("Cannot replace opkind of subgraph"),
        }
    }

    /// Returns the operation kind of the node (if any).
    pub fn opkind(&self) -> SupportedOp {
        match self {
            NodeType::Node(n) => n.opkind.clone(),
            NodeType::SubGraph { .. } => SupportedOp::Unknown(Unknown),
        }
    }

    //   /// Returns the lookups required by a graph
    //   pub fn required_lookups(&self) -> Vec<LookupOp> {
    //     match self {
    //       NodeType::Node(n) => n.opkind.required_lookups(),
    //       NodeType::SubGraph { model, .. } => model.required_lookups(),
    //     }
    //   }
}

/// The result of a forward pass.
#[derive(Clone, Debug)]
pub struct ForwardResult {
    /// The outputs of the forward pass.
    pub outputs: Vec<Tensor<Fp>>,
    /// The maximum value of any input to a lookup operation.
    pub max_lookup_inputs: i128,
    /// The minimum value of any input to a lookup operation.
    pub min_lookup_inputs: i128,
}

/// Representation of execution graph
pub type NodeGraph = BTreeMap<usize, NodeType>;

///
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    ///
    Single {
        ///
        outlet: usize,
        ///
        is_state: bool,
    },
    ///
    Stacked {
        ///
        outlet: usize,
        ///
        axis: usize,
        ///
        is_state: bool,
    },
}

impl OutputMapping {
    ///
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    ///
    pub fn outlet(&self) -> usize {
        match self {
            OutputMapping::Single { outlet, .. } => *outlet,
            OutputMapping::Stacked { outlet, .. } => *outlet,
        }
    }
}

/// Describes how each input to a subgraph (such as a Scan/Loop in ONNX) is mapped from the parent graph.
///
/// # Overview
/// `InputMapping` is used to specify how the inputs to a subgraph are fed from the outer graph.
/// This is essential for handling ONNX control flow operators like Scan, Loop, or custom subgraphs,
/// where each input may be treated differently (e.g., as a state variable, as a full tensor, or as a chunked/stacked input).
///
/// # Variants
/// - `Full`: The entire input tensor is passed as-is to the subgraph input.
/// - `State`: The input acts as a state variable, typically carried across iterations (e.g., hidden state in RNNs).
/// - `Stacked { axis, chunk }`: The input is split along the specified `axis` into chunks of size `chunk`,
///   and each chunk is fed to the subgraph in each iteration (used for sequence processing).
///
/// # Role in the Codebase
/// `InputMapping` is crucial for correctly wiring up subgraphs during model parsing and execution.
/// It allows the code to handle dynamic and complex ONNX models that use control flow or iterative constructs,
/// ensuring that data is fed into subgraphs in the correct shape and order. This enables support for models
/// with recurrent or iterative computation patterns.
///
/// # Example
/// For an ONNX Scan node processing a sequence, the input sequence tensor might be mapped as `Stacked`,
/// while an initial hidden state would be mapped as `State`.
///
/// # Usage
/// The mapping is determined during graph parsing (`nodes_from_graph`) and is later used during execution
/// (`forward`) to slice, reshape, or carry inputs as needed for each subgraph invocation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputMapping {
    ///
    Full,
    ///
    State,
    ///
    Stacked {
        ///
        axis: usize,
        ///
        chunk: usize,
    },
}

fn number_of_iterations(mappings: &[InputMapping], dims: Vec<&[usize]>) -> usize {
    let mut number_of_iterations =
        dims.iter()
            .zip(mappings)
            .filter_map(|(dims, mapping)| match mapping {
                InputMapping::Stacked { axis, chunk } => Some(
                    // number of iterations given the dim size along the axis
                    // and the chunk size
                    dims[*axis].div_ceil(*chunk),
                ),
                _ => None,
            });
    // assert all collected number of iterations are equal
    assert!(number_of_iterations.clone().all_equal());

    number_of_iterations.next().unwrap_or(1)
}

fn input_state_idx(input_mappings: &[InputMapping]) -> Vec<usize> {
    input_mappings
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r, InputMapping::State))
        .map(|(index, _)| index)
        .collect::<Vec<_>>()
}

fn output_state_idx(output_mappings: &[Vec<OutputMapping>]) -> Vec<usize> {
    output_mappings
        .iter()
        .flatten()
        .filter_map(|x| if x.is_state() { Some(x.outlet()) } else { None })
        .collect::<Vec<_>>()
}
