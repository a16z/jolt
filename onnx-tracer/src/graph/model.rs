use super::node::*;
use crate::{
    circuit::ops::{Input, Op, Unknown},
    decode_node,
    graph::{
        input::GraphData, tracer::Tracer, utilities::node_output_shapes, vars::VarScales,
        GraphError,
    },
    tensor::Tensor,
    RunArgs,
};
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
    pub tracer: Tracer,
}

impl Model {
    /// Creates a `Model` from a specified path to an Onnx file.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn new(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> Self {
        let graph = Self::load_onnx_model(reader, run_args);
        let om = Model {
            graph,
            tracer: Tracer::default(),
        };
        debug!("\n {}", om.table_nodes());
        om
    }

    /// Executes a forward pass through the parsed ONNX model using provided input tensors.
    ///
    /// # Purpose
    /// This function simulates running the ONNX model on input data, producing the model's outputs
    /// as if it were being executed in a standard inference engine. It is essential for testing,
    /// debugging, and validating the model conversion pipeline, as well as for extracting
    /// intermediate and final outputs for further processing or verification.
    ///
    /// # Arguments
    /// * `model_inputs` - A slice of [`Tensor<Fp>`] representing the input data for the model.
    ///   Each tensor in this slice should correspond to one of the model's input nodes, and must
    ///   have the correct shape and data type expected by the model. The order of tensors must
    ///   match the order of the model's input nodes.
    ///
    /// # Returns
    /// Returns a [`Result<ForwardResult, Box<dyn Error>>`] where:
    /// - `ForwardResult` contains:
    ///     - `outputs`: The output tensors produced by the model (in the order of the model's outputs).
    ///     - `max_lookup_inputs`/`min_lookup_inputs`: The maximum and minimum values encountered as inputs to any lookup operation during execution (useful for quantization or table sizing).
    /// - If an error occurs (e.g., shape mismatch, missing node, or execution failure), returns an error describing the issue.
    ///
    /// # How It Works (Step by Step)
    /// 1. **Prepare Results Map:** Initializes a map to store the output tensors of each node as the graph is executed.
    /// 2. **Reshape and Insert Inputs:** Reshapes each provided input tensor to match the expected input shape, and inserts it into the results map keyed by the input node index.
    /// 3. **Node Execution Loop:** Iterates through each node in the graph in topological order:
    ///     - Gathers the required input tensors for the node from the results map.
    ///     - Executes the node's operation (or recursively executes subgraphs for control flow nodes).
    ///     - Tracks min/max values for lookup operations.
    ///     - Stores the node's output(s) in the results map.
    /// 4. **Collect Outputs:** After all nodes have been executed, collects the output tensors corresponding to the model's output nodes.
    /// 5. **Return Results:** Packages the outputs and lookup statistics into a `ForwardResult` and returns it.
    ///
    /// # When to Use
    /// Use this function whenever you need to:
    /// - Simulate model inference on sample data (e.g., for testing or debugging).
    /// - Validate that the model conversion from ONNX to the internal graph representation is correct.
    /// - Extract intermediate or final outputs for further analysis.
    ///
    /// # Example Usage
    /// ```ignore
    /// let model = Model::new(&mut onnx_file, &run_args);
    /// let input_tensors = vec![...]; // Prepare input tensors matching model's input shapes
    /// let result = model.forward(&input_tensors)?;
    /// println!("Model outputs: {:?}", result.outputs);
    /// ```
    ///
    /// # Notes
    /// - Input tensors must be in the correct order and shape.
    /// - This function does not perform any hardware-accelerated inference; it executes the model using the internal Rust implementation.
    /// - Handles both standard nodes and subgraphs (e.g., for ONNX Scan/Loop constructs).
    pub fn forward(&self, model_inputs: &[Tensor<i128>]) -> Result<ForwardResult, Box<dyn Error>> {
        // A map that stores the output tensors of each node in the computation graph.
        //
        // # Purpose
        // `results` is used to keep track of the intermediate and final outputs produced by each node
        // (identified by their unique index) during the execution of the model. The key is a reference to
        // the node's index (`&usize`), and the value is a vector of `Tensor<Fp>`, representing the output
        // tensors generated by that node.
        //
        // # Why we need `results`
        // In a computational graph, nodes may depend on the outputs of previous nodes. By storing the
        // results in a `BTreeMap`, we can efficiently retrieve the outputs of any node as needed for
        // subsequent computations. This structure also ensures deterministic iteration order, which can be
        // important for reproducibility and debugging.
        //
        // # Usage
        // - When a node is executed, its output tensors are inserted into `results` under its index.
        // - When another node requires the output of a previous node, it can look up the corresponding
        //   entry in `results`.
        // - After the entire graph has been executed, `results` contains the outputs of all nodes, which
        //   can be used for further processing or for extracting the final model outputs.
        //
        // DESIGN NOTE: Why use `BTreeMap<&usize, Vec<Tensor<Fp>>>` for `results`?
        //
        // 1. Why a BTreeMap?
        //    - Deterministic ordering: BTreeMap iterates in sorted order, which helps with reproducibility and debugging.
        //    - Efficient lookup: We need to quickly retrieve the output(s) of any node by its index.
        //    - Nodes are indexed by their unique usize index in the graph.
        //
        // 2. Why does the value type store a Vec<Tensor<Fp>> instead of just Tensor<Fp>?
        //    - Some nodes (especially subgraphs or nodes with multiple outputs) can produce multiple output tensors.
        //    - The Vec allows us to store all outputs for a node, indexed by their outlet (output slot).
        //    - For most nodes, this Vec will have a single element, but for nodes with multiple outputs, each output is stored at its respective index.
        //
        // 3. How do you use this map to get the output of a node?
        //    - To get the output tensor(s) of node 10, you would do:
        //        `let outputs = results.get(&10);`
        //      - If you want the first output (most common case): `let output = &outputs[0];`
        //      - If the node has multiple outputs, you can access them by their outlet index: `let output = &outputs[outlet_index];`
        //
        // 4. Why is the key a reference (&usize) instead of just usize?
        //    - This is because we often have references to node indices from elsewhere in the graph, and using references avoids unnecessary copies.
        //    - BTreeMap allows lookups with either &usize or usize, but here the code is consistent with using references.
        //
        // 5. Summary:
        //    - This structure allows us to efficiently store and retrieve all intermediate and final outputs of the computation graph,
        //      supporting both single-output and multi-output nodes, and ensuring deterministic iteration order.
        let mut results: BTreeMap<&usize, Vec<Tensor<i128>>> = BTreeMap::new();
        let mut max_lookup_inputs = 0;
        let mut min_lookup_inputs = 0;
        // Retrieves the shapes of all input tensors for the current computational graph.
        //
        // # Intent
        // This line obtains the dimensions (shapes) of each input tensor that the model expects,
        // which is essential for validating input data, constructing subsequent layers, and ensuring
        // compatibility throughout the model's execution.
        //
        // # Why it's needed
        // Knowing the input shapes is crucial for tasks such as input validation, dynamic graph construction,
        // and for informing downstream operations about the expected data structure. It helps prevent
        // runtime errors due to shape mismatches and is often required when exporting, tracing, or
        // transforming the model.
        //
        // # How it's used
        // The returned `input_shapes` value is typically used to:
        // - Validate that provided input data matches the model's requirements.
        // - Dynamically allocate memory or buffers for inputs.
        // - Inform other components or tools (e.g., ONNX exporters, tracers) about the model's input signature.
        //
        // # When it's used
        // This is usually called during model initialization, tracing, or before running inference,
        // whenever the input specification of the model needs to be known or verified.
        let input_shapes = self.graph.input_shapes()?;
        // Insert model inputs into the results map after reshaping them to match expected input shapes.
        //
        // # Why this code is needed
        // The model's forward execution relies on a map (`results`) that holds the output tensors of each node.
        // For input nodes, we must provide the actual input tensors supplied by the user, but these may need to be reshaped
        // to match the expected dimensions as defined by the model (e.g., to handle batch size or symbolic shapes).
        //
        // # Intent
        // This code ensures that each input tensor provided to the model is reshaped to the correct shape and then
        // inserted into the results map under the corresponding input node index. This allows subsequent nodes in the
        // computation graph to retrieve the correct input data during execution.
        //
        // # How it works
        // - Iterates over all input node indices and their corresponding position in the input tensor list.
        // - For each input:
        //     - Clones the provided input tensor.
        //     - Reshapes it to match the expected shape for that input node.
        //     - Inserts the reshaped tensor into the `results` map under the input node's index.
        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            let mut input = model_inputs[i].clone();
            input.reshape(&input_shapes[i])?;
            results.insert(input_idx, vec![input]);
        }

        // --- Fetch Decode Execute ---
        for (idx, n) in self.graph.nodes.iter() {
            // Fetch and Decode
            let mut inputs = Self::node_inputs(idx, n, &results)?;
            let instr = decode_node((idx, n));
            self.tracer.capture_pre_state(instr.clone(), inputs.clone());
            if n.is_lookup() {
                Self::lookup_check(&inputs, &mut max_lookup_inputs, &mut min_lookup_inputs)?;
            }
            match n {
                NodeType::Node(n) => {
                    // Execute
                    let res = Op::<i128>::f(&n.opkind, &inputs)?;
                    debug!("opkind: {:#?}, instr: {instr:#?}, res: {res:#?}", n.opkind);
                    // see if any of the intermediate lookup calcs are the max
                    if !res.intermediate_lookups.is_empty() {
                        Self::lookup_check(
                            &res.intermediate_lookups,
                            &mut max_lookup_inputs,
                            &mut min_lookup_inputs,
                        )?;
                    }
                    debug!(
                        "------------ output node int {}: {} \n  ------------ scale: {}",
                        idx,
                        res.output.show(),
                        n.out_scale
                    );
                    results.insert(idx, vec![res.output.clone()]);
                    self.tracer.capture_post_state(res.output);
                }
                // --- SubGraph Node Execution ---
                //
                // This block handles the execution of a subgraph node (NodeType::SubGraph), which is fundamentally
                // different from executing a standard node (NodeType::Node). Subgraphs are used for control flow
                // constructs like ONNX Scan, Loop, or custom nested models, where a portion of the graph is executed
                // multiple times with varying inputs (e.g., sequence processing, RNNs).
                //
                // Why is this needed?
                // - Standard nodes perform a single computation given their inputs.
                // - Subgraph nodes encapsulate an entire model that must be executed repeatedly, often with sliced or
                //   stateful inputs, and may have complex input/output mappings.
                // - This code ensures correct iteration, input slicing, state management, and output collection for
                //   subgraph execution.
                //
                // Intent & Purpose:
                // - To simulate the iterative execution of a subgraph as required by ONNX control flow semantics.
                // - To correctly map parent graph inputs to subgraph inputs, handle state variables, and collect outputs.
                // - To recursively execute the subgraph and aggregate results, including lookup statistics.
                //
                // How it works (Step-by-Step):
                // 1. Clone the original inputs and input mappings for reference.
                // 2. Determine the number of iterations required by inspecting the input mappings and dimensions.
                //    (For example, if an input is chunked along an axis, the number of iterations is dim_size / chunk_size.)
                // 3. For each iteration:
                //    a. Slice or update the inputs as specified by the input mappings (e.g., Stacked inputs get a chunk).
                //    b. Recursively call `model.forward(&inputs)` to execute the subgraph for this iteration.
                //    c. Track min/max lookup values from subgraph execution for quantization/table sizing.
                //    d. Map subgraph outputs back to parent graph outputs using output mappings, handling stacking
                //       (concatenation) and state variables.
                //    e. Update stateful inputs for the next iteration using output states.
                // 4. After all iterations, insert the aggregated outputs into the parent graph's results map.
                //
                // Key differences from normal node execution:
                // - Iterative: Subgraphs may execute multiple times, while normal nodes execute once.
                // - Input/Output Mapping: Inputs/outputs may be sliced, stacked, or carried as state, requiring complex mapping.
                // - Recursion: Subgraph execution is recursive, calling `forward` on the sub-model.
                // - State Management: Handles state variables that persist across iterations.
                // - Output Aggregation: May need to concatenate outputs across iterations (stacked outputs).
                //
                // Gotchas:
                // - Input slicing must match the expected chunk size and axis; mismatches can cause runtime errors.
                // - State variables must be correctly updated between iterations.
                // - Output mappings can be complex; ensure correct outlet and axis handling.
                // - Recursion can lead to stack overflows if subgraphs are deeply nested.
                //
                // Summary:
                // This code is essential for supporting ONNX models with control flow, enabling correct execution of
                // iterative constructs and nested graphs. It ensures that subgraph semantics are faithfully reproduced,
                // including input slicing, state management, and output aggregation.
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
                    let mut full_results: Vec<Tensor<i128>> = vec![];
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
                        full_results.iter().map(|x| x.show()).collect_vec()
                    );
                    results.insert(idx, full_results);
                }
            }
        }
        // Collects the output tensors of the model from the results map.
        //
        // # Why we need this code
        // After executing all nodes in the computational graph, we need to extract the final outputs of the model.
        // The model's outputs are defined as specific nodes (and their output slots) in the graph.
        // This code gathers those outputs from the `results` map, which contains the outputs of every node.
        //
        // # What it does
        // - Iterates over the list of output node indices and outlet slots (`self.graph.outputs`).
        // - For each output, retrieves the corresponding tensor from the `results` map.
        // - Collects all output tensors into a vector, preserving the order defined by the model's outputs.
        // - Wraps the outputs and lookup statistics into a `ForwardResult` struct.
        //
        // # How it works
        // - Uses `.map()` to iterate over each output node and outlet.
        // - Looks up the node's outputs in the `results` map using the node index.
        // - Selects the correct output tensor by indexing into the vector with the outlet index.
        // - Handles missing results by returning a `GraphError`.
        // - Collects all outputs into a `Vec<Tensor<Fp>>`.
        //
        // # Intent
        // The intent is to provide the user with the final outputs of the model in the correct order,
        // as well as any statistics (such as min/max lookup inputs) gathered during execution.
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
        Ok(ForwardResult {
            outputs,
            max_lookup_inputs,
            min_lookup_inputs,
        })
    }

    /// Gathers the input tensors required for the current node's execution.
    ///
    /// # Intent
    /// This code block prepares the list of input tensors (`inputs`) that will be fed into the current node's operation.
    /// For each node in the graph, we must collect its input tensors from the results of previously executed nodes.
    ///
    /// # Why this code is needed
    /// In a computational graph, each node may depend on the outputs of other nodes (its inputs).
    /// Before executing a node, we must gather all its required input tensors in the correct order.
    /// This ensures that the node receives the correct data for computation, and is essential for correct model execution.
    ///
    /// # How it works
    /// - Initializes an empty `inputs` vector.
    /// - If the current node is an input node, retrieves its tensor from the `results` map.
    /// - Otherwise, iterates over the node's input connections (each a tuple of node index and outlet).
    ///   For each input:
    ///     - Looks up the output tensor of the source node from the `results` map.
    ///     - Pushes the required output (by outlet index) into the `inputs` vector.
    ///   - If any required input is missing, returns an error.
    ///
    /// # What it does
    /// After this block, `inputs` contains the tensors that should be passed to the current node's operation,
    /// in the order expected by the node. This enables the subsequent execution of the node's computation.
    fn node_inputs(
        idx: &usize,
        n: &NodeType,
        results: &BTreeMap<&usize, Vec<Tensor<i128>>>,
    ) -> Result<Vec<Tensor<i128>>, Box<dyn Error>> {
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
        Ok(inputs)
    }

    fn lookup_check(
        inputs: &[Tensor<i128>],
        max_lookup_inputs: &mut i128,
        min_lookup_inputs: &mut i128,
    ) -> Result<(), Box<dyn Error>> {
        let (mut min, mut max) = (0, 0);
        for i in inputs {
            max = max.max(i.iter().copied().max().ok_or("missing max")?);
            min = min.min(i.iter().copied().min().ok_or("missing min")?);
        }
        *max_lookup_inputs = (*max_lookup_inputs).max(max);
        *min_lookup_inputs = (*min_lookup_inputs).min(min);
        debug!("max lookup inputs: {max}");
        debug!("min lookup inputs: {min}");
        Ok(())
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
                    let om = Model {
                        graph: subgraph,
                        tracer: Tracer::default(),
                    }; // TODO: Figure out tracing for subgraphs
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
                    string = format!("{string} \n\n  MAIN GRAPH \n\n{table}",);
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
        format!("{string} \n{table}",)
    }

    pub fn add_node(
        &mut self,
        op: SupportedOp,
        inputs: Vec<Outlet>,
        out_dims: Vec<usize>,
    ) -> Result<usize, Box<dyn Error>> {
        let node_id = (0..self.graph.nodes.len() + 1)
            .find(|i| !self.graph.nodes.contains_key(i))
            .ok_or(GraphError::MissingNode(0))?;
        self.graph.nodes.insert(
            node_id,
            NodeType::Node(op.gen_node(inputs, out_dims, node_id)),
        );
        Ok(node_id)
    }

    pub fn add_inputs(&mut self, inputs: Vec<usize>) {
        self.graph.inputs.extend(inputs);
    }

    pub fn add_outputs(&mut self, outputs: Vec<Outlet>) {
        self.graph.outputs.extend(outputs);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
/// Represents a parsed computational graph consisting of ONNX nodes, inputs, and outputs.
///
/// `ParsedNodes` is the core data structure used to describe the internal representation of a computational graph
/// after it has been loaded and converted from an ONNX model. It contains all the nodes (operations and subgraphs),
/// as well as metadata about which nodes are considered inputs and outputs. This struct is central to the execution,
/// transformation, and analysis of models within this module.
///
/// - After loading an ONNX model, the graph is parsed into a `ParsedNodes` instance.
/// - The [`Model`] struct holds a `ParsedNodes` as its main graph representation.
/// - During model execution (inference), the `inputs` field is used to map user-provided tensors to the correct nodes,
///   and the `outputs` field is used to extract the final results after computation.
/// - When exporting, tracing, or analyzing the model, `ParsedNodes` provides a complete and queryable view of the graph structure.
///
/// - Encapsulates the entire computational graph, including all nodes and their relationships.
/// - Clearly separates the graph's structure (nodes) from its entry points (inputs) and exit points (outputs).
/// - Enables flexible manipulation, traversal, and execution of the graph for various purposes (inference, optimization, etc.).
/// - Supports both simple models and complex graphs with subgraphs (e.g., for control flow).
///
/// # Example
/// ```ignore
/// // Construct a simple graph: input -> add (with a constant) -> output
/// let mut nodes = BTreeMap::new();
/// nodes.insert(0, NodeType::Node(input_node));
/// nodes.insert(1, NodeType::Node(const_node));
/// nodes.insert(2, NodeType::Node(add_node));
///
/// let parsed = ParsedNodes {
///     nodes,
///     inputs: vec![0],           // Node 0 is the input node
///     outputs: vec![(2, 0)],     // Node 2's output (outlet 0) is the model output
/// };
///
/// // Use in a Model:
/// let model = Model { graph: parsed, tracer: Tracer::default() };
/// let result = model.forward(&[input_tensor]).unwrap();
/// println!("Model output: {:?}", result.outputs);
/// ```
pub struct ParsedNodes {
    /// The nodes in the graph.
    pub nodes: BTreeMap<usize, NodeType>,
    /// Indices of the input nodes for this computational graph.
    ///
    /// # Why we need this field
    /// This field specifies which nodes in the graph are considered as inputs—i.e., the entry points where external data is fed into the model.
    /// Each entry in this vector is the unique index (usize) of a node in the `nodes` map that acts as an input node.
    /// This is essential for:
    /// - Mapping user-provided input tensors to the correct nodes during model execution.
    /// - Determining the expected input signature (shapes, types) of the model.
    /// - Supporting models with multiple inputs (e.g., multi-branch networks).
    ///
    /// # What it does
    /// When running inference, the code uses this vector to know which nodes should receive the input tensors provided by the user.
    /// The order of indices in this vector determines the order in which input tensors should be supplied.
    ///
    /// # Example
    /// For a simple model: `input -> const -> add`, where "input" is node 0, "const" is node 1, and "add" is node 2,
    /// the `inputs` field would be: `vec![0]` (only node 0 is an input node).
    pub inputs: Vec<usize>,

    /// Output nodes and their outlet indices for this computational graph.
    ///
    /// # Why we need this field
    /// This field defines which nodes (and which output slots, if a node has multiple outputs) are considered as the outputs of the model.
    /// Each entry is a tuple `(node_index, outlet_index)`, where:
    /// - `node_index` is the index of the node in the `nodes` map.
    /// - `outlet_index` specifies which output of the node is used (for nodes with multiple outputs).
    ///
    /// This is essential for:
    /// - Collecting the final results after model execution.
    /// - Supporting models with multiple outputs (e.g., multi-task networks).
    /// - Mapping the internal graph outputs to the user-facing model outputs.
    ///
    /// # What it does
    /// After executing the graph, the code uses this vector to extract the correct output tensors from the results map.
    /// The order of entries determines the order of outputs returned to the user.
    ///
    /// # Example
    /// For the model: `input -> const -> add`, where "add" is node 2 and produces a single output at outlet 0,
    /// the `outputs` field would be: `vec![(2, 0)]` (the output of node 2, outlet 0, is the model's output).
    ///
    /// # Usage Example
    /// ```rust
    /// // Suppose we have a graph:
    /// // Node 0: Input
    /// // Node 1: Constant
    /// // Node 2: Add (inputs: Node 0, Node 1)
    /// let parsed_nodes = ParsedNodes {
    ///     nodes: /* ... */,
    ///     inputs: vec![0],           // Node 0 is the input node
    ///     outputs: vec![(2, 0)],     // Node 2's output (outlet 0) is the model output
    /// };
    /// // When running inference:
    /// // - The input tensor is mapped to node 0.
    /// // - After execution, the output is taken from node 2, outlet 0.
    /// ```
    pub outputs: Vec<Outlet>,
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
            NodeType::SubGraph { .. } => log::warn!("Cannot decrement const of subgraph"),
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
    pub outputs: Vec<Tensor<i128>>,
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

#[cfg(test)]
mod tests {
    use crate::circuit::{
        ops::{lookup::LookupOp, poly::PolyOp, InputType},
        utils::F32,
    };

    use super::*;

    #[test]
    fn test_model_builder_mat_mul() {
        let mut model = Model::default();

        let input_node = SupportedOp::Input(Input {
            scale: 1,
            datum_type: InputType::F32,
        });

        // Add matrix node input
        model
            .add_node(input_node.clone(), vec![], vec![2, 2])
            .unwrap();
        // Add vector node input
        model
            .add_node(input_node.clone(), vec![], vec![1, 2])
            .unwrap();

        let mat_mul_node = SupportedOp::Linear(PolyOp::Einsum {
            equation: "ij,bj->bi".to_string(),
        });

        model
            .add_node(mat_mul_node, vec![(0, 0), (1, 0)], vec![1, 2])
            .unwrap();
        model.add_inputs(vec![0, 1]);
        model.add_outputs(vec![(2, 0)]);

        // Test execution with vector-matrix multiplication
        // Vector: [1, 2]
        let input2 = Tensor::new(Some(&[1, 2]), &[1, 2]).unwrap();
        // Matrix: [[5, 6], [7, 8]]
        let input1 = Tensor::new(Some(&[5, 6, 7, 8]), &[2, 2]).unwrap();

        let result = model.forward(&[input1.clone(), input2.clone()]).unwrap();

        assert_eq!(result.outputs.len(), 1);
        assert_eq!(
            result.outputs[0],
            Tensor::new(Some(&[17, 23]), &[1, 2]).unwrap()
        );
    }

    #[test]
    fn test_model_builder_relu() {
        let mut model = Model::default();

        let input_node = SupportedOp::Input(Input {
            scale: 1,
            datum_type: InputType::F32,
        });

        model
            .add_node(input_node.clone(), vec![], vec![1, 4])
            .unwrap();

        let relu_node = SupportedOp::Nonlinear(LookupOp::ReLU);

        model.add_node(relu_node, vec![(0, 0)], vec![1, 4]).unwrap();
        model.add_inputs(vec![0]);
        model.add_outputs(vec![(1, 0)]);

        // Test execution with various inputs
        let input = Tensor::new(Some(&[-1, 0, 1, 2]), &[1, 4]).unwrap();
        let result = model.forward(&[input]).unwrap();

        // Expected result: [0.0, 0.0, 1.0, 2.0]
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(
            result.outputs[0],
            Tensor::new(Some(&[0, 0, 1, 2]), &[1, 4]).unwrap()
        );
    }

    #[test]
    fn test_model_builder_sigmoid() {
        let mut model = Model::default();

        let input_node = SupportedOp::Input(Input {
            scale: 1,
            datum_type: InputType::F32,
        });

        // input: shape [1, 3]
        model
            .add_node(input_node.clone(), vec![], vec![1, 3])
            .unwrap();

        // sigmoid node
        let sigmoid_node = SupportedOp::Nonlinear(LookupOp::Sigmoid { scale: F32(1.0) });
        model
            .add_node(sigmoid_node, vec![(0, 0)], vec![1, 3])
            .unwrap();

        model.add_inputs(vec![0]);
        model.add_outputs(vec![(1, 0)]);

        // x = [-2.0, 0.0, 2.0]
        let x = Tensor::new(Some(&[-2, 0, 2]), &[1, 3]).unwrap();

        let result = model.forward(&[x]).unwrap();
        assert_eq!(result.outputs.len(), 1);

        let _out: Vec<i128> = result.outputs[0].iter().copied().collect();

        // TODO(Alberto): Not sure how to handle precision yet
        // sigmoid(-2)≈0.119, sigmoid(0)=0.5, sigmoid(2)≈0.881
        // assert!((out[0] - 0.119).abs() < 1e-3);
        // assert!((out[1] - 0.5).abs() < 1e-3);
        // assert!((out[2] - 0.881).abs() < 1e-3);
    }

    #[test]
    fn test_model_builder_add() {
        let mut model = Model::default();

        let input_node = SupportedOp::Input(Input {
            scale: 1,
            datum_type: InputType::F32,
        });

        // two inputs: shape [1, 3]
        model
            .add_node(input_node.clone(), vec![], vec![1, 3])
            .unwrap(); // id 0
        model
            .add_node(input_node.clone(), vec![], vec![1, 3])
            .unwrap(); // id 1

        // elementwise add
        let add_node = SupportedOp::Linear(PolyOp::Add);
        model
            .add_node(add_node, vec![(0, 0), (1, 0)], vec![1, 3])
            .unwrap();

        model.add_inputs(vec![0, 1]);
        model.add_outputs(vec![(2, 0)]);

        let a = Tensor::new(Some(&[1, 2, 3]), &[1, 3]).unwrap();
        let b = Tensor::new(Some(&[4, 5, 6]), &[1, 3]).unwrap();

        let result = model.forward(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(
            result.outputs[0],
            Tensor::new(Some(&[5, 7, 9]), &[1, 3]).unwrap()
        );
    }
}
