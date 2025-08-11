use crate::{
    circuit::ops::{hybrid::HybridOp, poly::PolyOp},
    graph::{
        model::Model,
        node::SupportedOp,
        utilities::{
            create_const_node, create_div_node, create_input_node, create_node, create_polyop_node,
        },
    },
    tensor::Tensor,
};

type Wire = (usize, usize); // (node_id, output_idx)
const O: usize = 0; // single-output nodes use 0

struct ModelBuilder {
    model: Model,
    next_id: usize,
    scale: i32,
}

impl ModelBuilder {
    fn new(scale: i32) -> Self {
        Self {
            model: Model::default(),
            next_id: 0,
            scale,
        }
    }

    fn take(self, inputs: Vec<usize>, outputs: Vec<Wire>) -> Model {
        let mut m = self.model;
        m.set_inputs(inputs);
        m.set_outputs(outputs);
        m
    }

    fn alloc(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn input(&mut self, dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let n = create_input_node(self.scale, dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn const_tensor(
        &mut self,
        tensor: Tensor<i128>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let raw = Tensor::new(Some(&[] as &[f32]), &[0]).unwrap();
        let n = create_const_node(tensor, raw, self.scale, out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn poly(
        &mut self,
        op: PolyOp<i128>,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let n = create_polyop_node(op, self.scale, vec![a, b], out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn div(&mut self, divisor: i128, x: Wire, out_dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let n = create_div_node(divisor, self.scale, vec![x], out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn gather(
        &mut self,
        data: Wire,
        indices: Wire,
        dim: usize,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let gather_op = HybridOp::Gather {
            dim,
            constant_idx: None,
        };
        let gather_node = create_node(
            SupportedOp::Hybrid(gather_op),
            self.scale,
            vec![data, indices],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(gather_node);
        (id, O)
    }

    fn reshape(
        &mut self,
        input: Wire,
        new_shape: Vec<usize>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let reshape_node = create_node(
            SupportedOp::Linear(PolyOp::Reshape(new_shape)),
            self.scale,
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(reshape_node);
        (id, O)
    }

    fn sum(
        &mut self,
        input: Wire,
        axes: Vec<usize>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let sum_node = create_node(
            SupportedOp::Linear(PolyOp::Sum { axes }),
            self.scale,
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(sum_node);
        (id, O)
    }

    fn greater_equal(
        &mut self,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let gte_node = create_node(
            SupportedOp::Hybrid(HybridOp::GreaterEqual),
            0, // Binary output has scale 0
            vec![a, b],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(gte_node);
        (id, O)
    }
}

/* ********************** Testing Model's ********************** */

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn custom_addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let y = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, const, []), (2, add, [0, 1]), (3, sub, [0, 1]), (4, mul, [2, 3]), (5, output, [4])]
pub fn custom_addsubmulconst_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    let x = b.input(vec![1, 4], 2);
    let mut c = Tensor::new(Some(&[50i128, 60i128, 70i128, 80i128]), &[1, 4]).unwrap();
    c.set_scale(SCALE);
    let k = b.const_tensor(c, vec![1, 4], 2);

    let a = b.poly(PolyOp::Add, x, k, vec![1, 4], 1);
    let s = b.poly(PolyOp::Sub, x, k, vec![1, 4], 1);
    let y = b.poly(PolyOp::Mult, a, s, vec![1, 4], 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, div, [4]), (6, output, [5])]
pub fn custom_addsubmuldiv_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let y = b.div(2i128, t, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, div, [4]), (6, div, [5]), (7, output, [6])]
pub fn custom_addsubmuldivdiv_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let d1 = b.div(2i128, t, out_dims.clone(), 1);
    let y = b.div(5i128, d1, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn scalar_addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let dims = vec![1];

    let x = b.input(dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, dims.clone(), 1);
    let y = b.poly(PolyOp::Add, s, m, dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// Implements a simple embedding-based sentiment analysis model:
/// 1. Looks up embeddings for input word indices
/// 2. Sums the embeddings and normalizes (divides by -0.46149117, which we round up to -0.5, which is multiplying by -2)
/// 3. Adds a bias term (-54)
/// 4. Returns positive sentiment if result >= 0
pub fn embedding_sentiment_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Create the embedding tensor (shape [14, 1]) (embeddings taken from /models/sentiment_sum)
    let mut embedding = Tensor::new(
        Some(&[
            139i128, -200, -331, -42, -260, -290, -166, -171, -481, -294, 210, 291, 2, 328,
        ]),
        &[14, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor(embedding, vec![14, 1], 1);

    // Node 1: Input node for word indices (shape [1, 5])
    let input_indices = b.input(vec![1, 5], 1);

    // Node 2: Gather (lookup embeddings based on indices)
    let gathered = b.gather(embedding_const, input_indices, 0, vec![1, 5, 1], 1);

    // Node 3: Reshape (flatten the gathered embeddings)
    let reshaped = b.reshape(gathered, vec![1, 5], vec![1, 5], 1);

    // Node 4: Sum the embeddings along axis 1
    let summed = b.sum(reshaped, vec![1], vec![1, 1], 1);
    /*
       Node 6: Divide by constant with floating-point value
       Node 6: Multiply by constant (reciprocal of divisor)
       let divided = b.div_f64(-0.46149117, summed, vec![1, 1], 1);
       -1 / -0.46149117 ≈ -2.167
    */
    let mul_const = Tensor::new(Some(&[-2i128]), &[1, 1]).unwrap();
    let mul_wire = b.const_tensor(mul_const, vec![1, 1], 1);
    // Multiplication instead of division
    let multiplied = b.poly(PolyOp::Mult, summed, mul_wire, vec![1, 1], 1);

    // Node 7: Create the bias constant (-54)
    let mut bias = Tensor::new(Some(&[-54i128]), &[1, 1]).unwrap();
    bias.set_scale(SCALE);
    let bias_const = b.const_tensor(bias, vec![1, 1], 1);

    // Node 8: Add the bias
    let added = b.poly(PolyOp::Add, multiplied, bias_const, vec![1, 1], 1);

    // Node 9: Create the zero constant
    let mut zero = Tensor::new(Some(&[0i128]), &[1, 1]).unwrap();
    zero.set_scale(SCALE);
    let zero_const = b.const_tensor(zero, vec![1, 1], 1);

    // Node 10: Greater than or equal comparison
    let result = b.greater_equal(added, zero_const, vec![1, 1], 1);

    b.take(vec![input_indices.0], vec![result])
}
