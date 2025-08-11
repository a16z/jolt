use crate::{
    circuit::ops::poly::PolyOp,
    graph::{
        model::Model,
        utilities::{create_const_node, create_div_node, create_input_node, create_polyop_node},
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
        let raw = Tensor::new(Some(&[] as &[f32]), &[0]).unwrap(); // same as your code
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
}

/* ********************** Testing Model's ********************** */

pub fn custom_sentiment_model() -> Model {
    todo!()
}

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
