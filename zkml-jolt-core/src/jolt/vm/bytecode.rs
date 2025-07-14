use onnx_tracer::trace_types::{ONNXCycle, ONNXInstr};

pub struct BytecodePreprocessing {
    code_size: usize,
    bytecode: Vec<ONNXInstr>,
}

pub struct BytecodeProof {}

impl BytecodeProof {
    pub fn prove(preprocessing: &BytecodePreprocessing, trace: &[ONNXCycle]) {
        let K = preprocessing.code_size;
        let T = trace.len();

        // --- Shout PIOP ---
        // --- Hamming weight check ---
        // --- Booleanity check ---
        // --- raf evaluation ---
    }
}
