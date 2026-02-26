import init, {
    initThreadPool,
    init_inlines,
    init_tracing,
    get_trace_json,
    clear_trace,
    WasmProver,
    WasmVerifier,
} from '../pkg/jolt_wasm_sha2_prover.js';

let wasmExports = null;
const provers = {};
const verifiers = {};

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                wasmExports = await init();
                await initThreadPool(data.numThreads);
                init_tracing();
                init_inlines();
                self.postMessage({ type: 'init-done' });
                break;
            }

            case 'load-program': {
                const name = data.program;
                provers[name] = new WasmProver(
                    new Uint8Array(data.proverPreprocessing),
                    new Uint8Array(data.elfBytes)
                );
                verifiers[name] = new WasmVerifier(
                    new Uint8Array(data.verifierPreprocessing)
                );
                self.postMessage({ type: 'program-loaded', program: name });
                break;
            }

            case 'prove': {
                const prover = provers[data.program];
                const start = performance.now();
                let result;

                switch (data.program) {
                    case 'sha2':
                        result = prover.prove_sha2(new Uint8Array(data.input));
                        break;
                    case 'ecdsa':
                        result = prover.prove_ecdsa(
                            BigUint64Array.from(data.z.map(BigInt)),
                            BigUint64Array.from(data.r.map(BigInt)),
                            BigUint64Array.from(data.s.map(BigInt)),
                            BigUint64Array.from(data.q.map(BigInt)),
                        );
                        break;
                    case 'keccak':
                        result = prover.prove_keccak_chain(
                            new Uint8Array(data.input),
                            data.numIters
                        );
                        break;
                }

                const elapsed = performance.now() - start;
                const peakMemory = wasmExports.memory.buffer.byteLength;

                self.postMessage({
                    type: 'prove-done',
                    program: data.program,
                    proof: result.proof,
                    proofSize: result.proof_size,
                    compressedProofSize: result.compressed_proof_size,
                    programIo: result.program_io,
                    numCycles: result.num_cycles,
                    peakMemory,
                    elapsed,
                });
                break;
            }

            case 'verify': {
                const verifier = verifiers[data.program];
                const start = performance.now();
                const valid = verifier.verify(data.proof, data.programIo);
                const elapsed = performance.now() - start;

                self.postMessage({
                    type: 'verify-done',
                    program: data.program,
                    valid,
                    elapsed,
                });
                break;
            }

            case 'get-trace': {
                const traceJson = get_trace_json();
                self.postMessage({
                    type: 'trace',
                    trace: traceJson,
                });
                break;
            }

            case 'clear-trace': {
                clear_trace();
                self.postMessage({ type: 'trace-cleared' });
                break;
            }
        }
    } catch (err) {
        self.postMessage({ type: 'error', error: err.message || String(err) });
    }
};
