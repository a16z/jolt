import init, {
    initThreadPool,
    init_inlines,
    init_tracing,
    get_trace_json,
    clear_trace,
    WasmProver,
    WasmVerifier,
} from '../pkg/jolt_wasm_ecdsa_prover.js';

let prover = null;
let verifier = null;

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                await init();
                await initThreadPool(data.numThreads);
                init_tracing();
                init_inlines();

                prover = new WasmProver(
                    new Uint8Array(data.proverPreprocessing),
                    new Uint8Array(data.elfBytes)
                );

                verifier = new WasmVerifier(new Uint8Array(data.verifierPreprocessing));

                self.postMessage({ type: 'init-done' });
                break;
            }

            case 'prove': {
                const start = performance.now();
                const result = prover.prove_ecdsa(
                    BigUint64Array.from(data.z),
                    BigUint64Array.from(data.r),
                    BigUint64Array.from(data.s),
                    BigUint64Array.from(data.q)
                );
                const elapsed = performance.now() - start;

                self.postMessage({
                    type: 'prove-done',
                    proof: result.proof,
                    programIo: result.program_io,
                    elapsed,
                });
                break;
            }

            case 'verify': {
                const start = performance.now();
                const valid = verifier.verify(data.proof, data.programIo);
                const elapsed = performance.now() - start;

                self.postMessage({
                    type: 'verify-done',
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
