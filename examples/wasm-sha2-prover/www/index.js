const status = document.getElementById('status');
const proveBtn = document.getElementById('prove-btn');
const verifyBtn = document.getElementById('verify-btn');
const traceBtn = document.getElementById('trace-btn');
const output = document.getElementById('output');
const inputMessage = document.getElementById('input-message');

let worker = null;
let lastProofBytes = null;
let lastProgramIoBytes = null;

function setStatus(text, className) {
    status.textContent = text;
    status.className = className;
}

function log(msg) {
    output.textContent += msg + '\n';
    console.log(msg);
}

async function main() {
    try {
        output.textContent = '';
        setStatus('Loading files...', 'loading');

        const [proverPrepResp, verifierPrepResp, elfResp] = await Promise.all([
            fetch('./prover_preprocessing.bin'),
            fetch('./verifier_preprocessing.bin'),
            fetch('./guest.elf'),
        ]);

        if (!proverPrepResp.ok || !verifierPrepResp.ok || !elfResp.ok) {
            throw new Error('Failed to load preprocessing files. Run generate-preprocessing first.');
        }

        const [proverPrepBytes, verifierPrepBytes, elfBytes] = await Promise.all([
            proverPrepResp.arrayBuffer(),
            verifierPrepResp.arrayBuffer(),
            elfResp.arrayBuffer(),
        ]);

        log(`Prover preprocessing: ${(proverPrepBytes.byteLength / 1024 / 1024).toFixed(2)} MB`);
        log(`Verifier preprocessing: ${(verifierPrepBytes.byteLength / 1024 / 1024).toFixed(2)} MB`);
        log(`Guest ELF: ${(elfBytes.byteLength / 1024).toFixed(2)} KB`);

        setStatus('Starting worker...', 'loading');

        worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

        worker.onmessage = (e) => {
            const { type, error, proof, programIo, valid, elapsed, trace } = e.data;

            if (type === 'error') {
                setStatus('Error: ' + error, 'error');
                log('Error: ' + error);
                proveBtn.disabled = false;
                verifyBtn.disabled = lastProofBytes !== null;
                return;
            }

            if (type === 'init-done') {
                log('Ready');
                setStatus('Ready', 'ready');
                proveBtn.disabled = false;
                traceBtn.disabled = false;
                return;
            }

            if (type === 'trace') {
                const blob = new Blob([trace], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `jolt-trace-${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
                log('Trace downloaded');
                return;
            }

            if (type === 'prove-done') {
                lastProofBytes = proof;
                lastProgramIoBytes = programIo;

                log(`Proof generated in ${(elapsed / 1000).toFixed(2)}s`);
                log(`Proof size: ${proof.length} bytes (${(proof.length / 1024).toFixed(2)} KB)`);
                log(`Program IO size: ${programIo.length} bytes`);

                setStatus('Proof generated!', 'ready');
                proveBtn.disabled = false;
                verifyBtn.disabled = false;
                return;
            }

            if (type === 'verify-done') {
                log(`Verification completed in ${(elapsed / 1000).toFixed(2)}s`);
                log(`Result: ${valid ? 'VALID' : 'INVALID'}`);

                setStatus(valid ? 'Verification passed!' : 'Verification failed!', valid ? 'ready' : 'error');
                proveBtn.disabled = false;
                verifyBtn.disabled = false;
                return;
            }
        };

        worker.onerror = (e) => {
            setStatus('Worker error: ' + e.message, 'error');
            log('Worker error: ' + e.message);
            console.error(e);
        };

        const numThreads = Math.min(navigator.hardwareConcurrency || 4, 12);
        setStatus(`Initializing WASM (${numThreads} threads)...`, 'loading');

        worker.postMessage({
            type: 'init',
            data: {
                numThreads,
                proverPreprocessing: proverPrepBytes,
                verifierPreprocessing: verifierPrepBytes,
                elfBytes: elfBytes,
            },
        }, [proverPrepBytes, verifierPrepBytes, elfBytes]);

    } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        log('Error: ' + e.message);
        console.error(e);
    }
}

proveBtn.onclick = () => {
    const message = inputMessage.value;
    const input = new TextEncoder().encode(message);

    proveBtn.disabled = true;
    verifyBtn.disabled = true;
    setStatus('Proving...', 'proving');
    log(`\nProving SHA-256("${message}") [${input.length} bytes]`);

    worker.postMessage({
        type: 'prove',
        data: { input: Array.from(input) },
    });
};

verifyBtn.onclick = () => {
    if (!lastProofBytes || !lastProgramIoBytes) {
        log('No proof to verify. Generate a proof first.');
        return;
    }

    proveBtn.disabled = true;
    verifyBtn.disabled = true;
    setStatus('Verifying...', 'proving');
    log('\nStarting verification...');

    worker.postMessage({
        type: 'verify',
        data: {
            proof: lastProofBytes,
            programIo: lastProgramIoBytes,
        },
    });
};

traceBtn.onclick = () => {
    worker.postMessage({ type: 'get-trace' });
};

main();
