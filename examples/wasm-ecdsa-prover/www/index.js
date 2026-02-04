const status = document.getElementById('status');
const proveBtn = document.getElementById('prove-btn');
const verifyBtn = document.getElementById('verify-btn');
const traceBtn = document.getElementById('trace-btn');
const output = document.getElementById('output');

let worker = null;

// Test ECDSA signature - pre-computed for "hello world" signed with secp256k1
// Note: r, s, q are valid only for "hello world" - changing message will fail verification

// Current message hash as big-endian u64 array in reverse order
let z = [
    0x9088f7ace2efcde9n,
    0xc484efe37a5380een,
    0xa52e52d7da7dabfan,
    0xb94d27b9934d3e08n,
];

// Signature (valid for "hello world" only)
const r = [
    0xb8fc413b4b967ed8n,
    0x248d4b0b2829ab00n,
    0x587f69296af3cd88n,
    0x3a5d6a386e6cf7c0n,
];
const s = [
    0x66a82f274e3dcafcn,
    0x299a02486be40321n,
    0x6212d714118f617en,
    0x9d452f63cf91018dn,
];

// Public key
const q = [
    0x0012563f32ed0216n,
    0xee00716af6a73670n,
    0x91fc70e34e00e6c8n,
    0xeeb6be8b9e68868bn,
    0x4780de3d5fda972dn,
    0xcb1b42d72491e47fn,
    0xdc7f31262e4ba2b7n,
    0xdc7b004d3bb2800dn,
];

// Compute SHA256 hash using Web Crypto API
async function sha256(message) {
    const encoder = new TextEncoder();
    const data = encoder.encode(message);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(hashBuffer);
}

// Convert SHA256 bytes to big-endian u64 array in reverse order
function hashToU64Array(hashBytes) {
    const view = new DataView(hashBytes.buffer);
    const result = [];
    for (let i = 3; i >= 0; i--) {
        result.push(view.getBigUint64(i * 8, false)); // big-endian, reverse order
    }
    return result;
}

// Convert little-endian u64 array to hex string (big-endian display)
function u64ArrayToHex(arr) {
    let hex = '';
    for (let i = arr.length - 1; i >= 0; i--) {
        hex += arr[i].toString(16).padStart(16, '0');
    }
    return '0x' + hex;
}

// Update hash display when message changes
async function updateMessageHash() {
    const message = document.getElementById('input-message').value;
    const hashBytes = await sha256(message);
    z = hashToU64Array(hashBytes);
    document.getElementById('message-hash').textContent = u64ArrayToHex(z);
}

// Populate the UI with hex values
function populateInputs() {
    document.getElementById('input-r').value = u64ArrayToHex(r);
    document.getElementById('input-s').value = u64ArrayToHex(s);
    document.getElementById('input-q').value = u64ArrayToHex(q);

    // Set up message input listener
    const messageInput = document.getElementById('input-message');
    messageInput.addEventListener('input', updateMessageHash);

    // Initialize hash display
    updateMessageHash();
}

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
    populateInputs();

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
                // Download trace as JSON file
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

        const numThreads = navigator.hardwareConcurrency || 4;
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
    proveBtn.disabled = true;
    verifyBtn.disabled = true;
    setStatus('Proving...', 'proving');
    log('\n--- Starting proof generation ---');

    worker.postMessage({
        type: 'prove',
        data: { z, r, s, q },
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
    log('\n--- Starting verification ---');

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
