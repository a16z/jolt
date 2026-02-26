const PROGRAMS = ['sha2', 'ecdsa', 'keccak'];
const PROGRAM_FILES = {
    sha2:   { prover: 'sha2_prover.bin',   verifier: 'sha2_verifier.bin',   elf: 'sha2.elf'   },
    ecdsa:  { prover: 'ecdsa_prover.bin',  verifier: 'ecdsa_verifier.bin',  elf: 'ecdsa.elf'  },
    keccak: { prover: 'keccak_prover.bin', verifier: 'keccak_verifier.bin', elf: 'keccak.elf' },
};

const ECDSA_TEST_VECTOR = {
    z: ['0x9088f7ace2efcde9', '0xc484efe37a5380ee', '0xa52e52d7da7dabfa', '0xb94d27b9934d3e08'],
    r: ['0xb8fc413b4b967ed8', '0x248d4b0b2829ab00', '0x587f69296af3cd88', '0x3a5d6a386e6cf7c0'],
    s: ['0x66a82f274e3dcafc', '0x299a02486be40321', '0x6212d714118f617e', '0x9d452f63cf91018d'],
    q: [
        '0x0012563f32ed0216', '0xee00716af6a73670', '0x91fc70e34e00e6c8', '0xeeb6be8b9e68868b',
        '0x4780de3d5fda972d', '0xcb1b42d72491e47f', '0xdc7f31262e4ba2b7', '0xdc7b004d3bb2800d',
    ],
};

const status = document.getElementById('status');
const tabButtons = document.querySelectorAll('.tab-btn');
const pages = document.querySelectorAll('.tab-page');

let worker = null;
let wasmReady = false;
let activeTab = 'sha2';

// 'idle' | 'loading' | 'ready'
const programState = {};
for (const p of PROGRAMS) {
    programState[p] = { loadState: 'idle', proofBytes: null, programIoBytes: null };
}

function setStatus(text, className) {
    status.textContent = text;
    status.className = className;
}

function getOutput(program) {
    return document.querySelector(`#page-${program} .output`);
}

function log(program, msg) {
    const out = getOutput(program);
    out.textContent += msg + '\n';
    console.log(`[${program}] ${msg}`);
}

function setButtonsEnabled(program, prove, verify) {
    const page = document.getElementById(`page-${program}`);
    page.querySelector('.prove-btn').disabled = !prove;
    page.querySelector('.verify-btn').disabled = !verify;
}

function updateTabButtons(program) {
    const st = programState[program];
    const ready = st.loadState === 'ready';
    setButtonsEnabled(program, ready, ready && st.proofBytes !== null);
    document.querySelector(`#page-${program} .trace-btn`).disabled = !wasmReady;
}

function switchTab(program) {
    activeTab = program;
    for (const btn of tabButtons) {
        btn.classList.toggle('active', btn.dataset.program === program);
    }
    for (const page of pages) {
        page.classList.toggle('hidden', page.id !== `page-${program}`);
    }
}

for (const btn of tabButtons) {
    btn.addEventListener('click', () => switchTab(btn.dataset.program));
}

async function sha256Digest(bytes) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', bytes);
    return new Uint8Array(hashBuffer);
}

async function loadProgram(name) {
    const st = programState[name];
    if (st.loadState !== 'idle') return;
    st.loadState = 'loading';

    log(name, 'Loading preprocessing...');
    setStatus(`Loading ${name} preprocessing...`, 'loading');

    const files = PROGRAM_FILES[name];
    const [prover, verifier, elf] = await Promise.all([
        fetch(`./${files.prover}`).then(r => {
            if (!r.ok) throw new Error(`Failed to load ${files.prover}`);
            return r.arrayBuffer();
        }),
        fetch(`./${files.verifier}`).then(r => {
            if (!r.ok) throw new Error(`Failed to load ${files.verifier}`);
            return r.arrayBuffer();
        }),
        fetch(`./${files.elf}`).then(r => {
            if (!r.ok) throw new Error(`Failed to load ${files.elf}`);
            return r.arrayBuffer();
        }),
    ]);

    log(name, `Prover preprocessing: ${(prover.byteLength / 1024 / 1024).toFixed(2)} MB`);
    log(name, `Verifier preprocessing: ${(verifier.byteLength / 1024 / 1024).toFixed(2)} MB`);
    log(name, `Guest ELF: ${(elf.byteLength / 1024).toFixed(2)} KB`);

    log(name, 'Initializing prover & verifier...');
    worker.postMessage({
        type: 'load-program',
        data: {
            program: name,
            proverPreprocessing: prover,
            verifierPreprocessing: verifier,
            elfBytes: elf,
        },
    }, [prover, verifier, elf]);
}

async function ensureProgramLoaded(name) {
    const st = programState[name];
    if (st.loadState === 'ready') return true;
    if (st.loadState === 'idle') {
        try {
            await loadProgram(name);
        } catch (e) {
            st.loadState = 'idle';
            setStatus(`Error: ${e.message}`, 'error');
            log(name, `Error: ${e.message}`);
            return false;
        }
    }
    // Wait for program-loaded message
    return new Promise(resolve => {
        const check = () => {
            if (programState[name].loadState === 'ready') {
                resolve(true);
            } else {
                setTimeout(check, 50);
            }
        };
        check();
    });
}

async function main() {
    try {
        if (!crossOriginIsolated) {
            setStatus('This page requires a browser with SharedArrayBuffer support. Please open in Chrome or Safari (not an in-app browser).', 'error');
            return;
        }

        setStatus('Initializing WASM...', 'loading');

        worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

        worker.onmessage = (e) => {
            const msg = e.data;

            if (msg.type === 'error') {
                setStatus('Error: ' + msg.error, 'error');
                log(activeTab, 'Error: ' + msg.error);
                return;
            }

            if (msg.type === 'init-done') {
                wasmReady = true;
                setStatus('Ready — select a program and generate a proof', 'ready');
                for (const p of PROGRAMS) {
                    const page = document.getElementById(`page-${p}`);
                    page.querySelector('.prove-btn').disabled = false;
                    page.querySelector('.trace-btn').disabled = false;
                }
                return;
            }

            if (msg.type === 'program-loaded') {
                const p = msg.program;
                programState[p].loadState = 'ready';
                log(p, 'Ready');
                setStatus('Ready', 'ready');
                updateTabButtons(p);
                return;
            }

            if (msg.type === 'trace') {
                const blob = new Blob([msg.trace], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `jolt-trace-${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
                log(activeTab, 'Trace downloaded');
                return;
            }

            if (msg.type === 'prove-done') {
                const p = msg.program;
                const st = programState[p];
                st.proofBytes = msg.proof;
                st.programIoBytes = msg.programIo;

                log(p, `Proof generated in ${(msg.elapsed / 1000).toFixed(2)}s`);
                if (msg.numCycles != null) log(p, `RISC-V cycles: ${msg.numCycles.toLocaleString()}`);
                log(p, `Proof size: ${(msg.proofSize / 1024).toFixed(2)} KB`);
                log(p, `Proof size (compressed): ${(msg.compressedProofSize / 1024).toFixed(2)} KB`);
                if (msg.peakMemory != null) log(p, `Peak WASM memory: ${(msg.peakMemory / 1024 / 1024).toFixed(0)} MB`);

                setStatus('Proof generated!', 'ready');
                setButtonsEnabled(p, true, true);
                return;
            }

            if (msg.type === 'verify-done') {
                const p = msg.program;
                log(p, `Verification completed in ${(msg.elapsed / 1000).toFixed(2)}s`);
                log(p, `Result: ${msg.valid ? 'VALID' : 'INVALID'}`);
                setStatus(msg.valid ? 'Verification passed!' : 'Verification failed!', msg.valid ? 'ready' : 'error');
                setButtonsEnabled(p, true, true);
                return;
            }
        };

        worker.onerror = (e) => {
            setStatus('Worker error: ' + e.message, 'error');
            console.error(e);
        };

        const numThreads = Math.min(navigator.hardwareConcurrency || 4, 12);
        setStatus(`Initializing WASM (${numThreads} threads)...`, 'loading');

        worker.postMessage({
            type: 'init',
            data: { numThreads },
        });

    } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        console.error(e);
    }
}

function setupProveHandlers() {
    document.querySelector('#page-sha2 .prove-btn').addEventListener('click', async () => {
        const message = document.getElementById('sha2-message').value;
        const input = new TextEncoder().encode(message);
        setButtonsEnabled('sha2', false, false);
        if (!await ensureProgramLoaded('sha2')) { updateTabButtons('sha2'); return; }
        setStatus('Proving...', 'proving');
        log('sha2', `\nProving SHA-256("${message}") [${input.length} bytes]`);
        worker.postMessage({
            type: 'prove',
            data: { program: 'sha2', input: Array.from(input) },
        });
    });

    document.querySelector('#page-ecdsa .prove-btn').addEventListener('click', async () => {
        setButtonsEnabled('ecdsa', false, false);
        if (!await ensureProgramLoaded('ecdsa')) { updateTabButtons('ecdsa'); return; }
        setStatus('Proving...', 'proving');
        log('ecdsa', '\nProving ECDSA signature verification ("hello world")');
        worker.postMessage({
            type: 'prove',
            data: {
                program: 'ecdsa',
                z: ECDSA_TEST_VECTOR.z,
                r: ECDSA_TEST_VECTOR.r,
                s: ECDSA_TEST_VECTOR.s,
                q: ECDSA_TEST_VECTOR.q,
            },
        });
    });

    document.querySelector('#page-keccak .prove-btn').addEventListener('click', async () => {
        const message = document.getElementById('keccak-message').value;
        const numIters = parseInt(document.getElementById('keccak-iters').value, 10);
        if (numIters < 1 || numIters > 100) {
            log('keccak', 'Iterations must be between 1 and 100');
            return;
        }
        const messageBytes = new TextEncoder().encode(message);
        const input = await sha256Digest(messageBytes);
        setButtonsEnabled('keccak', false, false);
        if (!await ensureProgramLoaded('keccak')) { updateTabButtons('keccak'); return; }
        setStatus('Proving...', 'proving');
        log('keccak', `\nProving Keccak chain("${message}", ${numIters} iters)`);
        worker.postMessage({
            type: 'prove',
            data: { program: 'keccak', input: Array.from(input), numIters },
        });
    });
}

function setupVerifyHandlers() {
    for (const p of PROGRAMS) {
        document.querySelector(`#page-${p} .verify-btn`).addEventListener('click', () => {
            const st = programState[p];
            if (!st.proofBytes || !st.programIoBytes) {
                log(p, 'No proof to verify. Generate a proof first.');
                return;
            }
            setButtonsEnabled(p, false, false);
            setStatus('Verifying...', 'proving');
            log(p, '\nStarting verification...');
            worker.postMessage({
                type: 'verify',
                data: { program: p, proof: st.proofBytes, programIo: st.programIoBytes },
            });
        });
    }
}

function setupTraceHandlers() {
    for (const p of PROGRAMS) {
        document.querySelector(`#page-${p} .trace-btn`).addEventListener('click', () => {
            worker.postMessage({ type: 'get-trace' });
        });
    }
}

setupProveHandlers();
setupVerifyHandlers();
setupTraceHandlers();
main();
