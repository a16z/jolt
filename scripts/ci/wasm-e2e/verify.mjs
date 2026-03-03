import { readFileSync } from 'fs';
import { initSync, verify_fib } from './pkg/sample_project.js';

const wasmBytes = readFileSync('./pkg/sample_project_bg.wasm');
initSync(wasmBytes);

const pp = new Uint8Array(readFileSync('pp.bin'));
const proof = new Uint8Array(readFileSync('proof.bin'));
const io = new Uint8Array(readFileSync('io.bin'));

console.log(`preprocessing: ${pp.length} bytes`);
console.log(`proof: ${proof.length} bytes`);
console.log(`io: ${io.length} bytes`);

const result = verify_fib(pp, proof, io);
console.log(`verification result: ${result}`);

if (!result) {
    console.error('WASM verification failed');
    process.exit(1);
}
