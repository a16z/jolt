import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {blake2b} from '@noble/hashes/blake2.js';
import {keccak_256} from '@noble/hashes/sha3.js';
import {Common,Hardfork,Mainnet} from '@ethereumjs/common';
import {createEVM} from '@ethereumjs/evm';
import {compileModules} from './compile-modules.mjs';
import {resetWarmth} from './evm-warmth.mjs';
import {transactionCost} from './transaction-cost.mjs';
const build=compileModules();
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const at=(b,n)=>BigInt('0x'+b.subarray(n,n+32).toString('hex'));
const little=(b,n,len=8)=>BigInt('0x'+Buffer.from(b.subarray(n,n+len)).reverse().toString('hex'));
const le=n=>Buffer.from(word(n)).reverse();
const errorSelector=name=>Buffer.from(keccak_256(Buffer.from(name))).subarray(0,4).toString('hex');
const artifact=name=>build.complete.contracts[name+'.sol'][name];
const coordinator=artifact('SpartanVerifier');
const selector=Buffer.from(coordinator.evm.methodIdentifiers['verify(bytes,bytes,bytes,bytes,uint256[4][2])'],'hex');
function fixture(name){const base='test/spartan/'+name;return Object.fromEntries(['key','setup','inputs','proof','g2-affine'].map(k=>[k,fs.readFileSync(base+'/'+k+'.bin')]));}
function calldata(f){let offset=384;const heads=[],tails=[];for(const arg of [f.key,f.setup,f.inputs,f.proof]){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([selector,...heads,f['g2-affine'],...tails]);}
const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});
const gasLimit=30_000_000n;const deployments=[];const modules=[];
async function deploy(name,args=Buffer.alloc(0),reject=false){const initcode=Buffer.concat([Buffer.from(artifact(name).evm.bytecode.object,'hex'),args]);const result=await evm.runCall({data:initcode,gasLimit});deployments.push({name,reject,executionGas:result.execResult.executionGasUsed.toString(),transaction:transactionCost(initcode,result.execResult.executionGasUsed,true),exception:result.execResult.exceptionError?.error??null,address:result.createdAddress?.toString()});if(reject){assert.equal(result.execResult.exceptionError?.error,'revert');return;}assert.equal(result.execResult.exceptionError,undefined,name+' deployment');const code=await evm.stateManager.getCode(result.createdAddress);deployments.at(-1).actualRuntimeKeccak=Buffer.from(keccak_256(code)).toString('hex');assert.equal(Buffer.from(code).length,artifact(name).evm.deployedBytecode.object.length/2);return result.createdAddress;}
for(const name of build.names)modules.push(await deploy(name));
const messageStack=[];let moduleCalls=[];let leafData;
evm.events.on('beforeMessage',m=>{if(m.to?.toString()===modules[3]?.toString())leafData=Buffer.from(m.data);messageStack.push({address:m.to?.toString(),inputBytes:m.data.length,gasForwarded:m.gasLimit.toString()});});
evm.events.on('afterMessage',result=>{const m=messageStack.pop();if(m && modules.some(a=>a.toString()===m.address))moduleCalls.push({...m,executionGas:result.execResult.executionGasUsed.toString(),outputBytes:result.execResult.returnValue.length});});
const addressWord=a=>Buffer.concat([Buffer.alloc(12),a.bytes]);
const configs=new Map();
async function configured(f){const key=digest(f.key).toString('hex');if(!configs.has(key))configs.set(key,await deploy('SpartanVerifier',Buffer.concat([digest(f.key),digest(f.setup),...modules.map(addressWord)])));return configs.get(key);}
const records=[];
async function run(name,f,reject=false,{faultModule=null,externalCheckpoint=false}={}){
 const to=await configured(f);await resetWarmth(evm,modules);
 let original;
 if(faultModule!==null){const address=modules[faultModule].toString().slice(2);original=evm.precompiles.get(address);evm.precompiles.set(address,()=>({executionGasUsed:0n,returnValue:Uint8Array.of(1)}));}
 const data=externalCheckpoint?Buffer.concat([Buffer.from('ffffffff','hex'),calldata(f)]):calldata(f);
 moduleCalls=[];const result=await evm.runCall({to,data,gasLimit});
 if(faultModule!==null){const address=modules[faultModule].toString().slice(2);if(original)evm.precompiles.set(address,original);else evm.precompiles.delete(address);}
 const record={name,moduleCalls,coordinatorExecutionGas:(result.execResult.executionGasUsed-moduleCalls.reduce((sum,c)=>sum+BigInt(c.executionGas),0n)).toString(),accepted:!result.execResult.exceptionError,executionGas:result.execResult.executionGasUsed.toString(),exception:result.execResult.exceptionError?.error??null,returnHex:Buffer.from(result.execResult.returnValue).toString('hex'),calldataBytes:data.length,transaction:transactionCost(data,result.execResult.executionGasUsed)};records.push(record);
 if(reject)assert.equal(record.exception,'revert',name);else {assert.equal(record.exception,null,name);assert.deepEqual(moduleCalls.map(c=>c.address),modules.map(a=>a.toString()),name+' all four same-call stages');}
 return Buffer.from(result.execResult.returnValue);
}
let honestLeaf;
for(const name of ['toy','empty','empty-public']){
 const f=fixture(name),native=JSON.parse(fs.readFileSync('test/spartan/'+name+'/complete.json'));
 assert.equal(native.accepted,true);const result=await run(name,f);assert.equal(at(result,0),1n);
 assert.equal(result.subarray(32,64).toString('hex'),native.state,name+' native full transcript');
 if(name==='toy')honestLeaf=Buffer.from(leafData);
}
assert.equal(JSON.parse(fs.readFileSync('test/spartan/zero-products/complete.json')).accepted,false);
await run('zero-products-now-rejected',fixture('zero-products'),true);
// Derive intentional fixture mutation offsets by walking its wire grammar, not duplicating algebra.
function offsets(f){const log=n=>BigInt(n).toString(2).length-1;const r=log(little(f.key,100)),w=log(little(f.key,108)),n=log(little(f.key,116)),l=log(little(f.key,124)),t=log(little(f.key,132)),p=Number(little(f.key,92));let at=84+r*97+96+96*p+(4*(r+t)+2)*32+w*65;const privateValues=at;at+=96+32;const roots=at;at+=16*32;const halves=at;at+=6*32;const layers=[];for(const [network,depth,width]of [['ops',n,12],['mem',l,4]])for(let j=0;j<depth;j++){const rounds=at;at+=97*j;const ends=at;at+=2*width*32;const dots=at;if(network==='ops'&&j===depth-1)at+=18*32;layers.push({network,j,rounds,ends,dots});}return {privateValues,roots,halves,layers,next:at};}
const toy=fixture('toy');const clone=f=>Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));
function slabs(f){const o=offsets(f);const log=n=>BigInt(n).toString(2).length-1;const n=log(little(f.key,116)),l=log(little(f.key,124));let at=o.next;return [6,16,2].map((count,j)=>{const values=at;at+=32*count;const arity=[n+3,n+4,l+1][j];const opening=at;const witnesses=opening+32*(arity-1)+96*arity;at=witnesses+96;return {values,opening,witnesses,arity,end:at};});}
for(let phase=0;phase<3;phase++){
 const f=clone(toy),o=slabs(f)[phase];f.setup.subarray(64,96).copy(f.proof,o.witnesses);await run('slab-pairing-'+phase,f,true);
 assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector('PairingEquation()'));
}
for(let phase=0;phase<3;phase++){
 const f=clone(toy),o=slabs(f)[phase];f.proof[o.values]^=1;await run('slab-evaluation-'+phase,f,true);
}
{const f=clone(toy);le(1).copy(f.proof,slabs(f)[1].values+15*32);await run('nonzero-padding-slab',f,true);assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector('ZeroSlab()'));}
// Previously pending obligations are now mandatory.
{const f=clone(toy);const offset=slabs(f)[2].end;le((little(f.proof,offset,32)+1n)%21888242871839275222246405745257275088548364400416034343698204186575808495617n).copy(f.proof,offset);await run('witness-product-now-checked',f,true);assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector('WitnessProduct()'));}
{const f=clone(toy);const w=BigInt(little(f.key,108)).toString(2).length-1;const opening=slabs(f)[2].end+32;f.setup.subarray(64,96).copy(f.proof,opening+32*(w-1)+96*w);await run('witness-pairing-now-checked',f,true);assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector('PairingEquation()'));}
// Whole-verifier wire and application boundary failures.
for(const [name,mutate] of [
 ['proof-truncated',f=>{f.proof=f.proof.subarray(0,-1);} ],
 ['proof-trailing',f=>{f.proof=Buffer.concat([f.proof,Buffer.of(0)]);} ],
 ['noncanonical-witness-field',f=>{le(21888242871839275222246405745257275088548364400416034343698204186575808495617n).copy(f.proof,slabs(f)[2].end);} ],
 ['malformed-witness-point',f=>{Buffer.alloc(32,255).copy(f.proof,52);} ],
 ['public-input-changed',f=>{f.inputs[0]^=1;} ],
 ['key-binding-changed',f=>{f.proof[20]^=1;} ],
 ['zero-round-tag',f=>{f.proof[84]=0;} ],
 ['oversized-round-tag',f=>{f.proof[84]=4;} ],
]){const f=clone(toy);mutate(f);await run(name,f,true);}
// Diagnostic module calls test each leaf equation using genuine internally produced inputs.
// These checkpoints are NEVER accepted as external coordinator inputs.
const scalarR=21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const relative=(b,root,slot)=>root+Number(at(b,root+slot*32));
async function diagnostic(name,mutate,error){const data=Buffer.from(honestLeaf);mutate(data);await resetWarmth(evm,modules);const r=await evm.runCall({to:modules[3],data,gasLimit});assert.equal(r.execResult.exceptionError?.error,'revert',name);assert.equal(Buffer.from(r.execResult.returnValue).subarray(0,4).toString('hex'),errorSelector(error),name);records.push({name,scope:'standalone diagnostic, not authenticated input',accepted:false,executionGas:r.execResult.executionGasUsed.toString(),returnHex:Buffer.from(r.execResult.returnValue).toString('hex')});}
const change=(b,p)=>word((at(b,p)+1n)%scalarR).copy(b,p);
await diagnostic('dot-leaf',b=>{const sparse=4+Number(at(b,68));const ops=relative(b,sparse,2);change(b,ops+64);},'DotLeaf()');
await diagnostic('read-hash-leaf',b=>{const sparse=4+Number(at(b,68));const ops=relative(b,sparse,2);const trees=relative(b,ops,1);change(b,trees+32);},'OperationLeaf()');
await diagnostic('write-increment-must-alpha-squared',b=>{const sparse=4+Number(at(b,68));const ops=relative(b,sparse,2);const trees=relative(b,ops,1);word((at(b,trees+32)+1n)%scalarR).copy(b,trees+32*4);},'OperationLeaf()');
await diagnostic('memory-init-leaf',b=>{const sparse=4+Number(at(b,68));const mem=relative(b,sparse,3);const trees=relative(b,mem,1);change(b,trees+32);},'MemoryLeaf()');
await diagnostic('memory-audit-leaf',b=>{const sparse=4+Number(at(b,68));const mem=relative(b,sparse,3);const trees=relative(b,mem,1);change(b,trees+64);},'MemoryLeaf()');
await diagnostic('query-order-bound',b=>{const prefix=4+Number(at(b,36));const outer=relative(b,prefix,0);const point=relative(b,outer,0);change(b,point+32);},'MemoryLeaf()');
await deploy('SpartanVerifier',Buffer.concat([digest(toy.key),digest(toy.setup),addressWord(modules[1]),addressWord(modules[0]),addressWord(modules[2]),addressWord(modules[3])]),true);
for(let j=0;j<4;j++){const original=await evm.stateManager.getCode(modules[j]);await evm.stateManager.putCode(modules[j],Uint8Array.of(0));await run('changed-module-'+j,toy,true);await evm.stateManager.putCode(modules[j],original);}
for(let j=0;j<4;j++)await run('malformed-module-return-'+j,toy,true,{faultModule:j});
await run('external-checkpoint-selector-absent',toy,true,{externalCheckpoint:true});
fs.mkdirSync('evidence/complete',{recursive:true});fs.writeFileSync('evidence/complete/honest-leaf-calldata.hex',honestLeaf.toString('hex')+'\n');fs.writeFileSync('evidence/complete/results.json',JSON.stringify({scope:'complete clear Spartan algebraic verification under fixed authenticated key/setup, conditional native security assumptions unchanged',settings:build.settings,node:process.version,codeHashes:build.hashes,sourceHashes:Object.fromEntries(Object.entries(build.sources).map(([name,v])=>[name,hash(Buffer.from(v.content))])),runtimeBytes:Object.fromEntries([...build.names,'SpartanVerifier'].map(n=>[n,artifact(n).evm.deployedBytecode.object.length/2])),deployments,records,gasLimit:gasLimit.toString(),warmness:'Each call resets journal; precompiles05/06/07/08/09 explicitly warm; module accounts initially cold, EXTCODEHASH warms them before STATICCALL',gasScope:'actual EVM CREATE/code-deposit and call execution; excludes transaction intrinsic/calldata gas, no chain deployment'},null,2)+'\n');
console.log(JSON.stringify({deployments,records:records.map(({returnHex,...r})=>r)},null,2));
