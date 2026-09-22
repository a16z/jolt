import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {blake2b} from '@noble/hashes/blake2.js';
import {keccak_256} from '@noble/hashes/sha3.js';
import {Common,Hardfork,Mainnet} from '@ethereumjs/common';
import {createEVM} from '@ethereumjs/evm';
import {compileModules} from './compile-modules.mjs';
import {resetWarmth} from './evm-warmth.mjs';
const build=compileModules();
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const at=(b,n)=>BigInt('0x'+b.subarray(n,n+32).toString('hex'));
const little=(b,n,len=8)=>BigInt('0x'+Buffer.from(b.subarray(n,n+len)).reverse().toString('hex'));
const le=n=>Buffer.from(word(n)).reverse();
const errorSelector=name=>Buffer.from(keccak_256(Buffer.from(name))).subarray(0,4).toString('hex');
const artifact=name=>build.complete.contracts[name+'.sol'][name];
const coordinator=artifact('SpartanSparseCoordinator');
const selector=Buffer.from(coordinator.evm.methodIdentifiers['checkIncompleteSparse(bytes,bytes,bytes,bytes,uint256[4][2])'],'hex');
function fixture(name){const base='test/spartan/'+name;return Object.fromEntries(['key','setup','inputs','proof','g2-affine'].map(k=>[k,fs.readFileSync(base+'/'+k+'.bin')]));}
function calldata(f){let offset=384;const heads=[],tails=[];for(const arg of [f.key,f.setup,f.inputs,f.proof]){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([selector,...heads,f['g2-affine'],...tails]);}
const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});
const gasLimit=30_000_000n;const deployments=[];const modules=[];
async function deploy(name,args=Buffer.alloc(0),reject=false){const result=await evm.runCall({data:Buffer.concat([Buffer.from(artifact(name).evm.bytecode.object,'hex'),args]),gasLimit});deployments.push({name,reject,executionGas:result.execResult.executionGasUsed.toString(),exception:result.execResult.exceptionError?.error??null,address:result.createdAddress?.toString()});if(reject){assert.equal(result.execResult.exceptionError?.error,'revert');return;}assert.equal(result.execResult.exceptionError,undefined,name+' deployment');const code=await evm.stateManager.getCode(result.createdAddress);assert.equal(Buffer.from(code).length,artifact(name).evm.deployedBytecode.object.length/2);return result.createdAddress;}
for(const name of build.names.slice(0,3))modules.push(await deploy(name));
const messageStack=[];let moduleCalls=[];
evm.events.on('beforeMessage',m=>{messageStack.push({address:m.to?.toString(),inputBytes:m.data.length,gasForwarded:m.gasLimit.toString()});});
evm.events.on('afterMessage',result=>{const m=messageStack.pop();if(m && modules.some(a=>a.toString()===m.address))moduleCalls.push({...m,executionGas:result.execResult.executionGasUsed.toString(),outputBytes:result.execResult.returnValue.length});});
const addressWord=a=>Buffer.concat([Buffer.alloc(12),a.bytes]);
const configs=new Map();
async function configured(f){const key=digest(f.key).toString('hex');if(!configs.has(key))configs.set(key,await deploy('SpartanSparseCoordinator',Buffer.concat([digest(f.key),digest(f.setup),...modules.map(addressWord)])));return configs.get(key);}
const records=[];
async function run(name,f,reject=false,{faultModule=null,externalCheckpoint=false}={}){
 const to=await configured(f);await resetWarmth(evm,modules);
 let original;
 if(faultModule!==null){const address=modules[faultModule].toString().slice(2);original=evm.precompiles.get(address);evm.precompiles.set(address,()=>({executionGasUsed:0n,returnValue:Uint8Array.of(1)}));}
 const data=externalCheckpoint?Buffer.concat([Buffer.from('ffffffff','hex'),calldata(f)]):calldata(f);
 moduleCalls=[];const result=await evm.runCall({to,data,gasLimit});
 if(faultModule!==null){const address=modules[faultModule].toString().slice(2);if(original)evm.precompiles.set(address,original);else evm.precompiles.delete(address);}
 const record={name,moduleCalls,coordinatorExecutionGas:(result.execResult.executionGasUsed-moduleCalls.reduce((sum,c)=>sum+BigInt(c.executionGas),0n)).toString(),accepted:!result.execResult.exceptionError,executionGas:result.execResult.executionGasUsed.toString(),exception:result.execResult.exceptionError?.error??null,returnHex:Buffer.from(result.execResult.returnValue).toString('hex'),calldataBytes:data.length};records.push(record);
 if(reject)assert.equal(record.exception,'revert',name);else {assert.equal(record.exception,null,name);assert.deepEqual(moduleCalls.map(c=>c.address),modules.map(a=>a.toString()),name+' all three same-call stages');}
 return Buffer.from(result.execResult.returnValue);
}
function vector(b,n){return Array.from({length:Number(at(b,n))},(_,j)=>at(b,n+32*(j+1)));}
function decode(b){const root=Number(at(b,0));const ops=root+Number(at(b,root+64));const mem=root+Number(at(b,root+96));return {
 values:[at(b,root),at(b,root+32),...vector(b,ops+Number(at(b,ops))),...vector(b,ops+Number(at(b,ops+32))),...Array.from({length:9},(_,j)=>at(b,ops+64+32*j)),...vector(b,mem+Number(at(b,mem))),...vector(b,mem+Number(at(b,mem+32)))],
 state:b.subarray(root+224,root+256).toString('hex'),nextScalar:at(b,root+160),nextTag:at(b,root+192),
 };}
for(const name of ['toy','empty','empty-public','zero-products']){
 const f=fixture(name),native=JSON.parse(fs.readFileSync('test/spartan/'+name+'/sparse.json'));
 assert.equal(native.full_native_acceptance,name!=='zero-products');const result=decode(await run(name,f));
 assert.deepEqual(result.values,native.values_le.map(x=>BigInt('0x'+Buffer.from(x,'hex').reverse().toString('hex'))),name+' native network claims/points');assert.equal(result.state,native.state,name+' native transcript');
}
// Derive intentional fixture mutation offsets by walking its wire grammar, not duplicating algebra.
function offsets(f){const log=n=>BigInt(n).toString(2).length-1;const r=log(little(f.key,100)),w=log(little(f.key,108)),n=log(little(f.key,116)),l=log(little(f.key,124)),t=log(little(f.key,132)),p=Number(little(f.key,92));let at=84+r*97+96+96*p+(4*(r+t)+2)*32+w*65;const privateValues=at;at+=96+32;const roots=at;at+=16*32;const halves=at;at+=6*32;const layers=[];for(const [network,depth,width]of [['ops',n,12],['mem',l,4]])for(let j=0;j<depth;j++){const rounds=at;at+=97*j;const ends=at;at+=2*width*32;const dots=at;if(network==='ops'&&j===depth-1)at+=18*32;layers.push({network,j,rounds,ends,dots});}return {privateValues,roots,halves,layers,next:at};}
const toy=fixture('toy');const clone=f=>Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.from(v)]));
for(const [name,where]of [['root',o=>o.roots],['half-sum',o=>o.halves],['ops-zero-round-terminal',o=>o.layers.find(x=>x.network==='ops'&&x.j===0).ends],['mem-zero-round-terminal',o=>o.layers.find(x=>x.network==='mem'&&x.j===0).ends],['bottom-dot',o=>o.layers.findLast(x=>x.network==='ops').dots],['memory-round',o=>o.layers.find(x=>x.network==='mem'&&x.j===1).rounds+1]]){const f=clone(toy);f.proof[where(offsets(f))]^=1;await run(name,f,true);const expected=name==='root'?'RootRelation()':name==='half-sum'?'HalfSum()':'ProductTerminal(bool,uint256)';assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector(expected),name+' failing equation');}
// Root products may contain zero. Perturb a zero-product terminal into a nonzero product coherently.
{const f=fixture('zero-products');const o=offsets(f);const end=o.layers[0].ends;le(1).copy(f.proof,end);le(1).copy(f.proof,end+32);await run('zero-products-terminal-mismatch',f,true);assert.equal(records.at(-1).returnHex.slice(0,8),errorSelector('ProductTerminal(bool,uint256)'));}
// Slab claims and private witness binding are deliberately not authenticated by this gate.
{const f=clone(toy);f.proof[offsets(f).next]^=1;await run('slab-opening-still-pending',f);}
await deploy('SpartanSparseCoordinator',Buffer.concat([digest(toy.key),digest(toy.setup),addressWord(modules[1]),addressWord(modules[0]),addressWord(modules[2])]),true);
for(let j=0;j<3;j++){const original=await evm.stateManager.getCode(modules[j]);await evm.stateManager.putCode(modules[j],Uint8Array.of(0));await run('changed-module-'+j,toy,true);await evm.stateManager.putCode(modules[j],original);}
for(let j=0;j<3;j++)await run('malformed-module-return-'+j,toy,true,{faultModule:j});
await run('external-checkpoint-selector-absent',toy,true,{externalCheckpoint:true});
fs.mkdirSync('evidence/sparse',{recursive:true});fs.writeFileSync('evidence/sparse/results.json',JSON.stringify({scope:'same-call fixed-code modules; sparse roots/product networks only, slab/hash-leaf/witness obligations pending',settings:build.settings,node:process.version,codeHashes:build.hashes,sourceHashes:Object.fromEntries(Object.entries(build.sources).map(([name,v])=>[name,hash(Buffer.from(v.content))])),runtimeBytes:Object.fromEntries([...build.names,'SpartanSparseCoordinator'].map(n=>[n,artifact(n).evm.deployedBytecode.object.length/2])),deployments,records,gasLimit:gasLimit.toString(),warmness:'Each call resets journal; precompiles05/06/07/08/09 explicitly warm; module accounts initially cold, EXTCODEHASH warms them before STATICCALL',gasScope:'actual EVM CREATE/code-deposit and call execution; excludes transaction intrinsic/calldata gas, no chain deployment'},null,2)+'\n');
console.log(JSON.stringify({deployments,records:records.map(({returnHex,...r})=>r)},null,2));
