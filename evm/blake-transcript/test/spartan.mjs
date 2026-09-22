import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import solc from 'solc';
import { blake2b } from '@noble/hashes/blake2.js';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { createEVM } from '@ethereumjs/evm';
const digest=b=>Buffer.from(blake2b(b,{dkLen:32}));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const paths=['BlakeTranscript.sol','SpartanInputs.sol'];
const sources=Object.fromEntries(paths.map(p=>[p,{content:fs.readFileSync('contracts/'+p,'utf8')}]));
sources['Hash256Harness.sol']={content:'pragma solidity 0.8.30; import {Blake2b512} from "./BlakeTranscript.sol"; contract Hash256Harness { fallback(bytes calldata b) external returns(bytes memory) {return abi.encodePacked(Blake2b512.hash256(b));}}'};
// Value inspection calls the production decoder; this harness does not authenticate or verify a proof.
sources['DecodeValuesHarness.sol']={content:`pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {Blake2b512} from "./BlakeTranscript.sol";
contract DecodeValuesHarness {
    function decodeValues(bytes calldata key, bytes calldata setup, bytes calldata inputs, bytes calldata proof)
        external view returns (uint256, uint256, uint256, uint256, uint256, uint256) {
        SpartanInputs.Decoded memory d = SpartanInputs.decodeAuthenticated(
            Blake2b512.hash256(key), Blake2b512.hash256(setup), key, setup, inputs, proof);
        SpartanInputs.G1 memory last = d.points[d.points.length - 1];
        return (d.scalars[0], d.publicInputs[0], d.points[0].x, d.points[0].y, last.x, last.y);
    }
}`};
const settings={viaIR:true,optimizer:{enabled:true,runs:200},evmVersion:'prague',outputSelection:{'*':{'*':['evm.deployedBytecode','evm.methodIdentifiers'],'':['ast']}}};
const compiled=JSON.parse(solc.compile(JSON.stringify({language:'Solidity',sources,settings})));
for(const e of compiled.errors??[])if(e.severity==='error')throw Error(e.formattedMessage);
const contract=compiled.contracts['SpartanInputs.sol'].SpartanInputBoundary;
const unconfigured=Buffer.from(contract.evm.deployedBytecode.object,'hex');
const refs=contract.evm.deployedBytecode.immutableReferences;
const ast=compiled.sources['SpartanInputs.sol'].ast.nodes.find(n=>n.nodeType==='ContractDefinition'&&n.name==='SpartanInputBoundary');
const names=Object.fromEntries(ast.nodes.filter(n=>n.nodeType==='VariableDeclaration').map(n=>[n.id,n.name]));
assert.deepEqual(Object.keys(refs).map(x=>names[x]).sort(),['expectedKey','expectedSetup']);
function code(keyId,setupId){const b=Buffer.from(unconfigured);for(const [id,positions]of Object.entries(refs))for(const p of positions){assert.equal(p.length,32);(names[id]==='expectedKey'?keyId:setupId).copy(b,p.start);}return b;}
const word=n=>Buffer.from(BigInt(n).toString(16).padStart(64,'0'),'hex');
const le=(n,len=32)=>Buffer.from(BigInt(n).toString(16).padStart(len*2,'0'),'hex').reverse();
const selector=Buffer.from(contract.evm.methodIdentifiers['decodeAuthenticated(bytes,bytes,bytes,bytes)'],'hex');
function calldata(args,method=selector){let offset=args.length*32;const heads=[],tails=[];for(const arg of args){heads.push(word(offset));const tail=Buffer.concat([word(arg.length),arg,Buffer.alloc((32-arg.length%32)%32)]);tails.push(tail);offset+=tail.length;}return Buffer.concat([method,...heads,...tails]);}
const records=[];const gasLimit=30_000_000n;
async function execute(name,runtime,data,reject=false){const evm=await createEVM({common:new Common({chain:Mainnet,hardfork:Hardfork.Prague})});for(const a of [5,9])evm.journal.addAlwaysWarmAddress(a.toString(16).padStart(40,'0'));const res=await evm.runCode({code:runtime,data,gasLimit});records.push({name,accepted:!res.exceptionError,executionGas:res.executionGasUsed.toString(),calldataBytes:data.length,returnHex:Buffer.from(res.returnValue).toString('hex'),exception:res.exceptionError?.error??null});if(reject)assert.equal(res.exceptionError?.error,'revert',name);else assert.equal(res.exceptionError,undefined,name+': '+res.exceptionError);return Buffer.from(res.returnValue);}
const hashCode=Buffer.from(compiled.contracts['Hash256Harness.sol'].Hash256Harness.evm.deployedBytecode.object,'hex');
for(const v of JSON.parse(fs.readFileSync('test/spartan/blake256.json'))){const b=Buffer.from(v.input,'hex');assert.equal(digest(b).toString('hex'),v.digest);assert.equal((await execute('blake256-'+b.length,hashCode,b)).toString('hex'),v.digest);}
function fixture(name){const dir='test/spartan/'+name;return {key:fs.readFileSync(dir+'/key.bin'),setup:fs.readFileSync(dir+'/setup.bin'),inputs:fs.readFileSync(dir+'/inputs.bin'),proof:fs.readFileSync(dir+'/proof.bin'),prefix:fs.existsSync(dir+'/prefix.json')?JSON.parse(fs.readFileSync(dir+'/prefix.json')):null};}
function readWord(bytes,at){return BigInt('0x'+bytes.subarray(at,at+32).toString('hex'));}
const fixtures=Object.fromEntries(['toy','empty','norm'].map(n=>[n,fixture(n)]));
async function run(name,f,reject=false,keyPolicy=digest(f.key),setupPolicy=digest(f.setup)){
 if(name!=='norm') {const dir='evidence/spartan/corpus/'+name;fs.mkdirSync(dir,{recursive:true});for(const k of ['key','setup','inputs','proof'])fs.writeFileSync(dir+'/'+k+'.bin',f[k]);fs.writeFileSync(dir+'/key-id.bin',keyPolicy);fs.writeFileSync(dir+'/setup-id.bin',setupPolicy);fs.writeFileSync(dir+'/expected.txt',reject?'reject':'decode');}
 return execute(name,code(keyPolicy,setupPolicy),calldata([f.key,f.setup,f.inputs,f.proof]),reject);
}
for(const [name,f]of Object.entries(fixtures)){
 const out=await run(name,f);assert.equal(out.subarray(0,32).toString('hex'),digest(f.key).toString('hex'));
 if(f.prefix){assert.equal(out.subarray(192,224).toString('hex'),f.prefix.state,name+' native prefix');const at=Number(readWord(out,224));assert.equal(readWord(out,at),BigInt(f.prefix.tau_le.length));for(let j=0;j<f.prefix.tau_le.length;j++)assert.equal(readWord(out,at+32+32*j),BigInt('0x'+Buffer.from(f.prefix.tau_le[j],'hex').reverse().toString('hex')),name+' tau'+j);}
 if(name==='toy'){assert.equal(readWord(out,32),248n);assert.equal(readWord(out,64),36n);assert.equal(readWord(out,96),9n);}
 if(name==='norm'){assert.equal(readWord(out,32),1595n);assert.equal(readWord(out,64),100n);assert.equal(readWord(out,96),255n);}
 if(name==='empty'){assert(readWord(out,128)>0n,'honest shortened rounds');assert(readWord(out,160)>0n,'identity commitments accepted');}
}
const toy=fixtures.toy;const clone=f=>Object.fromEntries(Object.entries(f).map(([k,v])=>[k,Buffer.isBuffer(v)?Buffer.from(v):v]));
for(const field of ['key','setup','inputs','proof'])for(const op of ['truncate','trail']){const f=clone(toy);f[field]=op==='truncate'?f[field].subarray(0,-1):Buffer.concat([f[field],Buffer.of(0)]);await run(field+'-'+op,f,true,digest(toy.key),digest(toy.setup));}
for(const [name,offset,value]of [['bad-magic',0,0],['bad-version',16,2],['bad-key-id',20,0],['width-zero',84,0],['width-large',84,4]]){const f=clone(toy);f.proof[offset]=value;await run(name,f,true);}
const r=21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const valueContract=compiled.contracts['DecodeValuesHarness.sol'].DecodeValuesHarness;
const valueCode=Buffer.from(valueContract.evm.deployedBytecode.object,'hex');
const valueSelector=Buffer.from(valueContract.evm.methodIdentifiers['decodeValues(bytes,bytes,bytes,bytes)'],'hex');
async function values(name,f,reject=false){return execute(name,valueCode,calldata([f.key,f.setup,f.inputs,f.proof],valueSelector),reject);}
const scalarValues=[0n,1n,r-1n,BigInt('0x102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20'),r,r+1n,(1n<<256n)-1n];
for(let byte=0;byte<32;byte++)scalarValues.push(1n<<BigInt(8*byte));
for(const [j,value]of scalarValues.entries()){
 const f=clone(toy);le(value).copy(f.proof,85);le(value).copy(f.inputs,0);
 const out=await values('decoded-scalar-'+j,f,value>=r);
 if(value<r){assert.equal(readWord(out,0),value);assert.equal(readWord(out,32),value);}
}
const q=21888242871839275222246405745257275088696311157297823662689037894645226208583n;
for(const [name,encoding,x,y,reject]of [
 ['generator-small',1n,1n,2n,false],['generator-large',1n+(1n<<255n),1n,q-2n,false],
 ['infinity',1n<<254n,0n,0n,false],['infinity-nonzero',1n+(1n<<254n),0n,0n,true],
 ['invalid-flags',3n<<254n,0n,0n,true],['x-modulus',q,0n,0n,true],['noncurve',0n,0n,0n,true],
]){
 const f=clone(toy);le(encoding).copy(f.proof,52);le(encoding).copy(f.proof,f.proof.length-32);
 const out=await values('decoded-point-'+name,f,reject);
 if(!reject){assert.equal(readWord(out,64),x);assert.equal(readWord(out,96),y);assert.equal(readWord(out,128),x);assert.equal(readWord(out,160),y);}
}
for(const field of ['proof','inputs']){const f=clone(toy);le(r).copy(f[field],field==='proof'?85:0);await run(field+'-field-modulus',f,true);}
for(const [name,encoding]of [['noncurve',Buffer.alloc(32)],['noncanonical-infinity',le((1n<<254n)+1n)],['invalid-flags',le(3n<<254n)]]){const f=clone(toy);encoding.copy(f.proof,52);await run(name,f,true);}
{const f=clone(toy);f.proof[84]=1;f.proof.fill(0,117,181);f.proof[117]=1;await run('nonzero-round-padding',f,true);}
{const f=clone(toy);f.key[284]^=1;await run('unauthenticated-key',f,true,digest(toy.key));}
{const f=clone(toy);f.setup[16]^=1;await run('unauthenticated-setup',f,true,digest(toy.key),digest(toy.setup));}
{const f=clone(toy);for(const at of [76,100,124])le(1n<<33n,8).copy(f.key,at);await run('authenticated-overdepth-policy',f,true);}
{const f=clone(toy);for(const [at,v]of [[84,1030],[92,1025],[108,8],[124,8],[132,4096]])le(v,8).copy(f.key,at);await run('authenticated-overpublic-policy',f,true);}
// This is deliberately not a SNARK acceptance API.
{const f=clone(toy);f.proof[85]^=1;await run('canonical-algebra-tamper-not-verified',f);}
const report={scope:'authenticated canonical input decoding and native-v2 initial transcript only; NO proof verification',node:process.version,solc:solc.version(),settings,hardfork:'Prague',ethereumjs:'10.1.0',gasLimit:gasLimit.toString(),prewarmedAddresses:['0x05 MODEXP','0x09 Blake2F'],context:'fresh EVM per case; immutable policy slots patched to constructor-equivalent values; no deployment',gasScope:'bytecode execution, memory, and precompiles only; excludes intrinsic/calldata transaction gas, deployment and unimplemented algebra/PCS',sourceHashes:Object.fromEntries(Object.entries(sources).map(([p,s])=>[p,sha(Buffer.from(s.content))])),unconfiguredRuntimeBytes:unconfigured.length,unconfiguredRuntimeSha256:sha(unconfigured),fixtureHashes:Object.fromEntries(Object.entries(fixtures).map(([n,f])=>[n,Object.fromEntries(['key','setup','inputs','proof'].map(k=>[k,sha(f[k])]))])),records};
fs.mkdirSync('evidence/spartan',{recursive:true});fs.writeFileSync('evidence/spartan/results.json',JSON.stringify(report,null,2)+'\n');fs.writeFileSync('evidence/spartan/runtime.hex',unconfigured.toString('hex')+'\n');fs.writeFileSync('evidence/spartan/selector.json',JSON.stringify({selector:selector.toString('hex')}));fs.writeFileSync('evidence/spartan/immutable-references.json',JSON.stringify({refs,names},null,2)+'\n');console.log(JSON.stringify(report,null,2));
