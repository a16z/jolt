import assert from 'node:assert/strict';
// cleanup is async: awaiting it prevents a late reset from racing the next call.
export async function resetWarmth(evm,modules,{precompiles=true}={}){
 await evm.journal.cleanup();
 for(const address of modules)assert.equal(evm.journal.isWarmedAddress(address.bytes),false,'module begins cold');
 for(const n of [5,6,7,8,9]){
  const hex=n.toString(16).padStart(40,'0');const bytes=Buffer.from(hex,'hex');
  assert.equal(evm.journal.isWarmedAddress(bytes),false,'precompile before explicit warming');
  if(precompiles){evm.journal.addAlwaysWarmAddress(hex);assert.equal(evm.journal.isWarmedAddress(bytes),true,'precompile is warm');}
 }
}
