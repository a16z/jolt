// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanAlgebra} from "./SpartanAlgebra.sol";
import {SpartanPrefixModule} from "./SpartanPrefixModule.sol";
import {BlakeHashSponge, Bn254WideBlake} from "./BlakeTranscript.sol";

/// @notice Product networks only; slab PCS, hash leaves and witness relation remain unchecked.
contract SpartanSparseModule {
    uint256 constant R=SpartanInputs.R;
    error RootRelation();
    error HalfSum();
    error ProductTerminal(bool operations,uint256 layer);
    struct Cursor {uint256 scalar;uint256 tag;}
    struct NetworkResult {
        uint256[] point;
        uint256[] trees;
        uint256[3][3] dots;
    }
    struct MemoryResult {uint256[] point;uint256[] trees;}
    struct IncompleteResult {
        uint256 alpha;
        uint256 beta;
        NetworkResult operations;
        MemoryResult memoryNetwork;
        BlakeHashSponge.State transcript;
        uint256 nextScalar;
        uint256 nextTag;
        bytes32 transcriptState;
    }
    function take(SpartanInputs.Decoded memory d,uint256 at,uint256 count) private pure returns(uint256[] memory a) {
        a=new uint256[](count);for(uint256 j;j<count;++j)a[j]=d.scalars[at+j];
    }
    function network(SpartanInputs.Decoded memory d,Cursor memory cursor,uint256[] memory claims,uint256[] memory halves,bool operations)
        private view returns(NetworkResult memory result) {
        uint256 layers=operations?d.shape.n:d.shape.l;
        uint256 width=operations?12:4;
        SpartanAlgebra.label(d.transcript,operations?bytes32("product-ops-v2"):bytes32("product-mem-v2"));
        result.point=new uint256[](0);
        for(uint256 layer;layer<layers;++layer) {
            bool bottom=operations && layer+1==layers;
            SpartanAlgebra.label(d.transcript,bytes32("product-layer"));
            SpartanAlgebra.append(d.transcript,layer);
            SpartanAlgebra.label(d.transcript,bytes32("layer-weights"));
            uint256[] memory weights=new uint256[](width+(bottom?6:0));
            uint256 claim;
            for(uint256 j;j<weights.length;++j) {
                weights[j]=Bn254WideBlake.challenge(d.transcript);
                claim=addmod(claim,mulmod(weights[j],j<width?claims[j]:halves[j-width],R),R);
            }
            SpartanAlgebra.Sumcheck memory sc=SpartanAlgebra.sumcheck(d,cursor.scalar,cursor.tag,layer,3,claim);
            cursor.scalar+=3*layer;cursor.tag+=layer;
            uint256[] memory ends=take(d,cursor.scalar,2*width);cursor.scalar+=2*width;
            uint256[] memory dots=take(d,cursor.scalar,bottom?18:0);cursor.scalar+=dots.length;
            uint256 eq=SpartanAlgebra.eq(result.point,sc.point);
            uint256 expected;
            for(uint256 j;j<width;++j)
                expected=addmod(expected,mulmod(mulmod(weights[j],eq,R),mulmod(ends[2*j],ends[2*j+1],R),R),R);
            for(uint256 j;j<dots.length/3;++j)
                expected=addmod(expected,mulmod(weights[width+j],mulmod(mulmod(dots[3*j],dots[3*j+1],R),dots[3*j+2],R),R),R);
            // Layer zero has no messages: the initial weighted claim still must equal its terminal.
            if(sc.value!=expected)revert ProductTerminal(operations,layer);
            SpartanAlgebra.values(d.transcript,bytes32("product-ends"),ends);
            if(bottom)SpartanAlgebra.values(d.transcript,bytes32("dot-ends"),dots);
            SpartanAlgebra.label(d.transcript,bytes32("layer-eta"));
            uint256 eta=Bn254WideBlake.challenge(d.transcript);
            uint256 oneMinus=SpartanAlgebra.sub(1,eta);
            for(uint256 j;j<width;++j)claims[j]=addmod(mulmod(oneMinus,ends[2*j],R),mulmod(eta,ends[2*j+1],R),R);
            result.point=new uint256[](layer+1);result.point[0]=eta;
            for(uint256 j;j<layer;++j)result.point[j+1]=sc.point[j];
            if(bottom)for(uint256 m;m<3;++m)for(uint256 j;j<3;++j)
                result.dots[m][j]=addmod(mulmod(oneMinus,dots[6*m+j],R),mulmod(eta,dots[6*m+3+j],R),R);
        }
        result.trees=claims;
    }
    function check(SpartanInputs.Decoded memory d,SpartanPrefixModule.Result memory prefix,bytes calldata proof)
        external view returns(IncompleteResult memory result) {
        d.transcript=prefix.transcript;
        uint256[] memory values=take(d,d.privateValuesOffset,3);
        SpartanAlgebra.label(d.transcript,bytes32("matrix-query-v2"));
        SpartanAlgebra.values(d.transcript,bytes32("matrix-x"),prefix.outer.point);
        SpartanAlgebra.values(d.transcript,bytes32("matrix-y"),prefix.inner.point);
        SpartanAlgebra.values(d.transcript,bytes32("matrix-values"),values);
        SpartanAlgebra.label(d.transcript,bytes32("matrix-derefs"));
        Bn254WideBlake.append(d.transcript,proof[d.sparseCommitmentOffset:d.sparseCommitmentOffset+32]);
        SpartanAlgebra.label(d.transcript,bytes32("memory-hash"));
        result.alpha=Bn254WideBlake.challenge(d.transcript);result.beta=Bn254WideBlake.challenge(d.transcript);
        uint256[] memory roots=take(d,d.privateValuesOffset+3,16);
        uint256[] memory halves=take(d,d.privateValuesOffset+19,6);
        for(uint256 axis;axis<2;++axis) {
            uint256 k=axis*8;
            uint256 left=mulmod(mulmod(roots[k],roots[k+4],R),mulmod(roots[k+5],roots[k+6],R),R);
            uint256 right=mulmod(mulmod(roots[k+1],roots[k+2],R),mulmod(roots[k+3],roots[k+7],R),R);
            if(left!=right)revert RootRelation();
        }
        for(uint256 m;m<3;++m)if(addmod(halves[2*m],halves[2*m+1],R)!=values[m])revert HalfSum();
        SpartanAlgebra.values(d.transcript,bytes32("memory-roots"),roots);
        SpartanAlgebra.values(d.transcript,bytes32("dot-halves"),halves);
        uint256[] memory ops=new uint256[](12);uint256[] memory mem=new uint256[](4);
        for(uint256 axis;axis<2;++axis) {
            for(uint256 j;j<6;++j)ops[axis*6+j]=roots[axis*8+1+j];
            mem[2*axis]=roots[8*axis];mem[2*axis+1]=roots[8*axis+7];
        }
        Cursor memory cursor=Cursor(d.networkScalarOffset,d.networkTagOffset);
        result.operations=network(d,cursor,ops,halves,true);
        NetworkResult memory memoryResult=network(d,cursor,mem,new uint256[](0),false);
        result.memoryNetwork=MemoryResult(memoryResult.point,memoryResult.trees);
        result.transcript=d.transcript;result.nextScalar=cursor.scalar;result.nextTag=cursor.tag;
        bytes memory peek=BlakeHashSponge.peek(d.transcript);bytes32 state;
        assembly ("memory-safe"){state:=mload(add(peek,32))}
        result.transcriptState=state;
    }
}
