// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanAlgebra} from "./SpartanAlgebra.sol";
import {SpartanKzgPairing} from "./SpartanPublicOpening.sol";
import {SpartanPrefixModule} from "./SpartanPrefixModule.sol";
import {SpartanSparseModule} from "./SpartanSparseModule.sol";
import {BlakeHashSponge, Bn254WideBlake} from "./BlakeTranscript.sol";

/// @notice Sparse leaves and final witness checks for the fixed-code coordinator.
/// @dev Standalone module calls do not authenticate their supplied checkpoints.
contract SpartanLeafModule {
    uint256 constant R=SpartanInputs.R;
    error ZeroSlab();
    error WitnessProduct();
    error DotLeaf();
    error OperationLeaf();
    error MemoryLeaf();
    struct IncompleteResult {
        BlakeHashSponge.State transcript;
        bytes32 transcriptState;
    }
    function slab(SpartanInputs.Decoded memory d,bytes calldata key,bytes calldata setup,bytes calldata proof,
        uint256[4][2] calldata g2,SpartanSparseModule.IncompleteResult memory sparse,uint256 phase)
        private view returns(uint256[] memory evaluations) {
        uint256 count=phase==0?6:phase==1?16:2;
        uint256 selectorSize=phase==0?3:phase==1?4:1;
        evaluations=new uint256[](count);
        for(uint256 j;j<count;++j)evaluations[j]=d.scalars[d.slabValues[phase]+j];
        if(phase==1 && evaluations[15]!=0)revert ZeroSlab();
        SpartanAlgebra.values(d.transcript,phase==0?bytes32("deref-evals"):phase==1?bytes32("ops-evals"):bytes32("audit-evals"),evaluations);
        SpartanAlgebra.label(d.transcript,phase==0?bytes32("deref-selector"):phase==1?bytes32("ops-selector"):bytes32("audit-selector"));
        uint256[] memory selector=new uint256[](selectorSize);
        uint256[] memory suffix=phase==2?sparse.memoryNetwork.point:sparse.operations.point;
        uint256[] memory point=new uint256[](selectorSize+suffix.length);
        for(uint256 j;j<selectorSize;++j){selector[j]=Bn254WideBlake.challenge(d.transcript);point[j]=selector[j];}
        for(uint256 j;j<suffix.length;++j)point[selectorSize+j]=suffix[j];
        uint256 value;
        for(uint256 j;j<count;++j)value=addmod(value,mulmod(evaluations[j],SpartanAlgebra.eqIndex(selector,j),R),R);
        bytes calldata commitment=phase==0?proof[d.sparseCommitmentOffset:d.sparseCommitmentOffset+32]:key[380+phase*32:412+phase*32];
        SpartanInputs.G1 memory decoded=phase==0?d.points[d.dereferencePoint]:d.commitments[phase];
        SpartanAlgebra.PendingKzg memory opening=SpartanAlgebra.opening(d,setup,proof,commitment,decoded,point,value,d.slabOpenings[phase]);
        SpartanKzgPairing.check(d,opening,setup,g2);
    }
    function leaves(SpartanInputs.Decoded memory d,SpartanPrefixModule.Result memory prefix,
        SpartanSparseModule.IncompleteResult memory sparse,bytes calldata key,bytes calldata setup,bytes calldata proof,uint256[4][2] calldata g2)
        private view returns(IncompleteResult memory result) {
        d.transcript=sparse.transcript;
        uint256[] memory dr=slab(d,key,setup,proof,g2,sparse,0);
        uint256[] memory op=slab(d,key,setup,proof,g2,sparse,1);
        uint256[] memory audit=slab(d,key,setup,proof,g2,sparse,2);
        uint256 alpha2=mulmod(sparse.alpha,sparse.alpha,R);
        for(uint256 matrix;matrix<3;++matrix) {
            if(sparse.operations.dots[matrix][0]!=dr[matrix] || sparse.operations.dots[matrix][1]!=dr[3+matrix]
                || sparse.operations.dots[matrix][2]!=op[12+matrix])revert DotLeaf();
            for(uint256 axis;axis<2;++axis) {
                uint256 read=SpartanAlgebra.sub(addmod(addmod(op[6*axis+matrix],mulmod(sparse.alpha,dr[3*axis+matrix],R),R),
                    mulmod(alpha2,op[6*axis+3+matrix],R),R),sparse.beta);
                if(sparse.operations.trees[6*axis+matrix]!=read || sparse.operations.trees[6*axis+3+matrix]!=addmod(read,alpha2,R))revert OperationLeaf();
            }
        }
        // IdentityPolynomial evaluates the MSB-first binary index, not its bits reversed.
        uint256 identity;
        for(uint256 j;j<sparse.memoryNetwork.point.length;++j)identity=addmod(mulmod(identity,2,R),sparse.memoryNetwork.point[j],R);
        for(uint256 axis;axis<2;++axis) {
            uint256[] memory query=axis==0?prefix.outer.point:prefix.inner.point;
            uint256[] memory padded=new uint256[](sparse.memoryNetwork.point.length);
            for(uint256 j;j<query.length;++j)padded[padded.length-query.length+j]=query[j];
            uint256 init=SpartanAlgebra.sub(addmod(identity,mulmod(sparse.alpha,SpartanAlgebra.eq(padded,sparse.memoryNetwork.point),R),R),sparse.beta);
            if(sparse.memoryNetwork.trees[2*axis]!=init || sparse.memoryNetwork.trees[2*axis+1]!=addmod(init,mulmod(alpha2,audit[axis],R),R))revert MemoryLeaf();
        }
        result.transcript=d.transcript;
        bytes memory peek=BlakeHashSponge.peek(d.transcript);bytes32 state;
        assembly ("memory-safe"){state:=mload(add(peek,32))}
        result.transcriptState=state;
    }
    function check(SpartanInputs.Decoded memory d,SpartanPrefixModule.Result memory prefix,
        SpartanSparseModule.IncompleteResult memory sparse,bytes calldata key,bytes calldata setup,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(IncompleteResult memory) {
        return leaves(d,prefix,sparse,key,setup,proof,g2);
    }
    /// @dev Complete algebra only for coordinator-produced inputs; standalone checkpoints are unauthenticated.
    function checkFinal(SpartanInputs.Decoded memory d,SpartanPrefixModule.Result memory prefix,
        SpartanSparseModule.IncompleteResult memory sparse,bytes calldata key,bytes calldata setup,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(bytes32 state) {
        IncompleteResult memory leaf=leaves(d,prefix,sparse,key,setup,proof,g2);
        d.transcript=leaf.transcript;
        uint256 linear;
        for(uint256 j;j<3;++j)linear=addmod(linear,mulmod(prefix.weights[j],d.scalars[d.privateValuesOffset+j],R),R);
        uint256 value=d.scalars[d.witnessValue];
        if(prefix.inner.value!=mulmod(linear,value,R))revert WitnessProduct();
        SpartanAlgebra.label(d.transcript,bytes32("witness-evaluation"));SpartanAlgebra.append(d.transcript,value);
        SpartanAlgebra.PendingKzg memory opening=SpartanAlgebra.opening(d,setup,proof,proof[52:84],d.points[0],prefix.inner.point,value,d.witnessOpening);
        SpartanKzgPairing.check(d,opening,setup,g2);
        bytes memory peek=BlakeHashSponge.peek(d.transcript);
        assembly ("memory-safe"){state:=mload(add(peek,32))}
    }

}
