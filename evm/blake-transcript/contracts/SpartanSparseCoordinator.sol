// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanDecodeModule} from "./SpartanDecodeModule.sol";
import {SpartanPrefixModule} from "./SpartanPrefixModule.sol";
import {SpartanSparseModule} from "./SpartanSparseModule.sol";
import {SpartanModulePolicy} from "./SpartanModulePolicy.sol";

/// @notice Same-call product-network checkpoint. Slab, hash-leaf and witness checks remain.
contract SpartanSparseCoordinator {
    error ModuleIdentity();
    bytes32 public immutable expectedKey;
    bytes32 public immutable expectedSetup;
    address public immutable decodeModule;
    address public immutable prefixModule;
    address public immutable sparseModule;
    constructor(bytes32 keyId,bytes32 setupId,address decodeAddress,address prefixAddress,address sparseAddress) {
        requireCode(decodeAddress,SpartanModulePolicy.DECODE);
        requireCode(prefixAddress,SpartanModulePolicy.PREFIX);
        requireCode(sparseAddress,SpartanModulePolicy.SPARSE);
        expectedKey=keyId;expectedSetup=setupId;
        decodeModule=decodeAddress;prefixModule=prefixAddress;sparseModule=sparseAddress;
    }
    function requireCode(address target,bytes32 expected) private view {
        if(target.codehash!=expected)revert ModuleIdentity();
    }
    function checkIncompleteSparse(bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(SpartanSparseModule.IncompleteResult memory) {
        requireCode(decodeModule,SpartanModulePolicy.DECODE);
        requireCode(prefixModule,SpartanModulePolicy.PREFIX);
        requireCode(sparseModule,SpartanModulePolicy.SPARSE);
        SpartanInputs.Decoded memory d=SpartanDecodeModule(decodeModule).decode(expectedKey,expectedSetup,key,setup,inputs,proof);
        SpartanPrefixModule.Result memory prefix=SpartanPrefixModule(prefixModule).check(d,key,setup,proof,g2);
        return SpartanSparseModule(sparseModule).check(d,prefix,proof);
    }
}
