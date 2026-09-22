// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {SpartanInputs} from "./SpartanInputs.sol";
import {SpartanDecodeModule} from "./SpartanDecodeModule.sol";
import {SpartanPrefixModule} from "./SpartanPrefixModule.sol";
import {SpartanSparseModule} from "./SpartanSparseModule.sol";
import {SpartanLeafModule} from "./SpartanLeafModule.sol";
import {SpartanModulePolicy} from "./SpartanModulePolicy.sol";

/// @notice Same-call authenticated sparse checkpoint. Final witness product/opening remain.
contract SpartanLeafCoordinator {
    error ModuleIdentity();
    bytes32 public immutable expectedKey;
    bytes32 public immutable expectedSetup;
    address public immutable decodeModule;
    address public immutable prefixModule;
    address public immutable sparseModule;
    address public immutable leafModule;
    constructor(bytes32 keyId,bytes32 setupId,address decodeAddress,address prefixAddress,address sparseAddress,address leafAddress) {
        requireCode(decodeAddress,SpartanModulePolicy.DECODE);
        requireCode(prefixAddress,SpartanModulePolicy.PREFIX);
        requireCode(sparseAddress,SpartanModulePolicy.SPARSE);
        requireCode(leafAddress,SpartanModulePolicy.LEAF);
        expectedKey=keyId;expectedSetup=setupId;
        decodeModule=decodeAddress;prefixModule=prefixAddress;sparseModule=sparseAddress;leafModule=leafAddress;
    }
    function requireCode(address target,bytes32 expected) private view {
        if(target.codehash!=expected)revert ModuleIdentity();
    }
    function checkIncompleteSparseLeaves(bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof,uint256[4][2] calldata g2)
        external view returns(SpartanLeafModule.IncompleteResult memory) {
        requireCode(decodeModule,SpartanModulePolicy.DECODE);
        requireCode(prefixModule,SpartanModulePolicy.PREFIX);
        requireCode(sparseModule,SpartanModulePolicy.SPARSE);
        requireCode(leafModule,SpartanModulePolicy.LEAF);
        SpartanInputs.Decoded memory d=SpartanDecodeModule(decodeModule).decode(expectedKey,expectedSetup,key,setup,inputs,proof);
        SpartanPrefixModule.Result memory prefix=SpartanPrefixModule(prefixModule).check(d,key,setup,proof,g2);
        SpartanSparseModule.IncompleteResult memory sparse=SpartanSparseModule(sparseModule).check(d,prefix,proof);
        return SpartanLeafModule(leafModule).check(d,prefix,sparse,key,setup,proof,g2);
    }
}
