// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Vm} from "forge-std/Vm.sol";
import {Test} from "forge-std/Test.sol";
import "forge-std/console.sol";
import {Jolt} from "../../src/reference/JoltTypes.sol";
import {Fr} from "../../src/reference/Fr.sol";

contract TestBase is Test {

    function getProofData () internal returns (Jolt.BatchedGrandProductProof memory) {
        string[] memory cmds = new string[](3);
        cmds[0] = "sh";
        cmds[1] = "script/run.sh"; 
        cmds[2] = "proofs";
        bytes memory result = vm.ffi(cmds);
        (Jolt.BatchedGrandProductProof memory decodedProof) = abi.decode(result, (Jolt.BatchedGrandProductProof));

        return decodedProof;
    }

    function getClaims () internal returns (Fr[] memory) {

        string[] memory cmds = new string[](3);
        cmds[0] = "sh";
        cmds[1] = "script/run.sh"; 
        cmds[2] = "claims";
        bytes memory result = vm.ffi(cmds);
        (Fr[] memory claims) = abi.decode(result, (Fr[]));

        return claims;
    }
}
