pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {Fr, FrLib} from "../src/subprotocols/Fr.sol";
import {Transcript, FiatShamirTranscript} from "../src/subprotocols/FiatShamirTranscript.sol";
import {SpartanVerifier, SpartanProof} from "../src/subprotocols/SpartanVerifier.sol";

import "forge-std/console.sol";

struct VK {
    uint256 VK_g1_x;
    uint256 VK_g1_y;
    uint256[] VK_g2;
    uint256[] VK_beta_g2;
}

// Need to have a version which inits the immutables
contract DeployableSpartan is SpartanVerifier {
    constructor(VK memory vk) {
        VK_g1_x = vk.VK_g1_x;
        VK_g1_y = vk.VK_g1_y;
        VK_g2_x_c0 = vk.VK_g2[0];
        VK_g2_x_c1 = vk.VK_g2[1];
        VK_g2_y_c0 = vk.VK_g2[2];
        VK_g2_y_c1 = vk.VK_g2[3];
        VK_beta_g2_x_c0 = vk.VK_beta_g2[0];
        VK_beta_g2_x_c1 = vk.VK_beta_g2[1];
        VK_beta_g2_y_c0 = vk.VK_beta_g2[2];
        VK_beta_g2_y_c1 = vk.VK_beta_g2[3];
    }
}

contract TestSpartanVerifier is TestBase {
    using FrLib for Fr;

    struct SpartanExample {
        SpartanProof proof;
        VK vk;
        uint256[] commitments;
    }

    function testSpartanVerifier() public {
        // We use internal state of a jolt transcript at the point of verifying
        bytes32[] memory internal_data = new bytes32[](5);
        internal_data[0] = 0xc60338a7fc9488db12fd3c81b6719b62a2038420ecb8556bbf256a0b242d15b2;
        internal_data[1] = bytes32(uint256(12932));
        Transcript memory transcript = Transcript(internal_data);

        string[] memory cmds = new string[](2);
        cmds[0] = "sh";
        cmds[1] = "script/spartan_example.sh";
        bytes memory result = vm.ffi(cmds);
        (SpartanExample memory decodedProof) = abi.decode(result, (SpartanExample));

        DeployableSpartan spartan = new DeployableSpartan(decodedProof.vk);

        assert(spartan.verifySpartanR1CS(decodedProof.proof, decodedProof.commitments, transcript, 16, 17, 131072));
    }
}
